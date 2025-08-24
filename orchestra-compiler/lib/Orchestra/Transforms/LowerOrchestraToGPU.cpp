#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "Orchestra/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/XeGPU/IR/XeGPU.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::orchestra;

namespace {
void lowerToTMA(mlir::gpu::GPUFuncOp funcOp) {
  mlir::OpBuilder builder(funcOp.getContext());
  auto loc = funcOp.getLoc();

  // Create and initialize the mbarrier at the start of the function.
  builder.setInsertionPointToStart(&funcOp.getBody().front());
  auto mbarrierType = nvgpu::MBarrierGroupType::get(
      builder.getContext(),
      gpu::AddressSpaceAttr::get(builder.getContext(),
                                 gpu::AddressSpace::Workgroup));
  auto mbarrier = builder.create<nvgpu::MBarrierCreateOp>(loc, mbarrierType);

  auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto numThreads =
      builder.create<arith::ConstantIndexOp>(loc, 1);  // Placeholder
  builder.create<nvgpu::MBarrierInitOp>(loc,
                                        mbarrier.getResult(),
                                        c0,
                                        numThreads,
                                        /*predicate=*/nullptr);

  // Collect all transfer ops to be rewritten.
  llvm::SmallVector<TransferOp, 4> transferOps;
  funcOp.walk([&](TransferOp op) { transferOps.push_back(op); });

  llvm::DenseMap<mlir::Value, mlir::Value> destToMBarrierToken;

  for (auto op : transferOps) {
    builder.setInsertionPoint(op);
    auto transferLoc = op.getLoc();
    auto source = op.getSource();
    auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());

    if (!sourceType || !sourceType.hasStaticShape()) {
      op.emitError("requires a memref with static shape");
      return;
    }

    // Create TMA descriptor.
    SmallVector<Value, 4> boxDims;
    for (int64_t dim : sourceType.getShape()) {
      boxDims.push_back(
          builder.create<arith::ConstantIndexOp>(transferLoc, dim));
    }
    auto unrankedSourceType = UnrankedMemRefType::get(
        sourceType.getElementType(), sourceType.getMemorySpace());
    auto castedSource =
        builder.create<memref::CastOp>(transferLoc, unrankedSourceType, source);

    // Create a new memref on the GPU in shared memory.
    auto destType = mlir::MemRefType::get(
        sourceType.getShape(),
        sourceType.getElementType(),
        sourceType.getLayout(),
        mlir::gpu::AddressSpaceAttr::get(op.getContext(),
                                         mlir::gpu::AddressSpace::Workgroup));

    auto descriptorType = nvgpu::TensorMapDescriptorType::get(
        builder.getContext(),
        destType,
        nvgpu::TensorMapSwizzleKind::SWIZZLE_NONE,
        nvgpu::TensorMapL2PromoKind::L2PROMO_NONE,
        nvgpu::TensorMapOOBKind::OOB_ZERO,
        nvgpu::TensorMapInterleaveKind::INTERLEAVE_NONE);
    auto descriptor = builder.create<nvgpu::TmaCreateDescriptorOp>(
        transferLoc, descriptorType, castedSource, boxDims);

    auto dest = builder.create<mlir::memref::AllocOp>(transferLoc, destType);

    // Create zero indices for the copy.
    mlir::SmallVector<mlir::Value, 4> indices;
    for (unsigned i = 0; i < sourceType.getRank(); ++i) {
      indices.push_back(
          builder.create<mlir::arith::ConstantIndexOp>(transferLoc, 0));
    }

    // Create an async TMA load.
    builder.create<nvgpu::TmaAsyncLoadOp>(
        transferLoc,
        /*dst=*/dest.getResult(),
        /*barriers=*/mbarrier.getResult(),
        /*tensorMapDescriptor=*/descriptor.getResult(),
        /*coordinates=*/ValueRange(indices),
        /*mbarId=*/c0,
        /*multicastMask=*/nullptr,
        /*predicate=*/nullptr);

    // Arrive at the barrier and get a token.
    auto arriveToken = builder.create<nvgpu::MBarrierArriveOp>(
        transferLoc, mbarrier.getResult(), c0);
    destToMBarrierToken[dest.getResult()] = arriveToken.getToken();

    op.getResult().replaceAllUsesWith(dest.getResult());
    op.erase();
  }

  // Insert wait operations.
  for (auto const &[destBuffer, barrierToken] : destToMBarrierToken) {
    for (mlir::OpOperand &use : destBuffer.getUses()) {
      mlir::Operation *user = use.getOwner();
      if (isa<nvgpu::TmaAsyncLoadOp>(user)) {
        continue;
      }
      mlir::OpBuilder waitBuilder(user);
      auto waitLoc = user->getLoc();
      auto c1_i1 = waitBuilder.create<arith::ConstantIntOp>(waitLoc, 1, 1);

      auto scfWhile = waitBuilder.create<scf::WhileOp>(
          waitLoc,
          TypeRange{},
          ValueRange{},
          [&](OpBuilder &beforeBuilder, Location beforeLoc, ValueRange) {
            auto isComplete = beforeBuilder.create<nvgpu::MBarrierTestWaitOp>(
                beforeLoc,
                builder.getI1Type(),
                mbarrier.getResult(),
                barrierToken,
                c0);
            auto notComplete = beforeBuilder.create<arith::XOrIOp>(
                beforeLoc, isComplete.getWaitComplete(), c1_i1);
            beforeBuilder.create<scf::ConditionOp>(
                beforeLoc, notComplete, ValueRange{});
          },
          [&](OpBuilder &afterBuilder, Location afterLoc, ValueRange) {
            afterBuilder.create<scf::YieldOp>(afterLoc);
          });
    }
  }
}

// This is the original logic for lowering to NVGPU, extracted into its own
// pass.
class LowerOrchestraToNVGPUPass
    : public mlir::PassWrapper<LowerOrchestraToNVGPUPass,
                               mlir::OperationPass<mlir::gpu::GPUFuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToNVGPUPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::orchestra::OrchestraDialect,
                    mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect,
                    mlir::nvgpu::NVGPUDialect,
                    mlir::arith::ArithDialect,
                    mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::gpu::GPUFuncOp funcOp = getOperation();
    auto gpuModule = funcOp->getParentOfType<mlir::gpu::GPUModuleOp>();
    if (!gpuModule) {
      funcOp.emitError("must be nested inside a gpu.module");
      signalPassFailure();
      return;
    }

    auto smArchAttr = gpuModule->getAttrOfType<mlir::IntegerAttr>("sm_arch");
    int smArch = smArchAttr ? smArchAttr.getInt() : 0;

    if (smArch >= 100) {
      lowerToTMA(funcOp);
      return;
    }

    // Map from destination buffer to async token.
    llvm::DenseMap<mlir::Value, mlir::Value> asyncTokens;

    // Collect all transfer ops to be rewritten.
    llvm::SmallVector<TransferOp, 4> transferOps;
    funcOp.walk([&](TransferOp op) { transferOps.push_back(op); });

    mlir::OpBuilder builder(funcOp.getContext());
    for (auto op : transferOps) {
      builder.setInsertionPoint(op);
      auto loc = op.getLoc();
      auto source = op.getSource();
      auto sourceType = mlir::cast<mlir::MemRefType>(source.getType());

      if (!sourceType || !sourceType.hasStaticShape()) {
        op.emitError("requires a memref with static shape");
        signalPassFailure();
        return;
      }

      // Create a new memref on the GPU in shared memory.
      auto destType = mlir::MemRefType::get(
          sourceType.getShape(),
          sourceType.getElementType(),
          sourceType.getLayout(),
          mlir::gpu::AddressSpaceAttr::get(op.getContext(),
                                           mlir::gpu::AddressSpace::Workgroup));
      auto dest = builder.create<mlir::memref::AllocOp>(loc, destType);

      // Create zero indices for the copy.
      mlir::SmallVector<mlir::Value, 4> indices;
      for (unsigned i = 0; i < sourceType.getRank(); ++i) {
        indices.push_back(builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
      }

      // Create an async copy.
      auto asyncCopy = builder.create<mlir::nvgpu::DeviceAsyncCopyOp>(
          loc,
          mlir::nvgpu::DeviceAsyncTokenType::get(op.getContext()),
          dest.getResult(),
          indices,
          source,
          indices,
          builder.getIndexAttr(sourceType.getNumElements()),
          mlir::Value{},
          mlir::UnitAttr{});

      asyncTokens[dest.getResult()] = asyncCopy.getResult();

      op.getResult().replaceAllUsesWith(dest.getResult());
      op.erase();
    }

    // Second phase: insert wait ops.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<mlir::Value, 1>>
        waitsToInsert;
    for (auto &pair : asyncTokens) {
      mlir::Value buffer = pair.first;
      mlir::Value token = pair.second;
      for (mlir::OpOperand &use : buffer.getUses()) {
        mlir::Operation *user = use.getOwner();
        if (isa<mlir::nvgpu::DeviceAsyncCopyOp>(user)) {
          continue;
        }
        waitsToInsert[user].push_back(token);
      }
    }

    for (auto &pair : waitsToInsert) {
      mlir::Operation *user = pair.first;
      llvm::SmallVector<mlir::Value, 1> &tokens = pair.second;
      mlir::OpBuilder wait_builder(user);
      for (auto token : tokens) {
        wait_builder.create<mlir::nvgpu::DeviceAsyncWaitOp>(
            user->getLoc(), token, nullptr);
      }
    }
  }
};

// This is the refactored pipeline pass.
class LowerOrchestraToGPUPass
    : public mlir::PassWrapper<LowerOrchestraToGPUPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToGPUPass)

  mlir::StringRef getArgument() const final {
    return "lower-orchestra-to-gpu";
  }
  mlir::StringRef getDescription() const final {
    return "Lowers the Orchestra dialect to a specific GPU vendor dialect.";
  }

  LowerOrchestraToGPUPass() = default;
  LowerOrchestraToGPUPass(const LowerOrchestraToGPUPass &pass) {
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::orchestra::OrchestraDialect,
                    mlir::gpu::GPUDialect,
                    mlir::memref::MemRefDialect,
                    mlir::nvgpu::NVGPUDialect,
                    mlir::arith::ArithDialect,
                    mlir::xegpu::XeGPUDialect,
                    mlir::scf::SCFDialect>();
  }

  // Option to select the GPU architecture.
  Option<std::string> gpuArch{
      *this,
      "gpu-arch",
      llvm::cl::desc("The target GPU architecture (e.g., nvgpu, xegpu)"),
      llvm::cl::init("nvgpu")};

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpPassManager pm(mlir::ModuleOp::getOperationName());

    if (gpuArch == "nvgpu") {
      auto &gpuModulePM = pm.nest<mlir::gpu::GPUModuleOp>();
      gpuModulePM.addNestedPass<mlir::gpu::GPUFuncOp>(
          std::make_unique<LowerOrchestraToNVGPUPass>());
    } else if (gpuArch == "xegpu") {
      auto &gpuModulePM = pm.nest<mlir::gpu::GPUModuleOp>();
      gpuModulePM.addNestedPass<mlir::gpu::GPUFuncOp>(
          createLowerOrchestraToXeGPUPass());
    } else if (gpuArch == "rocdl") {
      auto &gpuModulePM = pm.nest<mlir::gpu::GPUModuleOp>();
      gpuModulePM.addNestedPass<mlir::gpu::GPUFuncOp>(
          createLowerOrchestraToROCDLPass());
    } else {
      module.emitError() << "unsupported GPU architecture: " << gpuArch;
      signalPassFailure();
      return;
    }

    if (failed(runPipeline(pm, module))) {
      signalPassFailure();
    }
  }
};
}  // namespace

namespace mlir {
namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerOrchestraToGPUPass() {
  return std::make_unique<LowerOrchestraToGPUPass>();
}

void registerLoweringToGPUPasses() {
  ::mlir::PassRegistration<LowerOrchestraToGPUPass>();
}
}  // namespace orchestra
}  // namespace mlir
