#include "Orchestra/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace orchestra;

namespace orchestra {

namespace {
// This is the original logic for lowering to NVGPU, extracted into its own pass.
class LowerOrchestraToNVGPUPass
    : public mlir::PassWrapper<LowerOrchestraToNVGPUPass,
                               mlir::OperationPass<mlir::gpu::GPUFuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToNVGPUPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<OrchestraDialect, mlir::gpu::GPUDialect,
                mlir::memref::MemRefDialect, mlir::nvgpu::NVGPUDialect,
                mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::gpu::GPUFuncOp funcOp = getOperation();
    // Map from destination buffer to async token.
    llvm::DenseMap<mlir::Value, mlir::Value> asyncTokens;

    // Collect all transfer ops to be rewritten.
    llvm::SmallVector<TransferOp, 4> transferOps;
    funcOp.walk(
        [&](TransferOp op) { transferOps.push_back(op); });

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
          sourceType.getShape(), sourceType.getElementType(),
          sourceType.getLayout(),
          mlir::gpu::AddressSpaceAttr::get(op.getContext(),
                                           mlir::gpu::AddressSpace::Workgroup));
      auto dest = builder.create<mlir::memref::AllocOp>(loc, destType);

      // Create zero indices for the copy.
      mlir::SmallVector<mlir::Value, 4> indices;
      for (unsigned i = 0; i < sourceType.getRank(); ++i) {
        indices.push_back(
            builder.create<mlir::arith::ConstantIndexOp>(loc, 0));
      }

      // Create an async copy.
      auto asyncCopy = builder.create<mlir::nvgpu::DeviceAsyncCopyOp>(
          loc, mlir::nvgpu::DeviceAsyncTokenType::get(op.getContext()),
          dest.getResult(), indices, source, indices,
          builder.getIndexAttr(sourceType.getNumElements()), mlir::Value{},
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
        wait_builder.create<mlir::nvgpu::DeviceAsyncWaitOp>(user->getLoc(),
                                                            token, nullptr);
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

  mlir::StringRef getArgument() const final { return "lower-orchestra-to-gpu"; }
  mlir::StringRef getDescription() const final {
    return "Lowers the Orchestra dialect to a specific GPU vendor dialect.";
  }

  LowerOrchestraToGPUPass() = default;
  LowerOrchestraToGPUPass(const LowerOrchestraToGPUPass& pass) {}

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    // This pass is a pipeline, so it should not have dialect dependencies itself.
    // The nested passes will declare their own dependencies.
  }

  // Option to select the GPU architecture.
  Option<std::string> gpuArch{
      *this, "gpu-arch",
      llvm::cl::desc("The target GPU architecture (e.g., nvgpu, xegpu)"),
      llvm::cl::init("nvgpu")};

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::PassManager pm(module.getContext());

    if (gpuArch == "nvgpu") {
      pm.addNestedPass<mlir::gpu::GPUFuncOp>(
          std::make_unique<LowerOrchestraToNVGPUPass>());
    } else if (gpuArch == "xegpu") {
      pm.addNestedPass<mlir::gpu::GPUFuncOp>(createLowerOrchestraToXeGPUPass());
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
} // namespace

std::unique_ptr<mlir::Pass> createLowerOrchestraToGPUPass() {
  return std::make_unique<LowerOrchestraToGPUPass>();
}

void registerLoweringToGPUPasses() {
  ::mlir::PassRegistration<LowerOrchestraToGPUPass>();
}

} // namespace orchestra
