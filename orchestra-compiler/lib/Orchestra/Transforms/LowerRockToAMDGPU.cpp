#include "Orchestra/Transforms/LowerRockToAMDGPU.h"

#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"

#include <string>

namespace {
struct LowerRockToAMDGPUPass
    : public mlir::PassWrapper<LowerRockToAMDGPUPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  LowerRockToAMDGPUPass() = default;
  LowerRockToAMDGPUPass(const LowerRockToAMDGPUPass &pass)
      : mlir::PassWrapper<LowerRockToAMDGPUPass,
                          mlir::OperationPass<mlir::ModuleOp>>(pass) {}

  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "lower-rock-to-amdgpu";
  }

  mlir::StringRef getDescription() const final {
    return "Lower Rock operations to AMDGPU operations.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::rock::RockDialect, mlir::gpu::GPUDialect,
                    mlir::amdgpu::AMDGPUDialect, mlir::arith::ArithDialect,
                    mlir::memref::MemRefDialect,
                    mlir::bufferization::BufferizationDialect,
                    mlir::vector::VectorDialect>();
  }

  Option<std::string> arch{*this, "arch",
                             llvm::cl::desc("Target GPU architecture"),
                             llvm::cl::init("amdgcn-amd-amdhsa")};
};
} // namespace

void LowerRockToAMDGPUPass::runOnOperation() {
  mlir::ModuleOp module = getOperation();

  // Add the container module attribute if it's not there.
  if (!module->hasAttr("gpu.container_module")) {
    mlir::OpBuilder builder(module.getContext());
    module->setAttr("gpu.container_module", builder.getUnitAttr());
  }

  // Find or create a single GPU module for the entire compilation unit.
  mlir::gpu::GPUModuleOp gpuModule;
  for (auto op : module.getOps<mlir::gpu::GPUModuleOp>()) {
    gpuModule = op;
    break;
  }
  if (!gpuModule) {
    mlir::OpBuilder builder(module.getBodyRegion());
    gpuModule = builder.create<mlir::gpu::GPUModuleOp>(module.getLoc(),
                                                     "orchestra_gpu_module");
    if (!arch.empty()) {
        gpuModule->setAttr("arch", builder.getStringAttr(arch));
    }
  }

  int kernelCounter = 0;
  module.walk([&](mlir::rock::GemmOp gemmOp) {
    mlir::OpBuilder builder(gemmOp);
    mlir::Location loc = gemmOp.getLoc();

    // 1. Convert input tensors to memrefs for the launch op
    auto rankedTensorTypeA =
        mlir::cast<mlir::RankedTensorType>(gemmOp.getMatrixA().getType());
    auto memrefTypeA =
        mlir::MemRefType::get(rankedTensorTypeA.getShape(),
                              rankedTensorTypeA.getElementType());
    auto memrefA_host = builder.create<mlir::bufferization::ToMemrefOp>(
        loc, memrefTypeA, gemmOp.getMatrixA());

    auto rankedTensorTypeB =
        mlir::cast<mlir::RankedTensorType>(gemmOp.getMatrixB().getType());
    auto memrefTypeB =
        mlir::MemRefType::get(rankedTensorTypeB.getShape(),
                              rankedTensorTypeB.getElementType());
    auto memrefB_host = builder.create<mlir::bufferization::ToMemrefOp>(
        loc, memrefTypeB, gemmOp.getMatrixB());

    auto rankedTensorTypeC =
        mlir::cast<mlir::RankedTensorType>(gemmOp.getMatrixC().getType());
    auto memrefTypeC =
        mlir::MemRefType::get(rankedTensorTypeC.getShape(),
                              rankedTensorTypeC.getElementType());
    auto memrefC_host = builder.create<mlir::bufferization::ToMemrefOp>(
        loc, memrefTypeC, gemmOp.getMatrixC());

    // 2. Create a unique GPU kernel inside the single GPU module
    mlir::OpBuilder gpuBuilder(gpuModule.getBodyRegion());
    std::string kernelName = "gemm_kernel_" + std::to_string(kernelCounter++);

    auto gpuFunc = gpuBuilder.create<mlir::gpu::GPUFuncOp>(
        loc, kernelName,
        gpuBuilder.getFunctionType({memrefTypeA, memrefTypeB, memrefTypeC}, {}));
    gpuFunc->setAttr("gpu.kernel", gpuBuilder.getUnitAttr());

    // The create call with a function type already creates the entry block.
    mlir::Block *entryBlock = &gpuFunc.front();
    mlir::OpBuilder bodyBuilder = mlir::OpBuilder::atBlockBegin(entryBlock);
    
    // --- Start of merged logic ---
    mlir::Value memrefA_device = gpuFunc.getArgument(0);
    mlir::Value memrefB_device = gpuFunc.getArgument(1);
    mlir::Value memrefC_device = gpuFunc.getArgument(2);

    auto elemType = mlir::cast<mlir::MemRefType>(memrefA_device.getType()).getElementType();
    auto vectorType = mlir::VectorType::get({16}, elemType);
    mlir::Value zero = bodyBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);

    mlir::Value vectorA = bodyBuilder.create<mlir::vector::TransferReadOp>(
        loc, vectorType, memrefA_device, mlir::ValueRange{zero, zero});
    mlir::Value vectorB = bodyBuilder.create<mlir::vector::TransferReadOp>(
        loc, vectorType, memrefB_device, mlir::ValueRange{zero, zero});
    mlir::Value vectorC = bodyBuilder.create<mlir::vector::TransferReadOp>(
        loc, vectorType, memrefC_device, mlir::ValueRange{zero, zero});

    auto mfmaOp = bodyBuilder.create<mlir::amdgpu::MFMAOp>(
        loc, vectorType,
        bodyBuilder.getI32IntegerAttr(32), bodyBuilder.getI32IntegerAttr(32),
        bodyBuilder.getI32IntegerAttr(16), bodyBuilder.getI32IntegerAttr(1),
        vectorA, vectorB, vectorC,
        bodyBuilder.getI32IntegerAttr(0), bodyBuilder.getI32IntegerAttr(0),
        mlir::amdgpu::MFMAPermBAttr::get(bodyBuilder.getContext(),
                                          mlir::amdgpu::MFMAPermB::none),
        bodyBuilder.getUnitAttr(), bodyBuilder.getUnitAttr(),
        bodyBuilder.getUnitAttr(), bodyBuilder.getUnitAttr());

    bodyBuilder.create<mlir::vector::TransferWriteOp>(loc, mfmaOp.getDestD(), memrefC_device,
                                                   mlir::ValueRange{zero, zero});
    // --- End of merged logic ---

    bodyBuilder.create<mlir::gpu::ReturnOp>(loc);

    // 3. Create launch op at the original location
    builder.setInsertionPoint(gemmOp);
    auto gridDim = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIndexAttr(1));
    auto blockDim = builder.create<mlir::arith::ConstantOp>(
        loc, builder.getIndexAttr(64));
    builder.create<mlir::gpu::LaunchFuncOp>(
        loc, gpuFunc, mlir::gpu::KernelDim3{gridDim, gridDim, gridDim},
        mlir::gpu::KernelDim3{blockDim, blockDim, blockDim},
        /*dynamic shmem size*/ nullptr,
        mlir::ValueRange{memrefA_host, memrefB_host, memrefC_host});

    // 4. Convert output memref back to tensor
    auto resultTensor =
        builder.create<mlir::bufferization::ToTensorOp>(loc, memrefC_host);

    // 5. Replace uses and erase
    gemmOp.replaceAllUsesWith(resultTensor.getResult());
    gemmOp.erase();
  });
}

void mlir::orchestra::registerLowerRockToAMDGPUPass() {
  mlir::PassRegistration<LowerRockToAMDGPUPass>();
}

namespace mlir {
namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerRockToAMDGPUPass() {
  return std::make_unique<LowerRockToAMDGPUPass>();
}
} // namespace orchestra
} // namespace mlir