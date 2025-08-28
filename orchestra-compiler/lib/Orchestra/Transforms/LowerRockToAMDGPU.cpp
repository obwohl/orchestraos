#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/IR/ROCDL.h"
#include "orchestra/Dialects/Rock/RockDialect.h"
#include "orchestra/Transforms/Passes.h"

namespace {
class LowerRockToAMDGPU
    : public mlir::PassWrapper<LowerRockToAMDGPU,
                               mlir::OperationPass<mlir::gpu::GPUModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerRockToAMDGPU)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::amdgpu::AMDGPUDialect, mlir::rock::RockDialect,
                    mlir::arith::ArithDialect, mlir::vector::VectorDialect,
                    mlir::rocdl::ROCDLDialect>();
  }

  void runOnOperation() override;

private:
  llvm::StringRef getPassName() const override {
    return "lower-rock-to-amdgpu";
  }
};
} // namespace

void LowerRockToAMDGPU::runOnOperation() {
  mlir::gpu::GPUModuleOp gpuModule = getOperation();
  mlir::MLIRContext *ctx = &getContext();

  gpuModule.walk([&](mlir::rock::GemmOp gemmOp) {
    mlir::OpBuilder builder(gemmOp);
    auto loc = gemmOp.getLoc();

    auto a = gemmOp.getA();
    auto b = gemmOp.getB();
    auto c = gemmOp.getC();

    // Create a zero-initialized accumulator.
    auto accType = c.getType().cast<mlir::VectorType>();
    auto elemType = accType.getElementType();
    auto zero = builder.create<mlir::arith::ConstantOp>(
        loc, elemType, builder.getFloatAttr(elemType, 0.0));
    auto acc = builder.create<mlir::vector::SplatOp>(loc, accType, zero);

    // TODO(user): These values should be derived from the 'arch' attribute
    // and the input/output shapes.
    int m = 32;
    int n = 32;
    int k = 4;
    int blocks = 1;

    auto mfmaOp = builder.create<mlir::amdgpu::MFMAOp>(
        loc, c.getType(), a, b, acc, m, n, k, blocks,
        /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);

    gemmOp.replaceAllUsesWith(mfmaOp.getResult(0));
    gemmOp.erase();
  });
}

std::unique_ptr<mlir::Pass> mlir::createLowerRockToAMDGPUPass() {
  return std::make_unique<LowerRockToAMDGPU>();
}
