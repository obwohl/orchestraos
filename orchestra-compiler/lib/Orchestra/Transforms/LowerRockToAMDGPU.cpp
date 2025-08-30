#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"
#include "Orchestra/Transforms/Passes.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

class GemmLoweringPattern : public mlir::OpConversionPattern<mlir::rock::GemmOp> {
public:
  using OpConversionPattern<mlir::rock::GemmOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::rock::GemmOp gemmOp, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Location loc = gemmOp.getLoc();
    auto tensorCType = gemmOp.getResult().getType().cast<mlir::RankedTensorType>();

    auto zeroAttr = mlir::DenseElementsAttr::get(
        tensorCType, rewriter.getF32FloatAttr(0.0f));
    mlir::Value zero = rewriter.create<mlir::arith::ConstantOp>(loc, tensorCType, zeroAttr);

    rewriter.replaceOp(gemmOp, zero);

    return mlir::success();
  }
};

struct RockToAMDGPUConversionPass
    : public mlir::PassWrapper<RockToAMDGPUConversionPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  mlir::StringRef getArgument() const final { return "lower-rock-to-amdgpu"; }
  mlir::StringRef getDescription() const final {
    return "Lower Rock dialect to AMDGPU dialect.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::amdgpu::AMDGPUDialect, mlir::arith::ArithDialect,
                    mlir::vector::VectorDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addIllegalDialect<mlir::rock::RockDialect>();
    target.addLegalDialect<mlir::amdgpu::AMDGPUDialect,
                           mlir::arith::ArithDialect,
                           mlir::vector::VectorDialect, mlir::scf::SCFDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<GemmLoweringPattern>(&getContext());

    if (mlir::failed(mlir::applyPartialConversion(
            getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // anonymous namespace

namespace mlir {
namespace rock {
std::unique_ptr<mlir::Pass> createLowerRockToAMDGPUConversionPass() {
  return std::make_unique<RockToAMDGPUConversionPass>();
}

void registerLowerRockToAMDGPU() {
  mlir::PassRegistration<RockToAMDGPUConversionPass>();
}
} // namespace rock
} // namespace mlir
