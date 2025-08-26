#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Orchestra/Rock/RockDialect.h"
#include "Orchestra/Rock/RockOps.h"
#include "Orchestra/Transforms/Passes.h"

using namespace mlir;

namespace {
// LinalgMatmulToRockGemm
struct LinalgMatmulToRockGemm : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern<linalg::MatmulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto rockGemm = rewriter.create<rock::GemmOp>(
        loc, op.getResult(0).getType(), op.getInputs()[0], op.getInputs()[1],
        op.getOutputs()[0], rewriter.getBoolAttr(false),
        rewriter.getBoolAttr(false), rewriter.getF32FloatAttr(1.0),
        rewriter.getF32FloatAttr(1.0));

    rewriter.replaceOp(op, rockGemm.getD());
    return success();
  }
};

// LinalgConvToRockConv
struct LinalgConvToRockConv : public OpRewritePattern<linalg::Conv2DNhwcHwcfOp> {
  using OpRewritePattern<linalg::Conv2DNhwcHwcfOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::Conv2DNhwcHwcfOp op,
                                     PatternRewriter &rewriter) const override {
    auto loc = op.getLoc();

    auto inputLayout = rewriter.getStringAttr("NHWC");
    auto filterLayout = rewriter.getStringAttr("KYXC");
    auto outputLayout = rewriter.getStringAttr("NHWK");
    auto strides = op.getStrides();
    auto dilations = op.getDilations();
    auto padding = rewriter.getArrayAttr({});

    auto rockConv = rewriter.create<rock::ConvOp>(
        loc, op.getResult(0).getType(), op.getInputs()[1], op.getInputs()[0],
        inputLayout, filterLayout, outputLayout,
        dilations, strides, padding);

    rewriter.replaceOp(op, rockConv.getOutput());
    return success();
  }
};
} // namespace

struct LowerLinalgToRockPass
    : public PassWrapper<LowerLinalgToRockPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToRockPass)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<rock::RockDialect, linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, tensor::TensorDialect, rock::RockDialect>();
    target.addLegalOp<func::FuncOp, func::ReturnOp>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgMatmulToRockGemm, LinalgConvToRockConv>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "lower-linalg-to-rock"; }
  StringRef getDescription() const final { return "Lower Linalg dialect to Rock dialect."; }
};

std::unique_ptr<Pass> mlir::createLowerLinalgToRockPass() {
  return std::make_unique<LowerLinalgToRockPass>();
}

void mlir::registerLowerLinalgToRockPasses() {
  PassRegistration<LowerLinalgToRockPass>();
}
