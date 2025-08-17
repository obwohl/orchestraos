/*===- LowerOrchestraToStandard.cpp - Orchestra to Standard lowering passes -----*- C++ -*-===//
 *
 * This file implements a pass to lower the Orchestra dialect to the Standard
 * dialect.
 *
 *===----------------------------------------------------------------------===*/

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "Orchestra/Transforms/Passes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"

namespace {

class CommitOpLowering : public mlir::OpConversionPattern<mlir::orchestra::CommitOp> {
public:
  using OpConversionPattern<mlir::orchestra::CommitOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::orchestra::CommitOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Value, 4> new_results;
    for (size_t i = 0; i < adaptor.getTrueValues().size(); ++i) {
      auto select = rewriter.create<mlir::arith::SelectOp>(
          op.getLoc(), adaptor.getCondition(), adaptor.getTrueValues()[i],
          adaptor.getFalseValues()[i]);
      new_results.push_back(select.getResult());
    }
    rewriter.replaceOp(op, new_results);
    return mlir::success();
  }
};

class LowerOrchestraToStandardPass
    : public mlir::PassWrapper<LowerOrchestraToStandardPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToStandardPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    mlir::ConversionTarget target(getContext());
    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addIllegalDialect<mlir::orchestra::OrchestraDialect>();

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<CommitOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }

  mlir::StringRef getArgument() const final { return "lower-orchestra-to-standard"; }
  mlir::StringRef getDescription() const final { return "Lower Orchestra dialect to standard dialects"; }
};

} // namespace

void mlir::orchestra::registerLoweringToStandardPasses() {
  mlir::PassRegistration<LowerOrchestraToStandardPass>();
}
