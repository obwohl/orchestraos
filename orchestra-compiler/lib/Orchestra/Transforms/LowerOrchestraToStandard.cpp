/*===- LowerOrchestraToStandard.cpp - Orchestra to Standard lowering passes
 *-----*- C++ -*-===//
 *
 * This file implements a pass to lower the Orchestra dialect to the Standard
 * dialect.
 *
 *===----------------------------------------------------------------------===*/

#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "Orchestra/Transforms/Passes.h"
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
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::orchestra;

namespace {

class LowerOrchestraToStandardPass
    : public mlir::PassWrapper<LowerOrchestraToStandardPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerOrchestraToStandardPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::arith::ArithDialect>();
  }

  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<OrchestraDialect>();

    RewritePatternSet patterns(&getContext());

    if (failed(applyPartialConversion(
            getOperation(), target, std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final {
    return "lower-orchestra-to-standard";
  }
  StringRef getDescription() const final {
    return "Lower Orchestra dialect to standard dialects";
  }
};

}  // namespace

void orchestra::registerLoweringToStandardPasses() {
  PassRegistration<LowerOrchestraToStandardPass>();
}
