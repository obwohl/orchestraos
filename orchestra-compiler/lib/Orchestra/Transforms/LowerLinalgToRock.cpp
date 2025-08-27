#include "Orchestra/Transforms/Passes.h"
#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {

using namespace mlir;
using namespace mlir::orchestra::rock;

// A conversion pattern that matches linalg.generic ops and converts them to
// rock.gemm ops.
class LinalgGenericToRockGemmPattern
    : public OpConversionPattern<linalg::GenericOp> {
public:
  using OpConversionPattern<linalg::GenericOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(linalg::GenericOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // This is a basic matcher. A more robust implementation would check the
    // indexing maps and iterator types to ensure it's a GEMM-like operation.
    if (op.getNumDpsInits() != 1 || op.getNumDpsInputs() != 2) {
      return failure();
    }

    // For now, we assume no transpose and a 'Set' store method.
    // A more complete implementation would infer this from the linalg.generic op.
    Rock_GemmOp::Properties properties;
    properties.aTransposed = false;
    properties.bTransposed = false;
    properties.cTransposed = false;
    properties.storeMethod = StoreMethod::Set;

    rewriter.replaceOpWithNewOp<Rock_GemmOp>(
        op, op.getResultTypes(),
        ValueRange{adaptor.getInputs()[0], adaptor.getInputs()[1],
                   adaptor.getDpsInits()[0]},
        properties);

    return success();
  }
};

struct LowerLinalgToRockPass
    : public PassWrapper<LowerLinalgToRockPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerLinalgToRockPass)

  void runOnOperation() override {
    ConversionTarget target(getContext());

    // The target is legal if it doesn't contain any linalg.generic ops.
    // We want to convert all of them to rock.gemm.
    target.addIllegalOp<linalg::GenericOp>();
    target.addLegalDialect<RockDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<LinalgGenericToRockGemmPattern>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      signalPassFailure();
    }
  }

  StringRef getArgument() const final { return "lower-linalg-to-rock"; }
  StringRef getDescription() const final {
    return "Lower Linalg operations to the Rock dialect";
  }
};

} // end anonymous namespace

namespace mlir {
namespace orchestra {
void registerLowerLinalgToRockPass() {
  PassRegistration<LowerLinalgToRockPass>();
}
} // namespace orchestra
} // namespace mlir
