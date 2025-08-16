#include "Orchestra/OrchestraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace orchestra;

//===----------------------------------------------------------------------===//
// CommitOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult CommitOp::verify() {
  if (getTrueValues().getTypes() != getFalseValues().getTypes()) {
    return emitOpError("requires 'true' and 'false' value types to match");
  }
  if (getTrueValues().getTypes() != getResultTypes()) {
    return emitOpError("requires result types to match operand types");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TransferOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult TransferOp::verify() {
  if (getSource().getType() != getResult().getType()) {
    return emitOpError("requires result type to match source type");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// ScheduleOp
//===----------------------------------------------------------------------===//

namespace {
// Erase an empty schedule that has no results.
struct EraseEmptySchedule : public mlir::OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ScheduleOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // An empty schedule has a single block with no operations.
    if (op.getBody().front().empty() && op.getNumResults() == 0) {
      rewriter.eraseOp(op);
      return mlir::success();
    }
    return mlir::failure();
  }
};
} // namespace

void ScheduleOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                             mlir::MLIRContext *context) {
  results.add<EraseEmptySchedule>(context);
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
