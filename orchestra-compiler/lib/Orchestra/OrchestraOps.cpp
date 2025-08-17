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

mlir::LogicalResult ScheduleOp::verify() {
  // Check that the schedule is a top-level operation.
  if (getOperation()->getParentOp() != nullptr &&
      !isa<mlir::ModuleOp>(getOperation()->getParentOp())) {
    return emitOpError("must be a top-level operation");
  }

  // Check the body of the schedule.
  auto &body = getBody();
  auto *block = &body.front();
  auto *terminator = block->getTerminator();

  // The block must end with an orchestra.yield.
  if (!isa<YieldOp>(terminator)) {
    return emitOpError("region must terminate with 'orchestra.yield'");
  }

  // The yield should have no operands, since the schedule has no results.
  if (terminator->getNumOperands() != 0) {
    return emitOpError("terminator should have no operands");
  }

  // All other operations must be orchestra.task ops.
  for (auto &op : block->without_terminator()) {
    if (!isa<TaskOp>(op)) {
      return op.emitError(
          "only 'orchestra.task' operations are allowed inside a "
          "'orchestra.schedule'");
    }
  }

  return mlir::success();
}

namespace {
// Erase a schedule that has no tasks.
struct EraseEmptySchedule : public mlir::OpRewritePattern<ScheduleOp> {
  using OpRewritePattern<ScheduleOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(ScheduleOp op,
                  mlir::PatternRewriter &rewriter) const override {
    // A schedule with no tasks has a single block with only a yield op.
    if (op.getBody().front().without_terminator().empty()) {
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
// TaskOp
//===----------------------------------------------------------------------===//

mlir::LogicalResult TaskOp::verify() {
  if (!getTarget()) {
    return emitOpError("requires a 'target' attribute");
  }
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
