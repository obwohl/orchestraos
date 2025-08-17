#include "Orchestra/OrchestraOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
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

namespace {
// Fold a commit op with a constant condition.
struct FoldConstantCommit : public mlir::OpRewritePattern<CommitOp> {
  using OpRewritePattern<CommitOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(CommitOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto constant =
        op.getCondition().getDefiningOp<mlir::arith::ConstantOp>();
    if (!constant) {
      return mlir::failure();
    }

    auto value = constant.getValue().dyn_cast<mlir::BoolAttr>();
    if (!value) {
      return mlir::failure();
    }

    if (value.getValue()) {
      rewriter.replaceOp(op, op.getTrueValues());
    } else {
      rewriter.replaceOp(op, op.getFalseValues());
    }
    return mlir::success();
  }
};
} // namespace

void CommitOp::getCanonicalizationPatterns(mlir::RewritePatternSet &results,
                                           mlir::MLIRContext *context) {
  results.add<FoldConstantCommit>(context);
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

  // The region of a task must have a single block.
  if (getBody().getBlocks().size() != 1) {
    return emitOpError("expected region to have a single block");
  }

  // The block must be terminated by an orchestra.yield op.
  auto terminator = getBody().front().getTerminator();
  if (!terminator) {
    return emitOpError("region must be terminated");
  }

  auto yieldOp = dyn_cast<YieldOp>(terminator);
  if (!yieldOp) {
    return emitOpError("region must terminate with 'orchestra.yield'");
  }

  // The number of yielded values must match the number of task results.
  if (yieldOp.getNumOperands() != getNumResults()) {
    return emitOpError("has ")
           << getNumResults() << " results but its "
           << "region yields " << yieldOp.getNumOperands() << " values";
  }

  // The types of the yielded values must match the task's result types.
  for (auto it : llvm::zip(getResultTypes(), yieldOp.getOperandTypes())) {
    if (std::get<0>(it) != std::get<1>(it)) {
      return yieldOp.emitOpError("type of yielded value ")
             << std::get<0>(it) << " does not match corresponding task result type "
             << std::get<1>(it);
    }
  }

  return mlir::success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
