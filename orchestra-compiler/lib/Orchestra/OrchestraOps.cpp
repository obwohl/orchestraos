#include "Orchestra/OrchestraOps.h"
#include "Orchestra/OrchestraDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace orchestra;

mlir::LogicalResult TaskOp::verify() {
  auto arch = getTargetArch().get("arch");
  if (!arch) {
    return emitOpError("requires a non-empty 'arch' in 'target_arch'");
  }
  if (!arch.isa<StringAttr>()) {
    return emitOpError("requires a string 'arch' in 'target_arch'");
  }
  if (arch.cast<StringAttr>().getValue().empty()) {
    return emitOpError("requires a non-empty 'arch' in 'target_arch'");
  }
  return mlir::success();
}

mlir::LogicalResult ScheduleOp::verify() {
  for (auto &op : getBody().front()) {
    if (!isa<TaskOp, ReturnOp>(op)) {
      return op.emitOpError("only 'orchestra.task' operations are allowed inside a 'orchestra.schedule'");
    }
  }
  return mlir::success();
}

// Stubs for other ops to link.
void CommitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                           MLIRContext *context) {
  results.add(+[](CommitOp op, PatternRewriter &rewriter) -> LogicalResult {
    auto condition = op.getCondition();
    if (auto cst = condition.getDefiningOp<arith::ConstantOp>()) {
      if (cst.getValue().cast<BoolAttr>().getValue()) {
        rewriter.replaceOp(op, op.getValues().take_front(op.getNumTrue()));
      } else {
        rewriter.replaceOp(op, op.getValues().drop_front(op.getNumTrue()));
      }
      return success();
    }
    return failure();
  });
}
void TransferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add(+[](TransferOp op, PatternRewriter &rewriter) -> LogicalResult {
    auto sourceOp = op.getSource().getDefiningOp<TransferOp>();
    if (!sourceOp) {
      return failure();
    }

    if (!sourceOp.getResult().hasOneUse()) {
      return failure();
    }

    if (op.getFrom() != sourceOp.getTo()) {
      return failure();
    }

    auto newPriority = op.getPriorityAttr();
    if (sourceOp.getPriorityAttr() &&
        (!newPriority ||
         sourceOp.getPriorityAttr().getValue().sgt(newPriority.getValue()))) {
      newPriority = sourceOp.getPriorityAttr();
    }

    rewriter.replaceOpWithNewOp<TransferOp>(
        op, op.getResult().getType(), sourceOp.getSource(), sourceOp.getFrom(),
        op.getTo(), newPriority);
    return success();
  });
}
mlir::LogicalResult CommitOp::verify() {
  auto num_true = getNumTrue();
  if (getValues().size() != 2 * num_true) {
    return emitOpError("has mismatched variadic operand sizes");
  }

  if (getResults().size() != num_true) {
    return emitOpError("requires number of results to match number of values in each branch");
  }

  for (size_t i = 0; i < num_true; ++i) {
    if (getValues()[i].getType() != getValues()[i + num_true].getType()) {
      return emitOpError("requires 'true' and 'false' value types to match");
    }
    if (getResults()[i].getType() != getValues()[i].getType()) {
      return emitOpError("requires result types to match operand types");
    }
  }

  return mlir::success();
}
mlir::LogicalResult TransferOp::verify() { return mlir::success(); }

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
