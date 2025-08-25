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
  if (getArch().empty()) {
    return emitOpError("requires a non-empty 'arch' property");
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
mlir::LogicalResult SelectOp::verify() {
  int32_t num_true = getNumTrue();
  // HACK: The generic op parser does not seem to initialize the property.
  // We manually read the attribute from the dictionary if the property has
  // its default value.
  if (num_true == 0) {
    if (auto attr = (*this)->getAttrOfType<IntegerAttr>("num_true")) {
      num_true = attr.getInt();
    }
  }

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
mlir::LogicalResult CommitOp::verify() {
  if (!getOperand().getType().isa<MemRefType>()) {
    return emitOpError("operand must be a MemRefType");
  }
  if (!getResult().getType().isa<MemRefType>()) {
    return emitOpError("result must be a MemRefType");
  }
  return mlir::success();
}
mlir::LogicalResult TransferOp::verify() { return mlir::success(); }

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
