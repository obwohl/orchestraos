#include "Orchestra/OrchestraOps.h"
#include "Orchestra/OrchestraDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

using namespace mlir;
using namespace orchestra;

// Verifies the schema of the 'target' attribute.
mlir::LogicalResult TaskOp::verify() {
  auto targetAttr = getTarget();
  if (!targetAttr) {
    return emitOpError("requires a 'target' attribute");
  }

  auto dictAttr = targetAttr.dyn_cast<DictionaryAttr>();
  if (!dictAttr) {
    return emitOpError("requires 'target' attribute to be a dictionary");
  }

  auto archAttr = dictAttr.get("arch").dyn_cast_or_null<StringAttr>();
  if (!archAttr) {
    return emitOpError(
        "requires a string 'arch' key in the 'target' dictionary");
  }

  if (archAttr.getValue().empty()) {
    return emitOpError("'arch' key in 'target' dictionary cannot be empty");
  }

  // Verify that 'device_id' exists and is an integer.
  auto deviceIdAttr = dictAttr.get("device_id");
  if (!deviceIdAttr) {
    return emitOpError("requires an 'device_id' key in the 'target' dictionary");
  }

  if (!deviceIdAttr.isa<IntegerAttr>()) {
    return emitOpError("'device_id' key in 'target' dictionary must be an integer");
  }

  return mlir::success();
}

// Verifies that the body of a schedule only contains orchestra.task operations.
mlir::LogicalResult ScheduleOp::verify() {
  for (auto &op : getBody().front()) {
    if (!isa<TaskOp, ReturnOp>(op)) {
      return op.emitOpError(
          "only 'orchestra.task' operations are allowed inside a "
          "'orchestra.schedule'");
    }
  }
  return mlir::success();
}

// This pattern folds a sequence of two transfer operations into a single one.
// For example:
//   %1 = orchestra.transfer %0 from @A to @B
//   %2 = orchestra.transfer %1 from @B to @C
// is folded into:
//   %2 = orchestra.transfer %0 from @A to @C
// This is a peephole optimization that reduces the number of data transfers.
void TransferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                             MLIRContext *context) {
  results.add(+[](TransferOp op, PatternRewriter &rewriter) -> LogicalResult {
    // Match the pattern: op's source must be another TransferOp.
    auto sourceOp = op.getSource().getDefiningOp<TransferOp>();
    if (!sourceOp) {
      return failure();
    }

    // The intermediate buffer must have only one use.
    if (!sourceOp.getResult().hasOneUse()) {
      return failure();
    }

    // The memory spaces must match up (@A -> @B, @B -> @C).
    if (op.getFrom() != sourceOp.getTo()) {
      return failure();
    }

    // The new transfer inherits the highest priority of the two.
    auto newPriority = op.getPriorityAttr();
    if (sourceOp.getPriorityAttr() &&
        (!newPriority ||
         sourceOp.getPriorityAttr().getValue().sgt(newPriority.getValue()))) {
      newPriority = sourceOp.getPriorityAttr();
    }

    // Replace the second transfer with a new one that directly connects the
    // source of the first transfer to the destination of the second.
    rewriter.replaceOpWithNewOp<TransferOp>(
        op, op.getResult().getType(), sourceOp.getSource(), sourceOp.getFrom(),
        op.getTo(), newPriority);
    return success();
  });
}

// Verifies the consistency of operand and result types and sizes.
mlir::LogicalResult CommitOp::verify() {
  int32_t num_true = getNumTrue();
  // HACK: The generic op parser does not seem to initialize the property.
  // We manually read the attribute from the dictionary if the property has
  // its default value. This is a workaround for a known MLIR issue.
  if (num_true == 0) {
    if (auto attr = (*this)->getAttrOfType<IntegerAttr>("num_true")) {
      num_true = attr.getInt();
    }
  }

  // The total number of values must be twice the number of 'true' branch
  // values.
  if (getValues().size() != 2 * num_true) {
    return emitOpError("has mismatched variadic operand sizes");
  }

  // The number of results must match the number of values in each branch.
  if (getResults().size() != num_true) {
    return emitOpError(
        "requires number of results to match number of values in each branch");
  }

  // Check that the types of the 'true' and 'false' branches match for each
  // operand, and that they also match the corresponding result type.
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

// Verifies that the 'from' and 'to' properties are not empty.
mlir::LogicalResult TransferOp::verify() {
  if (!getFrom() || getFrom().getRootReference().empty()) {
    return emitOpError("requires a non-empty 'from' attribute");
  }
  if (!getTo() || getTo().getRootReference().empty()) {
    return emitOpError("requires a non-empty 'to' attribute");
  }
  return mlir::success();
}

// This includes the C++ definitions for the operations, which are
// generated by TableGen from the .td files. This includes boilerplate
// code for parsing, printing, and building the operations.
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
