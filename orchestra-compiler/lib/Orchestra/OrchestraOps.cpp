#include "Orchestra/OrchestraOps.h"
#include "Orchestra/OrchestraDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"

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
void CommitOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}
void TransferOp::getCanonicalizationPatterns(RewritePatternSet &, MLIRContext *) {}
mlir::LogicalResult CommitOp::verify() { return mlir::success(); }
mlir::LogicalResult TransferOp::verify() { return mlir::success(); }

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
