#include "Orchestra/OrchestraOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

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
