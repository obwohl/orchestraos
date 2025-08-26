#include "Orchestra/Rock/RockDialect.h"
#include "Orchestra/Rock/RockOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::rock;

#include "Orchestra/Rock/RockOpsDialect.cpp.inc"

void RockDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Orchestra/Rock/RockOps.cpp.inc"
      >();
}
