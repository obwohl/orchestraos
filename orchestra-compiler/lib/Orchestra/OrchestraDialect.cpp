#include "Orchestra/OrchestraDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace orchestra;

#include "Orchestra/OrchestraOpsDialect.cpp.inc"

void OrchestraDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "Orchestra/OrchestraOps.h.inc"
  >();
}
