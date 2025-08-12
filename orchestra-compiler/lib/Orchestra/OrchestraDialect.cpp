#include "Orchestra/OrchestraDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

// For the registration function
#include "Orchestra/OrchestraRegistration.h"

// For the op class definitions
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.h.inc"
#undef GET_OP_CLASSES

using namespace mlir;
using namespace orchestra;

#include "Orchestra/OrchestraOpsDialect.cpp.inc"

void OrchestraDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "Orchestra/OrchestraOps.h.inc"
  >();
}

namespace orchestra {
// Definition of the registration function.
void ensureOrchestraDialectRegistered() {
  // This dummy reference is enough to force the compiler to
  // instantiate the MyOp class and its static registration logic.
  (void)MyOp::getOperationName();
}
} // namespace orchestra
