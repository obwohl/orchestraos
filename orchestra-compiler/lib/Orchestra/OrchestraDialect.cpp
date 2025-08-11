#include "Orchestra/OrchestraDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

// Include the auto-generated file for operation definitions.
// This must be included before the dialect's initialize method
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"

// Use the C++ namespace that was specified in the .td file.
namespace orchestra {

// Dialect initialization, called by MLIR framework.
void OrchestraDialect::initialize() {
  // Register operations, types, and attributes.
  // The 'addOperations' call uses a macro to expand to a list of
  // all operations defined in the .td file.
  addOperations<GET_OP_LIST>(); // GET_OP_LIST is defined in OrchestraOps.cpp.inc
}

} // namespace orchestra

// Include the auto-generated file for dialect definition.
// This must be included *after* the namespace block containing the
// 'initialize' implementation.
#include "Orchestra/OrchestraOpsDialect.cpp.inc"
