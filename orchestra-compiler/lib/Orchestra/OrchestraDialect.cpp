#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace orchestra;

//===----------------------------------------------------------------------===//
// OrchestraDialect
//===----------------------------------------------------------------------===//

// This include provides the implementation for the OrchestraDialect class.
#include "Orchestra/OrchestraDialect.cpp.inc"

// The `initialize` method is called by MLIR to register the dialect's
// operations, types, and attributes.
void OrchestraDialect::initialize() {
  // The `addOperations` template method is provided by the base Dialect class.
  // It takes a template pack of all operations to register.
  addOperations<
    // The GET_OP_LIST macro is a standard MLIR pattern that configures the
    // following include to expand to a comma-separated list of all C++
    // operation classes defined in the .td file.
#define GET_OP_LIST
#include "Orchestra/OrchestraOps.cpp.inc"
  >();
}

