#include "Orchestra/OrchestraDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace orchestra;

// Get the C++ class declarations for our ops.
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.h.inc"
#undef GET_OP_CLASSES

#include "Orchestra/OrchestraOpsDialect.cpp.inc"

// Get the C++ class definitions for our ops.
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
#undef GET_OP_CLASSES

void OrchestraDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "Orchestra/OrchestraOps.cpp.inc"
  >();
}

// This function provides an explicit hook for the main executable to call.
// It is marked 'extern "C"' to ensure a stable, unmangled name.
extern "C" void registerOrchestraDialect(mlir::DialectRegistry &registry) {
    registry.insert<orchestra::OrchestraDialect>();
}
