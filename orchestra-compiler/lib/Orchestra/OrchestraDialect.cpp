#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace orchestra;

#include "Orchestra/OrchestraOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.cpp.inc"
