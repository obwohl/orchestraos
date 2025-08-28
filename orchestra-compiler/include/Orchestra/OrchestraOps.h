#ifndef ORCHESTRA_OPS_H
#define ORCHESTRA_OPS_H

#include "Orchestra/OrchestraTarget.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// The GET_OP_CLASSES macro is a standard MLIR TableGen mechanism.
// It is replaced by the preprocessor with the C++ declarations of all the
// operations defined in the `OrchestraOps.td` file.
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.h.inc"

#endif  // ORCHESTRA_OPS_H
