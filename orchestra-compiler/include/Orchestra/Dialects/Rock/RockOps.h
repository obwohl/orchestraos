#ifndef ORCHESTRA_DIALECTS_ROCK_ROCKOPS_H_
#define ORCHESTRA_DIALECTS_ROCK_ROCKOPS_H_

// These headers are required for the types and interfaces used in the
// auto-generated file RockOps.h.inc.
#include "mlir/IR/BuiltinTypes.h"
// The correct path for BytecodeOpInterface is mlir/Bytecode/, not mlir/Interfaces/.
#include "mlir/Bytecode/BytecodeOpInterface.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Orchestra/Dialects/Rock/RockOps.h.inc"

#endif // ORCHESTRA_DIALECTS_ROCK_ROCKOPS_H_
