#ifndef MLIR_DIALECT_ROCK_OPS_H_
#define MLIR_DIALECT_ROCK_OPS_H_

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Orchestra/Rock/RockOps.h.inc"

#endif // MLIR_DIALECT_ROCK_OPS_H_
