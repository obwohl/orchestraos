#ifndef ORCHESTRA_OPS_H
#define ORCHESTRA_OPS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.h.inc"

#endif // ORCHESTRA_OPS_H
