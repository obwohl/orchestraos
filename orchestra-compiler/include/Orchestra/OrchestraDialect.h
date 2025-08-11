#ifndef ORCHESTRA_ORCHESTRADIRECT_H
#define ORCHESTRA_ORCHESTRADIRECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/Region.h" // Added for mlir::RegionRange
#include "mlir/IR/BuiltinOps.h" // Added
#include "mlir/IR/BuiltinAttributes.h" // Added
#include "mlir/IR/BuiltinTypes.h" // Added
#include "mlir/IR/OperationSupport.h" // Added for mlir::EmptyProperties

// Include the auto-generated header file for the dialect class declaration.
#include "Orchestra/OrchestraOpsDialect.h.inc"

// Include the auto-generated header file for the operation declarations.
#define GET_OP_CLASSES
#include "Orchestra/OrchestraOps.h.inc"

// Include the auto-generated header file for the type declarations.
// Only if custom types are defined in .td file.
// #define GET_TYPEDEF_CLASSES
// #include "Orchestra/OrchestraTypes.h.inc"

#endif // ORCHESTRA_ORCHESTRADIRECT_H
