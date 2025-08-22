#ifndef ORCHESTRA_DIALECT_TD
#define ORCHESTRA_DIALECT_TD

include "mlir/IR/OpBase.td"

def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "::mlir::orchestra";

  let dependentDialects = [
    "mlir::arith::ArithDialect"
  ];

  let usePropertiesForAttributes = 1;

  let summary = "A dialect for the Orchestra meta-OS compiler.";
  let description = [{
    The OrchestraIR dialect is the heart of the OrchestraOS compiler's
    proprietary technology. It is a mid-level dialect designed to serve as the
    crucial bridge between a purely logical, framework-agnostic representation
    of a computation and the concrete, hardware-specific representations
    required for final code generation.
  }];
}

#endif // ORCHESTRA_DIALECT_TD
