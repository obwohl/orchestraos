#ifndef ORCHESTRA_DIALECT_ROCK_OPS_TD
#define ORCHESTRA_DIALECT_ROCK_OPS_TD

include "mlir/IR/OpBase.td"
include "Orchestra/Dialects/Rock/RockDialect.td"

// NOTE: F32Tensor is a standard type constraint defined in MLIR's
// CommonTypeConstraints.td, which is included via OpBase.td. No special
// includes from other project files or local definitions are needed.

def Rock_GemmOp : Op<Rock_Dialect, "gemm"> {
  let summary = "rock gemm operation";
  let arguments = (ins
    StrAttr:$arch,
    F32Tensor:$matrix_a,
    F32Tensor:$matrix_b
  );
  let results = (outs F32Tensor:$matrix_c);
}

#endif // ORCHESTRA_DIALECT_ROCK_OPS_TD
