// This file is a corrected, minimal template for a Rock dialect operation.
// It demonstrates the correct "in-situ" property syntax that is proven to work
// in the project's environment.

#ifndef ORCHESTRA_DIALECT_ROCK_OPS_TD_TEMPLATE
#define ORCHESTRA_DIALECT_ROCK_OPS_TD_TEMPLATE

include "mlir/IR/OpBase.td"
// The final implementation will need to include its own dialect file, like so:
// include "Orchestra/Dialects/Rock/RockDialect.td"

def Rock_GemmOp_Template : Op<Rock_Dialect, "gemm"> {
  let summary = "A template for the rock.gemm operation.";

  let arguments = (ins
    // This is the correct, "in-situ" syntax for properties.
    // It is proven to work by the existing Orchestra_CommitOp.
    // DO NOT use the 'let properties = [...]' block.
    AttrProperty<"arch", "::mlir::StringAttr", "The target architecture.">:$properties,

    F32Tensor:$matrix_a,
    F32Tensor:$matrix_b
  );

  let results = (outs F32Tensor:$matrix_c);
}

#endif // ORCHESTRA_DIALECT_ROCK_OPS_TD_TEMPLATE
