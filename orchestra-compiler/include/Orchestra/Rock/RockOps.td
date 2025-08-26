#ifndef ROCK_OPS_TD
#define ROCK_OPS_TD

include "Rock.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Rock_GemmOp : Rock_Op<"gemm", [NoSideEffect]> {
  let summary = "gemm operation";
  let description = [{
    The `rock.gemm` operation computes the matrix multiplication of two inputs
    `A` and `B` and adds an optional third input `C` to the result. The result
    is stored in `D`. The operation is defined by the following equation:
    `D = alpha * A * B + beta * C`
    where `alpha` and `beta` are scalar attributes.
  }];

  let arguments = (ins
    Fp_Tensor:$A,
    Fp_Tensor:$B,
    Fp_Tensor:$C,
    DefaultValuedAttr<BoolAttr, "false">:$transA,
    DefaultValuedAttr<BoolAttr, "false">:$transB,
    DefaultValuedAttr<F32Attr, "1.0">:$alpha,
    DefaultValuedAttr<F32Attr, "1.0">:$beta
  );

  let results = (outs Fp_Tensor:$D);

  let assemblyFormat = [{
    `(`$A`, `$B`, `$C`)`
    `transA(`$transA`)` `transB(`$transB`)`
    `alpha(`$alpha`)` `beta(`$beta`)`
    `->` `$D`
    attr-dict
    `:` functional-type($A, $B, $C, $D)
  }];
}

def Rock_ConvOp : Rock_Op<"conv", [NoSideEffect]> {
  let summary = "conv operation";
  let description = [{
    The `rock.conv` operation computes the convolution of a filter with an
    input tensor.
  }];

  let arguments = (ins
    Fp_Tensor:$filter,
    Fp_Tensor:$input,
    DefaultValuedAttr<StrAttr, "\"NHWC\"">:$inputLayout,
    DefaultValuedAttr<StrAttr, "\"KYXC\"">:$filterLayout,
    DefaultValuedAttr<StrAttr, "\"NHWK\"">:$outputLayout,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$dilation,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$stride,
    DefaultValuedAttr<I64ArrayAttr, "{}">:$padding
  );

  let results = (outs Fp_Tensor:$output);

  let assemblyFormat = [{
    `(`$filter`, `$input`)`
    `inputLayout(`$inputLayout`)`
    `filterLayout(`$filterLayout`)`
    `outputLayout(`$outputLayout`)`
    `dilation(`$dilation`)`
    `stride(`$stride`)`
    `padding(`$padding`)`
    `->` `$output`
    attr-dict
    `:` functional-type($filter, $input, $output)
  }];
}

#endif // ROCK_OPS_TD
