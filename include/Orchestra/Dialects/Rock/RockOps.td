include "mlir/IR/OpBase.td"
include "Orchestra/Dialects/Rock/RockDialect.td"

class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits>;

def Rock_GemmOp : Rock_Op<"gemm", [NoSideEffect]> {
  let summary = "Matrix multiplication operation";
  let description = [{
    The `rock.gemm` operation performs a matrix multiplication of two tensors.
  }];

  let arguments = (ins
    AnyTensor:$a,
    AnyTensor:$b,
    StrAttr:$a_layout,
    StrAttr:$b_layout,
    StrAttr:$c_layout,
    StrAttr:$arch,
    I32Attr:$num_cu
  );

  let results = (outs AnyTensor:$c);

  let assemblyFormat = [{
    `(`$a `,` $b`)`
    `attr-dict`
    `:` type($a) `,` type($b) `->` type($c)
  }];
}
