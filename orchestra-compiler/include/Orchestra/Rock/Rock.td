#ifndef ROCK_TD
#define ROCK_TD

include "mlir/IR/OpBase.td"

def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::rock";
  let summary = "A dialect for rock operations.";
}

class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits>;

def Fp_Tensor : TensorOf<[AnyFloat]>;

#endif // ROCK_TD
