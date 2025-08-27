#ifndef ORCHESTRA_DIALECT_ROCK_OPS_TD
#define ORCHESTRA_DIALECT_ROCK_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def Rock_Dialect : Dialect {
  let name = "rock";
  let cppNamespace = "::mlir::orchestra::rock";
  let summary = "A dialect for lowering to AMD ROCm primitives.";
  let usePropertiesForAttributes = 1;
}

class Rock_Op<string mnemonic, list<Trait> traits = []> :
    Op<Rock_Dialect, mnemonic, traits>;

defvar GemmInputTypes = [F32, F16, BF16, I8, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN];
defvar GemmOutputTypes = [F32, F16, BF16, I32, I8, F8E5M2FNUZ, F8E4M3FNUZ, F8E5M2, F8E4M3FN];

class TensorOrMemRefOf<list<Type> allowedTypes> :
  AnyTypeOf<[TensorOf<allowedTypes>,
             MemRefOf<allowedTypes>],
             /*summary=*/"Constraints the type to be either a Tensor or MemRef of certain types of elements.",
             "::mlir::ShapedType">;

def StoreMethod : Enum<"StoreMethod", "Store method for results", [
  EnumVal<"Set", "set the value", 0>,
  EnumVal<"AtomicAdd", "atomically add to the value", 1>
]>;
def StoreMethodAttr : EnumAttr<Rock_Dialect, StoreMethod, "store method">;

// Using I32Attr as a placeholder for Rock_GemmFeaturesAttr from rocMLIR
def Rock_GemmFeaturesAttr : I32Attr;

def Rock_GemmOp : Rock_Op<"gemm", [NoSideEffect]> {
  let summary = "A generalized matrix multiplication operation for Rock devices.";

  let arguments = (
    ins
      TensorOrMemRefOf<GemmInputTypes>:$a,
      TensorOrMemRefOf<GemmInputTypes>:$b,
      TensorOrMemRefOf<GemmOutputTypes>:$c
  );

  let results = (outs Optional<AnyRankedTensor>:$result);

  let properties = (
    ins
      Defaulted<Bool, "false">:$aTransposed,
      Defaulted<Bool, "false">:$bTransposed,
      Defaulted<Bool, "false">:$cTransposed,
      Prop<StoreMethodAttr>:$storeMethod,
      Optional<Rock_GemmFeaturesAttr>:$features,
      Optional<I32>:$derivedBlockSize,
      Optional<I32>:$gridSize
  );

  let assemblyFormat = [{
    `(`$a`, `$b`, `$c`)`
    `attr_dict`
    `:` functional_type($a, $b, $c, $result)
  }];
}

#endif // ORCHESTRA_DIALECT_ROCK_OPS_TD
