// In orchestra-compiler/include/Orchestra/OrchestraOps.td

#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/RegionKindInterface.td"
include "Orchestra/OrchestraDialect.td"

//... other op definitions...

// Base class for Orchestra operations.
class Orchestra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

def Orchestra_TaskOp : Orchestra_Op<"task", []> {
  let summary = "Represents a schedulable task with a target architecture specifier.";

  let arguments = (ins
    StrAttr:$arch
  );

  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);

  // Enable the custom C++ verifier.
  let hasVerifier = 1;
}

def Orchestra_ScheduleOp : Orchestra_Op<"schedule", []> {
  let summary = "A container for a schedule of tasks.";
  let regions = (region SizedRegion<1>:$body);
  let hasVerifier = 1;
}

//
// --- orchestra.return op ---
//




def Orchestra_CommitOp : Orchestra_Op<"commit", []> {
  let summary = "Selects between two sets of values based on a condition.";

  let description = [{
    A conditional selection operation. If the `condition` is true, the
    first `num_true` values from the `values` operand are returned;
    otherwise, the remaining values are returned.
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$values,
    IntProp<"int32_t">:$num_true
  );

  let results = (outs Variadic<AnyType>:$results);

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def Orchestra_TransferOp : Orchestra_Op<"transfer", [AllTypesMatch<["source", "result"]>]> {
  let summary = "Explicitly represents data movement between memory spaces.";
  let description = [{
    Makes data movement between different logical memory spaces an explicit,
    first-class citizen of the IR. The `from` and `to` locations are
    represented by `SymbolRefAttr`, which will be resolved to physical memory
    spaces during a later lowering stage.
  }];

  let arguments = (ins AnyShaped:$source,
                       SymbolRefAttr:$from,
                       SymbolRefAttr:$to,
                       OptionalAttr<I32Attr>:$priority);
  let results = (outs AnyShaped:$result);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = "$source `from` $from `to` $to attr-dict `:` type($source)";
}

def Orchestra_ReturnOp : Orchestra_Op<"return", [Terminator]> {
  let summary = "A return operation for the orchestra dialect.";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs);
}

include "mlir/IR/PatternBase.td"

#endif // ORCHESTRA_OPS_TD
