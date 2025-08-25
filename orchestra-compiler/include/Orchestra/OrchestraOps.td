#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

// Base class for all operations in this dialect.
class Orchestra_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

// The orchestra.task operation definition.
def Orchestra_TaskOp : Orchestra_Op<"task", []> {
  let summary = "An orchestra task operation";

  // Define the property for this operation.
  let properties = (ins
    TargetArchProp:$target_arch
  );

  let arguments = (ins
    Variadic<AnyType>:$operands,
    StringProp:$task_id
  );
  let results = (outs Variadic<AnyType>:$results);

  // Use a custom assembly format for improved readability.
  let assemblyFormat = "`ID` `(` $task_id `)` `arch` `(` $target_arch `)` `operands` `(` $operands `)` `:` functional-type($operands, $results) attr-dict";

  // Enable the C++ verifier hook to enforce invariants.
  let hasVerifier = 1;

  let regions = (region AnyRegion:$body);
}

def Orchestra_ScheduleOp : Orchestra_Op<"schedule", [SingleBlock]> {
  let summary = "Container for a physically scheduled DAG of tasks.";
  let description = [{
    Contains a single region holding a Directed Acyclic Graph (DAG) of
    `orchestra.task` operations, where the edges of the DAG are represented by
    the SSA use-def chain. The schedule operation itself has no operands or
    results.
  }];

  let regions = (region AnyRegion:$body);
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
  let hasCustomAssemblyFormat = 1;
}

def Orchestra_YieldOp : Orchestra_Op<"yield", [Terminator]> {
  let summary = "Terminator for regions in OrchestraIR operations.";
  let description = [{
    A terminator operation for regions within `orchestra.schedule` and
    `orchestra.task`. It yields values from the region to the parent op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

#endif // ORCHESTRA_OPS_TD
