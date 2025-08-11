#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the dialect itself.
def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "::orchestra";
  let summary = "The Orchestra dialect, for orchestrating computation.";
  let description = [{
    The Orchestra dialect is a mid-level dialect designed to serve as the
    crucial bridge between a purely logical, framework-agnostic representation
    of a computation and the concrete, hardware-specific representations
    required for final code generation.
  }];
  let usePropertiesForAttributes = 0; // Temporarily disable properties for MLIR 18.1 compatibility
}

// Base class for Orchestra operations.
class Orchestra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Orchestra Operations
//===----------------------------------------------------------------------===//

def Orchestra_ScheduleOp : Orchestra_Op<"schedule"> {
  let summary = "Container for a physically scheduled DAG of tasks.";
  let description = [{
    This operation acts as a container for a physically scheduled subgraph.
    It contains a single region holding a Directed Acyclic Graph (DAG) of
    orchestra.task operations, where the edges of the DAG are represented
    by the SSA use-def chain.
  }];

  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$body);
  let hasCanonicalizer = 1;
}

def Orchestra_TaskOp : Orchestra_Op<"task"> {
  let summary = "An asynchronous unit of computation assigned to a resource.";
  let description = [{
    This operation encapsulates an atomic unit of computation assigned to a
    specific hardware resource. Its target attribute is defined as a
    DictionaryAttr, providing a flexible and extensible mechanism for
    specifying fine-grained placement constraints.
  }];

  let arguments = (ins Variadic<AnyType>:$operands,
                       DictionaryAttr:$target);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region SizedRegion<1>:$body);
}

def Orchestra_TransferOp : Orchestra_Op<"transfer", [AllTypesMatch<["source", "result"]>]> {
  let summary = "Explicitly represents data movement between memory spaces.";
  let description = [{
    This operation makes data movement between different logical memory spaces
    an explicit, first-class citizen of the IR. The from and to locations are
    represented by SymbolRefAttr, which will be resolved to physical memory
    spaces during a later lowering stage.
  }];

  let arguments = (ins AnyShaped:$source,
                       SymbolRefAttr:$from,
                       SymbolRefAttr:$to);
  let results = (outs AnyShaped:$result);
}

def Orchestra_CommitOp : Orchestra_Op<"commit", [SameVariadicOperandSize, SameVariadicResultSize]> {
  let summary = "Selects one of two SSA values based on a boolean condition.";
  let description = [{
    This operation is essential for the speculative execution pattern.
    It selects one of two sets of SSA values based on a boolean condition,
    materializing the result of the speculation.
  }];

  let arguments = (ins I1:$condition,
                       Variadic<AnyType>:$true_values,
                       Variadic<AnyType>:$false_values);
  let results = (outs Variadic<AnyType>:$results);

  let hasVerifier = 1;
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
