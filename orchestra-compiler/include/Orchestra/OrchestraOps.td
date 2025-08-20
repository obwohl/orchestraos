#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

include "OrchestraInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

// A base class for all operations in this dialect.
class Orchestra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

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

def Orchestra_TaskOp : Orchestra_Op<"task", [SingleBlock]> {
  let summary = "An asynchronous unit of computation assigned to a resource.";
  let description = [{
    Encapsulates an atomic unit of computation assigned to a specific hardware
    resource. Its `target` attribute provides a flexible mechanism for
    specifying fine-grained placement constraints.
  }];

  let arguments = (ins Variadic<AnyType>:$operands, DictionaryAttr:$target);
  let results = (outs Variadic<AnyType>:$results);

  let regions = (region AnyRegion:$body);
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::ValueRange":$operands, "mlir::TypeRange":$resultTypes,
                   "mlir::DictionaryAttr":$target),
      [{
        odsState.addOperands(operands);
        odsState.addTypes(resultTypes);
        odsState.addAttribute("target", target);
        mlir::Region *bodyRegion = odsState.addRegion();
        mlir::Block *bodyBlock = new mlir::Block();
        bodyRegion->push_back(bodyBlock);
        bodyBlock->addArguments(
            operands.getTypes(),
            llvm::SmallVector<mlir::Location>(operands.size(), odsState.location));
      }]>
  ];
}

def Orchestra_TransferOp : Orchestra_Op<"transfer", [AllTypesMatch<["source", "result"]>]> {
  let summary = "Explicitly represents data movement between memory spaces.";
  let description = [{
    Makes data movement between different logical memory spaces an explicit,
    first-class citizen of the IR. The `from` and `to` locations are
    represented by `SymbolRefAttr`, which will be resolved to physical memory
    spaces during a later lowering stage.
  }];

  let arguments = (ins AnyShaped:$source, SymbolRefAttr:$from, SymbolRefAttr:$to);
  let results = (outs AnyShaped:$result);
  let hasVerifier = 1;

  let assemblyFormat = "$source `from` $from `to` $to attr-dict `:` type($source)";
}

def Orchestra_CommitOp : Orchestra_Op<"commit", [AttrSizedOperandSegments]> {
  let summary = "Commits a set of values, conditioned on an i1 predicate.";

  let description = [{
    The `commit` operation has three logical operand segments:
    1. A single `i1` condition.
    2. A variadic list of values for the 'true' case.
    3. A variadic list of values for the 'false' case.

    The operation itself has no results. The segment sizes are stored in
    an attribute named `operand_segment_sizes`, managed by the
    `AttrSizedOperandSegments` trait.
  }];

  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$true_values,
    Variadic<AnyType>:$false_values
  );

  let results = (outs Variadic<AnyType>:$results);
  let hasVerifier = 1;
  let hasCanonicalizer = 1;

  let assemblyFormat = "$condition `true` `(` $true_values `)` `false` `(` $false_values `)` attr-dict `:` functional-type(operands, results)";
}

//===----------------------------------------------------------------------===//
// Canonicalization Patterns
//===----------------------------------------------------------------------===//

include "mlir/IR/PatternBase.td"

// C++ helper to merge attributes and create the new fused op.
def FuseTransferOps
    : NativeCodeCall<"fuseTransferOps($_builder, $0, $1)">;

// Constraint to check if an SSA value has exactly one use.
def HasOneUse : Constraint<CPred<"$0.getResult(0).hasOneUse()">>;

def TransferFusionPattern : Pat<
  // Match a transfer whose source is another transfer.
  (Orchestra_TransferOp:$final_op
    (Orchestra_TransferOp:$intermediate_op $source, $from1, $to1),
    $from2, $to2),

  // Rewrite by calling the C++ helper.
  (FuseTransferOps $intermediate_op, $final_op),

  // Apply C++ constraints.
  [(HasOneUse $intermediate_op)]
>;


def Orchestra_YieldOp : Orchestra_Op<"yield", [Terminator]> {
  let summary = "Terminator for regions in OrchestraIR operations.";
  let description = [{
    A terminator operation for regions within `orchestra.schedule` and
    `orchestra.task`. It yields values from the region to the parent op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

#endif // ORCHESTRA_OPS_TD
