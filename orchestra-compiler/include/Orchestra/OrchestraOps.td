#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the dialect itself. This provides the C++ namespace, summary, etc.
def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "::orchestra";
  let summary = "A dialect for a hypothetical orchestra compiler.";
  let description = [{
    The Orchestra dialect is for a hypothetical orchestra compiler.
  }];
}

// A base class for all operations in this dialect.
class Orchestra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

def YieldOp : Orchestra_Op<"yield", [Terminator]> {
  let summary = "Terminator for regions in OrchestraIR operations.";
  let description = [{
    A terminator operation for regions within `orchestra.schedule` and
    `orchestra.task`. It yields values from the region to the parent op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

#endif // ORCHESTRA_OPS_TD
