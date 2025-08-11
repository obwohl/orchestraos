#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Define the dialect itself. This provides the C++ namespace, summary, etc.
def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "orchestra";
  let summary = "A dialect for a hypothetical orchestra compiler.";
  let description = [{
    The Orchestra dialect is for a hypothetical orchestra compiler.
  }];
}

// A base class for all operations in this dialect.
class Orchestra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

// Example operation definition.
def MyOp : Orchestra_Op<"my_op"> {
  let summary = "An example operation.";
  let description = [{
    An example operation in the Orchestra dialect.
  }];

  let arguments = (ins);
  let results = (outs);
}

def YieldOp : Orchestra_Op<"yield", [Terminator]> {
  let summary = "Terminator for regions in OrchestraIR operations.";
  let description = [{
    A terminator operation for regions within `orchestra.schedule` and
    `orchestra.task`. It yields values from the region to the parent op.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);
}

def Orchestra_DummyOp : Orchestra_Op<"dummy_op"> {
  let summary = "A dummy operation for testing purposes.";
  let description = [{
    This is a dummy operation used to test the DummyPass.
  }];
  let arguments = (ins);
  let results = (outs);
}

#endif // ORCHESTRA_OPS_TD
