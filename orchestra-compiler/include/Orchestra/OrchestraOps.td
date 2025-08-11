#ifndef ORCHESTRA_OPS_TD
#define ORCHESTRA_OPS_TD

include "mlir/IR/OpBase.td"

// Define the dialect itself. This provides the C++ namespace, summary, etc.
def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "::orchestra";
  let summary = "A dialect for a hypothetical orchestra compiler.";
  let description = [{
    This dialect is part of the Orchestra compiler project.
  }];

  // Temporarily revert to the pre-MLIR-18 attribute storage mechanism
  // to avoid known compilation issues with MLIR 18.1.
  let usePropertiesForAttributes = 0;
}

// A base class for all operations in this dialect.
class Orchestra_Op<string mnemonic, list<Trait> traits = []> :
    Op<Orchestra_Dialect, mnemonic, traits>;

// A minimal, simple operation to establish a working build.
def MyOp : Orchestra_Op<"my_op"> {
  let summary = "An example operation.";
  let description = [{
    A simple placeholder operation.
  }];

  let arguments = (ins);
  let results = (outs);
}

#endif // ORCHESTRA_OPS_TD
