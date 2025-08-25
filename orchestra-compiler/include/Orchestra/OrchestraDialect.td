#ifndef ORCHESTRA_DIALECT_TD
#define ORCHESTRA_DIALECT_TD

include "mlir/IR/OpBase.td"

// Include all dialect components in the correct dependency order.
include "OrchestraProperties.td"
include "OrchestraOps.td"

// Finally, define the dialect itself.
def Orchestra_Dialect : Dialect {
  let name = "orchestra";
  let cppNamespace = "::orchestra";
  let summary = "The Orchestra dialect for task-based compilation.";

  // Enable the property-based storage model for all ops in this dialect.
  let usePropertiesForAttributes = 1;

  let description = [{
    The Orchestra dialect provides a high-level IR for describing asynchronous
    tasks and their data dependencies, suitable for targeting heterogeneous
    hardware.
  }];
}

#endif // ORCHESTRA_DIALECT_TD
