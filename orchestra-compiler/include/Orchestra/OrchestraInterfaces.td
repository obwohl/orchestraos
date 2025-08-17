#ifndef ORCHESTRA_INTERFACES_TD
#define ORCHESTRA_INTERFACES_TD

include "mlir/IR/OpBase.td"

def OrchestraFusionStrategyInterface : OpInterface<"OrchestraFusionStrategyInterface"> {
  let description = [{
    Interface for hardware-specific fusion profitability analysis.
  }];

  let methods = [
    InterfaceMethod<"Indicates whether it is profitable to fuse a producer operation "
                    "into a consumer operation.",
                    "bool", "isFusionProfitable",
                    (ins "mlir::Operation *":$producer, "mlir::Operation *":$consumer)>
  ];
}

#endif // ORCHESTRA_INTERFACES_TD
