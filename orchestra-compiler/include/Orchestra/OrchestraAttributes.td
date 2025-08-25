#ifndef ORCHESTRA_ATTRIBUTES_TD
#define ORCHESTRA_ATTRIBUTES_TD

include "mlir/IR/AttrTypeBase.td"

def TargetAttr : AttrDef<Orchestra_Dialect, "Target"> {
  let summary = "Attribute specifying task target information.";

  let parameters = [
    AttrParameter<"arch", "::mlir::StringAttr", "target architecture">,
    AttrParameter<"device_id", "::mlir::IntegerAttr", "target device ID">,
    AttrParameter<"options", "::mlir::DictionaryAttr", "optional target-specific parameters",
                  /*defaultValue=*/"::mlir::DictionaryAttr::get($_self.getContext())">
  ];

  let mnemonic = "target";

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

#endif // ORCHESTRA_ATTRIBUTES_TD
