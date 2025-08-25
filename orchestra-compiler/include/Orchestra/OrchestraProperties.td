#ifndef ORCHESTRA_PROPERTIES_TD
#define ORCHESTRA_PROPERTIES_TD

// The declarative contract for the custom struct property.
def TargetArchProp : StructProp<"TargetArch", "::orchestra::TargetArch",
                                "Target architecture property"> {
  let cppStorageType = "::orchestra::TargetArch";

  // The parser delegates to the C++ OpTrait via parseCustom.
  let parser =;

  // The printer can simply use the C++ operator<< overload, which will be
  // resolved by the AsmPrinter.
  let printer = [{ $_printer << $_self; }];
}

#endif // ORCHESTRA_PROPERTIES_TD
