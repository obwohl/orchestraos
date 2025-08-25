#ifndef ORCHESTRA_PROPERTIES_H
#define ORCHESTRA_PROPERTIES_H

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace orchestra {

struct TargetArch {
  std::string arch;
  // In the future, we can add more fields here, like:
  // int sm_version;
  // llvm::SmallVector<std::string> features;
};

} // namespace orchestra

namespace mlir {
template <>
struct OpTrait::Property<orchestra::TargetArch> {
  using Type = orchestra::TargetArch;

  static constexpr StringLiteral name = "target_arch";

  static orchestra::TargetArch parse(AsmParser &parser, Type &value) {
    if (parser.parseLess())
      return {};

    if (failed(parser.parseOptionalKeyword("arch"))) {
        parser.emitError(parser.getCurrentLocation(), "expected 'arch' key in target_arch property");
        return {};
    }

    if (parser.parseEqual())
      return {};

    std::string archValue;
    if (parser.parseString(&archValue))
      return {};

    if (parser.parseGreater())
      return {};

    value.arch = archValue;
    return value;
  }

  static void print(AsmPrinter &printer, const Type &value) {
    printer << "<{arch = \"" << value.arch << "\"}>";
  }
};
} // namespace mlir

#endif // ORCHESTRA_PROPERTIES_H
