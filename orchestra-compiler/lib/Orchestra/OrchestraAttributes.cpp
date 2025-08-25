#include "Orchestra/OrchestraAttributes.h"
#include "Orchestra/OrchestraDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/ADT/StringExtras.h"

#define GET_ATTRDEF_CLASSES
#include "Orchestra/OrchestraAttributes.cpp.inc"

namespace mlir {
namespace orchestra {

void OrchestraDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Orchestra/OrchestraAttributes.h.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TargetAttr
//===----------------------------------------------------------------------===//

Attribute TargetAttr::parse(AsmParser &parser, Type type) {
  if (parser.parseLess())
    return {};

  auto *ctx = parser.getContext();
  StringAttr arch;
  IntegerAttr device_id;
  DictionaryAttr options;

  // This is a simplified parser. A real one would be more robust.
  if (parser.parseAttribute(arch, "arch") || parser.parseComma() ||
      parser.parseAttribute(device_id, "device_id")) {
    parser.emitError(parser.getNameLoc(), "expected 'arch' and 'device_id'");
    return {};
  }

  // Parse optional parameters.
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseAttribute(options)) {
        parser.emitError(parser.getNameLoc(), "expected a dictionary of options");
        return {};
    }
  } else {
    options = DictionaryAttr::get(ctx);
  }

  if (parser.parseGreater())
    return {};
  return get(ctx, arch, device_id, options);
}

void TargetAttr::print(AsmPrinter &printer) const {
  printer << "<";
  printer.printAttribute(getArch());
  printer << ", ";
  printer.printAttribute(getDeviceId());
  if (getOptions() && !getOptions().empty()) {
    printer << ", ";
    printer.printAttribute(getOptions());
  }
  printer << ">";
}

LogicalResult TargetAttr::verify(
    function_ref<InFlightDiagnostic()> emitError, StringAttr arch,
    IntegerAttr device_id, DictionaryAttr options) {
  if (!arch) {
    return emitError() << "target attribute requires 'arch' attribute of type StringAttr";
  }
  if (arch.getValue().empty()) {
    return emitError() << "'arch' attribute cannot be an empty string";
  }
  if (!device_id) {
    return emitError() << "target attribute requires 'device_id' attribute of type IntegerAttr";
  }
  return success();
}

} // namespace orchestra
} // namespace mlir
