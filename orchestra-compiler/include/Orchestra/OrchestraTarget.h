#ifndef ORCHESTRA_TARGET_H
#define ORCHESTRA_TARGET_H

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace orchestra {

// A C++ helper class to provide a type-safe API for the 'target'
// attribute of the 'orchestra.task' operation.
class OrchestraTarget {
public:
  explicit OrchestraTarget(mlir::Attribute attr)
      : attr(attr.dyn_cast_or_null<mlir::DictionaryAttr>()) {}

  mlir::StringRef getArch() const {
    if (!attr) return "";
    auto archAttr = attr.get("arch").dyn_cast_or_null<mlir::StringAttr>();
    if (!archAttr) return "";
    return archAttr.getValue();
  }

  int32_t getDeviceId() const {
    if (!attr) return -1;
    auto deviceIdAttr = attr.get("device_id").dyn_cast_or_null<mlir::IntegerAttr>();
    if (!deviceIdAttr) return -1;
    return deviceIdAttr.getInt();
  }

  // Verifies the structure of the target attribute.
  mlir::LogicalResult verify(mlir::Operation *op) {
    if (!attr) {
      return op->emitOpError("requires a 'target' attribute to be a dictionary");
    }

    auto archAttr = attr.get("arch").dyn_cast_or_null<mlir::StringAttr>();
    if (!archAttr) {
      return op->emitOpError(
          "requires a string 'arch' key in the 'target' dictionary");
    }

    if (archAttr.getValue().empty()) {
      return op->emitOpError("'arch' key in 'target' dictionary cannot be empty");
    }

    auto deviceIdAttr = attr.get("device_id");
    if (!deviceIdAttr) {
      return op->emitOpError("requires an 'device_id' key in the 'target' dictionary");
    }

    if (!deviceIdAttr.isa<mlir::IntegerAttr>()) {
      return op->emitOpError("'device_id' key in 'target' dictionary must be an integer");
    }

    return mlir::success();
  }

private:
  mlir::DictionaryAttr attr;
};

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TARGET_H
