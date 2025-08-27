#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::orchestra::rock;

#include "Orchestra/Dialects/Rock/RockOpsDialect.cpp.inc"

void RockDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Orchestra/Dialects/Rock/RockOps.cpp.inc"
      >();
}
