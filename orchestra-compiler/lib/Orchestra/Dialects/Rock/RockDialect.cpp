#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

#include "Orchestra/Dialects/Rock/RockDialect.cpp.inc"

void mlir::rock::RockDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Orchestra/Dialects/Rock/RockOps.cpp.inc"
      >();
}
