#include "Orchestra/OrchestraDialect.h"

#include "Orchestra/OrchestraDialect.cpp.inc"

void mlir::orchestra::OrchestraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Orchestra/OrchestraOps.cpp.inc"
  >();
}
