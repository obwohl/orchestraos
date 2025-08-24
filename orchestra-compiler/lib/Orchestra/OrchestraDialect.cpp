#include "Orchestra/OrchestraDialect.h"

#include "Orchestra/OrchestraOps.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"

// Generated headers
#include "Orchestra/OrchestraDialect.cpp.inc"

using namespace mlir;
using namespace orchestra;

void OrchestraDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Orchestra/OrchestraOps.cpp.inc"
      >();
}
