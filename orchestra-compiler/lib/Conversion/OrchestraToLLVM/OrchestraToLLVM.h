#ifndef ORCHESTRA_CONVERSION_ORCHESTRATOLLVM_H
#define ORCHESTRA_CONVERSION_ORCHESTRATOLLVM_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {
class Pass;

namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
void registerOrchestraToLLVMPass();
} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_CONVERSION_ORCHESTRATOLLVM_H
