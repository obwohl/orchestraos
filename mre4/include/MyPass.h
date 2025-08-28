#ifndef MRE4_MYPASS_H
#define MRE4_MYPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<mlir::Pass> createMyPass();

} // namespace orchestra
} // namespace mlir

#endif // MRE4_MYPASS_H
