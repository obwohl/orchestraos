#ifndef ORCHESTRA_TRANSFORMS_LOWERLINALGTOROCK_H
#define ORCHESTRA_TRANSFORMS_LOWERLINALGTOROCK_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<mlir::Pass> createLowerLinalgToRockPass();

void registerLowerLinalgToRockPass();

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_LOWERLINALGTOROCK_H
