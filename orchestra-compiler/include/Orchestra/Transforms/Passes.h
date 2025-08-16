#ifndef ORCHESTRA_TRANSFORMS_PASSES_H
#define ORCHESTRA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<Pass> createDivergenceToSpeculationPass();

void registerOrchestraPasses();

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_PASSES_H
