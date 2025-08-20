#ifndef ORCHESTRA_TRANSFORMS_PASSES_H
#define ORCHESTRA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace orchestra {

std::unique_ptr<mlir::Pass> createDivergenceToSpeculationPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToStandardPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToGPUPass();

inline std::unique_ptr<mlir::Pass> createLowerOrchestraToXeGPUPass() {
  return nullptr;
}

void registerOrchestraPasses();

void registerLoweringToStandardPasses();
void registerLoweringToGPUPasses();

inline void registerLoweringToXeGPUPasses() {
  // Do nothing.
}

} // namespace orchestra

#endif // ORCHESTRA_TRANSFORMS_PASSES_H
