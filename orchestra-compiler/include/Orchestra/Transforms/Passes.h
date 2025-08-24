#ifndef ORCHESTRA_TRANSFORMS_PASSES_H
#define ORCHESTRA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<mlir::Pass> createDivergenceToSpeculationPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToStandardPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToGPUPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToROCDLPass();

inline std::unique_ptr<mlir::Pass> createLowerOrchestraToXeGPUPass() {
  return nullptr;
}

void registerOrchestraPasses();

void registerLoweringToStandardPasses();
void registerLoweringToGPUPasses();
void registerLoweringToROCDLPasses();

inline void registerLoweringToXeGPUPasses() {
  // Do nothing.
}

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_PASSES_H
