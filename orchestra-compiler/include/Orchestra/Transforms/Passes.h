#ifndef ORCHESTRA_TRANSFORMS_PASSES_H
#define ORCHESTRA_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<Pass> createDivergenceToSpeculationPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToStandardPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToGPUPass();
std::unique_ptr<mlir::Pass> createLowerOrchestraToXeGPUPass();

void registerOrchestraPasses();

void registerLoweringToStandardPasses();
void registerLoweringToGPUPasses();
void registerLoweringToXeGPUPasses();

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_PASSES_H
