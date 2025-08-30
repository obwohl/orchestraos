#ifndef ORCHESTRA_TRANSFORMS_LOWERRockToAMDGPU_H
#define ORCHESTRA_TRANSFORMS_LOWERRockToAMDGPU_H

namespace mlir {
class Pass;

namespace rock {
std::unique_ptr<mlir::Pass> createLowerRockToAMDGPUConversionPass();
void registerLowerRockToAMDGPU();
} // namespace rock
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_LOWERRockToAMDGPU_H
