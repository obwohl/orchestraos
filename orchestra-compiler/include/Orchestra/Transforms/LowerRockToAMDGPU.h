#ifndef ORCHESTRA_TRANSFORMS_LOWERRockTOAMDGPU_H
#define ORCHESTRA_TRANSFORMS_LOWERRockTOAMDGPU_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<Pass> createLowerRockToAMDGPUCPUPass();

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_LOWERRockTOAMDGPU_H
