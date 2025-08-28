#ifndef ORCHESTRA_TRANSFORMS_LOWERROCKTOAMDGPU_H
#define ORCHESTRA_TRANSFORMS_LOWERROCKTOAMDGPU_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace orchestra {

std::unique_ptr<mlir::Pass> createLowerRockToAMDGPUPass();

} // namespace orchestra
} // namespace mlir

#endif // ORCHESTRA_TRANSFORMS_LOWERROCKTOAMDGPU_H
