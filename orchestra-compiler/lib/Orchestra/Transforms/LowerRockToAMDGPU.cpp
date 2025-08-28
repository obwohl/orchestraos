#include "Orchestra/Transforms/LowerRockToAMDGPU.h"
#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPU.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Rock/IR/Rock.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace orchestra {
namespace {

struct LowerRockToAMDGPUCPUPass
    : public PassWrapper<LowerRockToAMDGPUCPUPass,
                         OperationPass<ModuleOp>> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<amdgpu::AMDGPUDialect, rock::RockDialect,
                    gpu::GPUDialect>();
  }

  void runOnOperation() override {
    // Implementation to follow in the next step.
  }
};

} // namespace

std::unique_ptr<Pass> createLowerRockToAMDGPUCPUPass() {
  return std::make_unique<LowerRockToAMDGPUCPUPass>();
}

} // namespace orchestra
} // namespace mlir
