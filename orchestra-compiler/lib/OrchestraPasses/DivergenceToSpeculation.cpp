#include "mlir/Pass/Pass.h"

namespace {
struct DivergenceToSpeculation
    : public mlir::Pass<DivergenceToSpeculation> {
  void runOnOperation() override {}
};

void registerDivergenceToSpeculationPass() {
    PassRegistration<DivergenceToSpeculation>();
}
} // namespace
