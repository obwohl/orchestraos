#include "Orchestra/Transforms/Passes.h"
#include "Orchestra/Transforms/SpeculateIfOp.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

using namespace mlir;
using namespace orchestra;

namespace {

struct DivergenceToSpeculationPass
    : public mlir::PassWrapper<DivergenceToSpeculationPass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DivergenceToSpeculationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<OrchestraDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<SpeculateIfOpPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final { return "divergence-to-speculation"; }
  StringRef getDescription() const final {
    return "Convert scf.if to orchestra speculative execution";
  }
};

} // anonymous namespace

std::unique_ptr<Pass> orchestra::createDivergenceToSpeculationPass() {
  return std::make_unique<DivergenceToSpeculationPass>();
}

void orchestra::registerOrchestraPasses() {
  PassRegistration<DivergenceToSpeculationPass>();
  registerLoweringToStandardPasses();
  registerLoweringToGPUPasses();
  registerLoweringToXeGPUPasses();
}
