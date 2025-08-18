#include "Orchestra/Transforms/Passes.h"
#include "Orchestra/Transforms/SpeculateIfOp.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace {

struct DivergenceToSpeculationPass
    : public mlir::PassWrapper<DivergenceToSpeculationPass, mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DivergenceToSpeculationPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::orchestra::OrchestraDialect, mlir::scf::SCFDialect>();
  }

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<mlir::orchestra::SpeculateIfOpPattern>(&getContext());
    if (failed(mlir::applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  mlir::StringRef getArgument() const final { return "divergence-to-speculation"; }
  mlir::StringRef getDescription() const final { return "Convert scf.if to orchestra speculative execution"; }
};

} // anonymous namespace

std::unique_ptr<mlir::Pass> mlir::orchestra::createDivergenceToSpeculationPass() {
  return std::make_unique<DivergenceToSpeculationPass>();
}

void mlir::orchestra::registerOrchestraPasses() {
  mlir::PassRegistration<DivergenceToSpeculationPass>();
  registerLoweringToStandardPasses();
  registerLoweringToGPUPasses();
  registerLoweringToXeGPUPasses();
}
