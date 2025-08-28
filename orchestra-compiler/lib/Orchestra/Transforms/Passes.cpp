#include "Orchestra/Transforms/Passes.h"
#include "Orchestra/Transforms/LowerRockToAMDGPU.h"

#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraOps.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Orchestra/Transforms/LowerLinalgToRock.h"

using namespace mlir;
using namespace mlir::orchestra;

namespace {

// Helper function to find all SSA Values used in a region but defined outside.
static llvm::SetVector<mlir::Value>
getUsedExternalValues(mlir::Region &region) {
  llvm::SetVector<mlir::Value> externalValues;
  region.walk([&](mlir::Operation *op) {
    for (mlir::Value operand : op->getOperands()) {
      if (!region.isAncestor(operand.getParentRegion())) {
        externalValues.insert(operand);
      }
    }
  });
  return externalValues;
}

// Helper to clone a region and remap its arguments. This is the original,
// correct implementation from the C++ pattern.
static void cloneAndRemap(mlir::Region &sourceRegion,
                          mlir::Region &destRegion,
                          const llvm::SetVector<mlir::Value> &externalValues,
                          mlir::PatternRewriter &rewriter) {
  mlir::IRMapping mapper;
  llvm::SmallVector<Type, 4> argTypes;
  for (auto val : externalValues) {
    argTypes.push_back(val.getType());
  }
  llvm::SmallVector<Location, 4> argLocs(argTypes.size(), sourceRegion.getLoc());
  destRegion.front().addArguments(argTypes, argLocs);
  auto destArgs = destRegion.getArguments();
  for (auto pair : llvm::zip(externalValues, destArgs)) {
    mapper.map(std::get<0>(pair), std::get<1>(pair));
  }

  rewriter.setInsertionPointToEnd(&destRegion.front());
  for (auto &op : sourceRegion.front().without_terminator()) {
    rewriter.clone(op, mapper);
  }

  auto sourceYield =
      mlir::cast<mlir::scf::YieldOp>(sourceRegion.front().getTerminator());
  llvm::SmallVector<mlir::Value> yieldOperands;
  for (mlir::Value operand : sourceYield.getOperands()) {
    yieldOperands.push_back(mapper.lookupOrDefault(operand));
  }
  // The destination region belongs to an orchestra.task, which needs an
  // orchestra.yield terminator.
  rewriter.create<orchestra::YieldOp>(sourceYield.getLoc(), yieldOperands);
}

#include "SpeculateIfOp.pdll.inc"

struct DivergenceToSpeculationPass
    : public mlir::PassWrapper<DivergenceToSpeculationPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DivergenceToSpeculationPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<OrchestraDialect,
                    scf::SCFDialect,
                    arith::ArithDialect,
                    pdl::PDLDialect,
                    pdl_interp::PDLInterpDialect>();
  }

  void runOnOperation() override {
    getContext().loadDialect<OrchestraDialect>();
    RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns<>(patterns);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }

  StringRef getArgument() const final {
    return "divergence-to-speculation";
  }
  StringRef getDescription() const final {
    return "Convert scf.if to orchestra speculative execution";
  }
};

}  // anonymous namespace

std::unique_ptr<Pass> orchestra::createDivergenceToSpeculationPass() {
  return std::make_unique<DivergenceToSpeculationPass>();
}

void orchestra::registerLoweringToROCDLPasses() {
  // Do nothing.
}

void mlir::orchestra::registerLoweringToAMDGPUPasses() {
  mlir::registerPass(
      []() -> std::unique_ptr<mlir::Pass> { return createLowerRockToAMDGPUPass(); });
}

void orchestra::registerOrchestraPasses() {
  PassRegistration<DivergenceToSpeculationPass>();
  registerLoweringToStandardPasses();
  registerLoweringToGPUPasses();
  registerLoweringToROCDLPasses();
  registerLoweringToXeGPUPasses();
  mlir::registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createLowerLinalgToRockPass();
  });
}
