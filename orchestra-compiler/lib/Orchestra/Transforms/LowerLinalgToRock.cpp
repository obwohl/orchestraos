#include "Orchestra/Transforms/LowerLinalgToRock.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "Orchestra/Dialects/Rock/RockDialect.h"
#include "Orchestra/Dialects/Rock/RockOps.h"

#define GET_OP_CLASSES
#include "Orchestra/Dialects/Rock/RockOps.cpp.inc"

namespace {
struct LowerLinalgToRockPass
    : public mlir::PassWrapper<LowerLinalgToRockPass,
                               mlir::OperationPass<mlir::func::FuncOp>> {
  void runOnOperation() override;

  mlir::StringRef getArgument() const final {
    return "lower-linalg-to-rock";
  }

  mlir::StringRef getDescription() const final {
    return "Lower Linalg operations to Rock operations.";
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::rock::RockDialect, mlir::linalg::LinalgDialect>();
  }
};
} // namespace

void LowerLinalgToRockPass::runOnOperation() {
  mlir::func::FuncOp func = getOperation();
  mlir::MLIRContext *context = &getContext();
  mlir::OpBuilder builder(context);

  func.walk([&](mlir::linalg::GenericOp genericOp) {
    if (genericOp.getNumDpsInits() != 1 || genericOp.getNumDpsInputs() != 2) {
      return;
    }

    builder.setInsertionPoint(genericOp);
    auto rockGemmOp = builder.create<mlir::rock::GemmOp>(
        genericOp.getLoc(), genericOp.getDpsInitOperand(0)->get().getType(),
        builder.getStringAttr(""), genericOp.getDpsInputOperand(0)->get(),
        genericOp.getDpsInputOperand(1)->get());

    genericOp.replaceAllUsesWith(mlir::ValueRange{rockGemmOp.getResult()});
    genericOp.erase();
  });
}

void mlir::orchestra::registerLowerLinalgToRockPass() {
  PassRegistration<LowerLinalgToRockPass>();
}

namespace mlir {
namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerLinalgToRockPass() {
  return std::make_unique<LowerLinalgToRockPass>();
}
} // namespace orchestra
} // namespace mlir
