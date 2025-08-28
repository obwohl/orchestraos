#include "Orchestra/Transforms/LowerLinalgToRock.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Builders.h"
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
  mlir::OpBuilder builder(func);

  func.walk([&](mlir::linalg::MatmulOp matmulOp) {
    if (!matmulOp.hasPureTensorSemantics())
      return;

    builder.setInsertionPoint(matmulOp);
    auto rockGemmOp = builder.create<mlir::rock::GemmOp>(
        matmulOp.getLoc(),
        matmulOp.getDpsInitOperand(0)->get().getType(),
        builder.getStringAttr(""), // arch
        matmulOp.getDpsInputOperand(0)->get(), // matrix_a
        matmulOp.getDpsInputOperand(1)->get(), // matrix_b
        matmulOp.getDpsInitOperand(0)->get()   // matrix_c
        );

    matmulOp.getResult(0).replaceAllUsesWith(rockGemmOp.getResult());
    matmulOp.erase();
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
