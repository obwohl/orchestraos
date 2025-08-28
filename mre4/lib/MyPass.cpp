#include "MyPass.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace {
class MyPass : public mlir::PassWrapper<MyPass, mlir::OperationPass<mlir::func::FuncOp>> {
public:
  void runOnOperation() override {}
  mlir::StringRef getArgument() const final { return "my-pass"; }
  mlir::StringRef getDescription() const final { return "My Pass"; }
};
} // namespace

namespace mlir {
namespace orchestra {

std::unique_ptr<mlir::Pass> createMyPass() {
  return std::make_unique<MyPass>();
}

} // namespace orchestra
} // namespace mlir
