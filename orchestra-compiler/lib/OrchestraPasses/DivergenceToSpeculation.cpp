#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringRef.h"

#include "Orchestra/OrchestraDialect.h"

namespace {
struct DivergenceToSpeculation
    : public mlir::PassWrapper<DivergenceToSpeculation,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DivergenceToSpeculation)

  void runOnOperation() override {
    // The following line is a test to ensure that the pass can see the
    // definitions of the Orchestra dialect's operations.
    orchestra::YieldOp *myOp = nullptr;
    (void)myOp; // Avoid unused variable warning.
  }

  llvm::StringRef getArgument() const final {
    return "divergence-to-speculation";
  }

  llvm::StringRef getDescription() const final {
    return "Convert divergence to speculation.";
  }
};
} // namespace

namespace mlir {
std::unique_ptr<mlir::Pass> createDivergenceToSpeculationPass() {
  return std::make_unique<DivergenceToSpeculation>();
}
} // namespace mlir
