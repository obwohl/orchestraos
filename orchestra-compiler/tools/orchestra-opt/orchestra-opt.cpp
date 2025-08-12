#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// Include our custom dialect.
#include "Orchestra/OrchestraDialect.h"
#include "OrchestraToLLVM.h"

// A function to create our custom conversion pass.
// This should be declared in a header file in a real project.
namespace mlir {
namespace orchestra {
std::unique_ptr<mlir::Pass> createLowerToLLVMPass();
} // namespace orchestra
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();

  // Register our custom pass.
  mlir::PassRegistration<mlir::Pass>(
      "orchestra-to-llvm", "Lower Orchestra dialect to LLVM dialect",
     []() -> std::unique_ptr<mlir::Pass> {
        return mlir::orchestra::createLowerToLLVMPass();
      });

  mlir::DialectRegistry registry;
  // Register all the standard dialects.
  registerAllDialects(registry);
  // Register our custom dialect.
  registry.insert<mlir::orchestra::OrchestraDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra Optimizer Driver", registry));
}
