#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Orchestra/OrchestraRegistration.h"
#include "OrchestraToLLVM.h"
#include "mlir/InitAllPasses.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all standard MLIR dialects.
  mlir::registerAllDialects(registry);

  // Register all standard MLIR passes.
  mlir::registerAllPasses();

  // Register Orchestra passes.
  mlir::orchestra::registerOrchestraToLLVMPass();

  // Explicitly register the Orchestra dialect. This is the crucial step
  // that ensures the dialect is available to the tool, bypassing the
  // issues with static initializers and LTO.
  registerOrchestraDialect(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
