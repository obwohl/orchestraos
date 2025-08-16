#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Orchestra/OrchestraDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register all standard MLIR dialects.
  registerAllDialects(registry);

  // Register the Orchestra dialect using the modern C++ template-based approach.
  registry.insert<orchestra::OrchestraDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
