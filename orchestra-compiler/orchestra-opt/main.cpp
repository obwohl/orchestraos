#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/Transforms/Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registerAllDialects(registry);

  registry.insert<mlir::orchestra::OrchestraDialect>();

  mlir::orchestra::registerOrchestraPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
