#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Orchestra/OrchestraDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<::orchestra::OrchestraDialect>();
  registerAllDialects(registry);
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
