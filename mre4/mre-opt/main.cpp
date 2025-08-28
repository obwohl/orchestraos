#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "Orchestra/OrchestraDialect.h"
#include "Passes.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::orchestra::OrchestraDialect>();
  mlir::registerAllPasses();
  mlir::orchestra::registerMyPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "MRE4 optimizer driver\n", registry));
}
