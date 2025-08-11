#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Orchestra/OrchestraDialect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register the dialects we want to support.
  registry.insert<orchestra::OrchestraDialect>();
  registry.insert<mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
