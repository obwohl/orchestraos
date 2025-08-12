#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/OrchestraRegistration.h"

int main(int argc, char **argv) {
  // This call is essential to prevent the compiler/linker from optimizing
  // away the dialect's operations.
  orchestra::ensureOrchestraDialectRegistered();

  mlir::DialectRegistry registry;

  // Register the dialects we want to support.
  registry.insert<orchestra::OrchestraDialect>();
  registry.insert<mlir::func::FuncDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
