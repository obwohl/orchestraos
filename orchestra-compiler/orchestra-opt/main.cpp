#include "Orchestra/OrchestraDialect.h"
#include "Orchestra/Transforms/Passes.h"
#include "Orchestra/Transforms/LowerLinalgToRock.h"
#include "Orchestra/Transforms/LowerLinalgToRock.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

// For the transform dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/TransformOps/SCFTransformOps.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/TransformOps/VectorTransformOps.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  registerAllDialects(registry);

  registry.insert<mlir::orchestra::OrchestraDialect,
                  mlir::transform::TransformDialect>();

  // Register transform dialect extensions.
  mlir::linalg::registerTransformDialectExtension(registry);
  mlir::scf::registerTransformDialectExtension(registry);
  mlir::vector::registerTransformDialectExtension(registry);

  mlir::registerAllPasses();
  mlir::orchestra::registerOrchestraPasses();
  mlir::orchestra::registerLowerLinalgToRockPass();
  mlir::registerPassManagerCLOptions();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Orchestra optimizer driver\n", registry));
}
