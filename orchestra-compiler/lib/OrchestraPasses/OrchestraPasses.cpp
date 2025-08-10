#include "OrchestraPasses.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Pass/Pass.h" // For mlir::Pass
#include <functional> // For std::function
#include "DummyPass.h" // Include DummyPass.h to make DummyPass visible

// Forward declaration for the pass creation function
std::unique_ptr<mlir::Pass> createDummyPass();

void registerOrchestraPasses() {
  // Register DummyPass using the PassRegistration utility.
  // This macro automatically handles pass name and description.
  mlir::PassRegistration<DummyPass>();
}
