#include "DummyPass.h" // Include the new header
#include "llvm/Support/raw_ostream.h"
#include <memory> // For std::unique_ptr and std::make_unique

// Manually define the TypeID
mlir::TypeID DummyPass::getTypeID() {
  static void *id = nullptr;
  if (!id)
    id = new char();
  return mlir::TypeID::getFromOpaquePointer(id);
}

void DummyPass::runOnOperation() {
  llvm::errs() << "DummyPass ran on an operation.\n";
}

std::unique_ptr<mlir::Pass> createDummyPass() {
  return std::make_unique<DummyPass>();
}


