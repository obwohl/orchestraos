# Defines the library for the Orchestra-to-LLVM conversion pass.
add_mlir_conversion_library(OrchestraToLLVM
  # 1. Source files for this library.
  OrchestraToLLVM.cpp

  # 2. List of CMake targets this library *depends on*.
  #    This is for build-time ordering, primarily for generated files.
  #    We depend on the Orchestra dialect's TableGen target because our
  #    C++ code will include its generated headers (e.g., OrchestraOps.h).
  DEPENDS
  MLIROrchestraOpsIncGen

  # 3. List of libraries to *link with*.
  #    This is for resolving symbols at link time. We need the compiled
  #    library for our source dialect (Orchestra) and our target
  #    dialect (LLVM).
  LINK_LIBS
  MLIROrchestra
  MLIRLLVMDialect
)
