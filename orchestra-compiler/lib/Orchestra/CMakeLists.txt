add_mlir_dialect_library(Orchestra
  # List of C++ source files to compile.
  OrchestraDialect.cpp

  # CRITICAL: This establishes the build-time dependency.
  DEPENDS
  MLIROrchestraOpsIncGen

  # Establish link-time dependencies.
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRParser
)
