# This file defines the compiled C++ library for the Orchestra dialect.

add_mlir_dialect_library(MLIROrchestra
  # Source files for the library.
  OrchestraDialect.cpp

  # This library DEPENDS on the TableGen target from include/Orchestra.
  # This ensures the.h.inc files are generated before this library is compiled.
  # The target name is MLIR<TdFileName>IncGen.
  DEPENDS
  MLIROrchestraOpsIncGen
)