# This file's responsibility is to compile the C++ sources into a library.
add_mlir_dialect_library(Orchestra
  # List of C++ source files to compile.
  OrchestraDialect.cpp
  OrchestraOps.cpp
  OrchestraInterfaces.cpp

  # CRITICAL: This establishes the build-time dependency.
  # It ensures that the.inc files are generated before the C++ sources are compiled.
  DEPENDS
  OrchestraIncGen

  # Establish link-time dependencies.
  # All dialects need MLIRIR. Add others as needed (e.g., MLIRParser).
  # PUBLIC visibility propagates this link dependency to any target that links against Orchestra.
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRParser
)

add_subdirectory(Dialects)