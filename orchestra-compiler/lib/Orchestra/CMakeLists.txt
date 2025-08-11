# This file's responsibility is to compile the C++ sources into a library.
add_mlir_dialect_library(Orchestra
  OrchestraDialect.cpp

  # CRITICAL: This establishes the build-time dependency.
  # It ensures that the .inc files are generated before OrchestraDialect.cpp is compiled.
  DEPENDS
  MLIROrchestraOpsIncGen

  # Establish link-time dependencies.
  LINK_LIBS
  PUBLIC
  MLIRIR
  MLIRSupport
  MLIRParser
)

target_include_directories(Orchestra PUBLIC ${CMAKE_BINARY_DIR}/include)
