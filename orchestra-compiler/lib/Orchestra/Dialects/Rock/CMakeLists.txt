# This file handles the C++ library compilation for the Rock dialect.
# The TableGen generation is handled by the CMakeLists.txt in the
# include/Orchestra/Dialects/Rock/ directory.
add_mlir_dialect_library(OrchestraRock
  # List of C++ source files to compile.
  RockDialect.cpp

  # Add a dependency on the IncGen target defined in the include/ tree.
  # This ensures the .h.inc and .cpp.inc files are generated before compiling.
  DEPENDS
  RockIncGen

  # Link against required MLIR libraries.
  LINK_LIBS
  PUBLIC
  MLIRIR
)
