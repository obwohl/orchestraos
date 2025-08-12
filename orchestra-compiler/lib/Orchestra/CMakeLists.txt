# This file's responsibility is to compile C++ sources into object files.

# Define the source files.
set(ORCHESTRA_SOURCES
  OrchestraDialect.cpp
)

# Create an OBJECT library. This compiles the sources to object files,
# but does not archive them into a static library.
add_library(Orchestra OBJECT
  ${ORCHESTRA_SOURCES}
)

# Add a dependency to ensure the .inc files are generated before compiling.
add_dependencies(Orchestra MLIROrchestraOpsIncGen)

# Associate the necessary MLIR libraries with the object library.
# These link dependencies will be forwarded to any target that
# consumes these object files.
target_link_libraries(Orchestra PUBLIC
  MLIRIR
  MLIRSupport
  MLIRParser
)