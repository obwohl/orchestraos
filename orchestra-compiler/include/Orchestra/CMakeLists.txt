# This file defines the TableGen targets for the Orchestra dialect.
# It generates the header files that define the C++ classes for the
# dialect and its operations.

include_directories(${MLIR_INCLUDE_DIRS})

# The first argument is the name of the.td file (without extension).
# The second argument is the C++ namespace for the dialect.
add_mlir_dialect(OrchestraOps orchestra)
