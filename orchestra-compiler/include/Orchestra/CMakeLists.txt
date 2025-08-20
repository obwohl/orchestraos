# This variable will hold the list of all generated header files (.h.inc).
set(ORCHESTRA_INCGEN_HDRS)

# This variable will hold the list of all generated source files (.cpp.inc).
set(ORCHESTRA_INCGEN_SRCS)

# Set the LLVM_TARGET_DEFINITIONS variable ONCE to the single, top-level .td file.
# This file, Orchestra.td, is expected to include all other dialect-related .td files.
set(LLVM_TARGET_DEFINITIONS Orchestra.td)

# Now, invoke mlir_tablegen for each required generated file.
# Each of these calls implicitly uses the same LLVM_TARGET_DEFINITIONS context
# established above, ensuring a consistent view for every generator.

# Generate dialect class declarations (e.g., OrchestraDialect).
mlir_tablegen(OrchestraDialect.h.inc --gen-dialect-decls)
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraDialect.h.inc")

# Generate operation class declarations (e.g., CommitOp).
mlir_tablegen(OrchestraOps.h.inc --gen-op-decls)
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraOps.h.inc")

# Generate operation class definitions (implementations).
mlir_tablegen(OrchestraOps.cpp.inc --gen-op-defs)
list(APPEND ORCHESTRA_INCGEN_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraOps.cpp.inc")

# Generate interface class declarations.
mlir_tablegen(OrchestraInterfaces.h.inc --gen-op-interface-decls)
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraInterfaces.h.inc")

# Generate interface class definitions (implementations).
mlir_tablegen(OrchestraInterfaces.cpp.inc --gen-op-interface-defs)
list(APPEND ORCHESTRA_INCGEN_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraInterfaces.cpp.inc")

# --- Generation for canonicalization patterns ---
mlir_tablegen(OrchestraCanonicalization.h.inc --gen-rewriters)
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraCanonicalization.h.inc")

# --- Create a single dependency target ---
# This creates a CMake target that represents all the generated files.
# The C++ library in the../lib/Orchestra directory can then simply
# add a dependency on "OrchestraIncGen" to ensure all headers are
# generated before any C++ files are compiled.
add_custom_target(OrchestraIncGen ALL
  DEPENDS
    ${ORCHESTRA_INCGEN_HDRS}
    ${ORCHESTRA_INCGEN_SRCS}
  )

# Propagate the lists of generated files to the parent scope, which can be
# useful for IDE integration or other build system logic.
set(ORCHESTRA_INCGEN_HDRS ${ORCHESTRA_INCGEN_HDRS} PARENT_SCOPE)
set(ORCHESTRA_INCGEN_SRCS ${ORCHESTRA_INCGEN_SRCS} PARENT_SCOPE)
