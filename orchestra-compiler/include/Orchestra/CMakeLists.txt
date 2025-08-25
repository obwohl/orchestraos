# This variable will hold the list of all generated header files (.h.inc).
set(ORCHESTRA_INCGEN_HDRS)

# This variable will hold the list of all generated source files (.cpp.inc).
set(ORCHESTRA_INCGEN_SRCS)

# --- Generation for OrchestraDialect.td ---
# The canonical pattern: first, set LLVM_TARGET_DEFINITIONS to the input .td file.
set(LLVM_TARGET_DEFINITIONS OrchestraDialect.td)
# Then, invoke mlir_tablegen with the *output* file name and generator flags.
mlir_tablegen(OrchestraDialect.h.inc -gen-dialect-decls -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraDialect.h.inc")

# Generate the dialect definitions source.
mlir_tablegen(OrchestraDialect.cpp.inc -gen-dialect-defs -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraDialect.cpp.inc")

# --- Generation for OrchestraOps.td ---
# Repeat the pattern for the operations file.
set(LLVM_TARGET_DEFINITIONS OrchestraOps.td)
# Generate the operation declarations header.
mlir_tablegen(OrchestraOps.h.inc -gen-op-decls -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraOps.h.inc")

# Generate the operation definitions source.
# The input file is still OrchestraOps.td from the previous 'set' command.
mlir_tablegen(OrchestraOps.cpp.inc -gen-op-defs -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraOps.cpp.inc")

# --- Generation for OrchestraInterfaces.td ---
# Repeat the pattern for the interfaces file.
set(LLVM_TARGET_DEFINITIONS OrchestraInterfaces.td)
# Generate the interface declarations header.
mlir_tablegen(OrchestraInterfaces.h.inc -gen-op-interface-decls -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraInterfaces.h.inc")

# Generate the interface definitions source.
mlir_tablegen(OrchestraInterfaces.cpp.inc -gen-op-interface-defs -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraInterfaces.cpp.inc")

# --- Generation for canonicalization patterns from OrchestraOps.td ---
set(LLVM_TARGET_DEFINITIONS OrchestraOps.td)
mlir_tablegen(OrchestraCanonicalization.h.inc -gen-rewriters -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraCanonicalization.h.inc")

# --- Generation for OrchestraAttributes.td ---
set(LLVM_TARGET_DEFINITIONS OrchestraAttributes.td)
mlir_tablegen(OrchestraAttributes.h.inc -gen-attrdef-decls -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_HDRS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraAttributes.h.inc")

mlir_tablegen(OrchestraAttributes.cpp.inc -gen-attrdef-defs -I ${MLIR_INCLUDE_DIRS})
list(APPEND ORCHESTRA_INCGEN_SRCS
  "${CMAKE_CURRENT_BINARY_DIR}/OrchestraAttributes.cpp.inc")

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
