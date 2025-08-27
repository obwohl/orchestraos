# This file handles the TableGen generation for the Rock dialect.
# It mimics the pattern from the parent directory's CMakeLists.txt.

set(LLVM_TARGET_DEFINITIONS RockDialect.td)
mlir_tablegen(RockDialect.h.inc -gen-dialect-decls)
mlir_tablegen(RockDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS RockOps.td)
# The include inside RockOps.td now uses a relative path ("../Orchestra.td"),
# so we no longer need to add an extra -I flag here.
mlir_tablegen(RockOps.h.inc -gen-op-decls)
mlir_tablegen(RockOps.cpp.inc -gen-op-defs)

# Add a custom target for all generated files. This is depended on by the
# C++ library target in the lib/ tree.
add_custom_target(RockIncGen ALL
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/RockDialect.h.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/RockDialect.cpp.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/RockOps.h.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/RockOps.cpp.inc"
)
