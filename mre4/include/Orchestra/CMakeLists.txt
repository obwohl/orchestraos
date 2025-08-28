include(TableGen)
set(LLVM_TARGET_DEFINITIONS OrchestraDialect.td)
mlir_tablegen(OrchestraDialect.h.inc -gen-dialect-decls)
mlir_tablegen(OrchestraDialect.cpp.inc -gen-dialect-defs)

set(LLVM_TARGET_DEFINITIONS OrchestraOps.td)
mlir_tablegen(OrchestraOps.h.inc -gen-op-decls)
mlir_tablegen(OrchestraOps.cpp.inc -gen-op-defs)

add_custom_target(OrchestraIncGen ALL
  DEPENDS
    "${CMAKE_CURRENT_BINARY_DIR}/OrchestraDialect.h.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/OrchestraDialect.cpp.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/OrchestraOps.h.inc"
    "${CMAKE_CURRENT_BINARY_DIR}/OrchestraOps.cpp.inc"
)
