# Process OrchestraDialect.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OrchestraDialect.td)
mlir_tablegen(OrchestraDialect.h.inc -gen-dialect-decls -dialect=orchestra)
mlir_tablegen(OrchestraDialect.cpp.inc -gen-dialect-defs -dialect=orchestra)

# Process OrchestraOps.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OrchestraOps.td)
mlir_tablegen(OrchestraOps.h.inc -gen-op-decls)
mlir_tablegen(OrchestraOps.cpp.inc -gen-op-defs)

# Process OrchestraInterfaces.td
set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/OrchestraInterfaces.td)
mlir_tablegen(OrchestraInterfaces.h.inc -gen-op-interface-decls)
mlir_tablegen(OrchestraInterfaces.cpp.inc -gen-op-interface-defs)

# Create a single, public IncGen target that aggregates all generated files.
add_public_tablegen_target(OrchestraIncGen)
