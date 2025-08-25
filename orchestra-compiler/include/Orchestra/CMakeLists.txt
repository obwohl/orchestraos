# --- SECTION 1: PROJECT SETUP AND PACKAGE FINDING ---
# This ensures we find the MLIR and LLVM installations and get the
# necessary paths and executables.

# Note: project() and find_package() calls are typically in the root CMakeLists.txt.
# This file assumes they have been called and variables like MLIR_INCLUDE_DIRS are available.

# --- SECTION 2: DEFINE FILE AND DIRECTORY VARIABLES ---
# We define all paths here for clarity and easy modification.

# The directory containing this CMakeLists.txt file.
set(ORCHESTRA_CURRENT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# The directory where generated files will be placed.
set(ORCHESTRA_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})

# The single, top-level TableGen source file.
set(ORCHESTRA_TD_FILE ${ORCHESTRA_CURRENT_INCLUDE_DIR}/OrchestraDialect.td)

# A list of ALL .td files that the top-level file depends on.
# This is critical to make CMake re-run tblgen when you edit any of them.
set(ORCHESTRA_TD_DEPS
  ${ORCHESTRA_TD_FILE}
  ${ORCHESTRA_CURRENT_INCLUDE_DIR}/OrchestraOps.td
  ${ORCHESTRA_CURRENT_INCLUDE_DIR}/OrchestraInterfaces.td
  ${ORCHESTRA_CURRENT_INCLUDE_DIR}/OrchestraProperties.td
)

# The critical list of include paths for the mlir-tblgen tool itself.
set(ORCHESTRA_TBLGEN_INCLUDE_DIRS
  -I ${ORCHESTRA_CURRENT_INCLUDE_DIR}
  -I ${MLIR_INCLUDE_DIRS}
)

# Create the output directory for generated files.
file(MAKE_DIRECTORY ${ORCHESTRA_BUILD_DIR})

# --- SECTION 3: THE DEFINITIVE BUILD RULE USING add_custom_command ---
# We bypass the failing mlir_tablegen function completely.

# This macro simplifies defining the custom commands.
macro(add_orchestra_tblgen_rule TBLGEN_GENERATOR OUTPUT_FILE)
  add_custom_command(
    OUTPUT  ${ORCHESTRA_BUILD_DIR}/${OUTPUT_FILE}
    COMMAND ${MLIR_TABLEGEN_EXE} -gen-${TBLGEN_GENERATOR} ${ORCHESTRA_TBLGEN_INCLUDE_DIRS} -o ${CMAKE_CURRENT_COMMAND_OUTPUT} ${ORCHESTRA_TD_FILE}
    DEPENDS ${ORCHESTRA_TD_DEPS}
    COMMENT "Generating Orchestra ${OUTPUT_FILE}"
  )
endmacro()

# Define the list of generated files.
set(ORCHESTRA_INCGEN_HDRS
  ${ORCHESTRA_BUILD_DIR}/OrchestraOps.h.inc
  ${ORCHESTRA_BUILD_DIR}/OrchestraDialect.h.inc
  ${ORCHESTRA_BUILD_DIR}/OrchestraInterfaces.h.inc
  ${ORCHESTRA_BUILD_DIR}/OrchestraCanonicalization.h.inc
)
set(ORCHESTRA_INCGEN_SRCS
  ${ORCHESTRA_BUILD_DIR}/OrchestraOps.cpp.inc
  ${ORCHESTRA_BUILD_DIR}/OrchestraDialect.cpp.inc
  ${ORCHESTRA_BUILD_DIR}/OrchestraInterfaces.cpp.inc
)

# Generate all the files.
add_orchestra_tblgen_rule(op-decls "OrchestraOps.h.inc")
add_orchestra_tblgen_rule(op-defs "OrchestraOps.cpp.inc")
add_orchestra_tblgen_rule(dialect-decls "OrchestraDialect.h.inc")
add_orchestra_tblgen_rule(dialect-defs "OrchestraDialect.cpp.inc")
add_orchestra_tblgen_rule(op-interface-decls "OrchestraInterfaces.h.inc")
add_orchestra_tblgen_rule(op-interface-defs "OrchestraInterfaces.cpp.inc")
add_orchestra_tblgen_rule(rewriters "OrchestraCanonicalization.h.inc")


# --- SECTION 4: CREATE A TARGET FOR ALL GENERATED FILES ---
# This bundles all the generation steps into a single dependency target.
add_custom_target(OrchestraIncGen ALL
  DEPENDS
    ${ORCHESTRA_INCGEN_HDRS}
    ${ORCHESTRA_INCGEN_SRCS}
)

# Propagate the lists of generated files to the parent scope.
set(ORCHESTRA_INCGEN_HDRS ${ORCHESTRA_INCGEN_HDRS} PARENT_SCOPE)
set(ORCHESTRA_INCGEN_SRCS ${ORCHESTRA_INCGEN_SRCS} PARENT_SCOPE)
