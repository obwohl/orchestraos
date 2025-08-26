# This file manually constructs the build rules for the PDLL file.
# This approach is used because the high-level MLIR CMake functions
# were not working as expected in this build environment.

# 1. Define the name of the generated include file.
set(SPECULATE_IF_OP_PDL_INC ${CMAKE_CURRENT_BINARY_DIR}/SpeculateIfOp.pdll.inc)

# 2. Define a custom command to generate the .inc file from the PDLL source.
#    This invokes the mlir-pdll compiler directly.
add_custom_command(
  OUTPUT ${SPECULATE_IF_OP_PDL_INC}
  COMMAND /usr/lib/llvm-20/bin/mlir-pdll
          -x=cpp
          -I ${CMAKE_SOURCE_DIR}/include
          -I ${CMAKE_SOURCE_DIR}/include/Orchestra
          -I /usr/lib/llvm-20/include
          ${CMAKE_CURRENT_SOURCE_DIR}/SpeculateIfOp.pdll
          -o ${SPECULATE_IF_OP_PDL_INC}
  # Add dependencies on the source PDLL and any .td files it includes.
  DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/SpeculateIfOp.pdll
          ${CMAKE_SOURCE_DIR}/include/Orchestra/Orchestra.td
          ${CMAKE_SOURCE_DIR}/include/Orchestra/OrchestraDialect.td
          ${CMAKE_SOURCE_DIR}/include/Orchestra/OrchestraOps.td
  VERBATIM
)

# 3. Create a custom target to drive the file generation.
add_custom_target(OrchestraSpeculateIfOpPDLGen
  DEPENDS ${SPECULATE_IF_OP_PDL_INC}
)

# 4. Define the library with only the handwritten C++ source files.
#    The generated file will be #included by Passes.cpp.
add_library(OrchestraTransforms
  Passes.cpp
  LowerOrchestraToStandard.cpp
  LowerOrchestraToGPU.cpp
  LowerOrchestraToXeGPU.cpp
  LowerLinalgToRock.cpp
  LowerOrchestraToROCDL.cpp
)

# 5. Add the binary directory to the include path, so that Passes.cpp can find
#    the generated .inc file.
target_include_directories(OrchestraTransforms
  PUBLIC ${CMAKE_CURRENT_BINARY_DIR}
)

# 6. Link the library against its dependencies.
target_link_libraries(OrchestraTransforms
  PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRSCFDialect
  MLIRNVGPUDialect
  MLIRROCDLDialect
  MLIRSupport
  LLVMSupport
  Orchestra
)

# 7. Ensure the custom generation target runs before this library is built.
add_dependencies(OrchestraTransforms OrchestraSpeculateIfOpPDLGen Orchestra)
