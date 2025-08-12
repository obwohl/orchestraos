# Define the orchestra-opt executable.
add_executable(orchestra-opt
  orchestra-opt.cpp
)

# Link the executable against all necessary libraries.
target_link_libraries(orchestra-opt PRIVATE
  # Our custom conversion pass library.
  OrchestraToLLVM

  # Our custom dialect library.
  MLIROrchestra

  # Standard MLIR libraries required for a tool like mlir-opt.
  MLIRPass
  MLIRConversion
  MLIRTransforms
  MLIRAnalysis
  MLIRParser
  MLIROptMain
)
