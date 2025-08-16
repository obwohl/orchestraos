add_executable(orchestra-opt main.cpp)

# Link against the dialect library and the MLIR libraries needed for an opt tool.
# The `Orchestra` target was defined in lib/Orchestra/CMakeLists.txt.
# Because we used PUBLIC linkage for its dependencies (MLIRIR, etc.),
# CMake will automatically link orchestra-opt against them as well.
target_link_libraries(orchestra-opt PRIVATE
  Orchestra
  OrchestraTransforms
  MLIRPass
  MLIRMlirOptMain
)
