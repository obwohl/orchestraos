# Set the minimum required version of CMake. MLIR requires a modern version.
cmake_minimum_required(VERSION 3.20.0)

# Define the project.
project(Orchestra LANGUAGES CXX C)

# Set the C++ standard. MLIR uses C++17 features.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the MLIR package provided by the system installation (e.g., from apt).
# The CONFIG mode is crucial for finding the MLIRConfig.cmake file.
# The version number ensures compatibility.
find_package(MLIR 18.1 REQUIRED CONFIG)
message(STATUS "Found MLIR ${MLIR_VERSION_STRING} in ${MLIR_DIR}")
message(STATUS "MLIR_INCLUDE_DIRS: ${MLIR_INCLUDE_DIRS}")

# Add the MLIR and LLVM CMake module paths to our project. This allows us to use
# helper functions like add_mlir_dialect_library.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddMLIR)
include(AddLLVM)
include(TableGen)
include(HandleLLVMOptions)

# Add the project's own include directory.
include_directories(include)

# Add subdirectories in the correct order.
# 1. Define the dialect's public interface and TableGen files.
add_subdirectory(include/Orchestra)
# 2. Define the dialect's implementation library. This depends on step 1.
add_subdirectory(lib/Orchestra)
# 3. Define the conversion pass. This depends on step 2.
add_subdirectory(lib/Conversion/OrchestraToLLVM)
# 4. Define the main executable tool. This depends on all previous steps.
add_subdirectory(tools/orchestra-opt)
