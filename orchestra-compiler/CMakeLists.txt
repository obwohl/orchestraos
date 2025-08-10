# Require a modern CMake version. MLIR itself requires at least 3.20.0.
cmake_minimum_required(VERSION 3.20.0)

# Define the project name and languages.
project(OrchestraCompiler LANGUAGES CXX C)

# Set the C++ standard. MLIR requires C++17.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the MLIR package provided by the Debian/Ubuntu installation.
# The user specified version 18.1. The CONFIG mode looks for MLIRConfig.cmake.
# The user must point CMake to the right location, e.g., by setting:
# -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir
find_package(MLIR 18.1 REQUIRED CONFIG)
message(STATUS "Found MLIR_DIR: ${MLIR_DIR}")
message(STATUS "Found LLVM_DIR: ${LLVM_DIR}")

# Add the MLIR and LLVM CMake module directories to the search path.
# This is essential for finding AddMLIR.cmake, AddLLVM.cmake, and TableGen.cmake.
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

# Include the necessary MLIR and LLVM CMake modules.
include(AddMLIR)
include(AddLLVM)
include(TableGen)
include(HandleLLVMOptions)

# Add the MLIR and LLVM include directories to the project's include path.
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})

# Add the project's own include directory and the build directory's include
# directory (for generated headers) to the path.
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# This is the core of the solution. Process the 'include' directory first
# to define the TableGen targets. Then, process the 'lib' directory, which
# can now safely depend on the targets from 'include'.
add_subdirectory(include)
add_subdirectory(lib)
