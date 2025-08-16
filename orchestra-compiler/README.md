# Orchestra Compiler

This directory contains the core components of the OrchestraOS compiler, built upon the Multi-Level Intermediate Representation (MLIR) framework. The compiler is designed to translate high-level computational graphs into optimized, hardware-specific primitives for heterogeneous compute platforms.

## Architecture Overview

The Orchestra compiler leverages MLIR's extensible, multi-level dialect system to represent programs at various levels of abstraction. It progressively lowers high-level representations through a series of custom dialects and optimization passes, ultimately targeting specific hardware backends.

For a comprehensive blueprint of the compiler's architectural vision, the OrchestraIR dialect, and the planned optimization passes, please refer to: [MLIR Implementation Plan Modernization](../../docs/architecture/mlir-implementation-plan.md)

## Prerequisites

This project requires a C++17 compiler and the following dependencies:
- LLVM and MLIR, version 20 or newer
- CMake (3.20.0 or newer)
- Ninja (recommended)

For Ubuntu 24.04, I have provided a complete environment setup script in a previous message.

## Build Instructions

To build the Orchestra compiler, create a build directory and run CMake and Ninja.

```bash
# From the root of the repository
mkdir -p orchestra-compiler/build
cd orchestra-compiler/build

# Configure the build using CMake.
# CMAKE_PREFIX_PATH should point to your LLVM/MLIR installation directory.
cmake -G Ninja .. -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20

# Run the build
ninja
```

This will produce the `orchestra-opt` executable in the `orchestra-compiler/build/orchestra-opt/` directory.