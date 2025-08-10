# Orchestra Compiler

This directory contains the core components of the OrchestraOS compiler, built upon the Multi-Level Intermediate Representation (MLIR) framework. The compiler is designed to translate high-level computational graphs into optimized, hardware-specific primitives for heterogeneous compute platforms.

## Architecture Overview

The Orchestra compiler leverages MLIR's extensible, multi-level dialect system to represent programs at various levels of abstraction. It progressively lowers high-level representations through a series of custom dialects and optimization passes, ultimately targeting specific hardware backends.

For a comprehensive blueprint of the compiler's architectural vision, the OrchestraIR dialect, and the planned optimization passes, please refer to: [MLIR Implementation Plan Modernization](../../docs/orchestra%20-%20PRIO%20-%20tech%20-%20%20MLIR%20Implementation%20Plan%20Modernization.md)

## Build Instructions

To build the Orchestra compiler, navigate to the `build` directory within this `orchestra-compiler` directory and run CMake and your build tool (e.g., Make or Ninja).

```bash
cd orchestra-compiler/build
cmake -G "Unix Makefiles" .. -DMLIR_DIR=/usr/lib/llvm-18/lib/cmake/mlir # Adjust MLIR_DIR as necessary
cmake --build .
```

For detailed instructions on setting up the CMake build system for out-of-tree MLIR dialects, including common pitfalls and their solutions, please consult: [Correcting MLIR Dialect CMake Build](../../docs/orchestra%20-%20PRIO%20-%20tech%20-%20Correcting%20MLIR%20Dialect%20CMake%20Build.md)

### Important Build Tip: CMake Structure for MLIR Dialects

A recurring challenge in MLIR development is correctly configuring CMake for out-of-tree dialects. The canonical solution, which this project adheres to, involves a strict separation of interface (`include/`) and implementation (`lib/`) directories, processed in a specific order by CMake.

*   The `include/` directory defines the dialect's public API (TableGen `.td` files) and is processed first to generate necessary C++ headers.
*   The `lib/` directory contains the dialect's C++ implementation, which depends on the headers generated from `include/`.

This structure is crucial for resolving dependency issues during the CMake configuration phase. Always refer to the `Correcting MLIR Dialect CMake Build` document for a deep dive into this pattern and to troubleshoot any build-related errors.