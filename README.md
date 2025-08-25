# OrchestraOS Compiler

Welcome to the OrchestraOS compiler project! This repository contains the source code for a "meta-OS" compiler that elevates scheduling and data movement to first-class citizens of the compiler's IR. The goal is to enable global, hardware-aware optimization of complex workloads, particularly in the domain of AI and machine learning.

The core of the project is the `orchestra-compiler`, which implements a custom MLIR dialect (`OrchestraIR`) and a set of tools for processing it.

For a deep dive into the project's vision, architecture, and technical specifications, please refer to the **[Architectural Blueprint](docs/architecture/blueprint.md)**.

## Getting Started

### Prerequisites

This project is developed and tested on Ubuntu 24.04. The following dependencies are required:

*   LLVM/MLIR 20
*   Clang 20
*   Ninja
*   Python 3 and `pip`
*   `lit` (LLVM Integrated Tester)

The `AGENTS.md` file contains a setup script that can be used to install all the necessary dependencies.

### Building the Compiler

The project uses a standard CMake-based workflow.

1.  **Configure the build:**
    This command should be run from the root of the repository. It tells CMake where the source code is (`-S`) and where to put the build artifacts (`-B`).

    ```bash
    cmake -S orchestra-compiler -B orchestra-compiler/build -G Ninja \
      -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
      -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
      -DLLVM_TOOLS_DIR=/usr/lib/llvm-20/bin
    ```

2.  **Build the compiler and its tools:**

    ```bash
    cmake --build orchestra-compiler/build
    ```

This will produce the `orchestra-opt` executable in the `orchestra-compiler/build/orchestra-opt/` directory.

### Running the Tests

The test suite is built on top of LLVM's `lit` and can be run using the `check-orchestra` CMake target.

```bash
cmake --build orchestra-compiler/build --target check-orchestra
```

This command will automatically discover and run all tests and report any failures. All tests must pass before any change is submitted.

## Contributing

This project has a strict set of guidelines for contributions and documentation. Please read the **[AGENTS.md](AGENTS.md)** file carefully before starting any work. It contains the full development workflow, documentation principles, and problem-solving strategies.

All documentation must adhere to the principles outlined in `AGENTS.md`. Key documentation, including this `README.md`, should be updated as the project evolves.
