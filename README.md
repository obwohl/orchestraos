# OrchestraOS Compiler

Welcome to the OrchestraOS Compiler project. This repository contains a custom compiler built with the Multi-Level Intermediate Representation (MLIR) framework, designed for optimizing and executing computational graphs on heterogeneous hardware.

## Project Overview

The OrchestraOS compiler is an ambitious project to build a "meta-OS" that elevates scheduling and data movement to first-class citizens of the compiler's IR. The goal is to enable global, hardware-aware optimization of complex workloads, particularly in the domain of AI and machine learning.

The core of the project is the `orchestra-compiler`, which implements a custom MLIR dialect (`OrchestraIR`) and a set of tools for processing it.

## Getting Started

To get started with the OrchestraOS compiler, you will need to set up the required dependencies and build the project.

### Prerequisites

This project requires a C++17 compiler and the following dependencies:
- LLVM and MLIR, version 20 or newer
- CMake (3.20.0 or newer)
- Ninja (recommended)

For **Ubuntu 24.04**, you can set up the complete environment by running the following commands:
```bash
# 1. Update package lists and install prerequisite tools
sudo apt-get update
sudo apt-get install -y wget software-properties-common ninja-build

# 2. Download and run the official LLVM installer script
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20

# 3. Install the full set of required LLVM, Clang, and MLIR development packages
sudo apt-get install -y \
    clang-tidy-20 \
    clang-format-20 \
    clang-tools-20 \
    llvm-20-dev \
    llvm-20-tools \
    libmlir-20-dev \
    mlir-20-tools \
    libomp-20-dev \
    libc++-20-dev \
    libc++abi-20-dev \
    libclang-common-20-dev \
    libclang-20-dev \
    libclang-cpp20-dev \
    liblldb-20-dev \
    libunwind-20-dev
```

### Build Instructions

Once the prerequisites are installed, you can build the compiler using CMake and Ninja:

```bash
# 1. Create a build directory
mkdir -p orchestra-compiler/build
cd orchestra-compiler/build

# 2. Configure the build using CMake
#    (The CMAKE_PREFIX_PATH should point to your LLVM/MLIR installation)
cmake -G Ninja .. -DCMAKE_PREFIX_PATH=/usr/lib/llvm-20

# 3. Run the build
ninja
```

This will produce the `orchestra-opt` executable in the `orchestra-compiler/build/orchestra-opt/` directory.

### Testing

The project includes a test suite using the LLVM Integrated Tester (lit). To run the tests, execute the following command from the root of the repository:

```bash
python3 tests/run.py
```

## Documentation

The project's documentation is located in the `docs/` directory. The most important documents to read are:

- **[Project Status](docs/project-status/status-quo.md):** This document provides a high-level overview of the current state of the project, what works, and what the immediate goals are.
- **[MLIR Implementation Plan](docs/architecture/mlir-implementation-plan.md):** This document contains the detailed technical blueprint for the compiler, including the design of the `OrchestraIR` dialect and the planned optimization passes.
- **[Contributing Guidelines](CONTRIBUTING.md):** This document outlines the standards and best practices for contributing to the project, particularly regarding documentation.
