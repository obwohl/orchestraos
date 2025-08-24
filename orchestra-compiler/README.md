# Orchestra Compiler

This directory contains the core components of the OrchestraOS compiler, built upon the Multi-Level Intermediate Representation (MLIR) framework. The compiler is designed to translate high-level computational graphs into optimized, hardware-specific primitives for heterogeneous compute platforms.

## Architecture Overview

The Orchestra compiler leverages MLIR's extensible, multi-level dialect system to represent programs at various levels of abstraction. It progressively lowers high-level representations through a series of custom dialects and optimization passes, ultimately targeting specific hardware backends.

For a comprehensive blueprint of the compiler's architectural vision, the OrchestraIR dialect, and the planned optimization passes, please refer to: [MLIR Implementation Plan Modernization](../../docs/architecture/mlir-implementation-plan.md)

## Getting Started

For detailed instructions on how to set up the development environment, build the compiler, and run the test suite, please refer to the main [README.md](../../README.md) at the root of the repository.

### GPU Lowering

The `--lower-orchestra-to-gpu` pass is the main entry point for lowering Orchestra IR to vendor-specific GPU dialects. The target hardware can be selected using the `--gpu-arch` option.

Supported architectures:
-   `--gpu-arch=nvgpu`: Targets NVIDIA GPUs using the `nvgpu` dialect.
-   `--gpu-arch=rocdl`: Targets AMD GPUs using the `rocdl` dialect.
-   `--gpu-arch=xegpu`: (Experimental) Targets Intel GPUs using the `xegpu` dialect.