# Orchestra Compiler

This directory contains the core components of the OrchestraOS compiler, built upon the Multi-Level Intermediate Representation (MLIR) framework. The compiler is designed to translate high-level computational graphs into optimized, hardware-specific primitives for heterogeneous compute platforms.

## Architecture Overview

The Orchestra compiler leverages MLIR's extensible, multi-level dialect system to represent programs at various levels of abstraction. It progressively lowers high-level representations through a series of custom dialects and optimization passes, ultimately targeting specific hardware backends.

For a comprehensive blueprint of the compiler's architectural vision, the OrchestraIR dialect, and the planned optimization passes, please refer to: [MLIR Implementation Plan Modernization](../../docs/architecture/mlir-implementation-plan.md)

## Getting Started

For detailed instructions on how to set up the development environment, build the compiler, and run the test suite, please refer to the main [README.md](../../README.md) at the root of the repository.

### Orchestra Optimizer (`orchestra-opt`)

`orchestra-opt` is the core optimizer tool for the Orchestra compiler. It takes MLIR source files as input and applies a series of specified transformation passes.

**Usage:**

```shell
./orchestra-compiler/build/orchestra-opt/orchestra-opt <input-file.mlir> [options]
```

**Common Options:**

*   `-o <filename>`: Specify the output filename. If not provided, the output is printed to standard output.
*   `--pass-pipeline=<string>`: A textual description of the pass pipeline to run. This is the primary way to apply optimizations.
*   `--lower-orchestra-to-gpu --gpu-arch=<string>`: A dedicated pass to lower the Orchestra dialect to a specific GPU vendor dialect. See the "GPU Lowering" section for more details on the available architectures.
*   `--lower-orchestra-to-standard`: Lowers the Orchestra dialect to a combination of standard MLIR dialects.
*   `--divergence-to-speculation`: Converts conditional control flow (`scf.if`) into speculative execution using Orchestra's mechanisms.
*   `--show-dialects`: Print the list of all registered dialects.
*   `--help`: Display the full list of available options and passes. This is a very long list, containing all standard MLIR passes as well as custom Orchestra passes.

### GPU Lowering

The `--lower-orchestra-to-gpu` pass is the main entry point for lowering Orchestra IR to vendor-specific GPU dialects. The target hardware can be selected using the `--gpu-arch` option.

Supported architectures:
-   `--gpu-arch=nvgpu`: Targets NVIDIA GPUs using the `nvgpu` dialect.
-   `--gpu-arch=rocdl`: Targets AMD GPUs using the `rocdl` dialect.
-   `--gpu-arch=xegpu`: (Experimental) Targets Intel GPUs using the `xegpu` dialect.