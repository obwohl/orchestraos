# Orchestra Transformation Passes

This directory contains the implementation of the core transformation passes for the Orchestra compiler.

## Lowering to GPU (`--lower-orchestra-to-gpu`)

The `LowerOrchestraToGPUPass` is the main entry point for lowering the `orchestra` dialect to specific GPU vendor dialects. It is a module-level pass that dispatches to more specific function-level passes based on the `gpu-arch` option.

### Usage

```sh
orchestra-opt --lower-orchestra-to-gpu="gpu-arch=<vendor>" ...
```

### Supported Vendors

*   **`nvgpu`**: Targets the NVIDIA GPU dialect (`nvgpu`). This is the default.
*   **`xegpu`**: Targets the Intel XeGPU dialect (`xegpu`). This path is currently experimental.

### NVIDIA GPU Lowering (`LowerOrchestraToNVGPUPass`)

This pass lowers `orchestra` operations to the `nvgpu` dialect. It contains architecture-aware logic to generate the most efficient code for the target GPU.

#### Blackwell (`sm_arch >= 100`)

For NVIDIA Blackwell and newer architectures, the pass lowers `orchestra.transfer` operations to use the **Tensor Memory Accelerator (TMA)**.

*   **Asynchronous Copy**: `nvgpu.tma.async.load`
*   **Synchronization**: `nvgpu.mbarrier`

This provides the highest performance for data transfers on the latest hardware.

#### Hopper and Older Architectures

For older architectures, the pass lowers `orchestra.transfer` to the standard asynchronous copy operations:

*   **Asynchronous Copy**: `nvgpu.device_async_copy`
*   **Synchronization**: `nvgpu.device_async_wait`

This ensures backward compatibility while still providing high performance on a wide range of NVIDIA GPUs.
