# Troubleshooting Build and Runtime Issues

This guide documents common issues that may arise when building and testing the Orchestra compiler, along with their solutions.

## 1. Build Failures

### 1.1. `FileCheck not found` CMake Error

During the CMake configuration step, you may encounter an error like this:

```
CMake Error at tests/CMakeLists.txt:5 (message):
  FileCheck not found
```

**Cause:** The build system cannot find the `FileCheck` executable, which is a standard LLVM testing utility. Although it is installed with LLVM, the CMake variable `LLVM_TOOLS_DIR` that hints at its location may not be set correctly by default.

**Solution:** Explicitly provide the path to the LLVM tools directory when running CMake.

```bash
cmake -G Ninja \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir \
  -DLLVM_TOOLS_DIR=/usr/lib/llvm-20/bin \
  ..
```

### 1.2. `zstd::libzstd_shared` Target Not Found

During the CMake configuration step, you may see an error related to a missing `zstd` target:

```
CMake Error at /usr/lib/llvm-20/lib/cmake/llvm/LLVMExports.cmake:73 (set_target_properties):
  The link interface of target "LLVMSupport" contains:

    zstd::libzstd_shared

  but the target was not found.
```

**Cause:** The pre-built LLVM/MLIR 20 packages for Ubuntu have a dependency on the Zstandard (`zstd`) compression library. The development headers and libraries for `zstd` are not installed by default.

**Solution:** Install the `libzstd-dev` package.

```bash
sudo apt-get update
sudo apt-get install -y libzstd-dev
```

## 2. Test Failures

### 2.1. `pyenv: llvm-lit: command not found`

When running the test suite (e.g., with `ninja check-orchestra`), you may encounter an error like this:

```
FAILED: tests/CMakeFiles/check-orchestra ...
...
pyenv: llvm-lit: command not found
```

**Cause:** This issue occurs when using `pyenv` to manage Python versions. The `lit` package is installed via `pip`, and `pyenv` creates a "shim" executable for it. The `llvm-lit` symlink created by the environment setup might point to this shim, which can fail to execute correctly when called from the build system's environment.

**Solution:** The symlink `/usr/lib/llvm-20/bin/llvm-lit` must point directly to the real `lit` executable, not the `pyenv` shim.

1.  **Find the real path** to the `lit` executable:
    ```bash
    pyenv which lit
    # Example output: /home/user/.pyenv/versions/3.9.12/bin/lit
    ```

2.  **Re-create the symlink** to point to this real path:
    ```bash
    LIT_PATH=$(pyenv which lit)
    sudo rm /usr/lib/llvm-20/bin/llvm-lit
    sudo ln -s $LIT_PATH /usr/lib/llvm-20/bin/llvm-lit
    ```

After fixing the symlink, re-run the build and tests.
