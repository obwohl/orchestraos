# Research Log: Resolving MLIR Pass Registration Failure

## Problem Statement
I am unable to correctly register a new MLIR pass (`LowerLinalgToRock`) in the `orchestra-compiler` project. My attempts to add the pass, whether in a separate file (`LowerLinalgToRock.cpp`) or by consolidating the code into the existing `Passes.cpp`, have consistently resulted in C++ compilation and linking errors.

The root cause appears to be a misunderstanding of the correct procedure for making a new pass visible to the MLIR pass manager and registering it within this project's specific two-part CMake build system. The errors suggest issues with symbol visibility, incorrect function signatures for pass creation, or improper use of MLIR's registration macros/functions.

---
### Cycle 1: Internal Codebase Search

*   **Question:** What is the correct, idiomatic way to add a new `OperationPass` and register it within the `orchestra-compiler` project, consistent with its existing patterns and CMake structure?
*   **Strategy:** Internal Codebase Search (`grep`).
*   **Findings:**
    *   `grep "PassRegistration"` revealed that passes like `LowerOrchestraToGPUPass` and `LowerOrchestraToStandardPass` are registered within their own implementation files (`.cpp`).
    *   `grep "create.*Pass"` revealed that the corresponding `create...Pass()` functions are declared in header files (`.h`) and defined in the `.cpp` files.
*   **Conclusion:** The project follows a pattern where each pass is implemented in its own `.cpp` file. The pass class itself is defined in an anonymous namespace, making it local to that file. A public "create" function is defined in the same file and declared in a corresponding header file. The pass is registered in the same `.cpp` file using `mlir::PassRegistration`. This contradicts my failed attempt to consolidate the pass logic into `Passes.cpp`. The error was in the execution of this pattern, not the pattern itself.
