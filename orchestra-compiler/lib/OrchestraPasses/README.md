# OrchestraPasses Library

This directory contains the implementation of various MLIR passes specific to the Orchestra dialect. These passes perform transformations and optimizations on the Orchestra Intermediate Representation (IR).

## DummyPass

The `DummyPass` is a simple, illustrative MLIR pass designed primarily for testing the pass registration mechanism and basic pass execution within the Orchestra compiler infrastructure.

*   **Purpose:** To serve as a minimal working example of an MLIR pass that operates on a module. Its `runOnOperation()` method simply prints a message to `llvm::errs()`.
*   **Location:** `lib/OrchestraPasses/DummyPass.cpp` and `lib/OrchestraPasses/DummyPass.h`.
*   **Behavior:** When executed, it prints "DummyPass ran on an operation." to the standard error stream.

## Important Findings & Hot Tips for MLIR Pass Development

Developing and integrating custom MLIR passes can be challenging due to the framework's strict requirements and subtle interactions. Here are some key lessons learned during the development of `DummyPass` and its integration:

### 1. Correct Pass Registration (`mlir::PassRegistration`)

The idiomatic and recommended way to register a custom MLIR pass is by using the `mlir::PassRegistration` utility. This macro simplifies the process by automatically handling the pass name, description, and creation function.

**Example Usage (in `OrchestraPasses.cpp`):**

```cpp
#include "mlir/Pass/Pass.h" // Required for mlir::PassRegistration
#include "DummyPass.h"      // Required to make DummyPass visible

void registerOrchestraPasses() {
  mlir::PassRegistration<DummyPass>();
}
```

### 2. Essential Pass Class Overrides

For `mlir::PassRegistration` to work correctly, your custom pass class (e.g., `DummyPass`) must override specific methods from `mlir::PassWrapper`.

*   **`static llvm::StringRef getArgument() const override`:**
    *   **Purpose:** This method provides the command-line argument (e.g., `-dummy-pass`) that users will use to invoke your pass via tools like `orchestra-opt`.
    *   **Pitfall:** Failing to override this method, or declaring it `static` when it should be a non-static `const` override, will lead to compilation errors or runtime crashes (e.g., `LLVM ERROR: Trying to register 'DummyPass' pass that does not override getArgument()`).
    *   **Solution:** Ensure your pass class includes this exact signature and returns the desired command-line argument string.

*   **`static mlir::TypeID getTypeID()`:**
    *   **Purpose:** Provides a unique identifier for your pass type, crucial for MLIR's internal type system and pass management.
    *   **Pitfall:** Omitting this or implementing it incorrectly can lead to runtime errors related to pass identification.
    *   **Solution:** A common pattern is to define it as shown in `DummyPass.h` and `DummyPass.cpp`.

### 3. Header Visibility and Include Paths

Ensuring that your pass class is visible to the registration function and that generated headers are found by the compiler is critical.

*   **Pass Header Inclusion:** The `.cpp` file where `mlir::PassRegistration<YourPass>()` is called (e.g., `OrchestraPasses.cpp`) *must* include the header file where your pass class is declared (e.g., `DummyPass.h`).
*   **Generated Headers:** Remember that TableGen-generated headers (e.g., `OrchestraOps.h.inc`) are typically placed in the build directory's include path (e.g., `orchestra-compiler/build/include`). Ensure this path is added to your compiler's include directories (e.g., in `CMakeLists.txt` using `${PROJECT_BINARY_DIR}/include`).

### 4. CMake Linking for Executables

For your custom passes to be available in an executable tool (like `orchestra-opt`), the library containing your passes (`libOrchestraPasses.a` in this case) must be explicitly linked to the executable.

*   **`target_link_libraries`:** In the `CMakeLists.txt` for your executable (e.g., `orchestra-compiler/tools/orchestra-opt/CMakeLists.txt`), ensure you link against your pass library.
    *   **Example:** `target_link_libraries(orchestra-opt PUBLIC OrchestraPasses)`
*   **Library Path:** Ensure the build directory's library path is added to `target_link_directories` so CMake can find your custom libraries.
    *   **Example:** `target_link_directories(orchestra-opt PUBLIC ${PROJECT_BINARY_DIR}/lib/OrchestraPasses)`

By paying close attention to these details, many common MLIR pass development and integration issues can be avoided. However, as noted in `orchestra-compiler/lib/Orchestra/README.md`, there are still unresolved issues with operation registration that can block pass functionality.