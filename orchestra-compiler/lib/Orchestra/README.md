# Orchestra Dialect C++ Implementation

This directory contains the C++ implementation files for the Orchestra dialect. While the `include/` directory defines the public interface, this directory provides the concrete logic for the dialect's operations, verification, and canonicalization patterns.

## Key Files

*   **`OrchestraDialect.cpp`**: Implements the initialization and registration of the Orchestra dialect. It's also where operation attributes and other dialect-level properties are defined.
*   **`OrchestraOps.cpp`**: Provides the C++ implementation for the custom operations defined in `OrchestraOps.td`. This includes:
    *   **Verification Logic (`verify`)**: Methods that check for semantic errors and ensure the IR is well-formed.
    *   **Canonicalization Patterns (`getCanonicalizationPatterns`)**: Patterns that simplify the IR into a more standard form (e.g., folding two consecutive `transfer` ops into one).
    *   **Builders**: Helper methods for creating instances of the operations programmatically.
*   **`OrchestraInterfaces.cpp`**: Implements any custom dialect interfaces.

## Common Pitfalls & Hot Tips

### Modernized `Properties` System

As of MLIR v20, the Orchestra dialect has been modernized to use the `Properties` system for its core operation attributes. This is a significant improvement over the older, string-based `DictionaryAttr` approach. When working with Orchestra operations in C++, you should **always** use the generated property accessors to get and set their inherent attributes.

**Benefits:**

*   **Compile-Time Type Safety:** Access attributes via generated C++ methods with proper types (e.g., `int32_t`, `mlir::StringAttr`).
*   **Improved Performance:** Faster than dictionary lookups.
*   **Enhanced Readability:** Clear, intention-revealing names (e.g., `op.getNumTrue()` instead of `op->getAttr("num_true")`).

### Using the Properties API

*   **`orchestra.task`**: Use `op.getArch()` to get the `arch` as a `StringRef`.
*   **`orchestra.commit`**: Use `op.getNumTrue()` to get the `num_true` value as an `int32_t`.

### The `verify` Method HACK in `orchestra.commit`

Due to a limitation in the generic MLIR op parser, the `num_true` property of the `orchestra.commit` operation is not always initialized correctly when parsing textual IR. The `verify` method for this op contains a necessary workaround that manually reads the `num_true` value from the raw attribute dictionary if the property has its default (zero) value. This is a crucial piece of logic that prevents incorrect verification failures.
