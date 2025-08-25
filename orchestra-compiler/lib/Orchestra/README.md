# Orchestra Dialect C++ API

This document provides a brief overview of the C++ API for the Orchestra dialect, with a focus on the modern, properties-based approach to accessing operation attributes.

## Modernized `Properties` System

As of MLIR v20, the Orchestra dialect has been modernized to use the `Properties` system for its core operation attributes. This is a significant improvement over the older, string-based `DictionaryAttr` approach.

The key benefits of this modernization are:

*   **Compile-Time Type Safety:** Operation attributes are now accessed through generated C++ methods with proper C++ types (e.g., `int32_t`, `mlir::StringAttr`), eliminating the risk of type errors at runtime.
*   **Improved Performance:** Accessing properties is faster than looking up attributes by string in a dictionary.
*   **Enhanced Readability:** The generated accessors have clear, intention-revealing names (e.g., `op.getNumTrue()` instead of `op->getAttr("num_true")`).

## Using the Properties API

When working with Orchestra operations in C++, you should use the generated property accessors to get and set their inherent attributes.

### `orchestra.task`

The `arch` attribute of the `task` operation is now a property.

*   **Getter:** `op.getArch()` returns a `StringRef`.

### `orchestra.select`

The `num_true` attribute of the `select` operation is now a property.

*   **Getter:** `op.getNumTrue()` returns an `int32_t`.

By using these generated accessors, you can write more robust, readable, and performant compiler passes for the Orchestra dialect.

## Core Operations

### `orchestra.commit`

The `orchestra.commit` operation represents the explicit commitment of a value to a specific memory space. This operation is a hint to the scheduler that the value is ready and can be moved to the target memory. It takes a single `MemRef` operand and produces a single `MemRef` result of the same type.

### `orchestra.select`

The `orchestra.select` operation is a conditional selection mechanism that is polymorphic and variadic. It takes a condition, two sets of SSA values, and a `num_true` attribute, and returns one of the sets of values based on the condition.
