# Orchestra Dialect C++ API

This document provides a brief overview of the C++ API for the Orchestra dialect, with a focus on the modern, properties-based approach to accessing operation attributes.

## Modernized `Properties` System

As of MLIR v20, the Orchestra dialect has been modernized to use the `Properties` system for its core operation attributes. This is a significant improvement over the older, string-based `DictionaryAttr` approach.

The key benefits of this modernization are:

*   **Compile-Time Type Safety:** Operation attributes are now accessed through generated C++ methods with proper C++ types (e.g., `int32_t`, `mlir::DictionaryAttr`), eliminating the risk of type errors at runtime.
*   **Improved Performance:** Accessing properties is faster than looking up attributes by string in a dictionary.
*   **Enhanced Readability:** The generated accessors have clear, intention-revealing names (e.g., `op.getNumTrue()` instead of `op->getAttr("num_true")`).

## Using the C++ API

When working with the `orchestra.commit` operation in C++, you should use the generated attribute accessors.

### `orchestra.commit`

The `num_true` attribute of the `commit` operation is an `I32Attr`.

*   **Getter:** `op.getNumTrue()` returns an `I32Attr`. To get the integer value, you can use `op.getNumTrue().getValue().getSExtValue()`.

## Custom Assembly Format for `orchestra.task`

The `orchestra.task` operation uses a custom assembly format to improve readability and parsing robustness.

### Syntax

```mlir
%results = orchestra.task (%operands,...) on "arch_name" {attributes} : (operand_types) -> (result_types) {
  // ... region with operations ...
}
```

### Example

```mlir
orchestra.task on "test" {} : () -> () {
  "orchestra.yield"() : () -> ()
}
```

This custom format clearly separates the target architecture from other attributes and makes the operation's signature more explicit in the assembly.
