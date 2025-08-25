This directory contains the header files for the Orchestra dialect, including the operation, type, and interface definitions. A key operation is `orchestra.task`, which uses a `target` dictionary attribute to specify hardware placement.

### Hot Tips: Working with Properties

When defining operations in this dialect, it is important to be aware of how MLIR's "Properties" system is configured for this project.

*   **Dialect-Wide Properties are Enabled:** The `Orchestra` dialect is configured with `usePropertiesForAttributes = 1;`. This is a powerful, non-invasive flag that automatically changes the underlying storage for all standard, inherent attributes (e.g., `SymbolRefAttr`, `IntegerAttr`) from the general attribute dictionary to more efficient, inline property storage.
*   **No Special Syntax Needed for Ops:** Because of this dialect-wide flag, you do **not** need to use special `*Prop` classes (like `IntProp`, `SymbolRefProp`, etc.) in your operation definitions in `OrchestraOps.td`. You should use the standard attribute types (e.g., `SymbolRefAttr`, `I32Attr`). The TableGen system will handle the conversion to property storage automatically. Attempting to use `*Prop` classes will result in build errors.
*   **Complex Attributes:** For complex, structured attributes like `DictionaryAttr`, this flag only optimizes the storage of the dictionary itself. It does not provide type-safe, structured access to its *contents*. For that, a custom C++ attribute class is required, as is being done for the `target` attribute on `orchestra.task`.
