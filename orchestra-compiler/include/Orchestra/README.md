# Orchestra Dialect

This directory contains the definition of the Orchestra dialect, its operations, and interfaces.

## `orchestra.task` Operation

The `orchestra.task` operation is a container for a schedulable unit of work. It has a single region that contains the operations to be executed.

### `target_arch` Attribute

The `target_arch` attribute is a dictionary attribute that specifies the target architecture for the task. It must contain a string attribute named `arch` that specifies the architecture type (e.g., "cpu", "gpu").

Example:
```mlir
"orchestra.task"() <{target_arch = {arch = "cpu"}}> ({
  ...
}) : () -> ()
```
