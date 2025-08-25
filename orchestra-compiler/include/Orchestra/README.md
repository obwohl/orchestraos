# Orchestra Dialect

This directory contains the definition of the Orchestra dialect, its operations, and interfaces.

## `orchestra.task` Operation

The `orchestra.task` operation is a container for a schedulable unit of work. It has a single region that contains the operations to be executed.

### `arch` Property

The `arch` property is a string attribute that specifies the target architecture for the task (e.g., "cpu", "gpu").

Example:
```mlir
"orchestra.task"() <{arch = "cpu"}> ({
  ...
}) : () -> ()
```
