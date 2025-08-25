# Orchestra Compiler Include Directory

This directory contains the public C++ header files for the Orchestra compiler.

## Structure

The header files are organized into subdirectories based on their component. The most important subdirectory is:

*   **`Orchestra/`**: Contains the core definitions for the Orchestra dialect, its operations, types, and interfaces, as well as the declarations for the custom transformation passes.

Files in this directory tree define the public API of the compiler's components. The corresponding implementation files are located in the `lib/` directory.
