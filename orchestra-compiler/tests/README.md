This directory contains the regression tests for the Orchestra compiler. The tests are written in MLIR and checked with `FileCheck`. The test suite is run using `lit`.

Tests for invalid operation schemas are generally split into separate files, one for each specific error condition, to ensure robust and clear test results.
