# Debugging MLIR TableGen and Parsing Issues

This document summarizes the key learnings from a complex debugging session involving the `orchestra.commit` operation. The issues involved a combination of TableGen syntax, the MLIR parser, and the interaction between declarative and imperative code.

## Key Learnings

*   **`functional-type` Ambiguity:** The `functional-type` directive in TableGen's `assemblyFormat` is not suitable for operations with multiple variadic operand lists. The parser cannot reliably distinguish how to partition the types among the different variadic groups, which can lead to silent mis-parsing.

*   **Robust Variadic Operands:** The standard, robust MLIR pattern for an operation that needs multiple groups of variadic operands is to use a single variadic operand list (e.g., `$values`) and an integer attribute (e.g., `$num_true`) to specify the split point. This declarative approach completely avoids parsing ambiguity and is the recommended pattern.

*   **Debugging `lit` Tests:** When a test fails because an `expected-error` is not produced, it is critical to inspect the `stdout` from the test command in the failure log. A discrepancy between the source `.mlir` file and the IR in `stdout` is a strong indicator of a parsing issue, even if the verifier code itself appears correct.

*   **TableGen Parser Strictness:** The `mlir-tblgen` parser is very strict about its syntax. For example, when defining an optional group `(...)` in `assemblyFormat`, it requires a strong keyword "anchor." Generic delimiters like `:` or `->` are often not considered valid anchors, leading to build failures.

## Process Takeaways

*   **Trust the Test, but Verify the Verifier:** We correctly treated the test as the ground truth for the op's behavior. However, when a verifier test fails, it's crucial to confirm that the verifier is actually running and not being bypassed by a parsing error. Jules' learning about checking the stdout for discrepancies is a key diagnostic technique here.

*   **Declarative Systems Have Edges:** TableGen and its declarative `assemblyFormat` are incredibly powerful, but they are not magic. When a declarative system repeatedly fails in a way that seems to defy its own rules, it's a strong signal that you have reached the edge of its capabilities for your specific problem. At that point, fighting the system is less productive than changing the problem to fit the system, which is what we ultimately did by redefining the op.

*   **The "Why" is as Important as the "What":** The breakthrough came when we stopped asking "what is the right syntax?" and started asking "why is the parser failing?". This led us to the conclusion that the ambiguity was inherent to the op's structure, which prompted the successful change in strategy.
