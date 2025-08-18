

# **A Comprehensive Diagnostic and Remediation Guide for MLIR Rewrite Pattern Failures**

## **Section 1: A Systematic Workflow for Diagnosing Rewrite Failures**

The failure of a Multi-Level Intermediate Representation (MLIR) rewrite pattern, manifesting as a "rollback" by the GreedyPatternRewriteDriver, is a canonical symptom of a deeper issue: the generation of semantically or structurally invalid Intermediate Representation (IR). The driver's behavior is not an error but a safety mechanism designed to preserve IR validity at all costs. An effective debugging process, therefore, is not one that questions the driver, but one that systematically uncovers the invalid IR and traces its origin within the failing pattern. This section outlines a rigorous, four-stage methodology to deconstruct these failures, moving from the opaque symptom to a precise, actionable diagnosis. This workflow is designed to first isolate the failure, then expose the malformed IR, trace its creation through the rewrite engine, and finally, for the most intractable cases, apply advanced instrumentation and bisection techniques.

### **1.1 Stage 1: Capturing a Minimal, Stable Reproducer**

The foundational step in any compiler debugging endeavor is to isolate the failure from the complexities of a large codebase or an extensive test suite. A minimal, self-contained reproducer is the primary asset for efficient analysis. It consists of the smallest possible .mlir input file and the exact mlir-opt command-line invocation required to trigger the verification failure.

The most direct method for capturing this state is to leverage the pass manager's built-in diagnostics. The mlir-opt tool provides a powerful flag specifically for this purpose: \--mlir-pass-pipeline-crash-reproducer=\<filename\>.1 When a pass fails—which includes verification failures that halt the pass pipeline—this option automatically serializes the state of the IR in the instant before the failing pass was executed into the specified file. It also prints the precise

mlir-opt command required to re-trigger the failure using this new file as input. This is the preferred, automated approach.

In scenarios where a reproducer is not automatically generated, a manual approach is necessary. This involves instrumenting a mlir-opt run to dump the IR state between every pass. The \-mlir-print-ir-before-all and \-mlir-print-ir-after-all flags are essential for this purpose.1 By redirecting the output to a log file, one can inspect the sequence of transformations. The objective is to identify the last valid IR dump immediately preceding the execution of the problematic pass (e.g.,

SpeculateAddress). This IR should be copied into a new .mlir file, which then serves as the input for the minimal test case. When constructing this command, it is also beneficial to use \--mlir-disable-threading to ensure deterministic execution and \--mlir-print-ir-module-scope to get a full view of the top-level module, which can be critical for diagnosing issues related to symbols or inter-procedural transformations.3

For failures occurring within a lit-based test suite, the \--split-input-file flag is a crucial component of test hygiene.1 It ensures that each test case within a file (separated by

// \-----) is processed independently, preventing a failure in one test from cascading and obscuring the root cause of a failure in another.

### **1.2 Stage 2: Exposing the Malformed IR**

The transactional nature of the GreedyPatternRewriteDriver means that upon detecting a verification failure, it discards the invalid IR, effectively hiding the evidence of the bug. The second stage of the diagnostic process is to disable this safety mechanism to permit the inspection of the raw, malformed IR that the pattern produced.

The cornerstone of this stage is the \--verify-each=0 command-line option.3 This flag instructs the pass manager to disable the verifier that normally runs after each successful pattern application. With the verifier disabled, the

GreedyPatternRewriteDriver will no longer detect the error and will not perform the rollback. The pass will proceed to completion, leaving the invalid IR in its final, broken state, which can then be printed and analyzed.

However, once the IR is in a structurally unsound state, even the process of printing it can trigger a crash. The default "pretty printers" for operations often make assumptions about the validity of the op's state (e.g., the number of operands or results). An attempt to print an op with an unexpected number of operands can lead to out-of-bounds access and a segmentation fault. To circumvent this, the \-mlir-print-op-generic flag should be used.3 This flag forces

mlir-opt to use a generic printer that renders every operation in its canonical, verbose form (e.g., "dialect.op\_name"(%operand) : (type) \-\> type). This format is an isomorphic representation of the underlying C++ data structures and is far more robust to invalid state, as it makes no assumptions about the operation's custom assembly format.

These flags can be combined with others for a comprehensive view of the failure. For instance, running with the verifier enabled but adding \--mlir-print-ir-after-failure 1 will dump the entire module's IR at the moment of failure. Similarly,

\-mlir-print-op-on-diagnostic 2 attaches the textual form of the failing operation as a note to the diagnostic message, providing immediate context for the error. This combination allows the developer to see both the specific failing operation and its surrounding context.

### **1.3 Stage 3: Tracing the Rewrite Engine's Logic**

With the ability to view the invalid IR, the investigation shifts to understanding the process that created it. The GreedyPatternRewriteDriver can be a "black box," applying dozens or hundreds of patterns in a complex sequence. This stage focuses on illuminating the internal decision-making process of the driver.

The most powerful tool for this purpose is the debug flag \-debug-only=dialect-conversion.3 Despite its name suggesting a focus on the dialect conversion infrastructure, this flag provides an exceptionally detailed trace of the

GreedyPatternRewriteDriver's execution. The output log will show:

* Which operations are added to the worklist.  
* Which patterns are attempting to match a given operation.  
* The success or failure of each match attempt.  
* Upon a successful rewrite, which new operations were created and which were erased.  
* Which new operations are being added back to the worklist.

This trace is indispensable for understanding not only why a specific pattern was applied but also for diagnosing complex interactions where the application of one pattern incorrectly enables the application of another.

For more targeted, pattern-specific debugging, developers can embed LLVM\_DEBUG macros within the C++ source of their matchAndRewrite function. These macros, when compiled in a debug build, will print custom messages or state information. The output can be enabled globally with the \--debug flag or, more effectively, targeted to a specific pass or file by defining DEBUG\_TYPE and using \--debug-only=\<your-tag\>.3 This allows for precise instrumentation without being overwhelmed by debug output from the entire system.

To make these debug traces more effective, it is a best practice to assign unique names and labels to patterns. The setDebugName method provides a unique identifier for a single pattern, while addDebugLabels can be used to group related patterns.8 These identifiers appear in the

\-debug-only=dialect-conversion logs, making it easier to track their application. Furthermore, they enable powerful filtering capabilities with the \--enable-patterns=\<label\> and \--disable-patterns=\<label\> flags.1 These flags allow a developer to bisect a failure caused by the interaction of multiple patterns by selectively enabling or disabling specific groups of rewrites.

### **1.4 Stage 4: Advanced Instrumentation and Bisection**

For the most subtle and deeply embedded failures, direct inspection of IR and debug logs may be insufficient. In these cases, more advanced techniques are required to directly instrument the compiler's execution flow or to automate the reduction of the failing test case.

**Action Tracing and Interactive Debugging:** MLIR's Action Tracing infrastructure provides a fine-grained mechanism for observing and controlling compiler transformations.5 An "Action" can be an event as granular as "try to apply one canonicalization pattern." By enabling the debugger hook with

\--mlir-enable-debugger-hook, developers can attach a debugger like GDB or LLDB and gain interactive control over the execution of each action. This allows for stepping through pattern applications one by one, inspecting the full state of the compiler and the IR before and after each minute transformation.

**Debug Counters for Bisection:** The debug counter system enables a powerful bisection technique for locating a fault within a long sequence of transformations.5 If a failure occurs after, for example, 100 pattern applications, it can be tedious to step through them manually. The

\-mlir-debug-counter flag allows one to specify that a particular action (e.g., a pattern application) should be skipped a certain number of times and then executed only a limited number of times. For example, one could instruct the compiler to skip the first 99 applications of a pattern and then stop, allowing inspection of the IR state just before the single failing transformation. The \-mlir-print-debug-counter flag can be used to get a summary of all actions and their counts, which helps in setting up the bisection.

**Expensive API Checks:** A common source of verification failures is the incorrect use of the pattern rewriter API. To detect these issues, MLIR can be built with the CMake flag \-DMLIR\_ENABLE\_EXPENSIVE\_PATTERN\_API\_CHECKS=ON.3 This enables a set of runtime assertions within the

GreedyPatternRewriteDriver that check for common contract violations, such as a pattern returning success without modifying the IR, returning failure after modifying the IR, or modifying IR directly instead of using the PatternRewriter API. These checks can immediately pinpoint logical errors in a pattern's implementation that might otherwise lead to subtle, downstream verification failures.

**Automated Test Case Reduction:** When a bug is triggered by a large and complex IR input, identifying the specific constructs responsible can be challenging. The mlir-reduce tool automates this process.11 Given a failing IR file and a script that can determine if a given input is "interesting" (i.e., still triggers the bug),

mlir-reduce will systematically remove operations, arguments, and attributes, and apply simplifying rewrite patterns, to produce a minimal version of the IR that still reproduces the failure. This reduced test case often makes the root cause of the bug immediately apparent.

The selection of these tools should not be arbitrary but should follow a logical progression. The distinction between a compiler *crash* (e.g., a segmentation fault) and a *verification failure* is the primary branching point. A crash points to a bug in the C++ logic of the compiler itself, making tools like \-mlir-print-stacktrace-on-diagnostic 2 and a traditional debugger the primary instruments. A verification failure, in contrast, indicates that the compiler's logic is working correctly—it has successfully detected semantically invalid IR. The focus must therefore shift from debugging the compiler's

*execution* to debugging the *data* (the IR) it produces. This is why flags that manipulate and inspect the IR data, such as \--verify-each=0 and \-mlir-print-op-generic, are the correct starting point for verification failures.

| Table 1: Essential mlir-opt Debugging Flags for Pattern Failures |  |  |
| :---- | :---- | :---- |
| **Flag** | **Use Case** | **Description** |
| \--mlir-pass-pipeline-crash-reproducer=\<file\> | **Reproducer Generation:** Automatically capture the IR state and command line just before a pass failure. | The most effective way to create a minimal, stable reproducer for a failing pass.1 |
| \--verify-each=0 | **Exposing Invalid IR:** Disable the verifier between pattern applications to prevent the driver from rolling back a failing rewrite. | Allows the invalid IR generated by a pattern to persist so it can be printed and inspected.3 |
| \-mlir-print-op-generic | **Safe IR Printing:** Print operations in their generic form, which is robust against structurally invalid IR that might crash custom pretty-printers. | Essential when working with IR that is known to be malformed, especially after using \--verify-each=0.3 |
| \-debug-only=dialect-conversion | **Tracing Rewrites:** Get a detailed log of the GreedyPatternRewriteDriver's decisions, including which patterns match and what IR they produce. | The primary tool for understanding the sequence of events and pattern interactions leading to a failure.3 |
| \-mlir-print-ir-before-all / \-after-all | **Manual IR Inspection:** Dump the IR before and after every pass in the pipeline. | Useful for manually creating a reproducer or bisecting which pass introduces an issue.1 |
| \-mlir-print-op-on-diagnostic | **Contextual Error Reporting:** When a diagnostic is emitted, attach the textual form of the operation as a note. | Provides immediate context for a verification error without needing to re-run with other flags.2 |
| \--enable-patterns=\<label\> / \--disable-patterns=\<label\> | **Pattern Filtering:** Selectively run only a subset of patterns based on their debug labels. | A powerful technique for bisecting failures caused by the interaction of multiple rewrite patterns.1 |
| \-mlir-print-stacktrace-on-diagnostic | **Debugging Crashes:** Print a C++ stack trace when a diagnostic is emitted. | More useful for debugging compiler crashes than verification failures, but can help pinpoint the source of an error diagnostic.2 |

## **Section 2: The Anatomy of an MLIR Verification Failure**

Understanding how to expose and trace a rewrite failure is the first step; understanding *what* the failure signifies is the second, and more critical, one. A verification failure is not a generic error. It is a precise signal that a specific, well-defined structural or semantic invariant of the IR has been violated. The MLIR verifier acts as the guardian of these invariants, and its failure messages are the key to diagnosing the logical error in a rewrite pattern. This section provides the theoretical foundation necessary to interpret these failures by dissecting the core invariants that the verifier enforces, including SSA dominance, region semantics, and operation-specific contracts.

### **2.1 The Verifier's Mandate: Guardian of IR Invariants**

The MLIR verifier is a fundamental component of the pass manager's infrastructure. It is designed to run before and after every pass, ensuring that each transformation preserves the integrity of the IR.12 The successful completion of the verifier is a non-negotiable prerequisite for the pass manager to commit the changes made by a pass. If the verifier fails, the pass is considered to have failed, and its modifications are discarded.

The verification process itself is a systematic traversal of the IR, starting from a given operation.13 The

OperationVerifier class recursively descends through the regions and blocks nested within the operation. At each level, it performs a series of checks. For an Operation, it invokes verifyOperation, which checks fundamental properties like ensuring no operands are null, and then dispatches to operation-specific and trait-based verifiers. For a Block, it checks that terminators are correctly placed (i.e., only as the last operation) and that block arguments are properly owned. This comprehensive check ensures that the IR remains in a consistent and valid state throughout the compilation pipeline.

When an invariant is found to be violated, the verifier reports it by emitting a diagnostic using methods like emitError or emitOpError.6 These diagnostics are captured by the pass manager, which then halts execution of the current pass pipeline and triggers the failure and rollback mechanism. Thus, the diagnostic message printed to the console is the primary clue to understanding which invariant was broken.

### **2.2 The Cornerstone: SSA Dominance and Value Scoping**

MLIR is fundamentally a Static Single Assignment (SSA)-based IR.15 This principle has two main components: every value (the result of an operation or a block argument) has exactly one definition, and that definition must

*dominate* all of its uses.

Dominance is a concept rooted in the Control Flow Graph (CFG) structure of the IR. Within an MLIR region that has CFG semantics (an SSACFG region), a definition A is said to dominate a use B if every possible path of execution from the entry point of the region to B must pass through A.17 This guarantees that whenever

B is executed, the value defined by A will have already been computed. The verifier rigorously enforces this property. It leverages a DominanceInfo analysis to check, for every operand of every operation, that the operation using the value is dominated by the operation or block that defines the value.18

A classic and frequent cause of verification failures in rewrite patterns is the violation of this dominance property. For a pass like SpeculateAddress, which likely involves code motion of memory accesses or their address calculations, this is a primary suspect. For example, if a pattern moves an operation that uses a value %v to a new location that is no longer dominated by the definition of %v (e.g., moving it from one branch of an if statement to a point before the if), the verifier will fail. Within a single block, the rules are simpler: operations are lexically ordered, and an operation may only use values defined by preceding operations in the same block or by definitions in dominating blocks.15

### **2.3 Region Semantics and Isolation**

The concept of regions is central to MLIR's ability to model nested, hierarchical structures.15 An operation can have one or more regions attached to it, where each region contains a list of blocks. These regions create distinct lexical scopes and are governed by specific semantic rules. MLIR distinguishes between two primary kinds of regions:

SSACFG regions and Graph regions.17

SSACFG regions, such as the body of a func.func or scf.for, represent traditional control flow and are subject to the strict SSA dominance rules described above. Graph regions, such as the body of a builtin.module, contain a list of blocks but have no implied control flow between them; their scoping rules are consequently different.

Among the most critical invariants related to regions is the IsIsolatedFromAbove trait.23 When an operation, such as

func.func, carries this trait, it makes a powerful assertion: no operation nested inside any of its regions is permitted to use an SSA value that is defined outside of the operation itself. This creates a strong semantic "firewall" around the operation. The verifier for this trait explicitly walks all operations within the regions and checks that every operand they use is defined either by another operation within the same region or as a block argument to one of the region's blocks.24

This isolation is not merely a matter of code hygiene; it is a core architectural feature that enables modularity and parallelism in the MLIR pass manager. Because a pass running on an IsIsolatedFromAbove operation is guaranteed not to be able to see or affect the IR outside that operation, the pass manager can safely schedule passes on different isolated operations (e.g., two different functions) to run concurrently on separate threads.27

A rewrite pattern that violates this isolation boundary is therefore committing a severe architectural error. A common way this occurs is when a pattern attempts to move an operation into an isolated region (like a function body) without correctly remapping its operands. If the moved operation still refers to an SSA value defined outside the function, it punches a hole in the isolation firewall, creating invalid IR that the verifier will immediately reject. The SpeculateAddress pass could easily cause such a violation if it attempts to hoist a calculation out of one function and move it into another, or if it clones an operation into a function without using an IRMapping to remap its operands to values available within the function's scope.

### **2.4 Operation-Specific and Trait-Based Invariants**

Beyond the general principles of SSA and region isolation, the verifier also enforces a multitude of invariants specific to individual operations and traits.

**Custom Verifiers:** Dialects can, and should, provide custom verification logic for their operations. This is specified in the Operation Definition Specification (ODS) via hasVerifier \= 1 and implemented as a verify() method in C++.12 This custom verifier is responsible for enforcing any domain-specific semantic invariants. For example, the verifier for

arith.cmpi checks that its predicate attribute is a valid integer comparison enum value. The verifier for a scf.parallel operation checks for more complex structural properties, such as ensuring its region terminates with a valid scf.reduce operation and that the number of induction variables matches the number of loop bound operands.30 A rewrite pattern that modifies such an operation must be careful to preserve all of these invariants.

**Trait Verifiers:** Many common properties are abstracted into traits, and these traits can provide their own verification logic via a verifyTrait hook.12 This allows invariants to be defined once and reused across many different operations. For example,

OpTrait::SameOperandsAndResultType automatically generates a verifier that checks if all operands and results of an operation have the same type.34 Other examples include

OpTrait::IsTerminator, which verifies that an operation appears at the end of a block, and OpTrait::SingleBlock, which ensures an operation's region contains exactly one block. A rewrite pattern that, for instance, changes the types of an operation's operands without also updating its result type could inadvertently violate the SameOperandsAndResultType trait, leading to a verification failure.

A verification failure, therefore, should be approached as a puzzle. The error message is a clue that points to a specific broken rule. By categorizing the potential rules—SSA dominance, region isolation, operation contracts, trait contracts—a developer can form concrete hypotheses. If the error mentions dominance, an SSA violation is likely. If it mentions a value being unavailable in a region, IsIsolatedFromAbove is the prime suspect. If it complains about a specific attribute or type mismatch, an operation- or trait-specific invariant has likely been broken. This structured approach transforms debugging from a blind search into a methodical process of elimination, with the verifier's own source code (primarily in lib/IR/Verifier.cpp) serving as the ultimate ground truth for the invariants being enforced.13

## **Section 3: Deconstructing the Greedy Pattern Rewrite Driver**

The "rollback" behavior observed by the user is the direct result of the GreedyPatternRewriteDriver's internal design. This driver is the most common engine for applying canonicalization and optimization patterns in MLIR. Its algorithm is designed for efficiency and, above all, for correctness. Understanding its worklist-driven mechanics, its strict API contract with patterns, and its transactional application model is essential for diagnosing failures and for authoring patterns that cooperate correctly with the system.

### **3.1 The Worklist-Driven Algorithm**

The GreedyPatternRewriteDriver employs a worklist-driven algorithm to iteratively apply a set of rewrite patterns until a fixed point is reached.36 The process is as follows:

1. **Initialization:** The driver is initialized with a set of operations to process. These operations are added to a worklist.  
2. **Iteration:** The driver enters a loop that continues as long as the worklist is not empty.  
3. **Selection:** An operation is popped from the worklist.  
4. **Pattern Matching:** The driver iterates through all registered patterns that can match this type of operation.  
5. **Prioritization and Application:** Patterns are typically sorted by a static benefit value, an integer that indicates the expected "goodness" of a transformation.8 The driver greedily applies the first matching pattern with the highest benefit.  
6. **Update Worklist:** If a pattern applies successfully and modifies the IR, the driver updates the worklist. Any newly created operations are added to the worklist. Additionally, any operations that used the results of the original, now-replaced operation are also added back to the worklist, as the change may have enabled new patterns to match on them.  
7. **Convergence:** The process continues until the worklist is empty, meaning no more patterns can be applied. This state is called a fixed point.

To prevent faulty patterns from causing infinite loops (e.g., a pattern A \-\> B and another pattern B \-\> A), the driver has a configurable iteration limit. The \--test-convergence flag can be used in testing to explicitly fail the pass if this limit is reached, which is a useful tool for debugging cyclic rewrite dependencies.1 For patterns that are known to be safely recursive (e.g., a loop peeling pattern that can apply to its own result), the pattern can declare this by calling

setHasBoundedRewriteRecursion, which signals to the driver that repeated application on its own output is expected and safe.8

### **3.2 The Pattern-Driver Contract: A Strict API**

For the worklist algorithm to function correctly, patterns must adhere to a strict API contract. The driver relies on patterns to report their actions accurately and to perform all modifications through a specific interface.

The most fundamental rule is that **all IR mutations must be performed using the PatternRewriter instance** passed to the matchAndRewrite method.8 This object serves as an essential intermediary between the pattern and the IR. It provides methods for all possible mutations: creating operations (

create), erasing operations (eraseOp), replacing operations (replaceOp), and performing in-place modifications. A pattern must *never* modify the IR directly (e.g., by calling an operation's erase() method).

The PatternRewriter is more than a collection of convenience functions; it is the mechanism by which the driver tracks changes to the IR. When a pattern calls rewriter.eraseOp(op), the rewriter implementation provided by the driver can record this action, notify any listeners, update its internal state, and manage the worklist. Most importantly, it allows the driver to stage all changes and either commit them or roll them back atomically. Direct modification of the IR bypasses this entire tracking system, corrupting the driver's state and leading to undefined behavior, incorrect results, or crashes.

To enforce this contract, MLIR provides the \-DMLIR\_ENABLE\_EXPENSIVE\_PATTERN\_API\_CHECKS=ON build flag.3 When enabled, this flag compiles in a set of runtime assertions that detect common API violations, such as:

* A pattern returning LogicalResult::success() without having made any modifications to the IR.  
* A pattern returning LogicalResult::failure() after having modified the IR.  
* A pattern modifying an operation directly, bypassing the rewriter.

Another key aspect of the contract is that the root operation of the match—the one passed as an argument to matchAndRewrite—must be addressed by the rewrite. The pattern is required to either replace it, erase it, or update it in-place.8 A pattern cannot simply match an operation as an anchor point and then modify some other, unrelated part of the IR while leaving the root operation untouched.

### **3.3 Transactional Application and the "Rollback" Mechanism**

This subsection directly addresses the user's primary observation: the rollback. The GreedyPatternRewriteDriver applies each successful pattern within a transactional context. This is a core design principle that ensures the overall correctness and robustness of the compiler. The sequence of events for a single pattern application is as follows:

1. The driver selects an operation from the worklist and finds a high-benefit pattern that successfully matches it.  
2. The driver invokes the pattern's matchAndRewrite method, passing it a PatternRewriter instance that is configured to stage changes rather than apply them immediately.  
3. The pattern executes its logic, using the rewriter to create, erase, and modify operations. These changes are recorded by the rewriter but not yet committed to the main IR.  
4. The pattern returns LogicalResult::success(), signaling to the driver that it has successfully matched and staged a rewrite.  
5. **This is the critical step: The driver now invokes the MLIR verifier on the IR, *as if* the staged changes had been applied.**  
6. **If verification passes,** the driver commits the staged changes, making them permanent. It then proceeds to update its worklist with any new or affected operations.  
7. **If verification fails,** the driver **discards all of the staged changes**. The IR is reverted to its exact state from before matchAndRewrite was called. This is the "rollback." The driver then typically marks the pattern as "failed to apply" for this operation and may try other, lower-benefit patterns.

This transactional model means that the "rollback" is not a bug in the driver; it is the driver's primary correctness feature. It guarantees that the compiler can never be left in an invalid, partially transformed state due to a faulty pattern. The bug lies entirely within the pattern, which has produced a set of changes that, when applied together, result in unverifiable IR. This reframes the debugging task entirely. The goal is not to determine why the driver is rolling back, but to determine why the IR being produced by the pattern is invalid, which connects directly to the verification principles discussed in Section 2\.

## **Section 4: Best Practices for Authoring Robust Rewrite Patterns**

Writing rewrite patterns that are correct by construction is key to building a stable and reliable compiler pass. The verification failures encountered in the SpeculateAddress pass are symptomatic of common pitfalls in pattern implementation, particularly those involving complex transformations that cross structural boundaries in the IR. This section provides prescriptive guidance on best practices for authoring robust patterns, covering mutation strategies, handling of regions, safe contextual matching, and the use of declarative frameworks to enhance correctness.

### **4.1 Mutation Strategy: Replace vs. In-Place Update**

The PatternRewriter offers two primary strategies for modifying an operation: replacement and in-place update. The choice between them has implications for safety and complexity.

**Replacement (The Safer Default):** The most common and generally safest approach is to replace the matched operation entirely. This is accomplished with methods like rewriter.replaceOp(op, newValues) or the more convenient rewriter.replaceOpWithNewOp\<NewOpTy\>(op,...).40 These methods perform an atomic swap: they create one or more new operations, replace all uses of the original operation's results with the results of the new operations, and then erase the original operation. This approach is robust because it avoids creating transient, intermediate states where an operation might be temporarily invalid. The new operation is constructed in its final, valid form before being integrated into the IR.

**In-Place Update (For Targeted Changes):** For transformations that only modify the attributes or operands of an operation without changing its fundamental structure or type, an in-place update can be more efficient. However, this requires careful management to ensure the driver's internal state remains consistent. Any in-place modification must be bracketed by calls to rewriter.startOpModification(op) and rewriter.finalizeOpModification(op).9 The

startOpModification call notifies the driver that the operation is about to be mutated. After the changes are made (e.g., using op-\>setOperand(...) or op-\>setAttr(...)), finalizeOpModification signals that the mutation is complete. This allows the driver to correctly update its tracking information and, if necessary, re-add the modified operation to the worklist for further canonicalization. The rewriter.modifyOpInPlace(op,...) helper provides a convenient RAII wrapper for this process. Failing to use this notification mechanism can lead to subtle bugs where the driver is unaware of a change, potentially causing it to miss subsequent pattern applications.

### **4.2 The Challenge of Regions: Propagating SSA Values**

Operations that contain regions, such as scf.for, scf.if, or func.func, introduce a significant level of complexity for rewrite patterns. A transformation that crosses a region boundary is an order of magnitude more complex than a local one and is a primary source of verification errors.

A common mistake is to add a new SSA value as an operand to a region-holding operation without also making that value accessible to the code *inside* the region. For example, if a pattern adds a new loop-carried variable to an scf.for operation, simply adding it to the iter\_args operand list is insufficient. The body of the scf.for is inside a region, which has its own scope. This will often lead to a verification failure, either because the new operand is not used inside (if the pattern stops there) or, more likely, because an attempt to use the value inside the region will fail as it is not in scope, potentially violating the IsIsolatedFromAbove trait if the parent operation has it.

The correct mechanism for passing an SSA value from an outer scope into a region is via **block arguments**.41 A pattern that performs such a transformation must meticulously execute the following steps:

1. Create the new, replacement region-holding operation with the additional external operand.  
2. Get a handle to the entry block of the relevant region within the new operation.  
3. Add a new block argument of the appropriate type to this entry block using block-\>addArgument(...).  
4. If other blocks branch to this entry block, their terminator operations must be updated to pass the new SSA value as an operand. For scf.for, this involves updating the scf.yield operation in the loop body to pass the value for the next iteration.  
5. Within the region, all intended uses of the external SSA value must be replaced with the newly created block argument.

This is an intricate process that requires careful manipulation of the IR. Any misstep, such as failing to update a terminator or providing a value of the wrong type, will result in a verification failure. The complexity of these region-aware rewrites underscores the need for thorough testing and a deep understanding of MLIR's scoping rules.

### **4.3 Safe Contextual Matching: Parents, Children, and Siblings**

The MLIR pass manager operates under a strict contract to enable safe multi-threaded execution. A pass running on a specific operation is prohibited from modifying any other operation that is not nested within it (with the exception of modifying the attributes of the current operation itself).28 This has direct implications for how rewrite patterns should be designed.

* **Safe Patterns:** It is safe for a pattern to match an operation and then inspect its ancestors (e.g., op-\>getParentOp() 42) or its descendants (i.e., operations within its regions). For example, a pattern matching a  
  func.func can safely iterate over the operations in its body to check for certain properties before deciding to apply a rewrite.  
* **Unsafe Patterns:** A pattern matching an operation A should generally avoid inspecting or relying on the state of A's siblings (e.g., A-\>getNextNode()). In a multi-threaded pass pipeline, another thread could be concurrently modifying or deleting that sibling operation, leading to data races or use-after-free errors. Patterns should be designed to be self-contained, relying only on the state of the matched operation and its hierarchical context (parents and children).

### **4.4 Declarative Rewrites for Increased Correctness**

While imperative C++ patterns offer maximum flexibility, they also place the full burden of correctness on the developer. For a large class of transformations—specifically, DAG-to-DAG rewrites—MLIR offers declarative frameworks that can significantly improve correctness and reduce boilerplate.

**TableGen-driven Rewrite Rules (DRR):** The Declarative Rewrite Rule system allows patterns to be specified concisely in TableGen files (.td).29 A rule consists of a source pattern to match and one or more result patterns to generate. The

Pat\<...\> class is used for this purpose.44

The benefits of this approach are substantial:

* **Reduced Boilerplate:** The mlir-tblgen tool automatically generates the C++ matchAndRewrite implementation, including all necessary dyn\_cast checks, operand/attribute extraction, and construction of new ops. This eliminates a major source of manual implementation errors.45  
* **Structural Clarity:** The DAG-based syntax makes the structure of the transformation explicit and self-documenting. It is immediately clear what subgraph is being matched and what it is being replaced with.  
* **Integrated Constraints:** Constraints on operands (types) and attributes can be specified directly within the source pattern. The generated code will enforce these constraints before attempting the rewrite.

**Pattern Description Language (PDLL):** For more advanced use cases, the Pattern Description Language (PDLL) is a more powerful successor to DRR.47 PDLL is a dedicated language for writing patterns that is designed to handle more complex scenarios, such as matching non-rooted DAGs, replacing multiple operations simultaneously, and even being compiled just-in-time at runtime.

Adopting a declarative-first approach is a powerful strategy for improving the robustness of a pass. For any transformation that can be expressed as a structural DAG-to-DAG rewrite, using DRR or PDLL should be the default choice. This leverages the MLIR infrastructure to enforce correctness and frees the developer to focus on the semantics of the transformation rather than the intricacies of its C++ implementation. Imperative C++ patterns should be reserved for cases that require complex conditional logic, region manipulation, or other logic that cannot be expressed declaratively.

## **Section 5: A Concrete Action Plan for SpeculateAddress**

This final section synthesizes the preceding analysis into a targeted, actionable plan for debugging the SpeculateAddress pass. It translates the general principles of diagnostics and robust pattern design into specific hypotheses and a concrete checklist tailored to the likely nature of the pass.

### **5.1 Formulating Hypotheses based on the Pass Name**

The name SpeculateAddress strongly implies that the pass performs transformations related to memory accesses (memref.load, memref.store) and their associated address calculations. This class of transformation is particularly prone to verification errors because it often involves code motion across control-flow boundaries. Based on this, we can formulate several high-probability hypotheses for the root cause of the verification failure:

* **Hypothesis A (Dominance Violation):** The pass is moving a memory operation or part of its address computation (e.g., an arith.addi or memref.subview) to a location that is no longer dominated by the definitions of its operands. A classic example would be hoisting a load memref.load %ptr\[...\] to a point before the SSA value %ptr has been defined. This is a direct violation of SSA principles.  
* **Hypothesis B (Region Isolation Violation):** The pass is hoisting a memory access out of a control-flow region (like an scf.if or scf.for) to speculate its execution. If this hoisted operation is then placed within the body of an operation with the IsIsolatedFromAbove trait (most commonly, func.func), but it still uses an operand defined outside that function, it will violate the region's isolation boundary.  
* **Hypothesis C (Invalid Type or Attribute):** The pass may be creating new operations with incorrect or inconsistent types. For example, it might construct a new memref.load but provide it with a result type that does not match the element type of the memref. This could violate an operation-specific verifier or a trait like SameOperandsAndResultElementType.34  
* **Hypothesis D (Incorrect Region Update):** If the pass modifies an operation with a region, such as moving a speculative load into the body of an scf.parallel loop, it may be failing to correctly update the region's structure. As detailed in Section 4.2, this would require adding a new block argument for the load's address pointer and updating the terminator, a complex and error-prone procedure.

### **5.2 A Tailored Debugging Checklist**

The following step-by-step checklist provides a concrete workflow for the user to test the above hypotheses using the tools from Section 1\.

1. **Generate Reproducer:** If not already done, use mlir-opt... \--mlir-pass-pipeline-crash-reproducer=reproducer.mlir on the failing test to generate a minimal, self-contained test case.  
2. **Expose the Invalid IR:** Run mlir-opt on the new reproducer with the SpeculateAddress pass, but add the flags to expose the invalid state:  
   Bash  
   mlir-opt reproducer.mlir \-speculate-address \--verify-each=0 \-mlir-print-op-generic \> invalid.mlir

   The file invalid.mlir now contains the malformed IR that was previously being rolled back.  
3. **Analyze the Invalid IR:** Carefully inspect invalid.mlir, looking for evidence that supports one of the hypotheses.  
   * **To Test Hypothesis A (Dominance):** Locate the new or moved operations created by your pass. For each operand of these operations, trace back to its defining operation. In the context of the control-flow graph, does the definition dominate the use? Manually inspect branches and loops to confirm that the definition will *always* execute before the use on any valid path.  
   * **To Test Hypothesis B (Isolation):** Scan for any operation inside a func.func (or other IsIsolatedFromAbove op) that uses an SSA value defined outside of that function's scope. Such a use is a definitive violation. The value must be passed in as a function argument.  
   * **To Test Hypotheses C and D (Op/Region Contracts):** Examine the new operations. Are their result types consistent with their operand types and attributes? If an operation with a region was modified (e.g., an scf.for), were its region's block arguments correctly added or updated to correspond to any new operands passed to the operation itself? Is the region's terminator (scf.yield) correctly updated?  
4. **Trace the Driver:** If static analysis of invalid.mlir is inconclusive, the issue might stem from an unexpected interaction of patterns. Run the reproducer again, this time with the driver trace enabled:  
   Bash  
   mlir-opt reproducer.mlir \-speculate-address \-debug-only=dialect-conversion &\> trace.log

   Inspect trace.log to see the precise sequence of rewrites that produced the invalid state. This can reveal if, for example, a canonicalization pattern ran after your pattern and modified the IR in an unexpected way that led to the verification failure.  
5. **Use Expensive API Checks:** If the bug remains elusive, it may be a subtle violation of the rewriter API. Recompile MLIR with \-DMLIR\_ENABLE\_EXPENSIVE\_PATTERN\_API\_CHECKS=ON and re-run the failing test. If the pattern is violating the API contract (e.g., returning success without making a change), this will trigger a runtime assertion, immediately pinpointing the faulty pattern logic.

### **5.3 Common Error Patterns and Resolutions (Illustrative Examples)**

To make the abstract guidance concrete, here are several "Incorrect vs. Correct" examples of patterns that perform transformations thematically similar to those in a SpeculateAddress pass.

#### **Example 1: Hoisting a Calculation (Potential Dominance Violation)**

**Scenario:** A pattern attempts to hoist an address calculation (arith.addi) out of an scf.if to avoid recomputing it on every execution.

**Incorrect Implementation:**

C++

// Incorrect: Blindly clones the op without checking operand dominance.  
LogicalResult matchAndRewrite(arith::AddIOp addOp, PatternRewriter \&rewriter) const override {  
  auto ifOp \= addOp-\>getParentOfType\<scf::IfOp\>();  
  if (\!ifOp) return failure();

  // This is unsafe\! %offset might be defined inside the 'if' region.  
  rewriter.setInsertionPoint(ifOp);  
  rewriter.clone(\*addOp.getOperation());  
  //... replace original addOp with cloned result...  
  return success();  
}

*Failure Mode:* If an operand to addOp is defined inside the scf.if region, the cloned operation outside the if will use a value that is not in scope, violating dominance.

**Correct Implementation:**

C++

// Correct: Verifies that all operands are defined outside the region before hoisting.  
LogicalResult matchAndRewrite(arith::AddIOp addOp, PatternRewriter \&rewriter) const override {  
  auto ifOp \= addOp-\>getParentOfType\<scf::IfOp\>();  
  if (\!ifOp) return failure();

  // Check that all operands are defined outside the IfOp's region.  
  for (Value operand : addOp.getOperands()) {  
    if (ifOp.getOperation()-\>isProperAncestor(operand.getDefiningOp())) {  
      // This operand is defined inside the 'if', cannot hoist.  
      return failure();  
    }  
  }

  // Now it is safe to hoist.  
  OpBuilder::InsertionGuard guard(rewriter);  
  rewriter.setInsertionPoint(ifOp);  
  Operation \*clonedOp \= rewriter.clone(\*addOp.getOperation());  
  rewriter.replaceOp(addOp, clonedOp-\>getResults());  
  return success();  
}

#### **Example 2: Adding a Loop-Carried Dependency to scf.for (Region Update)**

**Scenario:** A pattern needs to add a new value, computed before a loop, as a loop-carried variable to an scf.for.

**Incorrect Implementation:**

C++

// Incorrect: Adds an operand to scf.for but fails to update the region.  
LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter \&rewriter) const override {  
  Value newValue \= /\*... some value defined before the loop... \*/;

  SmallVector\<Value\> newIterArgs \= forOp.getInitArgs();  
  newIterArgs.push\_back(newValue);

  // This is incomplete. The loop body does not know about the new variable.  
  rewriter.modifyOpInPlace(forOp, \[&\]() {  
    forOp.getInitArgsMutable().assign(newIterArgs);  
  });  
  return success();  
}

*Failure Mode:* The scf.for verifier will fail because the number of init\_args (operands) no longer matches the number of block arguments in the region body, and the scf.yield is also now incorrect.

**Correct Implementation:**

C++

// Correct: Atomically replaces the loop and correctly wires the new variable  
// through the region's block arguments and terminator.  
LogicalResult matchAndRewrite(scf::ForOp forOp, PatternRewriter \&rewriter) const override {  
  Value newValue \= /\*... some value defined before the loop... \*/;

  // 1\. Prepare new operands and result types for the replacement loop.  
  SmallVector\<Value\> newInitArgs \= forOp.getInitArgs();  
  newInitArgs.push\_back(newValue);  
  SmallVector\<Type\> newResultTypes \= forOp.getResultTypes();  
  newResultTypes.push\_back(newValue.getType());

  // 2\. Create the new loop.  
  auto newForOp \= rewriter.create\<scf::ForOp\>(  
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(),  
      newInitArgs);

  // 3\. Move the old loop's body into the new loop.  
  rewriter.inlineRegionBefore(forOp.getRegion(), newForOp.getRegion(), newForOp.getRegion().end());

  // 4\. Add the new block argument to the loop body.  
  Block \*body \= \&newForOp.getRegion().front();  
  Value newBlockArg \= body-\>addArgument(newValue.getType(), newValue.getLoc());  
    
  // 5\. Update the terminator.  
  auto yieldOp \= cast\<scf::YieldOp\>(body-\>getTerminator());  
  SmallVector\<Value\> newYieldOperands \= yieldOp.getResults();  
  // Here, we assume the value is just carried through unmodified for simplicity.  
  // A real pattern would use the result of some computation inside the loop.  
  newYieldOperands.push\_back(newBlockArg);   
  rewriter.setInsertionPoint(yieldOp);  
  rewriter.create\<scf::YieldOp\>(yieldOp.getLoc(), newYieldOperands);  
  rewriter.eraseOp(yieldOp);

  // 6\. Replace the original loop.  
  SmallVector\<Value\> newResults \= newForOp.getResults();  
  Value newCarriedValue \= newResults.pop\_back\_val();  
  // The old results are replaced by the corresponding new results.  
  rewriter.replaceOp(forOp, newResults);  
  // The new carried value can now be used.  
  //...  
  return success();  
}

This final, complex example illustrates the level of care required when writing patterns that manipulate region structure. By following the systematic debugging workflow and adhering to the principles of robust pattern design, the developer of SpeculateAddress can effectively diagnose and resolve the verification failure, leading to a correct and stable compiler pass.

## **Conclusions**

The "rollback" of a rewrite pattern within MLIR's GreedyPatternRewriteDriver is a feature designed to enforce correctness, not a bug. It signals that a pattern has produced IR that violates one of the framework's fundamental invariants. A successful debugging campaign requires a methodical approach that shifts focus from the driver's behavior to the invalid output of the pattern itself.

The diagnostic process should be structured as a funnel: first, isolate the failure with a minimal reproducer; second, disable the verifier to expose the malformed IR; third, trace the rewrite engine's logic to understand the sequence of transformations; and finally, employ advanced instrumentation for deeply embedded issues. This workflow transforms debugging from an ad-hoc process into a systematic investigation.

Understanding the anatomy of a verification failure is paramount. Errors are not arbitrary; they correspond to violations of specific rules governing SSA dominance, region isolation (particularly the IsIsolatedFromAbove trait), and operation-specific contracts. By mapping an error message to the invariant it protects, a developer can quickly form and test precise hypotheses about the logical flaw in their pattern.

Finally, authoring robust patterns is a preventative measure against such failures. This involves preferring safe replacement over complex in-place updates, meticulously managing region boundaries and block arguments when performing cross-region transformations, respecting the pass manager's concurrency constraints, and leveraging declarative frameworks like DRR and PDLL to reduce implementation errors. By internalizing these principles and applying the systematic diagnostic workflow, developers can confidently build complex, correct, and robust transformations within the MLIR ecosystem.
