

# **Analysis and Resolution of the 'Pattern' Class Undefined Error in MLIR TableGen**

## **Deconstructing the 'Pattern' Class Undefined Error in MLIR TableGen**

The build error error: The class 'Pattern' is not defined is a precise diagnostic that points to a failure during the code generation phase orchestrated by the mlir-tblgen utility. This error does not originate from the C++ compiler (like Clang or GCC) but from the TableGen tool itself. Understanding this distinction is the first step toward a correct diagnosis. The error indicates that when mlir-tblgen was parsing a .td (TableGen description) file to generate C++ code for canonicalization patterns, it encountered a reference to a TableGen class named Pattern that had not been previously defined or included in the parsing context. This is fundamentally a dependency resolution failure within the TableGen language ecosystem.

### **The Role of TableGen in MLIR's Declarative Rewrite Rule (DRR) System**

TableGen is a domain-specific language and tool used extensively throughout LLVM and MLIR to manage records of domain-specific information.1 Instead of writing repetitive and error-prone boilerplate C++ code, developers define high-level specifications in

.td files. A TableGen backend then processes these records to generate the necessary C++ source code.3 In MLIR, this mechanism is central to two key systems: the Operation Definition Specification (ODS), for defining dialect operations, and the Declarative Rewrite Rule (DRR) system, for defining graph transformations and canonicalizations.4

The DRR system allows compiler developers to specify complex graph-to-graph rewrites in a concise, declarative format, abstracting away the intricacies of the C++ mlir::RewritePattern classes.4 This is accomplished through TableGen's

class and def constructs. A TableGen class serves as an abstract template, defining a set of fields and parameters that can be inherited by other records. A def, in contrast, is a concrete record instance, often specializing a class to define a specific entity.1 The error message's reference to a

class is significant, as it signals a failure to find the fundamental template from which the concrete rewrite rule definitions (defs) are derived.

### **The 'Pattern' and 'Pat' Classes: Cornerstones of DRR**

Within MLIR's DRR system, the Pattern class is the primary TableGen construct used to define a rewrite rule. It is a parameterized class that takes a source pattern (a directed acyclic graph, or DAG, of operations to match) and a list of result patterns (one or more DAGs to replace the matched structure).4 For the common scenario where a single source DAG is replaced by a single result DAG, the DRR system provides a convenient subclass named

Pat.4

It is critical to recognize that Pattern and Pat are not C++ classes at the point where the error occurs. They are TableGen classes defined within the MLIR framework's own .td files. When a developer defines a new rewrite rule, they create a def that inherits from one of these classes, for example: def MyRewrite : Pat\<(MyOp $arg), (AnotherOp $arg)\>;. The mlir-tblgen tool, when invoked with the \-gen-rewriters backend, processes these def records and generates the corresponding C++ subclasses of mlir::RewritePattern.4

### **Initial Diagnosis: A Dependency Resolution Failure**

The error The class 'Pattern' is not defined is therefore an unambiguous signal that the TableGen parser, while processing the user's .td file, could not find the definition for the Pattern class. This is not a C++ linkage or compilation error but a parsing-time failure within the code generation step itself. The problem lies entirely within the TableGen source files and their interdependencies. The parser encountered a def that claimed to inherit from Pattern (or Pat) before the definition of that base class was made available to it. This points directly to a missing dependency—specifically, a failure to include the core TableGen file that defines the DRR infrastructure.

This situation reveals a key aspect of the MLIR development experience. While TableGen provides a powerful abstraction to simplify compiler development, its implementation details can become visible during debugging. The error forces the developer to look past the high-level declarative syntax and engage with the mechanics of the underlying tooling: the TableGen language's own scoping and inclusion rules, the mlir-tblgen command-line interface, and the build system's orchestration of the entire process. The error is not merely a "missing header" but a breakdown in the connective tissue between these distinct layers of the compiler framework.

## **Primary Cause Analysis: Missing or Incorrect TableGen Includes**

The most direct and frequent cause of the "Pattern class not defined" error is the omission of a critical include directive in the .td file that defines the canonicalization patterns. The TableGen parser processes files in a linear fashion, and all definitions must be declared before they are used, much like in C++.

### **The Canonical Source: mlir/IR/PatternBase.td**

The Pattern and Pat TableGen classes, which form the foundation of the Declarative Rewrite Rule (DRR) system, are defined in a single, specific file within the MLIR source tree: mlir/IR/PatternBase.td.4 This file contains the complete specification for these classes, including their parameters for source patterns, result patterns, and additional constraints. The definition of the

Pattern class, for instance, is specified as:

Code-Snippet

class Pattern\<dag sourcePattern, list\<dag\> resultPatterns,  
              list\<dag\> additionalConstraints \=,  
              list\<dag\> supplementalPatterns \=,  
              dag benefitsAdded \= (addBenefit 0)\>;

This signature clearly shows that Pattern is a TableGen class that accepts DAGs and lists of DAGs as template arguments, which correspond to the structure of the rewrite rule.4 Any

.td file that uses Pattern or its derivative Pat must have access to this definition.

### **The 'include' Directive in TableGen**

Analogous to the \#include preprocessor directive in C and C++, TableGen provides an include directive to incorporate the contents of one .td file into another.12 When the parser encounters

include "path/to/file.td", it effectively textually includes the specified file, making all of its classes and definitions available for subsequent lines in the current file. This is the standard mechanism for managing dependencies between different sets of TableGen records.

### **The Most Likely Fix**

Given that Pattern is defined in mlir/IR/PatternBase.td, the definitive solution is to ensure that the .td file containing the pattern definitions includes this file. The include directive must appear before the first use of Pat or Pattern.

Consider the following minimal example of a failing pattern definition file:

**Failing (MyDialectPatterns.td):**

Code-Snippet

// This file includes the operation definitions, but not the pattern base classes.  
include "MyDialect/MyDialectOps.td"

// This line will fail because the 'Pat' class has not been defined yet.  
def MyRewrite : Pat\<(MyOp $arg), (AnotherOp $arg)\>;

The mlir-tblgen parser would read this file, see the def for MyRewrite, and attempt to resolve its superclass, Pat. Having not seen a definition for Pat, it would emit the error.

The correction is straightforward:

**Corrected (MyDialectPatterns.td):**

Code-Snippet

// The critical addition: include the file that defines 'Pat' and 'Pattern'.  
include "mlir/IR/PatternBase.td"

// Include the operation definitions for the dialect.  
include "MyDialect/MyDialectOps.td"

// This definition is now valid because 'Pat' is a known class.  
def MyRewrite : Pat\<(MyOp $arg), (AnotherOp $arg)\>;

By adding include "mlir/IR/PatternBase.td", the parser first processes the definitions of the core DRR classes, making them available when it subsequently encounters the MyRewrite definition.

### **Necessary Co-requisite Includes**

Successfully defining patterns requires more than just PatternBase.td. A well-formed pattern file typically depends on a small hierarchy of includes:

1. **mlir/IR/PatternBase.td**: As established, this is essential for the Pattern and Pat classes themselves.  
2. **mlir/IR/OpBase.td**: This file defines the foundational TableGen classes for MLIR operations, such as Op, Dialect, and various traits.5 Since patterns operate on MLIR ops, and  
   PatternBase.td itself may have dependencies on these core constructs, including OpBase.td is a standard and safe practice.12  
3. **Dialect-Specific ODS File (e.g., MyDialectOps.td)**: The patterns must be able to reference the specific operations they intend to match and generate. Therefore, the .td file containing the Operation Definition Specification for the target dialect must also be included.12

The linear, top-down parsing model of TableGen implies an ordering for these includes. A robust convention is to include the most general files first, followed by more specific ones. This suggests that mlir/IR/PatternBase.td should be included before the dialect-specific Ops.td file, ensuring that all necessary pattern-matching infrastructure is available before the operations that might be used within those patterns are defined. This practice treats the .td files not as simple configuration but as source code with a strict parsing model, which is key to avoiding subtle dependency errors.

## **Secondary Cause Analysis: Build System and Configuration Issues**

If the necessary include directives are present in the .td file, but the error persists, the root cause likely lies one level higher in the compilation stack: the build system. The build system, typically CMake in MLIR-based projects, is responsible for invoking the mlir-tblgen executable with the correct arguments. A misconfiguration here can prevent mlir-tblgen from finding the files specified in the include directives, leading to the same "class not defined" error.

### **How 'mlir-tblgen' Resolves Include Paths**

The string provided in an include directive (e.g., "mlir/IR/PatternBase.td") is not an absolute path. The mlir-tblgen tool resolves this path by searching through a list of specified include directories. This search list is primarily populated using the \-I \<directory\> command-line flag.14 For the include to succeed, a directory path passed via

\-I must be a parent of the path in the include string. For example, to resolve include "mlir/IR/PatternBase.td", mlir-tblgen would need to be invoked with an argument like \-I /path/to/llvm-project/mlir/include.

### **Inspecting Your CMake Configuration**

In an MLIR project, these command-line arguments are not constructed manually but are generated by CMake scripts. The CMakeLists.txt files are therefore the source of truth for the build process. MLIR provides a set of helper functions to simplify this process, most notably mlir\_tablegen and add\_mlir\_dialect.16

The mlir\_tablegen function is a wrapper that constructs the full mlir-tblgen command. Crucially, it automatically adds the necessary \-I flags pointing to the core MLIR and LLVM include directories where files like PatternBase.td reside.16 A typical invocation for generating rewrite patterns looks like this:

CMake

\# Specifies which.td file is the main input for this target.  
set(LLVM\_TARGET\_DEFINITIONS MyDialectPatterns.td)

\# Invokes mlir-tblgen on the input file to generate the C++ include.  
\# The \-gen-rewriters flag selects the correct backend.  
mlir\_tablegen(MyDialectPatterns.h.inc \-gen-rewriters)

If a developer bypasses this helper function and attempts to use a more generic CMake function like add\_custom\_command, they become responsible for manually adding all the required \-I paths. This is a common source of error, as it is easy to omit the path to the main MLIR include directory. The persistence of the "Pattern class not defined" error, even with correct .td includes, strongly suggests that the build system is failing to provide mlir-tblgen with the path to the MLIR installation's include directory.

### **Target Dependencies**

Another potential build system issue is incorrect dependency management. The C++ code for a pass will typically \#include the file generated by mlir-tblgen (e.g., \#include "MyDialect/MyDialectPatterns.h.inc"). The build system must be configured to ensure that the mlir-tblgen command is executed *before* the C++ compiler is invoked on the file that depends on its output.

MLIR's CMake infrastructure manages this through explicit dependencies. The add\_mlir\_dialect\_library function accepts a DEPENDS argument for this purpose.18 A missing dependency would not cause the

mlir-tblgen error directly, but rather a C++ compiler error like "file not found." However, in complex build scenarios, misconfigured dependencies can lead to stale generated files being used, which can mask or create confusing secondary errors.

The following table summarizes common CMake misconfigurations and their solutions, providing a quick diagnostic guide for developers to audit their build scripts.

| Symptom | Incorrect CMake Snippet | Corrected CMake Snippet & Explanation |  |
| :---- | :---- | :---- | :---- |
| Pattern class not found, even with correct include directive. | add\_custom\_command(OUTPUT... COMMAND mlir-tblgen MyDialectPatterns.td \-o...) | set(LLVM\_TARGET\_DEFINITIONS MyDialectPatterns.td) mlir\_tablegen(MyDialectPatterns.h.inc \-gen-rewriters) Explanation: Avoid using add\_custom\_command for TableGen. The mlir\_tablegen function 16 is specifically designed to correctly configure all necessary include paths ( | \-I) to the main MLIR source tree. Manual invocation often omits these essential paths. |
| C++ compiler error: MyDialectPatterns.h.inc: No such file or directory. | add\_mlir\_dialect\_library(MLIRMyDialect MyDialect/Passes.cpp) | add\_mlir\_dialect\_library(MLIRMyDialect MyDialect/Passes.cpp DEPENDS MLIRMyDialectPatternsIncGen) Explanation: The DEPENDS keyword 18 establishes a dependency in the build graph, ensuring that the TableGen code generation step completes before the C++ compiler attempts to include the generated file. |  |
| TableGen error: dialect-specific op (MyOp) is undefined in the pattern file. | set(LLVM\_TARGET\_DEFINITIONS MyDialectPatterns.td) | set(LLVM\_TARGET\_DEFINITIONS MyDialectPatterns.td) \# Ensure the dialect's root source dir is in the include path include\_directories(${CMAKE\_CURRENT\_SOURCE\_DIR}) **Explanation:** The directory containing the dialect's Ops.td file must be in the include search path, allowing include "MyDialect/MyDialectOps.td" to be resolved. This is often handled by a project's root CMake file but can be misconfigured. |  |

## **A Systematic Debugging Workflow for TableGen Failures**

Resolving TableGen issues requires a systematic approach that treats the build process not as a black box, but as a data transformation pipeline. By isolating each stage of the pipeline, from the build system orchestrator to the code generation tool itself, the precise point of failure can be identified and corrected.

### **Step 1: Isolate the 'mlir-tblgen' Command**

The first step is to extract the exact command that the build system is executing. Build systems like make and ninja typically hide these details for a cleaner output. To reveal them, run the build with a verbose flag.

* For Makefiles: make VERBOSE=1  
* For Ninja: ninja \-v

Scan the verbose output for the line that invokes mlir-tblgen and processes the problematic .td file. This full command line, including all flags and paths, is the ground truth for the failure and the primary artifact for the debugging process.

### **Step 2: Reproduce the Failure Manually**

With the isolated command, the next step is to reproduce the error outside the build system. Navigate to the build directory in a terminal (this is important, as paths in the command are often relative to this location) and paste the full mlir-tblgen command. If it fails with the same error, the problem has been successfully isolated from the complexities of the build system's dependency tracking and parallelism. This confirms that the issue lies with the command's arguments or the content of the input .td file.

### **Step 3: Inspect TableGen's Internal State**

Instead of treating mlir-tblgen as a tool that only generates C++, it can be used as a diagnostic utility to inspect its own parsing state. This is achieved by using alternative backends that dump the internal record representation instead of generating code.14

* **Using \-print-records**: Modify the isolated command, replacing the code generation flag (e.g., \-gen-rewriters) with \-print-records. This instructs mlir-tblgen to parse all the input .td files and print a textual representation of all the classes and definitions it has successfully processed.1 Pipe this output to a file or a pager (e.g.,  
  ... | less). Search this output for class Pattern.  
  * If class Pattern is **not** found, it confirms that the include "mlir/IR/PatternBase.td" directive is either missing from the source .td file or is failing to resolve due to an incorrect include path (-I flag).  
  * If class Pattern **is** found, the issue may be more subtle, such as a typo in the user's def (def MyRewrite : pat\<...\> instead of Pat\<...\>).  
* **Using \-dump-json**: For more structured analysis, the \-dump-json backend can be used.14 This produces a JSON object representing all parsed records, which can be programmatically inspected with tools like  
  jq.

This introspection transforms debugging from guesswork into a methodical process. A hypothesis (e.g., "PatternBase.td is not being included") can be formed and then directly tested by observing the tool's internal state.

### **Step 4: Iterative Fixing and Verification**

Based on the findings from the previous step, apply the appropriate fix:

* If class Pattern was missing from the \-print-records output, add include "mlir/IR/PatternBase.td" to the top of the source .td file.  
* If the manual command fails with a "file not found" error on the include line itself, the problem is with the \-I paths. Examine the paths in the isolated command and trace them back to the CMakeLists.txt file to identify the misconfiguration.

After each attempted fix, re-run the isolated command with \-print-records to verify that class Pattern now appears in the output. Once it does, switch the flag back to \-gen-rewriters and run the command again. It should now execute successfully without errors.

### **Step 5: Examine the Generated Output**

The final verification step is to inspect the output file generated by the successful mlir-tblgen command (e.g., MyDialectPatterns.h.inc). This file should contain the generated C++ code.8 Open it and confirm that it contains a C++ class definition corresponding to each

def in the patterns .td file. This confirms that the entire pipeline, from declarative TableGen definition to generated C++ code, is functioning correctly.

## **Contextual Analysis: MLIR Versioning and Potential API Shifts (MLIR v20)**

A crucial part of diagnosing any compiler infrastructure issue is considering the possibility of breaking changes or API evolution in the specific version being used. The query specifically mentions MLIR v20, raising the question of whether the error is a symptom of a recent change in the framework.

### **Analyzing MLIR v20 Release Notes**

A review of the official MLIR release notes corresponding to LLVM 20 reveals no documented breaking changes to the core TableGen infrastructure or the Declarative Rewrite Rule (DRR) system.20 The changes highlighted for the LLVM 20 release cycle primarily concern the unification of various MLIR runner tools into a single

mlir-runner executable.20 The documentation and core APIs for defining operations (ODS) and patterns (DRR) have remained stable through this release.

### **Stability of the Core DRR API**

The Pattern and Pat TableGen classes, along with the \-gen-rewriters backend, are mature and foundational components of the MLIR ecosystem. They are used extensively across dozens of upstream dialects. The absence of any mention of their modification in the release notes or deprecation announcements is a strong indicator of their stability.21 Therefore, it can be concluded with high confidence that the "Pattern class not defined" error is not the result of a bug or an intentional API change in MLIR v20. The root cause is overwhelmingly likely to be a project-specific configuration issue in the user's

.td files or build scripts.

### **Relevant Recent Trends in MLIR (LLVM 17-19)**

While not the direct cause of the error, it is valuable to be aware of the broader evolutionary trends in MLIR that have occurred in recent releases, as they shape the context in which dialects are developed.

* **Properties Beyond Attributes**: A significant feature introduced around LLVM 17/18 is "Properties".20 This provides a new mechanism for storing an operation's inherent data that is distinct from the traditional  
  Attribute dictionary. This primarily affects ODS definitions (Ops.td files) where developers can opt-in to this new storage mechanism. While this changes the C++ API for accessing some operation data, it does not alter the fundamental syntax of DRR for matching and rewriting operations.  
* **Deprecation of APIs**: The MLIR project maintains a list of deprecated features and APIs.21 Reviewing this list shows active refactoring in areas like the C++  
  OpBuilder API and the removal of older operations. The DRR TableGen classes, however, are conspicuously absent from this list, further reinforcing their stability.

### **The Rise of PDLL as a DRR Alternative**

The most significant trend related to pattern rewriting in MLIR is the development and promotion of a new system called PDL (Pattern Description Language) and its user-facing syntax, PDLL.22 PDLL is a new declarative language for defining rewrite patterns that is itself an MLIR dialect. It was designed from the ground up to overcome some of the known limitations of the TableGen-based DRR system.

The impetus for PDLL was the difficulty DRR has in expressing patterns for modern, complex MLIR operations, specifically those involving:

* Variadic operands or results  
* Operations with attached regions  
* Complex constraints that span multiple operations or attributes 22

The user's own operation definitions, which include Variadic\<AnyType\> results and a SizedRegion, place their work directly in this category of advanced operations.5 While DRR can handle some of these cases, the syntax can become cumbersome. PDLL provides a more natural and powerful way to express such patterns. This suggests that while the immediate build error is a simple configuration issue within the DRR system, the nature of the user's work is pushing against the boundaries where DRR is most effective. The most valuable long-term strategic advice is not just to fix the

include but to be aware that a more powerful and better-suited tool (PDLL) exists for the compiler development path they are on.

## **Best Practices for Defining Canonicalization Patterns**

Adhering to established conventions and best practices when defining canonicalization patterns can significantly improve the maintainability, readability, and robustness of an MLIR dialect.

### **Structuring '.td' Files for Maintainability**

For clarity and separation of concerns, it is highly recommended to split TableGen definitions into multiple files based on their purpose:

* **MyDialect.td**: The main dialect definition file, containing the Dialect def itself.  
* **MyDialectOps.td**: Contains all the Operation Definition Specification (ODS) defs for the operations in the dialect.  
* **MyDialectPatterns.td**: Contains all the Declarative Rewrite Rule (DRR) defs for the dialect's canonicalization and other patterns.  
* **MyDialectTypes.td / MyDialectAttributes.td**: For dialects with custom types and attributes, these can also be separated into their own files.

A standard header should be used for pattern files (MyDialectPatterns.td) to ensure all dependencies are met:

Code-Snippet

\#ifndef MY\_DIALECT\_PATTERNS  
\#define MY\_DIALECT\_PATTERNS

include "mlir/IR/PatternBase.td"  
include "MyDialect/IR/MyDialectOps.td"

//... pattern definitions ('def's) go here...

\#endif // MY\_DIALECT\_PATTERNS

### **Integrating Generated Patterns into a C++ Pass**

The C++ code generated from the Patterns.td file must be integrated into the dialect to be used by the canonicalization pass. The standard mechanism for this is to implement the getCanonicalizationPatterns method on each operation that has patterns defined for it.8

In the operation's C++ implementation file (e.g., MyDialectOps.cpp), this method is populated by adding the generated patterns to the provided RewritePatternSet.

C++

\#**include** "MyDialect/IR/MyDialectPatterns.h.inc"

void MyOp::getCanonicalizationPatterns(mlir::RewritePatternSet \&results,  
                                       mlir::MLIRContext \*context) {  
  results.add\<MyRewritePattern1, MyRewritePattern2\>(context);  
}

The dialect must then be configured to populate all of its operations' patterns. This is typically done in the dialect's main C++ file. When the global canonicalizer pass runs, it queries each operation for its patterns and applies them greedily.24 For simpler, function-style canonicalizations that do not require the full power of DAG matching, an operation can specify

let hasCanonicalizeMethod \= 1; in its ODS definition and provide a C++ implementation of a canonicalize method.9

### **Choosing Between 'Pat' and 'Pattern'**

The choice between the Pat and Pattern TableGen classes depends on the complexity of the replacement:

* **Use Pat**: For the vast majority of canonicalizations where a single DAG of operations is replaced by another single DAG of operations. It is a convenient shorthand that improves readability.4  
* **Use Pattern**: This more general class must be used for more complex scenarios, such as when a single root operation is replaced by multiple new operations, or when an operation with multiple results is being rewritten and its results need to be replaced by values from different generated ops.4

### **Advanced DRR Features**

For patterns where the logic cannot be expressed purely declaratively, DRR provides escape hatches:

* **NativeCodeCall**: This directive allows embedding a snippet of C++ code directly into a pattern. It can be used in the source pattern to apply a complex C++ predicate as an additional constraint, or in the result pattern to compute a result value or attribute using C++ logic that is beyond the scope of simple DAG construction.4  
* **Constraints**: Beyond NativeCodeCall, the source pattern DAG can directly include constraints on operands (e.g., (MyOp (SomeOp:$input)) to match MyOp only when its input is from SomeOp) and attributes (e.g., (MyOp $input, I32Attr\<"42"\>)) to make matches more specific without resorting to C++ code.

## **Conclusion**

The build error error: The class 'Pattern' is not defined is a common but solvable issue encountered when developing declarative rewrite rules in MLIR. The analysis indicates that this is not a bug or a breaking change in MLIR v20, but rather a project-level configuration error.

The primary cause is the omission of the include "mlir/IR/PatternBase.td" directive in the .td file where canonicalization patterns are defined. This file provides the essential Pattern and Pat TableGen class definitions required by the DRR system. Secondary causes can stem from a misconfigured build system, specifically a CMakeLists.txt file that fails to provide the mlir-tblgen tool with the correct include paths to the MLIR source tree.

A systematic debugging workflow—isolating the failing command, reproducing it manually, and using mlir-tblgen's own diagnostic backends like \-print-records—provides a robust methodology for pinpointing the exact cause of this and similar TableGen failures. This approach transforms the tool from a black box into an introspectable part of the compilation pipeline.

Finally, while the immediate problem can be resolved with a simple configuration fix, the context of the user's work—defining operations with variadic results and regions—suggests a strategic recommendation. These complex operations are precisely the kind that test the limits of the TableGen DRR system. For future development, an investigation into MLIR's modern pattern-writing infrastructure, PDLL, is strongly advised. PDLL is specifically designed to handle these advanced cases with greater power and clarity, representing the future direction of pattern rewriting in the MLIR ecosystem. By resolving the current issue and adopting these best practices, developers can build more robust and maintainable MLIR-based compilers.
