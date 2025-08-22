

# **An Authoritative Engineering Guide to MLIR Dialect Definition with TableGen (v20)**

## **Introduction**

### **The Power and Peril of a Declarative System**

The Multi-Level Intermediate Representation (MLIR) stands as a modern, extensible compiler infrastructure designed to address the challenges of software fragmentation and the increasing complexity of heterogeneous hardware.1 Its power emanates from a modular dialect system, which provides a framework for creating domain-specific Intermediate Representations (IRs) that can capture high-level semantics, progressively lower to machine-specific code, and coexist within a single compilation pipeline.1

### **TableGen as the Single Source of Truth**

TableGen is the cornerstone of this extensibility. It is a domain-specific language and tool, inherited from the broader LLVM project, that allows compiler engineers to define the core components of a dialect—operations, attributes, and types—in a concise, declarative format.1 By specifying these components in TableGen description (

.td) files, developers create a single source of truth from which the mlir-tblgen utility can automatically generate a significant amount of the necessary C++ boilerplate code, including class definitions, parsers, printers, and verifiers.1 This declarative approach, known as the Operation Definition Specification (ODS), is the standard and recommended method for dialect development, as it drastically reduces implementation effort and enhances maintainability.5

### **The Learning Curve: From Cryptic Errors to Core Principles**

While powerful, the declarative abstraction provided by TableGen is notoriously "leaky." Developers new to the ecosystem are often confronted with build failures that are difficult to diagnose. An error in a .td file rarely manifests at its source; instead, it frequently triggers a cascade of failures that culminates in a cryptic C++ compilation error deep within an auto-generated .inc file or a non-obvious failure of the mlir-tblgen tool itself.8 These issues are not bugs in the framework but symptoms of a violation of its underlying design principles, implicit contracts, and canonical patterns.

This guide adopts a unique pedagogical approach. It leverages forensic analyses of these common, real-world build failures as a tool to illuminate the core mechanics of the MLIR and TableGen ecosystem. By deconstructing why these errors occur, this report aims to build a robust mental model for the reader, transforming confusion into a deep understanding of the framework's architecture.

### **Guide Structure and Objectives**

This document is structured to follow the logical progression of dialect development, from foundational setup to advanced features, with each section grounded in the analysis of specific failure modes. It begins by establishing the bedrock of a robust build system and the core dialect definition. It then proceeds to the definition of operations, focusing on the strict contracts imposed by ODS traits. Subsequently, it covers the nuances of crafting human-readable textual IR through custom assembly formats. The guide concludes with advanced topics, including the definition of custom attributes and declarative rewrite patterns.

The primary objective is to equip the compiler engineer with a comprehensive and authoritative reference for developing, debugging, and maintaining MLIR dialects effectively. All information and recommendations have been verified against the MLIR v20 baseline, ensuring relevance and accuracy for modern dialect development.

## **Section 1: Foundations: Dialect Anatomy and Build System Configuration**

This section establishes the bedrock of any custom dialect: the core dialect definition and a robust, scalable build system. The structure of the build system is not a matter of style but a technical necessity to prevent subtle and confusing errors that can halt development. A correctly configured foundation is the prerequisite for all subsequent work.

### **1.1. The Core Dialect Definition in TableGen**

The central record for any dialect is an instance of the Dialect TableGen class, which is defined in the core mlir/IR/DialectBase.td file.5 This definition serves as the anchor for all operations, attributes, and types belonging to the dialect. It is a best practice to place this definition in its own file (e.g.,

OrchestraDialect.td) to create a clear dependency hierarchy.8

The primary fields of the Dialect class are:

* **name**: A string literal that defines the dialect's textual namespace, which prefixes all of its operations in the IR (e.g., "orchestra" results in operations like orchestra.commit).4  
* **cppNamespace**: A string that specifies the C++ namespace for all generated code. It is a robust practice to use a fully qualified global namespace (e.g., "::mlir::orchestra") to prevent name collisions and ambiguity when integrating with other dialects or projects.4  
* **dependentDialects**: A list of other Dialect C++ classes that this dialect depends on. This field is critical for the correctness of transformation passes. If a pass intends to generate operations from another dialect (e.g., a lowering pass that creates arith or linalg operations), that dialect must be listed here. This ensures the MLIRContext has loaded the necessary dialect definitions, preventing runtime errors when the pass manager executes.8

A complete, modern dialect definition is as follows:

Code-Snippet

// In OrchestraDialect.td  
\#ifndef ORCHESTRA\_DIALECT\_TD  
\#define ORCHESTRA\_DIALECT\_TD

include "mlir/IR/DialectBase.td"

def Orchestra\_Dialect : Dialect {  
  let name \= "orchestra";  
  let summary \= "A dialect for high-performance fused operations.";  
  let description \=;

  // The C++ namespace for generated classes.  
  let cppNamespace \= "::mlir::orchestra";

  // Declare dependencies on dialects used by transformation passes.  
  let dependentDialects \=;

  // Explicitly conform to the modern Properties system.  
  let usePropertiesForAttributes \= 1;  
}

\#endif // ORCHESTRA\_DIALECT\_TD

### **1.2. The MLIR v20 Paradigm Shift: usePropertiesForAttributes**

A critical aspect of targeting MLIR v20 is understanding the "Properties" system. This is a mechanism introduced to store the *inherent attributes* of an operation (those central to its identity, like the predicate of a comparison op) in a dedicated C++ struct rather than in the generic DictionaryAttr.8 This improves both compiler performance, by avoiding string-based lookups, and type safety.

As of the LLVM 18 release, the TableGen flag let usePropertiesForAttributes \= 1; became the default behavior for all dialects.8 For MLIR v20, this is the established and required paradigm. Any custom C++ code written for an older version of MLIR, particularly in custom operation builders that manually add attributes to the

OperationState, will likely fail to compile against MLIR v20 due to this fundamental change in the generated C++ class structure.8 Explicitly setting this flag in the dialect definition, as shown above, serves as clear documentation that the dialect is designed to conform to the modern API and is the single most important change for resolving a class of version-related build failures.8

### **1.3. The Canonical CMake Build Configuration**

The convenience function add\_mlir\_dialect is often presented in introductory tutorials as the primary way to configure a dialect's build.11 While suitable for simple dialects contained within a single

.td file, it is insufficient for production-scale dialects and is the source of numerous, hard-to-diagnose build failures.

#### **The "Context Contamination" Problem**

The fundamental issue with high-level CMake functions is that they can obscure the underlying mechanics of mlir-tblgen. These functions may combine multiple .td files (e.g., one for operations and one for custom attributes) into a single invocation of the mlir-tblgen tool. This creates a "context contamination" problem: the tool is configured with a generator backend intended for one type of definition (e.g., operations) but is fed records of another type (e.g., attributes). The generator then incorrectly applies its internal logic, leading to errors like Unexpected overlap or the misapplication of OpDef name-resolution logic to an AttrDef record.8 The architectural solution is to abandon this high-level abstraction in favor of a granular approach that gives the developer explicit control over each code generation step.

#### **The Granular mlir\_tablegen Pattern**

The canonical and robust pattern for invoking mlir-tblgen in CMake is a two-step process that relies on an external state variable rather than direct arguments for input file specification. This design choice prioritizes consistency with the broader LLVM build system, which uses the tablegen macro for a wide variety of code generation tasks beyond MLIR.12 The pattern is:

1. Set the CMake variable LLVM\_TARGET\_DEFINITIONS to the path of the input .td file.  
2. Invoke the mlir\_tablegen function, providing only the desired *output* file name and the necessary generator flags.

The necessity of this non-obvious pattern is revealed by a common cascade of build errors encountered when attempting more intuitive approaches.

#### **Case Study: The Is a directory Error Cascade**

A forensic analysis of a typical debugging session demonstrates why the LLVM\_TARGET\_DEFINITIONS pattern is required.8

* Attempt 1: Intuitive but Incorrect Invocation  
  A developer, refactoring away from add\_mlir\_dialect, might logically try to pass the input file as a direct argument:  
  CMake  
  \# Incorrect Invocation  
  mlir\_tablegen(OrchestraOps.h.inc \-gen-op-decls OrchestraOps.td)

  This results in the error: mlir-tblgen: Too many positional arguments specified\!.8 The  
  mlir\_tablegen macro, when LLVM\_TARGET\_DEFINITIONS is unset, defaults to using the current source directory as an implicit input file. The command line passed to the tool therefore contains two positional arguments: the explicitly provided .td file and the implicitly added directory path, which the tool rejects.8  
* Attempt 2: The Definitive Clue  
  The next logical step is to remove the explicit .td argument to resolve the argument count mismatch:  
  CMake  
  \# Incorrect Invocation  
  mlir\_tablegen(OrchestraOps.h.inc \-gen-op-decls)

  This leads to the final, blocking error: mlir-tblgen: Could not open input file... Is a directory.8 This error is the crucial piece of evidence. With the explicit argument gone, the implicit behavior takes over completely, and the build system passes only the current directory path to  
  mlir-tblgen. The tool correctly fails because a directory is not a valid TableGen source file.8 This sequence proves that the input file must be specified through an alternative, state-based mechanism.

That mechanism is LLVM\_TARGET\_DEFINITIONS. This variable provides the necessary context to the generic tablegen macro, allowing it to construct a correct command line with a single positional argument for the input file.8

### **1.4. Managing Dependencies and Libraries**

A complete build configuration requires managing the dependencies between the generated code and the C++ source files that consume it.

* **The IncGen Target:** It is a best practice to create an aggregate CMake target that represents the successful completion of all mlir\_tablegen commands. This is typically done with add\_public\_tablegen\_target.11 This  
  IncGen target acts as a crucial synchronization point in the build graph.8  
* **add\_mlir\_dialect\_library:** This high-level function correctly encapsulates the complexity of creating the final shared or static library for the dialect.11 Its most important parameter is the  
  DEPENDS clause, which must list the IncGen target. This guarantees that CMake will execute all mlir-tblgen commands to generate the .inc files *before* it attempts to compile any of the C++ source files that \#include them, thus preventing "file not found" errors during compilation.8  
* **Linking:** Dependencies on other MLIR and LLVM libraries should be specified using the LINK\_LIBS and LINK\_COMPONENTS keywords, respectively.11

A complete CMakeLists.txt for the directory containing the .td files would look like this:

CMake

\# In include/Orchestra/CMakeLists.txt

\# \--- Generation for OrchestraDialect.td \---  
set(LLVM\_TARGET\_DEFINITIONS OrchestraDialect.td)  
mlir\_tablegen(OrchestraDialect.h.inc \-gen-dialect-decls)

\# \--- Generation for OrchestraOps.td \---  
set(LLVM\_TARGET\_DEFINITIONS OrchestraOps.td)  
mlir\_tablegen(OrchestraOps.h.inc \-gen-op-decls)  
mlir\_tablegen(OrchestraOps.cpp.inc \-gen-op-defs)

\# \--- (Add similar blocks for Attributes.td, Interfaces.td, etc.) \---

\# Create a single, public IncGen target that aggregates all generated files.  
add\_public\_tablegen\_target(OrchestraIncGen)

The following table summarizes the key mlir-tblgen generator flags required for a complete dialect definition.

| Generator Flag | Output File Suffix | Purpose | C++ \#include Location |
| :---- | :---- | :---- | :---- |
| \-gen-dialect-decls | Dialect.h.inc | Generates the C++ Dialect class declaration. | In the main Dialect header (e.g., OrchestraDialect.h). |
| \-gen-op-decls | Ops.h.inc | Generates C++ class declarations for all operations. | In the main Dialect header (e.g., OrchestraDialect.h). |
| \-gen-op-defs | Ops.cpp.inc | Generates C++ method definitions for all operations. | In the main Dialect source file (e.g., OrchestraDialect.cpp). |
| \-gen-attrdef-decls | Attributes.h.inc | Generates C++ class declarations for TableGen-defined attributes. | In a dedicated attribute header (e.g., OrchestraAttributes.h). |
| \-gen-attrdef-defs | Attributes.cpp.inc | Generates C++ method definitions for TableGen-defined attributes. | In a dedicated attribute source file (e.g., OrchestraAttributes.cpp). |
| \-gen-rewriters | \*Patterns.h.inc | Generates C++ RewritePattern classes from DRR patterns. | In the source file for canonicalization/folding patterns. |

## **Section 2: Defining Operations: The ODS Contract**

This section delves into the heart of a dialect: its operations. The Operation Definition Specification (ODS) framework provides a rich declarative syntax, but this power comes with the responsibility of adhering to strict, often implicit, contracts. We will use two related but distinct build failures to expose the precise contract of the AttrSizedOperandSegments trait, establishing a crucial pattern for handling operations with complex variadic operand structures.

### **2.1. The OpDef Record: Arguments, Results, and Traits**

The primary construct for defining an operation in TableGen is a def record that inherits from the Op class, defined in mlir/IR/OpBase.td.5 The body of this definition specifies all facts about the operation.

* **arguments and results**: These fields, which must be of type dag, define the operation's signature. They are led by the special ins and outs markers. The ins dag lists the operation's operands (runtime SSA values) and attributes (compile-time constant values). The outs dag lists the operation's results.5  
* **OpTrait**: Traits are the primary mechanism for specifying properties, injecting common functionality, and enforcing verification logic on an operation.5 Adding a trait to an operation's definition, such as  
  Terminator or SideEffectFree, causes mlir-tblgen to generate additional C++ code and base classes for the operation's C++ class.

### **2.2. The Challenge: Multiple Variadic Operand Groups**

A common design pattern involves operations that accept multiple, independently-sized groups of variadic operands. For example, a conditional commit operation might take a set of values for a "true" branch and a different set for a "false" branch.8 The in-memory representation of an MLIR operation stores all of its SSA value operands in a single flat list. When multiple variadic groups are present, the framework cannot statically determine where one group ends and the next begins. This parsing ambiguity must be resolved with explicit, per-instance information.8

The standard ODS mechanism for this is the AttrSizedOperandSegments trait. This trait signals that the operation will carry an integer array attribute that specifies the size of each operand segment at runtime.8

### **2.3. The AttrSizedOperandSegments Contract: A Two-Part Case Study**

The contract for using AttrSizedOperandSegments is unforgiving and poorly documented, often leading to a frustrating cycle of build failures. Its rules are best understood by examining the two distinct errors that arise from violating its two primary conditions.

#### **Case Study Part A: The Missing Attribute (parse error in template argument list)**

The first failure mode occurs when a developer correctly adds the AttrSizedOperandSegments trait to an operation but fails to provide the necessary attribute.

* **Symptom**: The build fails not in mlir-tblgen, but much later during C++ compilation, with a cryptic message like error: parse error in template argument list originating from the dialect's registration code.8  
* **Forensic Analysis**: This is a classic downstream symptom of an upstream code generation error. The AttrSizedOperandSegments trait instructs mlir-tblgen to inject C++ code into the operation's class that expects to find and use an attribute named operandSegmentSizes and its corresponding accessor methods (e.g., getOperandSegmentSizes()). When the .td file fails to declare this attribute in the arguments list, mlir-tblgen does not generate the attribute's storage or its accessors. The result is an internally inconsistent C++ class: it has member functions and is used in templates that depend on an interface that has not been fully generated. When the C++ compiler tries to instantiate a template (like addOperations\<\>) with this malformed class, a fatal template substitution failure occurs, which is often reported as a generic parse error.8  
* **Resolution**: The immediate fix for this specific failure is to add the required attribute to the operation's arguments list, fulfilling the compile-time contract expected by the trait's generated code.8  
  Code-Snippet  
  // Definition that causes a C++ compile-time failure  
  def MyOp : MyDialect\_Op\<"my\_op",\> {  
    let arguments \= (ins  
      Variadic\<AnyType\>:$ins1,  
      Variadic\<AnyType\>:$ins2  
      // MISSING: I32ArrayAttr:$operandSegmentSizes  
    );  
   ...  
  }

#### **Case Study Part B: The Redundant Attribute (Unexpected overlap error)**

Having fixed the first error, a developer might then encounter the second, which is caused by the exact opposite problem.

* **Symptom**: The build fails immediately during the mlir-tblgen step with the error Unexpected overlap when generating 'getOperandSegmentSizesAttrName'.8  
* **Forensic Analysis**: This error is a deliberate validation feature of the code generator, not a bug. It occurs because the developer has provided two conflicting directives:  
  1. By including the AttrSizedOperandSegments trait, they have invoked a code generation path that *implicitly* manages an attribute named operandSegmentSizes. This path is hardwired to generate the storage, accessors, and internal helpers for an attribute with this specific, reserved name.8  
  2. By including I32ArrayAttr:$operandSegmentSizes in the arguments list, they have issued a separate, explicit directive to generate a public getter for an attribute of the same name.  
     The mlir-tblgen tool correctly identifies this logical contradiction. It foresees that it is being asked to generate two different C++ class members with the same symbol name and terminates with a fatal error to prevent emitting syntactically invalid code.8  
* **Resolution**: The definitive solution is to *remove* the explicit I32ArrayAttr:$operandSegmentSizes from the arguments list and trust the trait to manage the attribute's existence implicitly. The declarative .td file should only specify the operation's logical arguments and traits, not the implementation details of those traits.8

### **2.4. The Division of Labor: Declarative .td vs. Imperative C++ Builder**

Synthesizing the findings from these two case studies reveals a fundamental principle of ODS design: there is a clear division of labor between the declarative .td specification and the imperative C++ implementation.

* The **.td file** defines the operation's static structure and interface. For AttrSizedOperandSegments, this means simply including the trait in the trait list.  
* The **C++ builder** defines the operation's runtime instantiation logic. The value of the operandSegmentSizes attribute is runtime state—it depends on the number of SSA values passed to a specific instance of the operation. Therefore, the responsibility for creating and populating this attribute correctly belongs to the C++ builder.8

This "razor's edge" contract—where omitting the attribute causes a downstream C++ failure and explicitly defining it causes an immediate TableGen failure—highlights the importance of finding and adhering to canonical, in-tree examples. Relying on documentation alone is often insufficient; the most reliable strategy is to treat the implementation of a production-quality, core dialect operation as the ground truth.

#### **The Canonical Example: cf.switch**

The cf.switch operation from MLIR's standard Control Flow dialect serves as the definitive, real-world model for the correct application of the AttrSizedOperandSegments trait.8

* **.td Definition**: The TableGen definition for cf.switch in ControlFlowOps.td clearly demonstrates the correct pattern. It includes AttrSizedOperandSegments in its trait list but its arguments block conspicuously omits any mention of operandSegmentSizes, providing irrefutable evidence of the correct declarative syntax.8  
* **C++ Builder Implementation**: The logic for populating the attribute resides in the C++ builder for cf.switch. A walkthrough of this builder reveals the required implementation pattern 8:  
  1. The builder's C++ signature accepts the operands for the different logical groups as distinct ValueRange arguments.  
  2. It begins by adding all operands from these disparate ranges into the single, flat operand list of the OperationState.  
  3. It then calculates the segment sizes by inspecting the .size() of the ValueRange arguments passed to the builder and stores them in a SmallVector\<int32\_t\>.  
  4. This vector of sizes is used to create an I32ArrayAttr instance via builder.getI32ArrayAttr(...).  
  5. Finally, this newly created attribute is added to the OperationState's attribute dictionary using the reserved name. The robust way to do this is to use the static helper method generated by the trait: state.addAttribute(getOperandSegmentSizesAttrName(state.name), sizeAttr);.

This clear division of responsibility—static interface in TableGen, runtime logic in the C++ builder—is the key to correctly implementing operations with multiple variadic operand groups.

The following table makes the implicit contract of the AttrSizedOperandSegments trait explicit, providing a clear checklist for developers.

| Requirement | Specification | Rationale / Consequence of Violation |
| :---- | :---- | :---- |
| **Trait Declaration** | The operation's def in the .td file must include AttrSizedOperandSegments in its trait list. | This is the primary directive that invokes the specialized code generation logic. |
| **Attribute Definition** | The operation's arguments list in the .td file must **NOT** include a definition for operandSegmentSizes. | The trait manages this attribute implicitly. Explicitly defining it creates a conflict, leading to the Unexpected overlap build failure from mlir-tblgen. |
| **Attribute at Runtime** | An instance of the operation must have an I32ArrayAttr (or similar) named operandSegmentSizes in its attribute dictionary. | This attribute provides the runtime information needed to delineate the variadic operand groups. Its absence leads to C++ code with unresolved dependencies, causing downstream parse error in template argument list compilation errors. |
| **C++ Builder Logic** | A custom C++ builder must be implemented. This builder is responsible for calculating the size of each operand segment at runtime, creating an I32ArrayAttr with these sizes, and adding it to the OperationState using the reserved name. | This is the correct mechanism for populating the runtime-dependent segment size information. Failure to provide a correct builder will result in operations being created without the mandatory attribute, leading to the C++ compilation failure. |
| **Verification** | The trait injects a verifyTrait method that checks at runtime if the sum of the segment sizes matches the total number of SSA value operands. | This is a runtime check for semantic correctness. The build failures occur because the compile-time contract is violated before any runtime verification can be executed. |

## **Section 3: Crafting Human-Readable IR: Custom Assembly Formats**

Defining a human-readable textual representation for operations is a critical aspect of dialect development. This textual form is not merely for debugging; it enables reliable round-tripping of the IR, which is essential for writing test cases and understanding the behavior of compiler transformations.2 This section deconstructs the declarative

assemblyFormat grammar and provides a complete guide to the programmatic C++ fallback for when the declarative system is insufficient.

### **3.1. Declarative vs. Programmatic: A Decision Framework**

MLIR provides two primary pathways for defining the assembly format of a custom operation: a declarative approach using the assemblyFormat string in TableGen, and a programmatic approach requiring a direct C++ implementation via the OpAsmOpInterface.8 The choice between them is a key architectural decision in dialect design.

* **Use Declarative assemblyFormat when:**  
  * The desired syntax can be mapped to the canonical MLIR structure: op-name operands attributes: type region. This structure is highly flexible and can be customized with keywords and optional groups to achieve significant readability.8  
  * All components of the operation (operands, results, attributes, regions) are explicitly defined in the ODS arguments, results, and regions lists, making them available as variables in the format string.  
  * The primary goal is to minimize boilerplate C++ code, maximize maintainability, and leverage the full power of TableGen's code generation. This should be the default choice for most operations.8  
* **Use Programmatic C++ (OpAsmOpInterface) when:**  
  * The syntax is highly irregular and cannot be expressed as a linear sequence of the standard components (e.g., the keyword-interspersed format of scf.for).8  
  * Parsing requires complex, context-dependent logic, such as inferring an attribute's value from an operand's type in a non-trivial way.  
  * The printed form must be customized in ways that deviate from the in-memory representation, such as eliding default attributes or printing derived information.  
  * Additional assembly-related behaviors are required, such as providing custom names for SSA values (getAsmResultNames), which is only available through the OpAsmOpInterface.8

### **3.2. Mastering the Declarative assemblyFormat Grammar**

The declarative assemblyFormat grammar is not a general-purpose parsing language but a domain-specific "macro" system for customizing a fixed, underlying generic printer. Its limitations and strict ordering requirements stem directly from this design. The parser for the custom format is not starting from scratch; it is a state machine that expects to see elements in an order that allows it to progressively build up the OperationState. It needs to parse operands *before* it can fully resolve and verify the functional type, and it needs to parse all named arguments *before* it can know what is left for the generic attr-dict. This mental model clarifies why the ordering is so rigid.

#### **Case Study: The expected literal, variable, directive... Error**

This common mlir-tblgen error serves as an excellent entry point for deconstructing the grammar.8

* **Root Cause**: The error is caused by violating the grammar's unspoken structure, which is designed as a customizable overlay on MLIR's generic operation form: op-name (ssa-operands) attribute-dictionary: functional-type (regions).2 Placing a directive like  
  functional-type at the beginning of the format string is a frequent mistake that violates the expected parsing order, as the parser needs to see the operands that constitute the type before the type itself.8

#### **Core Components and Ordering**

The assemblyFormat string is a space-separated sequence of three primary element types 8:

* **Literals**: Keywords or punctuation enclosed in backticks, such as \`to\` or \`-\>\`, which serve as fixed syntactic markers.  
* **Variables**: Identifiers prefixed with a dollar sign, such as $operands or $my\_attr, which create a binding to a component of the operation defined in TableGen.  
* **Directives**: Special keywords that invoke built-in parsing and printing logic. Key directives include:  
  * attr-dict: Handles any attributes not explicitly bound to a variable elsewhere in the format.  
  * functional-type(operands, results): Parses or prints the full (operand\_types) \-\> (result\_types) signature.  
  * region: Parses or prints the operation's attached regions.

The canonical and syntactically valid ordering for these components is: **operands \-\> attributes \-\> attr-dict \-\> functional-type \-\> region**.8

#### **Handling Variadic and Optional Elements**

The grammar's ability to handle optional elements, such as variadic operand lists, is achieved through the optional group, denoted by (...). For this to be parsed unambiguously, the group must contain an "anchor"—a variable marked with a caret (^). The presence of the value bound to the anchor variable determines whether the entire group is processed. For a variadic operand list $operands, the format ($operands^) ensures that the parentheses are only printed if the list is non-empty, a crucial feature for clean syntax.5 The

func.return operation provides a canonical example of this pattern.8

### **3.3. The Programmatic Fallback: OpAsmOpInterface**

When the declarative format is insufficient, OpAsmOpInterface provides complete control over parsing and printing.8 This is the modern, preferred mechanism over the legacy

hasCustomAssemblyFormat \= 1 flag.

#### **Definitive C++ Signatures**

The most common point of failure when implementing custom parsers and printers is a mismatch between the C++ method signatures and the declarations generated by TableGen. The following signatures are non-negotiable 8:

C++

// In the operation's C++ implementation file (e.g., OrchestraOps.cpp)

// The 'parse' method is static and populates an OperationState.  
static mlir::ParseResult MyOp::parse(mlir::OpAsmParser \&parser,  
                                    mlir::OperationState \&result);

// The 'print' method is a const member function.  
void MyOp::print(mlir::OpAsmPrinter \&printer) const;

#### **Implementation Patterns**

The logic inside these methods interacts with the OpAsmParser and OpAsmPrinter objects to consume and produce the textual format. The OperationState in the parse method is a temporary representation of the operation being built, to which parsed operands, attributes, types, and regions are added.8

The following table provides a canonical mapping from ODS argument types to the C++ API calls used to handle them within the parse and print methods. This serves as an invaluable reference for bridging the gap between the declarative and imperative worlds.

| ODS Argument/Result Type | parse C++ Handling | print C++ Handling |
| :---- | :---- | :---- |
| SomeAttr:$name | SomeAttr attr; if (parser.parseAttribute(attr, "name", result.attributes)) return failure(); | printer.printAttribute(getName()); |
| OptionalAttr\<T\>:$name | if (succeeded(parser.parseOptionalAttribute(...))) {... } | if (getName()) { printer.printAttribute(...); } |
| AnyType:$operand | OpAsmParser::UnresolvedOperand operand; parser.parseOperand(operand); | printer.printOperand(getOperand()); |
| Variadic\<AnyType\>:$operands | SmallVector\<OpAsmParser::UnresolvedOperand\> operands; parser.parseOperandList(operands); | printer.printOperands(getOperands()); |
| Variadic\<AnyType\>:$results | SmallVector\<Type\> resultTypes; parser.parseTypeList(resultTypes); result.addTypes(resultTypes); | printer.printTypes(getOperation()-\>getResultTypes()); |
| AnyRegion:$body | Region \*body \= result.addRegion(); parser.parseRegion(\*body, {}); | printer.printRegion(getBody(), /\*printEntryBlockArgs=\*/false); |

#### **The Canonical Example: func.func**

The func.func operation stands as the canonical example of a complex operation with a fully custom C++ parser and printer.8 Its TableGen definition simply states

let hasCustomAssemblyFormat \= 1;.8 The entire logic for parsing and printing its sophisticated syntax—including the symbol name (

@my\_func), argument lists with attributes (%arg0: i32 {attr}), result lists, and the region body—is handled in C++ helper functions like parseFunctionOp and printFunctionOp found in FunctionImplementation.cpp.8 A study of this implementation reveals the patterns necessary for handling syntax that would be impossible to express in the declarative format, making it a vital reference for advanced use cases.8

## **Section 4: Advanced Definitions: Custom Attributes and Rewrite Patterns**

This final section covers two advanced but essential topics: defining custom attributes with complex C++ storage and defining declarative rewrite rules for canonicalization. The build failures associated with these features reveal a unifying principle: the mlir-tblgen backends operate in distinct, isolated contexts, and the project's file structure and build invocations must respect this separation. The recommended structure of separating definitions into \*Ops.td, \*Attributes.td, and \*Patterns.td is not merely a stylistic convention but a hard technical requirement dictated by the design of the code generation tools.

### **4.1. Custom Attributes with Complex C++ Storage**

Defining a custom MLIR attribute with a complex C++ storage type, such as an llvm::ArrayRef of pairs, requires a two-part definition: a manually implemented C++ storage class and a corresponding declarative definition in TableGen.4

* **The C++ Storage Class**: For attribute parameters that are not simple value types and require heap allocation, MLIR mandates a custom C++ storage class.22 This class inherits from  
  mlir::AttributeStorage and represents a strict contract with the MLIRContext's uniquing system. It must implement a specific set of methods:  
  * A type alias KeyTy that defines the unique key for the attribute.  
  * A static factory method, construct, that allocates memory using the provided AttributeStorageAllocator and, critically, performs a deep copy of any heap-allocated data. This is essential for correct memory lifetime management.8  
  * An operator== for comparison and a hashKey method for efficient lookup.8  
* **The TableGen AttrDef**: This declarative record links the attribute to its C++ implementation. It must specify the storageClass (the exact C++ class name) and the storageNamespace (the full C++ namespace of the storage class).8

#### **Case Study: The Initializer of 'attrName' could not be fully resolved Error**

This build failure exposes the necessity of separating attribute definitions from operation definitions.

* **Symptom**: A mlir-tblgen error indicating that it is attempting to resolve the attrName field of an AttrDef using logic intended for an OpDef's opName, which requires a mnemonic field that attributes do not have.8  
* **Forensic Analysis**: This error is a direct result of the "Context Contamination" problem described in Section 1\. When AttrDef and OpDef records reside in the same .td file and are processed by a high-level CMake function like add\_mlir\_dialect, the op-centric generator (-gen-op-defs) is invoked. This generator incorrectly applies its Op-specific name resolution logic to all def records it encounters, including the AttrDef, causing the evaluation to fail.8  
* **Resolution**: The solution is to enforce context separation:  
  1. **File Separation**: Place all AttrDef records in a dedicated file, such as OrchestraAttributes.td.  
  2. **Granular Build**: Use a specific mlir\_tablegen invocation for that file with the appropriate generator flags: \-gen-attrdef-decls and \-gen-attrdef-defs.8

### **4.2. Declarative Rewrite Rules (DRR) for Canonicalization**

TableGen-based Declarative Rewrite Rules (DRR) provide a concise, declarative syntax for specifying the graph-to-graph transformations used in passes like canonicalization.23

* **The Pattern and Pat Classes**: These are the core TableGen classes, defined in mlir/IR/PatternBase.td, used to define rewrite rules. A def that inherits from Pat or Pattern specifies a source DAG to match and one or more result DAGs for replacement.8

#### **Case Study: The The class 'Pattern' is not defined Error**

This common error highlights a simple but critical dependency management requirement within TableGen itself.

* **Symptom**: A mlir-tblgen error stating that the Pattern class is not defined, which occurs when the tool is parsing a .td file to generate C++ rewrite patterns.8  
* **Forensic Analysis**: This is a fundamental dependency resolution failure within the TableGen language's parsing context. The parser processes files in a linear fashion. It encountered a def that inherits from Pat or Pattern *before* the definition of that base class was made available to it.8  
* **Resolution**: The definitive solution is to add include "mlir/IR/PatternBase.td" at the top of the .td file containing the pattern definitions. This ensures the parser processes the definitions of the core DRR classes before it encounters any rules that use them.8

#### **Integration and Best Practices**

A robust dialect structure separates definitions into \*Dialect.td, \*Ops.td, and \*Patterns.td files.8 The C++ code generated from the

\*Patterns.td file (using the \-gen-rewriters backend) must be integrated into the dialect's C++ code. The standard mechanism is to implement the getCanonicalizationPatterns method on each operation that has patterns. This method adds the generated C++ RewritePattern classes to the RewritePatternSet provided by the canonicalizer pass driver.8

#### **The Future: A Note on PDLL**

While DRR is powerful, it has limitations, especially when expressing patterns for modern MLIR operations involving regions, variadic operands, or complex cross-operation constraints. The MLIR project has developed a more modern and powerful alternative called PDL (Pattern Description Language) and its user-facing syntax, PDLL.26 PDLL is itself an MLIR dialect for defining rewrite patterns and is better suited for these advanced use cases. For dialects that push the boundaries of ODS, an investigation into PDLL is a strategic recommendation for future development.8

## **Conclusion: A Synthesis of Best Practices for Dialect Maintenance**

The persistent build failures analyzed throughout this guide are not arbitrary bugs but direct consequences of violating the architectural principles of the MLIR framework. The root causes consistently trace back to an incompatibility between a dialect's declarative definitions and the modern MLIR C++ API, often exacerbated by misconfigured build systems or a misunderstanding of the strict contracts imposed by ODS features. This analysis has demonstrated that resolving these issues requires a holistic update to a dialect's definitions and build configuration, realigning them with the canonical patterns of the MLIR v20 ecosystem.

### **Recap of Core Principles**

The forensic deconstruction of these common failure modes illuminates a set of core principles that are essential for successful dialect development:

* **The Primacy of the Build System**: A granular, explicit CMake configuration using direct mlir\_tablegen calls is not a stylistic choice but the foundational requirement for a stable, scalable, and debuggable dialect. High-level abstractions like add\_mlir\_dialect should be avoided for anything beyond trivial, single-file dialects.  
* **Contracts are Inviolable**: ODS traits and interfaces impose strict, often implicit, contracts that must be fully understood and respected. As the AttrSizedOperandSegments case study demonstrates, these contracts govern both the declarative .td files and the imperative C++ code, and violations manifest as build failures that can be difficult to trace.  
* **Declarative Defines Structure, Imperative Defines Runtime**: A clear mental model of the boundary between TableGen and C++ is crucial. TableGen is for defining the static structure and interface of an IR component. C++ (in builders, custom parsers, verifiers, etc.) is for implementing the dynamic, runtime logic that cannot be expressed declaratively.  
* **Context is King**: The mlir-tblgen tool is not a monolithic entity but a collection of distinct backends, each operating in a specific context (e.g., for ops, attributes, or patterns). The recommended file structure of separating these definitions is a direct technical requirement to prevent context contamination and ensure each generator receives only the inputs it is designed to process.

### **Actionable Checklist for Dialect Development**

Synthesizing the lessons from this guide yields an actionable checklist of best practices for developing and maintaining a robust MLIR dialect 8:

* **Trust, but Verify the Generated Code**: When a C++ compiler fails inside a .inc file, the first and most effective debugging step is to inspect the generated code directly. Compare its structure to a known-good operation from a core MLIR dialect.  
* **Treat Traits as Invasive Contracts**: Before using any non-trivial trait, consult its documentation and, more importantly, find a canonical in-tree usage example. Assume it may impose requirements on arguments, attributes, or C++ implementation beyond simply adding its name to a list.  
* **Use In-Tree Dialects as Canonical Examples**: The LLVM/MLIR repository is the ultimate source of truth. Mimicking the tested usage patterns of core dialects (func, arith, scf, etc.) is often faster and more reliable than relying on documentation alone.  
* **Isolate and Minimize**: When a build fails after modifying a .td file, systematically comment out definitions and re-introduce them incrementally to pinpoint the exact feature causing the failure.  
* **Separate Concerns in .td Files**: Maintain distinct files for the core Dialect definition, OpDefs, AttrDefs, and Pattern defs. This is a technical requirement for the build system, not just a stylistic preference.  
* **Isolate C++ Logic from TableGen**: For complex builders or verifiers, implement the logic in a C++ function within a .cpp file. The TableGen C++ snippet should be a simple one-line call to this function, decoupling the imperative logic from the declarative definitions and making it easier to update when MLIR's C++ APIs change.

### **Final Word: Navigating a Living Ecosystem**

MLIR is a rapidly evolving framework.8 The principles and debugging methodologies presented in this guide—inspecting generated code, understanding toolchain mechanics, and learning from canonical examples—are the essential skills for successfully navigating this ecosystem. By adopting the modern builder patterns, correctly specifying dependencies, and explicitly conforming to the Properties system, a dialect can be placed on a stable and maintainable foundation, allowing development to confidently proceed on the higher-level goals of implementing novel compiler transformations and extending the frontiers of domain-specific computation.

#### **Referenzen**

1. MLIR Dialects in Catalyst \- PennyLane Documentation, Zugriff am August 22, 2025, [https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html](https://docs.pennylane.ai/projects/catalyst/en/stable/dev/dialects.html)  
2. MLIR Language Reference, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/LangRef/](https://mlir.llvm.org/docs/LangRef/)  
3. MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/](https://mlir.llvm.org/)  
4. MLIR — Defining a New Dialect \- Math ∩ Programming, Zugriff am August 22, 2025, [https://www.jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/](https://www.jeremykun.com/2023/08/21/mlir-defining-a-new-dialect/)  
5. Operation Definition Specification (ODS) \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)  
6. mlir/docs/DefiningDialects/Operations.md · e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e · llvm-doe / llvm-project \- GitLab, Zugriff am August 22, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e/mlir/docs/DefiningDialects/Operations.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e/mlir/docs/DefiningDialects/Operations.md)  
7. Chapter 2: Emitting Basic MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)  
8. source\_documents.pdf  
9. llvm-project/mlir/docs/DefiningDialects/Operations.md at main \- GitHub, Zugriff am August 22, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/DefiningDialects/Operations.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/DefiningDialects/Operations.md)  
10. MLIR Release Notes \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/ReleaseNotes/](https://mlir.llvm.org/docs/ReleaseNotes/)  
11. Creating a Dialect \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Tutorials/CreatingADialect/](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)  
12. Building LLVM with CMake — LLVM 22.0.0git documentation, Zugriff am August 22, 2025, [https://llvm.org/docs/CMake.html](https://llvm.org/docs/CMake.html)  
13. mlir/cmake/modules/AddMLIR.cmake \- toolchain/llvm-project \- Git at Google, Zugriff am August 22, 2025, [https://android.googlesource.com/toolchain/llvm-project/+/refs/heads/master-legacy/mlir/cmake/modules/AddMLIR.cmake](https://android.googlesource.com/toolchain/llvm-project/+/refs/heads/master-legacy/mlir/cmake/modules/AddMLIR.cmake)  
14. mlir/cmake/modules/AddMLIR.cmake \- llvm-project \- Git at Google \- Git repositories on llvm, Zugriff am August 22, 2025, [https://llvm.googlesource.com/llvm-project/+/refs/tags/llvmorg-14.0.0-rc1/mlir/cmake/modules/AddMLIR.cmake](https://llvm.googlesource.com/llvm-project/+/refs/tags/llvmorg-14.0.0-rc1/mlir/cmake/modules/AddMLIR.cmake)  
15. D85464 \[MLIR\] \[CMake\] Support building MLIR standalone \- LLVM Phabricator archive, Zugriff am August 22, 2025, [https://reviews.llvm.org/D85464](https://reviews.llvm.org/D85464)  
16. D76047 \[MLIR\] Add support for out of tree external projects using MLIR, Zugriff am August 22, 2025, [https://reviews.llvm.org/D76047?id=250393](https://reviews.llvm.org/D76047?id=250393)  
17. Traits \- MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Traits/](https://mlir.llvm.org/docs/Traits/)  
18. mlir::OpTrait::AttrSizedOperandSegments\< ConcreteType ... \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1OpTrait\_1\_1AttrSizedOperandSegments.html](https://mlir.llvm.org/doxygen/classmlir_1_1OpTrait_1_1AttrSizedOperandSegments.html)  
19. 'cf' Dialect \- MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/)  
20. Customizing Assembly Behavior \- MLIR \- LLVM, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/DefiningDialects/Assembly/](https://mlir.llvm.org/docs/DefiningDialects/Assembly/)  
21. 'func' Dialect \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Dialects/Func/](https://mlir.llvm.org/docs/Dialects/Func/)  
22. mlir/docs/AttributesAndTypes.md · main · undefined · GitLab, Zugriff am August 22, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/AttributesAndTypes.md](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/AttributesAndTypes.md)  
23. Table-driven Declarative Rewrite Rule (DRR) \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)  
24. mlir/docs/Tutorials/QuickstartRewrites.md · 5082acce4fd3646d5760c02b2c21d9cd2a1d7130 · llvm-doe / llvm-project \- GitLab, Zugriff am August 22, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Tutorials/QuickstartRewrites.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/5082acce4fd3646d5760c02b2c21d9cd2a1d7130/mlir/docs/Tutorials/QuickstartRewrites.md)  
25. Quickstart tutorial to adding MLIR graph rewrite, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/](https://mlir.llvm.org/docs/Tutorials/QuickstartRewrites/)  
26. PDLL \- PDL Language \- MLIR, Zugriff am August 22, 2025, [https://mlir.llvm.org/docs/PDLL/](https://mlir.llvm.org/docs/PDLL/)