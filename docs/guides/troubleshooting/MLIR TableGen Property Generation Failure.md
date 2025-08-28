

# **Resolving MLIR TableGen Property Generation Failures in Complex Projects**

### **Executive Summary: The Diagnosis of a TableGen Name Collision**

The root cause of the ‘arch’ does not name a type compilation error is a **name collision** within the TableGen processing environment. The identifier arch, intended to be the name of a property, is being incorrectly resolved by mlir-tblgen as a TableGen def record that also happens to be named arch. This record is being introduced from one of the .td files included during the processing of OrchestraOps.td. This name shadowing causes the code generator to emit the invalid C++ using archTy \= arch; instead of the correct using archTy \= ::llvm::StringRef;.

The immediate path to resolution involves identifying and renaming the conflicting def arch record within the project's TableGen files. As a matter of defensive programming, it is also best practice to always use fully-qualified C++ type names in Property definitions to mitigate such ambiguities. This report provides a detailed analysis of this failure mechanism, a corrected and robust definition for the orchestra.task operation, a canonical example demonstrating the use of various property types, and advanced strategies for debugging and defining MLIR operations with properties in version 20.1.8.

## **Chapter 1: The Mechanics of mlir-tblgen Property Codegen**

### **Introduction to ODS and the Role of mlir-tblgen**

The Multi-Level Intermediate Representation (MLIR) framework relies on an Operation Definition Specification (ODS) to manage the complexity of defining dialects, operations, attributes, and types.1 This specification is implemented using TableGen, a tool that translates declarative

.td description files into C++ boilerplate code.2 This table-driven approach automates the generation of accessor methods, builders, verifiers, and parser/printer logic, significantly reducing manual effort and ensuring consistency across the IR.4

The mlir-tblgen executable is a specialized backend for LLVM's generic tblgen tool. It is equipped with several "generators," such as OpDefinitionsGen, which are responsible for parsing the TableGen records defined in .td files and emitting the corresponding C++ header (.h.inc) and source (.cpp.inc) files.5

### **Deconstructing the Property Class in TableGen**

The Property class in ODS is a mechanism for declaring operation-specific data that is stored *inline* with the operation instance itself.6 This is a fundamental departure from

Attributes, which are uniqued within the global MLIRContext and are immutable.6 Properties are designed for data that is frequently accessed, may be simple (like integers or booleans), and does not benefit from context-wide uniquing. This design avoids the performance overhead of hash-consing and context lookups, making it ideal for performance-critical metadata.

The syntax used to declare a property within an operation's arguments block is parsed directly by the OpDefinitionsGen backend:

Property\<"name", "cpp\_type"\>:$arg\_name

This declaration provides three critical pieces of information to the code generator:

1. **Property Key:** The string literal "name" serves as the key in the operation's property dictionary.  
2. **C++ Storage Type:** The string literal "cpp\_type" specifies the exact C++ type that will be used for storage within the operation.  
3. **Argument/Getter Name:** The identifier :$arg\_name defines the name of the C++ argument in builders and the base name for the generated getter method (e.g., getArg\_name()).

### **The Code Generation Pathway**

When mlir-tblgen processes an operation containing a Property definition, it generates several C++ components. The component that is failing in this case is a using alias, which is intended to provide a convenient typedef for the property's storage type. The generator is designed to emit a C++ statement of the form:

using \<arg\_name\>Ty \= \<resolved\_cpp\_type\>;

The core of the issue lies in how \<resolved\_cpp\_type\> is derived from the "cpp\_type" string provided in the .td file. The generator does not simply treat this as an opaque string to be copied into the output. Instead, it performs a resolution step within the context of all known TableGen records.

The successful execution of the minimal, standalone test case demonstrates the intended behavior: with no other conflicting records present, mlir-tblgen correctly interprets "llvm::StringRef" as a C++ type string and emits it verbatim into the using alias. The failure within the larger "Orchestra" project, despite using the identical Property syntax, proves that the processing *environment* is the differentiating factor. In the context of TableGen, this environment is the complete set of all records loaded from all .td files and their transitive includes.

## **Chapter 2: Root Cause Analysis: Identifier Shadowing in TableGen's Global Record Scope**

### **TableGen's Global Namespace**

When mlir-tblgen is invoked, it first parses all specified .td files and recursively processes their include directives. All def and class records from these files are loaded into a single, flat, global collection of records.8 Unlike C++, TableGen does not have a robust namespacing system for

def records. A record defined as def MyRecord in one file is globally visible by that name to all other files processed in the same mlir-tblgen invocation. While this enables powerful cross-file composition, it is a frequent source of subtle and difficult-to-diagnose name collision bugs in large projects that aggregate multiple dialects or shared utility definitions.10

### **The Collision and Shadowing Mechanism**

The evidence strongly suggests that the "Orchestra" project includes a .td file, either directly or indirectly, that contains a TableGen definition named arch. This could be an EnumAttr definition, a Trait, or any other record type, for example: def arch : I32EnumAttr\<...\>;.

This leads to the following failure sequence during code generation:

1. mlir-tblgen begins execution, parsing OrchestraOps.td and all its included files. In this process, it loads a record defined as def arch into its global record map.  
2. The generator proceeds to parse the Orchestra\_TaskOp definition and encounters the argument Property\<"arch", "::llvm::StringRef"\>:$arch.  
3. When generating the C++ code for this property, the OpDefinitionsGen backend needs to determine the C++ type for the archTy alias.  
4. The name resolution logic in mlir-tblgen version 20.1.8 appears to have a flawed precedence rule. It checks if a TableGen record exists with the same name as the property's argument name (arch).  
5. It finds the def arch record from Step 1 and incorrectly concludes that this record *is* the type definition. It therefore uses the name of the TableGen record, arch, as the C++ type name.  
6. This def arch record effectively "shadows" the correct C++ type string, "::llvm::StringRef", that was explicitly provided as the second template argument to the Property class. The generator subsequently emits the erroneous C++ code.

### **Visualizing the Failure**

The discrepancy in the generated C++ code between the minimal test case and the full project build provides incontrovertible evidence of this name resolution failure. The following table illustrates the difference.

**Table 1: Comparison of Generated C++ Code (Minimal vs. Project Context)**

| Context | TableGen Definition | Generated C++ Code (in ...Ops.h.inc) | Analysis |
| :---- | :---- | :---- | :---- |
| Minimal Standalone | Property\<"arch", "::llvm::StringRef"\>:$arch | using archTy \= ::llvm::StringRef; | **Correct:** The C++ type string is correctly propagated. |
| "Orchestra" Project | Property\<"arch", "::llvm::StringRef"\>:$arch | using archTy \= arch; | **Incorrect:** The property name arch has been substituted as the C++ type, indicating resolution to a TableGen def arch. |

## **Chapter 3: The Canonical Solution for orchestra.task**

### **Step 1: The Direct Fix \- Auditing and Renaming**

The most direct solution is to eliminate the name collision. This requires finding the TableGen record def arch within the project's include paths and renaming it to something more specific and less likely to collide.

A command such as the following can be used to locate the offending definition within the project's source tree:

Bash

grep \-r "def arch" /path/to/orchestra-compiler/include/

Once located, the definition should be renamed to be more descriptive and unique. For example, if it defines an enumeration of architectures, a name like def Orchestra\_ArchEnum or def TargetArchEnum would be a significant improvement and would resolve the collision.

### **Step 2: The Defensive Fix \- Robust Property Definition**

After resolving the underlying name collision, the Orchestra\_TaskOp definition can be implemented. The original definition was syntactically correct; the failure was caused by the external environmental factor. The following definition represents the robust and recommended syntax.

Code-Snippet

def Orchestra\_TaskOp : Orchestra\_Op\<"task",\> {  
  let summary \= "Represents a schedulable task with a target architecture specifier.";

  let arguments \= (ins  
    // BEST PRACTICE: Always use fully-qualified C++ namespaces to avoid  
    // ambiguity with TableGen records.  
    Property\<"arch", "::llvm::StringRef"\>:$arch,  
    Property\<"device\_id", "int32\_t"\>:$device\_id,  
    Property\<"target\_props", "::mlir::DictionaryAttr"\>:$target\_props  
  );

  let results \= (outs Variadic\<AnyType\>:$results);  
  let regions \= (region SizedRegion:$body);  
  let hasVerifier \= 1;  
}

It is critical to emphasize that this definition will only compile successfully once the def arch name collision within the project's included .td files is eliminated. Using fully-qualified C++ types like ::llvm::StringRef and ::mlir::DictionaryAttr is a crucial defensive practice that minimizes ambiguity for the TableGen parser, though it was not sufficient to overcome the specific name resolution behavior in this case.

## **Chapter 4: A Definitive, Version-Specific Example: my\_dialect.compute\_kernel**

To provide a canonical, working reference for MLIR version 20.x, this section presents a complete TableGen definition for a hypothetical my\_dialect.compute\_kernel operation. This example demonstrates the correct syntax for string, integer, and dictionary properties, and clarifies the idiomatic approach to handling optionality.

### **Addressing Optional Properties**

The investigation into OptionalProperty and OptionalProp correctly concluded that these are not the standard mechanisms for defining optional properties in MLIR. The absence of these constructs in the core MLIR dialects and tests indicates that optionality is handled at the C++ type system level. This approach offers greater flexibility and aligns with the design of properties as a bridge to native C++ data structures.

* For mlir::Attribute subclasses like ::mlir::DictionaryAttr, optionality is inherent. A default-constructed DictionaryAttr is null. The generated C++ getter will return this null attribute, which evaluates to false in a boolean context, signaling its absence.  
* For native types like int32\_t, the C++ type itself can be changed to std::optional\<int32\_t\> to represent optionality. This requires ensuring that the necessary headers are available and that the property's storage and access logic can handle the std::optional wrapper.

### **The Canonical Example**

This example is structured across two files, MyDialect.td and MyDialectOps.td, which is a common practice for organizing dialects.

Code-Snippet

// \=== In file: MyDialect/MyDialect.td \===  
include "mlir/IR/OpBase.td"

def MyDialect\_Dialect : Dialect {  
  let name \= "my\_dialect";  
  let cppNamespace \= "::mlir::my\_dialect";  
  // This is the critical flag to enable the properties feature for this dialect.  
  let usePropertiesForAttributes \= 1;  
}

// Base class for ops in this dialect for consistency.  
class MyDialect\_Op\<string mnemonic, list\<Trait\> traits \=\> :  
    Op\<MyDialect\_Dialect, mnemonic, traits\>;

// \=== In file: MyDialect/MyDialectOps.td \===  
include "MyDialect/MyDialect.td"

def MyDialect\_ComputeKernelOp : MyDialect\_Op\<"compute\_kernel",\> {  
  let summary \= "A canonical example of an op with properties.";  
  let description \=;

  let arguments \= (ins  
    // 1\. A required string property.  
    // The C++ getter \`getKernelName()\` will return a \`::llvm::StringRef\`.  
    Property\<"kernel\_name", "::llvm::StringRef"\>:$kernel\_name,

    // 2\. A required 32-bit integer property.  
    // The C++ getter \`getCacheSizeKb()\` will return an \`int32\_t\`.  
    Property\<"cache\_size\_kb", "int32\_t"\>:$cache\_size\_kb,

    // 3\. An "optional" DictionaryAttr property.  
    // Optionality is handled by the C++ type. A default-constructed  
    // DictionaryAttr is null and represents absence. The getter \`getTuningParams()\`  
    // returns a \`::mlir::DictionaryAttr\` which can be checked for null.  
    // The default builder will require this argument, but it can be passed \`nullptr\`.  
    Property\<"tuning\_params", "::mlir::DictionaryAttr"\>:$tuning\_params  
  );

  let results \= (outs AnyType:$result);

  // To create a truly optional argument in the C++ builder API, we must  
  // define custom builders using \`extraClassDeclaration\`.  
  let extraClassDeclaration \=;  
}

The corresponding .cpp file would then implement the custom builder to delegate to the full builder, passing nullptr for the optional tuning\_params. This advanced technique is the standard way to create clean, user-friendly C++ APIs for operations with optional arguments.

## **Chapter 5: Advanced Considerations and Debugging Strategies**

### **Properties vs. Attributes: A Deeper Dive**

The decision to use a Property versus an Attribute is a key architectural choice when designing an MLIR dialect.

* **Performance:** Properties offer higher performance for simple, non-shared data by avoiding the overhead of MLIRContext lookups, hashing, and allocation. They store data directly within the Operation object.6  
* **Memory:** Attributes can be more memory-efficient if the same value is shared across many operations (e.g., a StringAttr for "gpu"). The MLIRContext ensures only one copy of that attribute's storage exists, and all operations reference it. Properties would result in a separate copy of the data in each operation.  
* **Mutability:** A key advantage of properties is that their C++ storage object can be designed to be mutable in-place, whereas attributes are strictly immutable constants by design.6

### **Bugs and Feature Interactions in the Property System**

The name resolution shadowing identified in Chapter 2 is a classic example of a subtle tooling bug that manifests only in complex environments. The property system, being a relatively newer feature in MLIR, has seen evolution and bug fixes. For instance, later versions of MLIR include fixes for parsing optional keys in a prop-dict assembly format, which validates the observation that the property subsystem is complex and has had corner cases that needed refinement.12

OpInterface definitions do not directly cause this type of TableGen name resolution failure. An interface defines a C++ contract. If an operation claims to implement an interface but its TableGen definition is flawed (e.g., due to this name collision), the compilation failure will occur when the C++ implementation of the operation attempts to satisfy the interface's method requirements and cannot find the correctly-typed getter. The error would be a C++ type mismatch or missing method, not a code generation failure in the .h.inc file.

### **Expert-Level Debugging with llvm-tblgen**

To diagnose these types of issues, it is essential to inspect the state of the TableGen records as seen by the code generator. The llvm-tblgen tool (a superset of mlir-tblgen) provides flags for this purpose.

The most powerful debugging tool is the \--print-records flag. By running llvm-tblgen with this flag and the same include paths used by the build system, one can see the complete, resolved set of all records.

Bash

llvm-tblgen \--print-records \-I /path/to/orchestra/include \-I /path/to/llvm/mlir/include OrchestraOps.td \> records.txt

The output file, records.txt, can then be searched for def arch to confirm the existence and definition of the conflicting record. It can also be inspected for the final state of the Orchestra\_TaskOp record to see how TableGen has interpreted its arguments. This method provides a definitive view of the input to the C++ code generator, bypassing any assumptions about how the .td files are being processed.8

### **Final Best Practices Summary**

The following table summarizes the key recommendations for defining and debugging MLIR properties, particularly when working with a fixed version like 20.1.8.

**Table 2: Best Practices for MLIR Property Definitions (v20.1.8)**

| Guideline | Rationale & Elaboration |
| :---- | :---- |
| **Avoid Name Collisions** | Before adding a property named foo, search your project's .td files for def foo. Rename any conflicting defs to be more specific (e.g., MyDialect\_FooEnum). TableGen's global namespace for defs is a common source of errors. |
| **Use Fully-Qualified C++ Types** | Always specify C++ types with their full namespace (e.g., ::llvm::StringRef, ::mlir::DictionaryAttr). This makes the type string unambiguous to the TableGen parser and is a robust defensive measure. |
| **Handle Optionality in C++** | Do not use non-existent classes like OptionalProperty. For properties that can be absent, use C++ types that have a "null" or "empty" state, such as ::mlir::Attribute subclasses (which can be null) or std::optional\<T\>. |
| **Define Clean Builders** | For properties that are logically optional, use extraClassDeclaration to provide overloaded build methods that supply default values (e.g., nullptr for a DictionaryAttr), creating a more ergonomic C++ API. |
| **Know Your Debugging Tools** | Use llvm-tblgen \--print-records to inspect the "final" state of all TableGen records as seen by the code generator. This is the definitive way to diagnose scope and resolution issues. |
| **Choose Property vs. Attribute Wisely** | Use Property for simple, frequently-accessed, op-local data where performance is key. Use Attribute for complex or widely-shared constant data to leverage context uniquing for memory savings. |

#### **Referenzen**

1. Operation Definition Specification (ODS) \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)  
2. Table-driven Declarative Rewrite Rule (DRR) \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/DeclarativeRewrites/](https://mlir.llvm.org/docs/DeclarativeRewrites/)  
3. Operation Definition Syntax (ODS) provides a concise way of defining... \- ResearchGate, Zugriff am August 26, 2025, [https://www.researchgate.net/figure/Operation-Definition-Syntax-ODS-provides-a-concise-way-of-defining-new-Ops-in-MLIR\_fig4\_339497737](https://www.researchgate.net/figure/Operation-Definition-Syntax-ODS-provides-a-concise-way-of-defining-new-Ops-in-MLIR_fig4_339497737)  
4. MLIR — Using Tablegen for Passes \- Math ∩ Programming, Zugriff am August 26, 2025, [https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/](https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)  
5. llvm-project/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp at main \- GitHub, Zugriff am August 26, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp](https://github.com/llvm/llvm-project/blob/main/mlir/tools/mlir-tblgen/OpDefinitionsGen.cpp)  
6. D141742 Introduce MLIR Op Properties \- LLVM Phabricator archive, Zugriff am August 26, 2025, [https://reviews.llvm.org/D141742](https://reviews.llvm.org/D141742)  
7. Defining Dialect Attributes and Types \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)  
8. tblgen \- Description to C++ Code — LLVM 22.0.0git documentation, Zugriff am August 26, 2025, [https://llvm.org/docs/CommandGuide/tblgen.html](https://llvm.org/docs/CommandGuide/tblgen.html)  
9. mlir/test/mlir-tblgen/rewriter-static-matcher.td · llvmorg-16.0.3 \- Gricad-gitlab, Zugriff am August 26, 2025, [https://www.gricad-gitlab.univ-grenoble-alpes.fr/tracing-llvm/llvm/-/blob/llvmorg-16.0.3/mlir/test/mlir-tblgen/rewriter-static-matcher.td](https://www.gricad-gitlab.univ-grenoble-alpes.fr/tracing-llvm/llvm/-/blob/llvmorg-16.0.3/mlir/test/mlir-tblgen/rewriter-static-matcher.td)  
10. mlir::tblgen Namespace Reference \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/doxygen/namespacemlir\_1\_1tblgen.html](https://mlir.llvm.org/doxygen/namespacemlir_1_1tblgen.html)  
11. Avoiding name collisions in QML \- Stack Overflow, Zugriff am August 26, 2025, [https://stackoverflow.com/questions/38991144/avoiding-name-collisions-in-qml](https://stackoverflow.com/questions/38991144/avoiding-name-collisions-in-qml)  
12. \[Mlir-commits\] \[mlir\] \[mlir\]\[tblgen\] Fix bug around parsing optional prop-dict keys (PR \#120045) \- Mailing Lists, Zugriff am August 26, 2025, [https://lists.llvm.org/pipermail/mlir-commits/2024-December/085551.html](https://lists.llvm.org/pipermail/mlir-commits/2024-December/085551.html)  
13. tblgen \- Description to C++ Code \- Ubuntu Manpage, Zugriff am August 26, 2025, [https://manpages.ubuntu.com/manpages/noble/man1/tblgen-19.1.html](https://manpages.ubuntu.com/manpages/noble/man1/tblgen-19.1.html)