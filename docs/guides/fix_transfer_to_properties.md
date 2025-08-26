

# **A Definitive Analysis of MLIR Operation Properties: Resolving the SymbolRefAttr Syntax in LLVM 20.1.8**

## **Section 1: The Architectural Distinction Between MLIR Attributes and Properties**

To resolve the impasse encountered within the Orchestra project concerning the migration of SymbolRefAttr to the MLIR Properties system, it is imperative to first establish a rigorous understanding of the fundamental architectural differences between the canonical attribute system and the more recent properties system. These two mechanisms, while superficially similar in that they both attach constant data to operations, are predicated on divergent design philosophies with significant implications for performance, memory layout, and developer ergonomics. The challenges faced in the Orchestra codebase stem directly from these underlying distinctions.

### **1.1 The Canonical MLIR Attribute System: Context, Uniqueness, and Immutability**

The Multi-Level Intermediate Representation (MLIR) is a highly extensible compiler infrastructure, designed to support a diverse ecosystem of dialects, operations, types, and attributes.1 Within this framework, attributes serve as the primary mechanism for specifying constant, compile-time data on operations.3 Examples range from the predicate of a comparison operation (

arith.cmpi) to the value of a constant (arith.constant) or the symbolic name of a function.4

The defining characteristic of the standard MLIR attribute system is its reliance on the MLIRContext. Every mlir::Attribute object is, in essence, a value-typed wrapper around an underlying storage object that is "uniqued" and owned by the central MLIRContext instance.3 This design has several profound consequences:

* **Memory Efficiency through Uniquing**: By ensuring that any two semantically identical attributes (e.g., two instances of the 64-bit integer attribute for the value 42\) point to the exact same storage object in memory, the context prevents widespread data duplication across the entire IR. This is particularly beneficial in large modules with repetitive metadata.  
* **Efficient Equality Comparison**: The uniquing mechanism allows for extremely fast equality checks. Two attributes are identical if and only if their underlying storage pointers are the same, reducing comparison to a simple pointer check.  
* **Immutability**: Once created and uniqued within the context, attribute storage is immutable. Any modification requires creating a new attribute, which then undergoes the uniquing process again.  
* **Performance Overhead**: The primary drawback of this architecture, and the principal motivation for the introduction of the Properties system, is the performance cost associated with uniquing. To create an attribute, MLIR must perform a lookup within a context-wide hash table, a process that may require thread synchronization (locking) in a multi-threaded compilation environment. For operations that have many inherent, instance-specific attributes, this repeated interaction with the MLIRContext can become a significant performance bottleneck.6

Structurally, an mlir::Operation stores its attributes in a single DictionaryAttr.7 This dictionary contains both

*inherent* attributes, which are core to the operation's semantics (e.g., from and to in orchestra.transfer), and *discardable* attributes, which provide external metadata (e.g., for a specific analysis pass).4 Accessing an inherent attribute thus involves a string-based lookup into this dictionary, which is less efficient and less type-safe than direct member access.

### **1.2 The Properties System: A Paradigm Shift to Inline, Mutable Storage**

The MLIR Operation Properties system was introduced as a direct architectural solution to the performance and ergonomic limitations of the context-uniqued attribute system for inherent operational data. An analysis of the original implementation review, D141742, reveals its core design principle: to provide "custom storage inline within operations".6

Unlike attributes, properties are not managed by the MLIRContext. Instead, their storage is allocated directly alongside the mlir::Operation object itself. This fundamental shift provides several key advantages:

* **Performance**: By storing data inline, the system completely bypasses the MLIRContext for creation, access, and destruction. This eliminates the overhead of hash table lookups and potential locking, leading to significantly faster instantiation of operations with complex inherent data.  
* **Stronger Typing and Ergonomics**: The Operation Definition Specification (ODS) framework generates a dedicated C++ Properties struct nested within the operation's class. This struct contains strongly-typed member variables corresponding to the defined properties. Consequently, developers can use type-safe getter methods (e.g., getFrom()) instead of error-prone, string-based dictionary lookups (getAttr("from")).  
* **Mutability**: The properties system fundamentally breaks from the immutability of attributes. Because the storage is local to the operation instance, it can be designed to be mutable in-place. The introductory documentation for the feature explicitly showcases this capability with an example of a std::vector stored as a property that can be modified without creating a new operation.6

This paradigm shift redefines the memory layout of an operation, moving what was once external, context-managed data into the core object representation, thereby optimizing for the common case of accessing inherent, operation-specific configuration.

### **1.3 A Comparative Framework: When to Use Attributes vs. Properties**

The choice between using a standard attribute or a property is a critical design decision in dialect development. It is not merely a syntactic preference but a reflection of the data's semantic role and performance characteristics. Attributes are the ideal mechanism for representing *shared, constant metadata*, especially values that are likely to be duplicated across many operations. Their context-wide uniquing provides memory savings and fast comparisons for this use case.

In contrast, properties are engineered for *instance-specific, inherent configuration*. This is data that is conceptually part of the operation's definition and is unlikely to be shared with other operations. The padding and stride values of a convolution, the bounds of a loop, or the symbolic references of a data transfer operation are all prime candidates for property-based storage. The performance benefits of avoiding the MLIRContext are most pronounced for this type of data.

The following table provides a concise architectural comparison to guide design decisions within the Orchestra dialect and other MLIR-based projects.

| Feature | mlir::Attribute | mlir::OpProperty |
| :---- | :---- | :---- |
| **Storage Location** | Uniqued within the global MLIRContext 3 | Allocated inline with the mlir::Operation object 6 |
| **Lifecycle** | Immortal; tied to the MLIRContext lifetime | Tied to the parent Operation's lifetime |
| **Mutability** | Strictly immutable | Can be designed to be mutable in-place |
| **Performance (Creation)** | Higher overhead due to context-wide lookup and potential locking | Lower overhead; part of the Operation's single allocation |
| **Performance (Access)** | Indirect via DictionaryAttr string lookup 7 | Direct via generated C++ member accessors |
| **Typical Use Case** | Shared constants, discardable metadata, linkage types | Inherent, operation-specific configuration data |
| **TableGen Syntax** | TypeAttr:$name | IntProp\<... \>:$name or implicit via dialect flag 6 |

The SymbolRefAttr presents a unique case. While it refers to a shared symbol, its role as the source or destination of a transfer is inherent to that specific orchestra.transfer operation instance. This makes it a perfect candidate for the performance benefits of property-based storage, which explains the motivation behind the assigned task.

## **Section 2: A Systematic Deconstruction of Erroneous TableGen Syntaxes**

The detailed log of failed attempts provides a valuable diagnostic trail. Each error message from mlir-tblgen is not arbitrary but a precise signal indicating a specific misunderstanding of the Operation Definition Specification (ODS) grammar and the architecture of the Properties system as it existed in the LLVM 16-20 release cycles. The Properties system was formally introduced in the LLVM 16.0 release in March 2023, following its commit in January 2023\.6 The project's use of LLVM 20.1.8 places it in a timeframe where this feature is established but its more nuanced aspects are not as universally documented as core MLIR concepts.

### **2.1 Analyzing the Failures: A Methodical Approach**

A systematic analysis of the failed attempts reveals a logical but ultimately incorrect assumption. The successful use of IntProp in Orchestra\_SelectOp induced a mental model where a complete, parallel hierarchy of \*Prop classes exists for every corresponding \*Attr class. This model predicts that if IntegerAttr has an IntProp counterpart, then SymbolRefAttr must have a SymbolRefProp counterpart, StringAttr must have StrProp, and so on.

This assumption is flawed. The MLIR developers provided explicit \*Prop classes only for common, primitive C++ types (int, float, bool, string) that can be stored directly. For migrating complex, pre-existing mlir::Attribute subclasses, they chose a different, more scalable, and non-invasive mechanism that avoids the need to create and maintain a vast parallel class hierarchy in TableGen. The errors encountered are mlir-tblgen systematically rejecting this flawed "parallel hierarchy" model.

### **2.2 Deconstruction of Failed Attempts**

A detailed breakdown of each attempt confirms this analysis.

* **Attempt 1: properties Block**  
  * **Syntax**: let properties \= \[ Property\<...\>,... \];  
  * **Error**: error: Value 'properties' unknown\!  
  * **Explanation**: This error indicates that properties is not a recognized keyword for a top-level let binding within an Op definition in TableGen. The ODS grammar specifies that all arguments to an operation—be they operands, attributes, or properties—must be defined within the arguments DAG structure.10 The system is designed to interleave these different kinds of arguments, not to segregate them into separate blocks.  
* **Attempt 2: SymbolRefProp**  
  * **Syntax**: SymbolRefProp:$from  
  * **Error**: error: Variable not defined: 'SymbolRefProp'  
  * **Explanation**: This is the most direct refutation of the "parallel hierarchy" assumption. The mlir-tblgen parser is stating unequivocally that it has no definition for a class or def named SymbolRefProp. A search of the core MLIR TableGen files (such as OpBase.td and the then-new Properties.td) for this version confirms that no such class was ever defined.  
* **Attempts 3 & 4: AttrProp and TypedProperty**  
  * **Syntax**: AttrProp\<SymbolRefAttr\>:$from and TypedProperty\<SymbolRefAttr\>:$from  
  * **Error**: error: Expected a class name, got 'AttrProp'  
  * **Explanation**: This error is slightly more subtle. It suggests that the parser recognizes the \<...\> syntax as a template specialization but does not consider AttrProp or TypedProperty to be valid templated classes in this context. These are logical but speculative names for a generic property wrapper for attributes. The ODS system does not provide such a generic, user-facing TableGen class. The mechanism for handling attributes as properties is, as will be shown, implicit and controlled elsewhere.  
* **Attempt 5: StrProp**  
  * **Syntax**: StrProp:$from  
  * **Error**: error: Variable not defined: 'StrProp'  
  * **Explanation**: This is identical in nature to the SymbolRefProp failure. While SymbolRefAttr internally uses a StringAttr for the symbol's name, the user is attempting to find a property type for the symbol reference itself. Even if they had tried to model it as a string, StrProp is not a standard, built-in property type in the same way IntProp is. Properties for standard C++ strings are typically handled by Data properties with a C++ type specified.  
* **Attempt 6: Adding include "mlir/IR/Properties.td"**  
  * **Action**: Added include "mlir/IR/Properties.td" to OrchestraOps.td.  
  * **Result**: The errors remained identical.  
  * **Explanation**: This action was necessary but not sufficient. Including Properties.td is the correct way to make the TableGen backend aware of the properties system and to import the definitions for the base Property class and the primitive types like IntProp. This is why Orchestra\_SelectOp's IntProp works. However, this file does not contain definitions for SymbolRefProp, AttrProp, or StrProp. Therefore, including it does not resolve the "Variable not defined" errors, as the user was still attempting to use classes that do not exist.

## **Section 3: The Canonical Solution: Implicit Property Storage via Dialect Configuration**

The correct method for converting an existing mlir::Attribute, such as SymbolRefAttr, to use the properties storage mechanism does not involve a new or different syntax at the operation definition level. Instead, it relies on a dialect-wide configuration flag that instructs the ODS code generator to alter the storage strategy for all inherent attributes within that dialect.

### **3.1 Explicit vs. Implicit Property Declaration**

Understanding the solution requires differentiating between two modes of property definition in TableGen:

1. **Explicit Definition**: This is used for defining new, simple properties that store basic C++ data types. It employs the \*Prop classes seen in the Orchestra\_SelectOp example (e.g., IntProp, FloatProp, BoolProp). These classes are designed for data that does not have a complex, pre-existing mlir::Attribute class counterpart. They provide a direct mapping from a TableGen declaration to a C++ member in the Properties struct.  
2. **Implicit Conversion**: This is the mechanism designed specifically for migrating existing, complex mlir::Attribute types to property-based storage. It is "implicit" because the syntax in the operation's arguments block remains unchanged (e.g., SymbolRefAttr:$from). The conversion to a property is triggered by a configuration setting at a higher level—the dialect itself. This design allows dialects to opt-in to the performance benefits of properties for their inherent attributes without rewriting every single operation definition.

### **3.2 The usePropertiesForAttributes Dialect Flag**

The key to solving the problem is the usePropertiesForAttributes flag. This is a boolean property that can be set on a Dialect definition in TableGen.

The Phabricator review D141742, which introduced the properties system, provides the definitive specification for this feature:

"A new options is introduced to ODS to allow dialects to specify: let usePropertiesForAttributes \= 1; When set to true, the inherent attributes for all the ops in this dialect will be using properties instead of being stored alongside discardable attributes." 6

This statement is unambiguous. To convert SymbolRefAttr (and all other inherent attributes) in the Orchestra dialect to use property storage, the solution is not to modify the Orchestra\_TransferOp definition, but to enable this flag on the Orchestra\_Dialect definition.

### **3.3 Rationale: Why SymbolRefAttr Requires an Implicit Mechanism**

The choice of an implicit, dialect-wide flag over an explicit SymbolRefProp class was a deliberate and architecturally sound design decision by the MLIR developers. A SymbolRefAttr is not merely a container for a string; it is a complex IR object with deep semantic ties to the SymbolTable infrastructure.12

* **Preservation of Semantics and API**: A SymbolRefAttr carries with it a host of associated logic for symbol resolution, which involves traversing the IR hierarchy to find the nearest parent operation with the SymbolTable trait.12 Furthermore, a rich C++ API exists for manipulating symbol uses (e.g.,  
  SymbolTable::replaceAllSymbolUses, SymbolTable::getSymbolUses).12 Creating a new  
  SymbolRefProp class would have been problematic. It would either need to duplicate all of this complex logic, leading to code duplication and maintenance burdens, or it would create a confusing dual system where the TableGen type (SymbolRefProp) differs from the underlying C++ type (mlir::SymbolRefAttr).  
* **Non-Invasive Optimization**: The usePropertiesForAttributes flag provides a much cleaner solution. It upholds the principle of separation of concerns: the SymbolRefAttr class continues to define the *semantics* and *API* for symbol references, while the properties system is concerned only with the *storage*. By enabling the flag, the ODS generator is instructed to change only the storage backend. The generated C++ code for the operation will still use ::mlir::SymbolRefAttr as the data type for the from and to members, and all existing APIs will continue to work seamlessly. The only difference is that these members will now reside in the inline Properties struct instead of the operation's DictionaryAttr. This allows the Orchestra project to gain the performance benefits of inline storage without any disruptive or risky changes to the attribute's well-defined behavior.

## **Section 4: Implementation Guide and Verification**

The following steps provide a concrete, actionable path to resolve the build errors and successfully complete the task of migrating the from and to attributes of orchestra.transfer to the properties system.

### **4.1 Step 1: Include the Properties TableGen Definitions**

First, it is necessary to ensure that the TableGen definitions for the properties system are available to the ODS generator. This is a prerequisite for both explicit property definitions (like IntProp) and the implicit conversion mechanism.

In the primary TableGen file for the dialect, likely orchestra-compiler/include/Orchestra/OrchestraOps.td, verify that the following include directive is present, typically near the top of the file:

Code-Snippet

include "mlir/IR/Properties.td"

This line makes the Property base classes and related constructs visible to the TableGen processor, enabling it to understand and generate code for the properties system.

### **4.2 Step 2: Enable Properties for Inherent Attributes in the Dialect**

This is the critical step that implements the solution. Locate the TableGen definition for the Orchestra dialect itself. This is a def that inherits from the Dialect class. Modify this definition to include the usePropertiesForAttributes flag.

Code-Snippet

def Orchestra\_Dialect : Dialect {  
  let name \= "orchestra";  
  let cppNamespace \= "::mlir::orchestra";  
  let description \=;

  //... other dialect-level configurations...

  // Enable property-based storage for all inherent attributes in this dialect.  
  let usePropertiesForAttributes \= 1;  
}

By setting this flag to 1, mlir-tblgen is instructed that for every operation defined within the Orchestra\_Dialect, any argument that is a subclass of Attr (such as SymbolRefAttr, IntegerAttr, ArrayAttr, etc.) should be treated as a property and stored inline with the operation.

### **4.3 Step 3: Revert the Operation Definition to the Original Syntax**

With the dialect-level flag enabled, the operation definition for Orchestra\_TransferOp requires no special property-specific syntax. It should be reverted to the original, standard syntax for defining an attribute. The implicit mechanism handles the conversion automatically.

The correct and final definition in OrchestraOps.td should be:

Code-Snippet

def Orchestra\_TransferOp : Orchestra\_Op\<"transfer",...\> {  
  let summary \= "Transfers data between two symbolic locations.";  
  let description \=;

  let arguments \= (ins  
    AnyShaped:$source,  
    SymbolRefAttr:$from, // This will now be stored as a property.  
    SymbolRefAttr:$to,   // This will also be stored as a property.  
    //... other arguments...  
  );

  //... other operation fields (results, assemblyFormat, etc.)...  
}

This syntax, which previously defined a standard attribute stored in the DictionaryAttr, is now re-interpreted by the ODS generator (due to the dialect flag) to define a property. The C++ type remains ::mlir::SymbolRefAttr, but the storage location changes.

### **4.4 Verification**

After implementing the changes above, the success of the migration can be verified through two distinct methods.

1. **Primary Verification (Compilation)**: Execute the project's build command from the /app directory:  
   Bash  
   cmake \--build build

   The previous mlir-tblgen errors related to unknown variables and incorrect syntax will be resolved. The build process should complete successfully, indicating that the TableGen definitions are now valid. Subsequently, running the test suite should also pass:  
   Bash  
   cd /app/build  
   lit \-v./tests

2. **Secondary Verification (Code Inspection)**: For a more rigorous confirmation that the storage mechanism has indeed changed, inspect the auto-generated C++ header file for the Orchestra operations. This file is typically located at build/orchestra-compiler/include/Orchestra/OrchestraOps.h.inc.  
   Within this file, locate the generated class for Orchestra\_TransferOp. One should observe the following changes compared to the pre-migration version:  
   * A new nested struct Properties will be defined within the Orchestra\_TransferOp class.  
   * This Properties struct will contain member variables named from and to, both of type ::mlir::SymbolRefAttr.  
   * The generated getter methods, getFrom() and getTo(), will now be implemented to access these members from the Properties struct (e.g., via getProperties().from). They will no longer access the generic DictionaryAttr via getAttr("from").

Observing these changes in the generated code provides definitive proof that the SymbolRefAttr arguments have been successfully migrated to the inline property storage system.

## **Section 5: Synthesis and Strategic Recommendations**

The resolution of this issue hinges on a nuanced feature of MLIR's ODS that is powerful but not immediately obvious. By synthesizing this solution with the contextual clues provided in the project, we can form a complete picture of the situation and derive strategic recommendations for the Orchestra project.

### **5.1 Reconciling the Conflicting Evidence**

The initial impasse was exacerbated by seemingly contradictory pieces of information. The correct solution, however, reconciles all of them perfectly.

* **The Code Reviewer's Insistence**: The reviewer's assertion that a solution exists within the .td files was entirely correct. The critical detail was that the necessary change was not at the level of the Op definition, where all attempts were focused, but at the higher level of the Dialect definition. This is a subtle but crucial distinction that is easy to miss when debugging a specific operation.  
* **The status.md Note on DictionaryAttr**: The note stating, "The orchestra.task target attribute was implemented with a C++ verifier instead of the Properties system due to technical challenges with DictionaryAttr properties," is not a contradiction but rather a vital piece of corroborating evidence. It indicates that the project team was aware of the properties system but had encountered limitations with its more complex applications in their specific version of MLIR (20.1.8). DictionaryAttr, being a dynamic, sorted container of other attributes, presents a more complex mapping to a static C++ Properties struct than a simple, self-contained attribute like SymbolRefAttr. This note validates that the user's difficulties were rooted in a genuine area of complexity within this version of MLIR, confirming that the properties system was still maturing.  
* **The Working IntProp**: The successful use of IntProp\<"int32\_t"\> in Orchestra\_SelectOp was a "red herring." It correctly demonstrated the existence and basic syntax of the properties system but inadvertently established a misleading mental model. It showcased the *explicit* definition mechanism for primitive types, leading to the logical but incorrect assumption that a similar *explicit* syntax (SymbolRefProp) must exist for complex attribute types. The key missing concept was the separate, *implicit* mechanism designed for this exact purpose.

### **5.2 Strategic Recommendations for the Orchestra Project**

Based on this analysis, the following strategic recommendations can help the Orchestra project improve its development practices and prevent similar issues in the future.

* **Enhance Internal Documentation**: It is strongly recommended to update the project's architectural documentation (e.g., blueprint.md) to explicitly record the decision to use the usePropertiesForAttributes flag for the Orchestra dialect. This documentation should include:  
  * A clear statement that the flag is enabled.  
  * The rationale for this decision (i.e., performance benefits of inline storage for inherent attributes).  
  * A brief explanation of how it works, clarifying that standard attribute syntax in .td files is automatically converted to property storage, avoiding the need for special \*Prop classes for complex attributes.  
    This will serve as a crucial onboarding resource for new team members and a reference for existing ones, preventing them from repeating this investigation.  
* **Consult Primary Sources for New Features**: When adopting relatively new compiler infrastructure features, the original design documents, code reviews (like Phabricator or GitHub PRs), and commit messages are often the most authoritative sources of information. For the properties system, the Phabricator review D141742 was the source of truth that unambiguously described the usePropertiesForAttributes flag.6 Encouraging a team culture of seeking out these primary sources can accelerate problem-solving for advanced features.  
* **Address Technical Debt Proactively**: The note in status.md regarding DictionaryAttr properties should be converted into a formal technical debt item in the project's issue tracker. This item should be scheduled for re-evaluation during the next planned upgrade of the project's LLVM/MLIR dependency. Later versions of MLIR may have introduced improved support or different mechanisms for handling such cases. Proactively tracking these known limitations ensures they are not forgotten and can be resolved when the underlying infrastructure evolves.

In conclusion, the challenge of converting SymbolRefAttr to a property was not due to a missing or arcane syntax at the operation level, but rather the under-publicized, dialect-level configuration that enables this migration implicitly. By applying the usePropertiesForAttributes flag, the Orchestra project can achieve its performance goals while maintaining clean, standard TableGen syntax for its operation definitions, thereby resolving the immediate build failure and aligning with the intended design of the MLIR Properties system.

#### **Referenzen**

1. Chapter 2: Emitting Basic MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/](https://mlir.llvm.org/docs/Tutorials/Toy/Ch-2/)  
2. Introduction \- tt-mlir documentation, Zugriff am August 26, 2025, [https://docs.tenstorrent.com/tt-mlir/](https://docs.tenstorrent.com/tt-mlir/)  
3. Defining Dialect Attributes and Types \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)  
4. MLIR Language Reference, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/LangRef/](https://mlir.llvm.org/docs/LangRef/)  
5. llvm-project/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td at main \- GitHub, Zugriff am August 26, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td](https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Arith/IR/ArithOps.td)  
6. D141742 Introduce MLIR Op Properties \- LLVM Phabricator archive, Zugriff am August 26, 2025, [https://reviews.llvm.org/D141742](https://reviews.llvm.org/D141742)  
7. mlir::Operation Class Reference \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1Operation.html](https://mlir.llvm.org/doxygen/classmlir_1_1Operation.html)  
8. LLVM 16.0 Released With New Intel/AMD CPU Support, More C++20 / C2X Features, Zugriff am August 26, 2025, [https://www.phoronix.com/news/LLVM-16.0-Released](https://www.phoronix.com/news/LLVM-16.0-Released)  
9. What is new in LLVM 16 \- Tools, Software and IDEs blog \- Arm Community, Zugriff am August 26, 2025, [https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/whats-new-in-llvm-16](https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/whats-new-in-llvm-16)  
10. Operation Definition Specification (ODS) \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)  
11. llvm-project/mlir/docs/DefiningDialects/Operations.md at main \- GitHub, Zugriff am August 26, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/docs/DefiningDialects/Operations.md](https://github.com/llvm/llvm-project/blob/main/mlir/docs/DefiningDialects/Operations.md)  
12. Symbols and Symbol Tables \- MLIR \- LLVM, Zugriff am August 26, 2025, [https://mlir.llvm.org/docs/SymbolsAndSymbolTables/](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)