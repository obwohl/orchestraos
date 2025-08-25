

# **An Analysis of Systemic TableGen Failures in MLIR Dialect Modernization**

## **Executive Summary: A Diagnosis of Build System and TableGen Configuration Faults**

The persistent and varied build failures encountered during the modernization of the orchestra.task operation are not indicative of a fundamental bug within the MLIR Properties system for the specified LLVM development version (20.1.8). A thorough analysis of the failure patterns across multiple methodical attempts reveals that the root cause is a systemic misconfiguration within the project's CMake build system. This misconfiguration creates an incomplete and context-less environment for the mlir-tblgen tool, preventing it from recognizing advanced dialect features such as the Properties system.

The two primary failures—the inability to parse the properties keyword and the subsequent parser corruption triggered by an AttrConstraint definition—are distinct symptoms of the same underlying pathology. The build system is failing to correctly orchestrate the flow of information and dependencies between the dialect's primary definition (OrchestraDialect.td) and its constituent operation definitions (OrchestraOps.td). Specifically, the mlir-tblgen invocations responsible for processing the operation definitions are being executed without the necessary dialect-level context, include paths, or specialized generation rules.

This report provides a multi-pathed strategy for resolution. The primary and recommended path involves rectifying the CMake configuration to enable the Properties system as intended, which represents the most idiomatic and maintainable long-term solution. As a robust and powerful alternative, a detailed guide for implementing a custom C++ attribute is presented. This "escape hatch" offers a solution that achieves the desired type safety and structured verification while being significantly less dependent on the intricacies of the TableGen build process, thereby de-risking the modernization effort against any further environmental brittleness.

## **Deconstruction of the Properties System Failure**

A forensic analysis of the chronological attempts to integrate the MLIR Properties system reveals a clear pattern of failures. These failures, while manifesting with different error messages, collectively point not to a flaw in the Properties system itself, but to a foundational issue in how the build environment invokes the TableGen tooling.

### **Hypothesis 1: Ineffective usePropertiesForAttributes Flag due to Build Invocation Context**

The initial attempt to use the modern properties block failed with Value 'properties' unknown\!. This error immediately suggests that the dialect-level setting to enable this feature, let usePropertiesForAttributes \= 1;, was not in effect during the parsing of the operation's definition. This setting, defined in OrchestraDialect.td, signals to the ODS (Operation Definition Specification) backend of mlir-tblgen that operations within this dialect can use the properties field.1 The Properties system itself was introduced as a modern alternative to inherent attributes, storing data inline with the operation rather than in the uniqued

MLIRContext, and became the default behavior in LLVM 17\.3

The core of the issue lies in the nature of mlir-tblgen as a command-line executable. It is a stateless tool that processes only the .td files provided to it in a single invocation.5 The common and recommended practice of separating the dialect definition from its operation, attribute, and type definitions into different files is intended to improve modularity and layering.7 However, this practice places the onus on the build system—in this case, CMake—to ensure that all necessary files are part of the same processing context.

The persistence of the Value 'properties' unknown\! error, even after correcting the internal include directive in Attempt 2, strongly indicates that the build system's invocation is the source of the problem. The CMake configuration is almost certainly generating an mlir-tblgen command that processes OrchestraOps.td in isolation, without the context of OrchestraDialect.td where usePropertiesForAttributes is defined. The flag is not incorrect; it is simply not in scope when the Orchestra\_TaskOp is being parsed.

### **Hypothesis 2: Incomplete TableGen Include Path and Dependency Resolution**

The most compelling evidence for a systemic build configuration issue is the failure of Attempt 3, which tried to use the older, interleaved property syntax (StrProp, IntProp). The resulting error, Expected a class name, got 'StrProp', is highly diagnostic. StrProp is not a custom definition; it is a fundamental class provided by MLIR's core TableGen libraries, typically made available via include "mlir/IR/OpBase.td".9 For

mlir-tblgen to recognize this class, two conditions must be met: the .td file must contain the correct include statement, and the mlir-tblgen executable must be launched with the correct include paths (-I flags) to locate the file specified in that statement.

The failure to resolve StrProp implies that the mlir-tblgen process is missing the critical \-I flag that points to the LLVM/MLIR installation's include directory. Without this, it cannot find mlir/IR/OpBase.td or any other standard MLIR TableGen definitions. This explains why the miniaturization in Attempt 5, which used a minimal, valid property definition, still failed. The issue is not the complexity of the TaskOp or DictionaryAttr, but a global failure of the build environment to provide the necessary context to the code generation tools.

Modern MLIR build systems use CMake helper functions like add\_mlir\_dialect to manage these complexities automatically.10 These functions ensure that

mlir-tblgen is invoked with the correct include paths, not only to the MLIR system headers but also to the project's own build directory (e.g., \-I${CMAKE\_CURRENT\_BINARY\_DIR}), which is essential for resolving dependencies on other generated .inc files.11 The observed failures suggest the project's

CMakeLists.txt is either not using these modern helpers or is using them incorrectly, resulting in incomplete command-line invocations for mlir-tblgen.

### **Hypothesis 3: Toolchain Version Brittleness**

The use of a development snapshot, LLVM/MLIR 20.1.8, introduces a degree of uncertainty. Such versions, captured between major releases, can contain transient bugs, experimental features behind non-standard flags, or breaking changes that are not yet fully documented. The project's own BUILD\_ISSUE\_ANALYSIS.md note about a "brittle" environment corroborates this risk.

While the evidence strongly favors a CMake configuration error as the primary cause, the toolchain's nature cannot be entirely dismissed as a potential contributing factor. Development on core components like the Properties system and its bytecode integration is ongoing, and it is plausible that this specific version has unique requirements or instabilities.12

This environmental risk underscores the importance of having a robust fallback strategy. If rectifying the build system proves to be a protracted effort due to the esoteric nature of the toolchain, pivoting to a solution that relies on more stable, long-standing MLIR features becomes a strategically sound decision. This consideration directly motivates the detailed exploration of implementing a custom C++ attribute (Path B) as a viable and powerful alternative.

## **Unraveling the AttrConstraint Parser State Corruption**

The failure encountered in Attempt 6, while trying to implement a declarative AttrConstraint, serves as a powerful diagnostic clue that corroborates the primary thesis of a build system misconfiguration. The bizarre error message is a direct result of invoking mlir-tblgen without the necessary context to understand the custom constraint definition.

### **The Missing Link: gen-attr-constraint-decls and \-defs**

The error message unexpected def type; only defs deriving from TypeConstraint or Attr or Property are allowed is the key. MLIR's documentation is explicit that when a user defines a new constraint by subclassing AttrConstraint or TypeConstraint, they must also add specific mlir\_tablegen rules to their build system.13 These rules use the

\-gen-attr-constraint-decls and \-gen-attr-constraint-defs backends to generate the C++ code that makes these constraints known entities within the MLIR system.

The user's description of Attempt 6 makes no mention of adding these required build rules to CMakeLists.txt. Without them, the def IsValidTaskTarget : AttrConstraint\<...\> is an unrecognized construct to the main ODS backend (-gen-op-defs). When this backend parses OrchestraOps.td, it encounters a def whose base class, AttrConstraint, is not one of the top-level entities it is designed to process (such as Op, Dialect, or Property).14 The parser correctly identifies this as an invalid definition in its current context, leading to the failure.

### **TableGen Parser Behavior on Unrecognized Definitions**

The fact that the error message points to the def *following* the problematic IsValidTaskTarget definition is characteristic of TableGen's parser behavior. TableGen is known for its often-cryptic error messages and fragile error recovery mechanisms.15 When the parser encounters the unrecognized

AttrConstraint definition, it fails internally. Its subsequent attempt at error recovery is minimal; it likely consumes tokens in a confused state until it finds a recognizable landmark, such as the next def keyword. At this point, it tries to resume parsing, but its internal state is already corrupted, leading to an immediate second failure. The error is then reported at this new, misleading location.

This behavior strongly reinforces the conclusion that the user is not fighting a series of unrelated syntax errors, but rather a single, fundamental build integration problem. The mlir-tblgen tool is consistently being invoked without the full context—be it dialect-level settings, system include paths, or specialized generation rules—that it requires to correctly interpret the provided .td files.

## **Strategic Pathways to a Modern, Verifiable Attribute**

Based on the comprehensive diagnosis of a systemic build configuration failure, several strategic paths are available to achieve the goal of a modern, verifiable attribute for the orchestra.task operation. These paths range from directly addressing the root cause to implementing robust alternatives that mitigate dependency on the fragile build environment.

### **Path A (Recommended): Rectifying the CMake Configuration**

The most direct and idiomatic solution is to correct the CMakeLists.txt file to properly configure the mlir-tblgen invocations. This will enable the use of the Properties system as originally intended and ensure the long-term health of the dialect's build process.

1. **Centralize Dialect Generation and Dependencies:** The project should leverage MLIR's standard CMake functions. The add\_mlir\_dialect function is designed to handle the generation of all core include files for a dialect from a primary .td file.10 The library target for the operations (  
   OrchestraOps) must declare a PUBLIC link dependency on the library for the dialect definition (OrchestraDialect). This ensures CMake correctly orders the build targets and makes generated headers from the dialect available to the operations.  
2. **Verify mlir-tblgen Invocation Details:** The exact command lines being executed by the build system can be inspected by running make VERBOSE=1 or ninja \-v. The command processing OrchestraOps.td must be verified to include:  
   * \-I flags pointing to the CMake build directory (e.g., ${CMAKE\_CURRENT\_BINARY\_DIR}), where generated headers like OrchestraDialect.h.inc will reside.11  
   * \-I flags pointing to the system's LLVM/MLIR include directory, which is necessary to resolve standard headers like mlir/IR/OpBase.td.  
   * A command structure where the OrchestraDialect.td definitions are processed in the same context as OrchestraOps.td. This is often handled by add\_mlir\_dialect which combines inputs appropriately.  
3. **Add AttrConstraint Generation Rules:** To resolve the failure from Attempt 6, the following explicit mlir\_tablegen rules must be added to the CMakeLists.txt file, as prescribed by the documentation.13  
   CMake  
   \# Add rules to generate C++ declarations and definitions for attribute constraints.  
   mlir\_tablegen(OrchestraAttrConstraints.h.inc \-gen-attr-constraint-decls)  
   mlir\_tablegen(OrchestraAttrConstraints.cpp.inc \-gen-attr-constraint-defs)

   \# Create a target that other libraries can depend on to ensure these files are generated first.  
   add\_public\_tablegen\_target(OrchestraAttrConstraintsIncGen)

   The library target that uses these constraints must then add OrchestraAttrConstraintsIncGen to its DEPENDS list.

### **Path B (The Robust Escape Hatch): Implementing a Custom TaskTargetAttr**

If the build environment proves intractable due to the development toolchain's instability, or if greater programmatic control is desired, creating a custom attribute class in C++ is a powerful alternative. This approach bypasses the most complex TableGen features, relying instead on stable C++ APIs.

1. **Define Storage and Attribute Classes:** A new C++ header (e.g., OrchestraAttributes.h) is required. Inside, define a storage class inheriting from mlir::AttributeStorage. This class will contain the structured data members (e.g., StringRef arch, int device\_id) and a KeyTy type alias for uniquing within the MLIRContext.17 Following this, define the  
   TaskTargetAttr class itself, inheriting from the CRTP base mlir::Attribute::AttrBase\<TaskTargetAttr, mlir::Attribute, TaskTargetAttrStorage\>.19 This class will expose typed accessor methods like  
   getArch() and getDeviceId().  
2. **Implement In-Attribute Verification:** The validation logic currently in the operation's verify() method can be moved directly into the attribute by implementing the static verifyConstructionInvariants method within the TaskTargetAttr class. This provides stronger encapsulation, ensuring that an invalid TaskTargetAttr can never be constructed.  
3. **Implement Custom Parser and Printer:** To provide a clean and readable assembly format like \#orchestra.target\<"gpu", device\_id \= 0\>, the dialect's printAttribute and parseAttribute virtual methods must be implemented in OrchestraDialect.cpp. These hooks allow for full control over the textual representation of the custom attribute, avoiding the verbosity of a generic DictionaryAttr.18  
4. **Register and Use the Attribute:** The new attribute must be registered with the dialect during initialization by calling addAttributes\<TaskTargetAttr\>(); in the OrchestraDialect::initialize() method.21 Finally, the  
   Orchestra\_TaskOp definition in OrchestraOps.td can be simplified to accept the new, fully-typed attribute directly: let arguments \= (ins TaskTargetAttr:$target);.

### **Path C (Pragmatic Fallback): Fortifying the C++ verify() Method**

As a minimal-effort alternative, the existing C++ verifier approach can be made more robust and maintainable without a full migration. This path is suitable if project constraints preclude larger changes. The core of this approach is to more effectively use the DictionaryAttr C++ API.22 Instead of relying on raw iteration and

dyn\_cast, the verification logic should be structured to first check for the presence and type of each expected key.

For example, the verifier should use dictAttr.get("arch") to retrieve the attribute associated with the "arch" key. It should then check if the result is non-null and if it can be cast to mlir::StringAttr before attempting to access its value. This pattern of "check presence, check type, then check value" for each field provides clearer error diagnostics and is less prone to crashes from unexpected attribute structures.

### **Decision-Making Summary**

The choice between these paths depends on project-specific constraints such as time, risk tolerance, and the desired level of architectural purity. The following table summarizes the trade-offs.

| Modernization Path | Implementation Complexity | Build System Dependency | Type Safety | Maintainability |
| :---- | :---- | :---- | :---- | :---- |
| **A: Properties System** | Low (Declarative) | High (Requires correct CMake) | High (Typed accessors) | High |
| **B: Custom C++ Attribute** | High (Boilerplate C++) | Low (Bypasses most tblgen features) | Highest (Full C++ control) | Moderate to High |
| **C: Enhanced C++ Verifier** | Low (Incremental) | Very Low (Existing setup) | Low (Manual casting) | Low to Moderate |

## **Concluding Recommendations and Implementation Roadmap**

The comprehensive analysis concludes that the unresolvable build failures stem from a solvable CMake configuration defect, not an intrinsic bug in MLIR's Properties system. The AttrConstraint failure was a critical diagnostic clue that confirmed this thesis by highlighting missing, specialized mlir-tblgen generation rules. The path forward should prioritize fixing this foundational issue to enable modern, idiomatic MLIR development.

A prioritized implementation roadmap is recommended:

1. **Debug and Verify:** Begin by using make VERBOSE=1 or ninja \-v to inspect the exact mlir-tblgen command lines being executed. This will provide empirical evidence of the missing \-I include paths and/or the isolated processing context for OrchestraOps.td.  
2. **Implement Path A (CMake Rectification):** Apply the CMake corrections detailed in Section IV, Path A.  
   * First, add the required mlir\_tablegen rules for AttrConstraint generation. This is an isolated and highly probable fix.  
   * Next, refactor the dialect and op library definitions to ensure correct dependency management, likely by adopting add\_mlir\_dialect and add\_mlir\_dialect\_library with proper PUBLIC dependencies.  
3. **Incremental Verification:** Re-introduce the minimal MyTestOp from Attempt 5 with a single property. If this now builds successfully, the core configuration problem has been resolved.  
4. **Complete the Migration:** Proceed with the full migration of Orchestra\_TaskOp to the Properties system as originally planned.  
5. **Contingency Plan:** If, after a reasonable, time-boxed effort, the build system issues persist—a possibility given the use of a development toolchain—the project should pivot to **Path B (Custom C++ Attribute)**. This path is presented as a guaranteed, high-quality solution that achieves all the technical goals of modernization while de-risking the project from further build system debugging. Path C remains a last-resort option if time constraints are exceptionally severe.

#### **Referenzen**

1. mlir::tblgen::Dialect Class Reference \- LLVM, Zugriff am August 25, 2025, [https://mlir.llvm.org/doxygen/classmlir\_1\_1tblgen\_1\_1Dialect.html](https://mlir.llvm.org/doxygen/classmlir_1_1tblgen_1_1Dialect.html)  
2. mlir/include/mlir/TableGen/Argument.h · 8f31c6dde730fd1e4d64a6126474f51446fef453 · llvm-doe / llvm-project · GitLab, Zugriff am August 25, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/8f31c6dde730fd1e4d64a6126474f51446fef453/mlir/include/mlir/TableGen/Argument.h](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/8f31c6dde730fd1e4d64a6126474f51446fef453/mlir/include/mlir/TableGen/Argument.h)  
3. MLIR Release Notes, Zugriff am August 25, 2025, [https://mlir.llvm.org/docs/ReleaseNotes/](https://mlir.llvm.org/docs/ReleaseNotes/)  
4. Open MLIR Meeting 2-9-2023: Deep Dive on MLIR Internals, Operation\&Attribute, towards Properties \- YouTube, Zugriff am August 25, 2025, [https://www.youtube.com/watch?v=7ofnlCFzlqg](https://www.youtube.com/watch?v=7ofnlCFzlqg)  
5. TableGen BackEnds — LLVM 22.0.0git documentation, Zugriff am August 25, 2025, [https://llvm.org/docs/TableGen/BackEnds.html](https://llvm.org/docs/TableGen/BackEnds.html)  
6. Operation Definition Specification (ODS) \- MLIR \- LLVM, Zugriff am August 25, 2025, [https://mlir.llvm.org/docs/DefiningDialects/Operations/](https://mlir.llvm.org/docs/DefiningDialects/Operations/)  
7. mlir/docs/DefiningDialects/\_index.md · 3665da3d0091ab765d54ce643bd82d353c040631 · llvm-doe / llvm-project · GitLab, Zugriff am August 25, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/3665da3d0091ab765d54ce643bd82d353c040631/mlir/docs/DefiningDialects/\_index.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/3665da3d0091ab765d54ce643bd82d353c040631/mlir/docs/DefiningDialects/_index.md)  
8. Defining Dialects \- MLIR \- LLVM, Zugriff am August 25, 2025, [https://mlir.llvm.org/docs/DefiningDialects/](https://mlir.llvm.org/docs/DefiningDialects/)  
9. Step-by-step Guide to Adding a New Dialect in MLIR \- Perry Gibson, Zugriff am August 25, 2025, [https://gibsonic.org/blog/2024/01/11/new\_mlir\_dialect.html](https://gibsonic.org/blog/2024/01/11/new_mlir_dialect.html)  
10. Creating a Dialect \- MLIR \- LLVM, Zugriff am August 25, 2025, [https://mlir.llvm.org/docs/Tutorials/CreatingADialect/](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)  
11. D85464 \[MLIR\] \[CMake\] Support building MLIR standalone \- LLVM Phabricator archive, Zugriff am August 25, 2025, [https://reviews.llvm.org/D85464](https://reviews.llvm.org/D85464)  
12. D155340 Add support for versioning properties in MLIR bytecode \- LLVM Phabricator archive, Zugriff am August 25, 2025, [https://reviews.llvm.org/D155340?id=540726](https://reviews.llvm.org/D155340?id=540726)  
13. Constraints \- MLIR \- LLVM, Zugriff am August 25, 2025, [https://mlir.llvm.org/docs/DefiningDialects/Constraints/](https://mlir.llvm.org/docs/DefiningDialects/Constraints/)  
14. mlir/docs/DefiningDialects/Operations.md · e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e · llvm-doe / llvm-project \- GitLab, Zugriff am August 25, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e/mlir/docs/DefiningDialects/Operations.md](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/e6c01432b6fb6077e1bdf2e0abf05d2c2dd3fd3e/mlir/docs/DefiningDialects/Operations.md)  
15. MLIR — Using Tablegen for Passes \- Math ∩ Programming, Zugriff am August 25, 2025, [https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/](https://www.jeremykun.com/2023/08/10/mlir-using-tablegen-for-passes/)  
16. TableGen has been the most annoying part of working with LLVM, it is unfortunate... | Hacker News, Zugriff am August 25, 2025, [https://news.ycombinator.com/item?id=38620388](https://news.ycombinator.com/item?id=38620388)  
17. mlir/docs/Tutorials/DefiningAttributesAndTypes.md · 2ecd39d6aee1d194d7a960fef6736007dac92f0b \- GitLab, Zugriff am August 25, 2025, [https://praios.lf-net.org/littlefox/llvm-project/-/blob/2ecd39d6aee1d194d7a960fef6736007dac92f0b/mlir/docs/Tutorials/DefiningAttributesAndTypes.md](https://praios.lf-net.org/littlefox/llvm-project/-/blob/2ecd39d6aee1d194d7a960fef6736007dac92f0b/mlir/docs/Tutorials/DefiningAttributesAndTypes.md)  
18. mlir/docs/AttributesAndTypes.md · main · Kevin Sala / llvm-project \- GitLab, Zugriff am August 25, 2025, [https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/AttributesAndTypes.md](https://git.cels.anl.gov/ksalapenades/llvm-project/-/blob/main/mlir/docs/AttributesAndTypes.md)  
19. llvm-project-with-mlir/mlir/g3doc/DefiningAttributesAndTypes.md at master \- GitHub, Zugriff am August 25, 2025, [https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/DefiningAttributesAndTypes.md](https://github.com/joker-eph/llvm-project-with-mlir/blob/master/mlir/g3doc/DefiningAttributesAndTypes.md)  
20. Defining Dialect Attributes and Types \- MLIR \- LLVM, Zugriff am August 25, 2025, [https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/](https://mlir.llvm.org/docs/DefiningDialects/AttributesAndTypes/)  
21. Defining Dialect Attributes and Types, Zugriff am August 25, 2025, [https://chromium.googlesource.com/external/github.com/llvm/llvm-project/+/refs/heads/dev-newmaster/mlir/docs/Tutorials/DefiningAttributesAndTypes.md](https://chromium.googlesource.com/external/github.com/llvm/llvm-project/+/refs/heads/dev-newmaster/mlir/docs/Tutorials/DefiningAttributesAndTypes.md)  
22. My Project: mlir::DictionaryAttr Class Reference, Zugriff am August 25, 2025, [https://bollu.github.io/mlir/html/classmlir\_1\_1DictionaryAttr.html](https://bollu.github.io/mlir/html/classmlir_1_1DictionaryAttr.html)