

# **An In-Depth Analysis of add\_mlir\_dialect\_library and Source File Management for Out-of-Tree MLIR 18.1 Dialects**

## **I. Executive Summary**

This report provides a comprehensive analysis of the CMake build errors encountered during out-of-tree dialect development for MLIR 18.1, specifically the Cannot find source file: SOURCES and No SOURCES given to target errors. These issues arise from a common but critical misunderstanding of the argument-passing mechanism within the LLVM and MLIR CMake infrastructure.

The central finding is that the add\_mlir\_dialect\_library macro, and its underlying counterparts, do not recognize a SOURCES keyword. Source files must be provided as a direct, keyword-less list of positional arguments following the library name. The build system's error message is literal: it interprets the SOURCES token as a filename, which it then fails to locate. This behavior is not a regression or breaking change in MLIR 18.1 but is a long-standing characteristic of the core LLVM build system's design.

The root of this confusion stems from syntactic inconsistencies within the broader LLVM/MLIR CMake ecosystem. While the core C++ library-building macros rely on archaic positional argument patterns, newer components, such as the Python bindings, have adopted modern CMake practices that do use an explicit SOURCES keyword. This divergence creates a challenging environment for developers, where "tribal knowledge" of these inconsistencies becomes necessary.

This analysis deconstructs the macro call chain from add\_mlir\_dialect\_library down to add\_llvm\_library to reveal the precise mechanism of argument forwarding. It then presents two canonical, correct patterns for specifying source files. Furthermore, it evaluates the modern CMake command target\_sources as a superior, more robust alternative. The report strongly recommends adopting a decoupled pattern, where add\_mlir\_dialect\_library is used to declare the target and target\_sources is used to populate it with source files. This approach aligns with modern CMake best practices, enhancing project modularity, readability, and long-term maintainability. A complete "golden" CMakeLists.txt template embodying this recommendation is provided as a definitive guide for out-of-tree dialect development.

## **II. Deconstructing the add\_mlir\_dialect\_library Call Chain**

To understand the origin of the build errors, it is essential to trace the flow of arguments from the user-facing add\_mlir\_dialect\_library macro through the layers of the MLIR and LLVM build systems. The issue is not a bug in a single function but a consequence of a specific architectural design based on argument forwarding.

### **The Wrapper Hierarchy**

The LLVM project's CMake infrastructure is built upon a hierarchy of macros and functions, where higher-level, domain-specific functions provide a layer of abstraction over more generic, core functions. The add\_mlir\_dialect\_library command is a prime example of this design.

1. **add\_mlir\_dialect\_library**: This is the function developers are encouraged to use when creating a library for an MLIR dialect.1 Its primary responsibility, beyond creating the library target itself, is to add the library's name to a global property,  
   MLIR\_DIALECT\_LIBS. This property is later consumed by other parts of the build system, for instance, to automatically link all available dialects into tools like mlir-opt.2 Critically, this function does not parse its own arguments. As seen in its definition within  
   AddMLIR.cmake, it simply forwards all arguments it receives (${ARGV}) to the next layer.3  
   CMake  
   \# From mlir/cmake/modules/AddMLIR.cmake  
   function(add\_mlir\_dialect\_library name)  
     set\_property(GLOBAL APPEND PROPERTY MLIR\_DIALECT\_LIBS ${name})  
     add\_mlir\_library(${ARGV})  
   endfunction(add\_mlir\_dialect\_library)

2. **add\_mlir\_library**: This function serves a similar purpose but at a more general MLIR level. It appends the library name to the MLIR\_ALL\_LIBS global property, which is used for tasks like constructing a monolithic libMLIR.so shared library.5 Like its predecessor, it performs no significant argument parsing of its own and forwards its complete argument list (  
   ${ARGV}) to the foundational LLVM layer.3  
3. **add\_llvm\_library**: This is the final and most critical macro in the chain, defined in LLVM's core AddLLVM.cmake module. It is the workhorse responsible for creating almost all static and shared libraries within the entire LLVM project. All arguments from the initial add\_mlir\_dialect\_library call ultimately arrive here.

This establishes a clear and direct call chain: add\_mlir\_dialect\_library → add\_mlir\_library → add\_llvm\_library. The key takeaway is that the arguments, including any source files and keywords, are passed unmodified through each layer. Therefore, the argument parsing behavior is dictated entirely by the final macro in the sequence, add\_llvm\_library.

### **Argument Parsing in add\_llvm\_library**

An examination of the add\_llvm\_library macro's implementation reveals the core of the issue. Unlike many modern CMake functions that use cmake\_parse\_arguments to handle named keywords, add\_llvm\_library follows an older, positional-argument paradigm.

The macro's definition shows that it consumes the first argument as the library name and then passes the remainder of the arguments, collected in the special CMake variable ${ARGN}, directly to another helper macro, llvm\_process\_sources.7

CMake

\# From llvm/cmake/modules/AddLLVM.cmake (simplified)  
macro(add\_llvm\_library name)  
  llvm\_process\_sources( ALL\_FILES ${ARGN} )  
  add\_library( ${name} ${ALL\_FILES} )  
 ...  
endmacro(add\_llvm\_library)

The llvm\_process\_sources macro is designed to iterate through a flat list of strings and interpret them as source files. It does not have any logic to recognize or handle a SOURCES keyword.

This architectural choice explains the user's error with perfect clarity. When a developer writes:

CMake

add\_mlir\_dialect\_library(MyDialectLib SOURCES MyDialect.cpp MyOps.cpp)

The following sequence of events occurs:

1. add\_mlir\_dialect\_library is called. The argument name is MyDialectLib, and ${ARGV} becomes the list MyDialectLib;SOURCES;MyDialect.cpp;MyOps.cpp.  
2. It calls add\_mlir\_library with this full list.  
3. add\_mlir\_library is called. The argument name is MyDialectLib, and ${ARGV} is the same list.  
4. It calls add\_llvm\_library with this full list.  
5. add\_llvm\_library is called. The argument name is MyDialectLib, and the special variable ${ARGN} becomes the list of remaining arguments: SOURCES;MyDialect.cpp;MyOps.cpp.  
6. llvm\_process\_sources is then called with this list. It iterates through the items.  
7. The first item is the string "SOURCES". The build system interprets this as a relative path to a source file.  
8. CMake attempts to find a file named SOURCES in the current directory. When it fails, it emits the error: Cannot find source file: SOURCES.

The error is not a parsing failure; it is the correct and expected behavior of a system designed for positional arguments when it is given an unrecognized token that it can only interpret as a filename. The problem lies not in the build system's execution but in the developer's syntactic assumption.

## **III. The SOURCES Keyword: A Case of Mistaken Identity**

The developer's assumption that a SOURCES keyword is required is not arbitrary or illogical. It is an entirely reasonable conclusion drawn from observing other, more modern parts of the very same LLVM/MLIR build system. The error arises from a fundamental inconsistency in API design philosophy across different components of the project.

### **Pinpointing the Error**

As established in the previous section, the direct and singular cause of the build failure is the presence of the SOURCES keyword in the add\_mlir\_dialect\_library call. The macro expects a simple, flat list of source files following the library name. The keyword disrupts this expectation.

### **Source of Syntactic Confusion**

The LLVM project is a massive and long-lived codebase, and its CMake infrastructure has evolved organically over many years. This has led to different components adopting different API styles. A prime example that illustrates the source of this confusion can be found in MLIR's Python bindings infrastructure.

The file mlir/cmake/modules/AddMLIRPython.cmake defines a function, declare\_mlir\_python\_sources, for managing Python source files. Its implementation explicitly uses the cmake\_parse\_arguments command to define and process keyword-based arguments, including SOURCES.8

CMake

\# From mlir/cmake/modules/AddMLIRPython.cmake  
function(declare\_mlir\_python\_sources name)  
  cmake\_parse\_arguments(ARG "" "ROOT\_DIR;ADD\_TO\_PARENT" "SOURCES;SOURCES\_GLOB" ${ARGN})  
 ...  
endfunction()

A developer who has interacted with this part of the build system, or seen examples from it, would logically conclude that SOURCES is a standard keyword for specifying source files throughout MLIR's CMake API. This is a classic case of mistaken identity, where a pattern valid in one context is incorrectly applied to another.

This divergence in API design reflects a deeper architectural reality. The core C++ library-building infrastructure, centered around add\_llvm\_library, is one of the oldest and most performance-critical parts of the build system. Its interface, while archaic, is stable and deeply embedded. Changing it would be a monumental effort with far-reaching consequences. In contrast, newer components like the Python bindings were developed more recently and their authors chose to adopt modern CMake idioms (cmake\_parse\_arguments) for their superior clarity, explicitness, and self-documenting nature.

This creates a knowledge barrier. Successfully navigating the MLIR build system requires not just reading the documentation for a specific function, but also possessing an implicit understanding of the historical context and design philosophy of the subsystem one is interacting with. This reliance on "tribal knowledge" is a significant source of friction for both new and experienced developers, leading directly to the type of persistent, non-obvious errors described in the query. The problem is systemic, not a fault of the user.

To clarify this inconsistency, the following table provides a direct comparison of the argument-passing styles.

| CMake Function | Argument Style | Correct Example Usage |
| :---- | :---- | :---- |
| add\_mlir\_dialect\_library | Positional | add\_mlir\_dialect\_library(MyLib MyDialect.cpp MyOps.cpp) |
| declare\_mlir\_python\_sources | Keyword-based | declare\_mlir\_python\_sources(MyPySources SOURCES \_\_init\_\_.py utils.py) |

This side-by-side view validates the developer's observation of the SOURCES pattern while simultaneously correcting its application. It transforms the issue from a simple bug into an understandable consequence of the ecosystem's internal evolution.

## **IV. Canonical and Correct Usage Patterns**

Having established the root cause of the error, the solution is to adhere to the positional argument syntax expected by add\_mlir\_dialect\_library. There are two primary patterns for this, both of which are widely used within the LLVM project and its tutorials.

### **Pattern 1: Direct, Keyword-less Argument Listing**

For dialects with a small number of source files, the most straightforward approach is to list them directly in the add\_mlir\_dialect\_library call. This is the most common pattern seen in tutorials and simple examples.2

The correct syntax is simply the library name followed by a space-separated list of C++ source files.

**Example:**

CMake

\# lib/Standalone/CMakeLists.txt  
add\_mlir\_dialect\_library(MLIRStandalone  
  StandaloneDialect.cpp  
  StandaloneOps.cpp  
)

This pattern is concise and easy to read for simple libraries contained within a single directory.

### **Pattern 2: Using a Variable for Source Lists**

For more complex dialects with numerous source files, or files organized into subdirectories, managing the list directly in the function call becomes unwieldy. The standard CMake practice in this scenario is to collect the source file paths into a variable and then expand that variable in the function call.

This pattern improves organization and maintainability without changing the fundamental argument-passing mechanism. CMake expands the ${MY\_DIALECT\_SOURCES} variable into a semicolon-separated string, which the macro then processes as the same flat list of positional arguments.

**Example:**

CMake

\# lib/MyDialect/CMakeLists.txt  
set(MY\_DIALECT\_SOURCES  
  IR/MyDialect.cpp  
  IR/MyOps.cpp  
  IR/MyTypes.cpp  
  Transforms/Canonicalization.cpp  
)

add\_mlir\_dialect\_library(MLIRMyDialect ${MY\_DIALECT\_SOURCES})

### **Complete lib/CMakeLists.txt Example**

A complete and robust CMakeLists.txt for a dialect's lib/ directory must also handle dependencies, particularly the dependency on header files generated by TableGen from the corresponding include/ directory. This is accomplished using the DEPENDS keyword, which is correctly parsed by the underlying add\_llvm\_library macro.

The following example integrates the source variable pattern with dependency management, representing a canonical structure for an out-of-tree dialect library definition.1

CMake

\# This file is located at \<project\_root\>/lib/CMakeLists.txt

\# Add the subdirectory containing the dialect's C++ sources  
add\_subdirectory(MyDialect)

CMake

\# This file is located at \<project\_root\>/lib/MyDialect/CMakeLists.txt

\# Collect all C++ source files for the dialect library into a variable  
\# for better organization.  
set(DIALECT\_SOURCES  
  MyDialect.cpp  
  MyOps.cpp  
)

\# Define the dialect library.  
\# The first argument is the target name.  
\# The subsequent arguments are the source files, passed without any keyword.  
add\_mlir\_dialect\_library(MLIRMyDialect  
  ${DIALECT\_SOURCES}

  \# Specify a dependency on the TableGen target that generates the.h.inc files.  
  \# This ensures the headers are generated before the C++ files are compiled.  
  \# This target is typically defined in the include/MyDialect/CMakeLists.txt file.  
  DEPENDS  
  MLIRMyDialectOpsIncGen

  \# Specify the other MLIR libraries that this dialect library depends on.  
  \# These are specified with the LINK\_LIBS keyword.  
  LINK\_LIBS  
  PUBLIC  
    MLIRIR  
    MLIRDialect  
    MLIRSupport  
)

This structure correctly defines the library, specifies its sources positionally, and ensures the build order is correct with respect to code generation, resolving the user's issue while following established project conventions.

## **V. A Modern Alternative: The target\_sources Command**

While the positional argument pattern is functionally correct and widely used within the LLVM codebase, it deviates from the best practices of modern CMake. For developers building new, clean-slate out-of-tree dialects, a more robust, readable, and maintainable approach is available: the target\_sources command. This section addresses the user's query about alternative methods and presents this pattern as the recommended best practice.

### **Introduction to Modern CMake Philosophy**

Modern CMake (generally considered versions 3.1 and newer) promotes a philosophy of separating the *creation* of a target from the *modification* of its properties. In this paradigm, a command like add\_library() is used to simply declare the existence and name of a target. Subsequent commands, such as target\_sources(), target\_include\_directories(), and target\_link\_libraries(), are then used to populate the properties of that target.10

This approach has several advantages over the older style of passing all information into a single, monolithic add\_library() call:

* **Decoupling:** The definition of what a target *is* (a library named MyLib) is separated from the details of *how it is built* (its sources, include paths, and dependencies). This improves clarity and reduces cognitive overhead.  
* **Modularity:** Properties can be added to a target from anywhere in the project structure after the target has been declared. For example, a top-level CMakeLists.txt can declare a library, and CMakeLists.txt files in various subdirectories can each use target\_sources() to add their respective source files to that same library. This is a powerful feature for organizing large projects that is difficult or impossible to achieve with the traditional add\_llvm\_library pattern.11  
* **Readability and Maintainability:** The code becomes more explicit and self-documenting. A call to target\_sources unambiguously states its purpose, aligning the project's build scripts with general CMake best practices rather than relying on LLVM-specific idioms.

### **Implementation Guide for MLIR Dialects**

Adopting this modern pattern for an MLIR dialect is straightforward. The key is to continue using add\_mlir\_dialect\_library to ensure the target is correctly registered with the MLIR build system, but to call it *without* any source files. The sources are then added in a separate, subsequent step.

Step 1: Create and Register the Library Target  
First, call add\_mlir\_dialect\_library with only the target name and its non-source properties, such as DEPENDS and LINK\_LIBS. This creates the CMake target and ensures it is added to the global MLIR\_DIALECT\_LIBS property.

CMake

\# Step 1: Create the library target and register it as a dialect library.  
\# Note that no source files are provided in this call.  
add\_mlir\_dialect\_library(MLIRMyDialect  
  \# Dependencies on TableGen targets and other libraries are still  
  \# specified here, as they are properties of the library itself.  
  DEPENDS  
  MLIRMyDialectOpsIncGen

  LINK\_LIBS  
  PUBLIC  
    MLIRIR  
    MLIRDialect  
)

Step 2: Add Sources with target\_sources  
Next, use the target\_sources command to populate the newly created target with its C++ source files. The PRIVATE keyword is used to specify that these source files are part of the library's implementation and are not part of its public interface that consumers would need to compile against.10

CMake

\# Step 2: Add the C++ source files to the target using the modern, explicit command.  
\# The 'PRIVATE' scope indicates these sources are for building MLIRMyDialect itself,  
\# not for targets that link against it.  
target\_sources(MLIRMyDialect  
  PRIVATE  
    MyDialect.cpp  
    MyOps.cpp  
)

This two-step process achieves the same result as the traditional pattern but with significantly improved structure and clarity.

### **Comparison of Library Definition Methodologies**

To help developers choose the appropriate pattern, the following table compares the traditional positional-argument method with the modern target\_sources approach.

| Methodology | Syntax Example | Pros | Cons | Recommendation |
| :---- | :---- | :---- | :---- | :---- |
| **Direct Argument (Positional)** | add\_mlir\_dialect\_library(Name file1.cpp...) | \- Concise for simple, single-directory libraries. \- The most common pattern in the existing LLVM/MLIR codebase. | \- Couples target creation with source specification. \- Less modular; all sources must be known at the point of call. \- Uses a non-standard CMake pattern, increasing reliance on "tribal knowledge." | Suitable for quick tests, trivial examples, or when strictly adhering to the prevailing style of the upstream LLVM source tree. |
| **Decoupled (target\_sources)** | add\_mlir\_dialect\_library(Name) target\_sources(Name PRIVATE file1.cpp...) | \- Aligns with modern CMake best practices. \- Highly modular and scalable for complex projects with multiple source directories. \- Improved readability and long-term maintainability. | \- Slightly more verbose for the most trivial cases. | **Recommended best practice** for all new out-of-tree dialect development. It produces more robust and maintainable build configurations. |

While both methods are valid, the decoupled target\_sources approach is demonstrably superior from a software engineering perspective. It is the recommended pattern for any serious, long-term out-of-tree dialect project.

## **VI. Version-Specific Analysis for MLIR 18.1**

A critical aspect of the user's query is whether the observed build issues are the result of a recent regression or breaking change in the MLIR 18.1 release cycle. A thorough review of the release artifacts and historical build system files confirms that this is not the case. The behavior is long-standing and by design.

### **Review of Release Notes and Deprecations**

An analysis of the official MLIR release notes for LLVM 18 and surrounding versions reveals no changes related to the core CMake library-building macros.12 The major developments highlighted for the LLVM 17 and 18 cycles include significant IR-level features:

* **Properties:** A new mechanism for storing operation data, with usePropertiesForAttributes \= 1; becoming the default in LLVM 18\.12  
* **Bytecode:** The introduction and maturation of a compact, versioned, and lazy-loadable binary representation for MLIR.12  
* **Actions:** A new framework for tracing and debugging compiler transformations.12

Further examination of release announcements for the 18.1.x point releases 13 and the official MLIR deprecation list 17 shows a focus on bug fixes, C++ API refinements (e.g., deprecating

OpBuilder::create), and changes to specific dialects (e.g., GPU), but a complete absence of any modifications to add\_mlir\_dialect\_library or add\_llvm\_library.

### **Analysis of CMake Module History**

To confirm that the current behavior is not recent, one can examine the state of the relevant CMake modules from older LLVM/MLIR versions. For example, the version of AddMLIR.cmake from the Android NDK r22 release branch (which is several major versions old) shows the exact same structure for add\_mlir\_dialect\_library and add\_mlir\_library: they are simple wrappers that forward all arguments (${ARGV}) to the next layer.3 Similarly, a version from the LLVM 11.x release branch shows the same design.4

This historical evidence proves conclusively that the reliance on add\_llvm\_library and its positional argument-passing mechanism is a stable, core aspect of the build system's architecture. The behavior has not changed in the MLIR 18.1 timeframe.

### **Conclusion of Version Analysis**

The evidence is definitive: the build errors encountered are not the result of a bug or breaking change in MLIR 18.1. They are the result of a syntactic misunderstanding of a long-established API pattern that has been consistent for many major release cycles. Developers can confidently target MLIR 18.1 using the canonical patterns described in this report.

## **VII. Synthesis and Final Recommendations**

This analysis has deconstructed the MLIR build system's library creation mechanism, identified the precise cause of the Cannot find source file: SOURCES error, and evaluated both canonical and modern solutions. The following recommendations synthesize these findings into a clear, actionable guide for out-of-tree dialect developers.

### **Immediate Corrective Action**

The most critical and immediate action to resolve the build errors is to **remove the SOURCES keyword** from all add\_mlir\_dialect\_library invocations. The source files must be passed as a keyword-less, space-separated list of positional arguments immediately following the library target name.

### **Recommended Best Practice for Future Development**

For all new and refactored out-of-tree dialect projects, the strongly recommended best practice is to adopt the **decoupled add\_mlir\_dialect\_library and target\_sources pattern**. This approach offers superior modularity, readability, and maintainability by separating the declaration of a library target from the specification of its source files. Adhering to this modern CMake paradigm will make projects more robust, easier for new contributors to understand, and less reliant on esoteric knowledge of the LLVM build system's internal quirks.

### **The "Golden" lib/CMakeLists.txt Template**

The following CMakeLists.txt file serves as a complete, heavily commented, and robust template for defining a dialect's implementation library. It embodies all the best practices discussed in this report, providing a copy-pasteable foundation for any out-of-tree MLIR dialect project.

This template should be placed in the lib/MyDialect/ subdirectory of a project that follows the canonical include//lib/ structure.

CMake

\# \==============================================================================  
\# CMakeLists.txt for the library of an out-of-tree MLIR dialect  
\#  
\# This template uses the modern \`target\_sources\` approach for maximum  
\# clarity, modularity, and maintainability.  
\#  
\# Location: \<project\_root\>/lib/MyDialect/CMakeLists.txt  
\# \==============================================================================

\# \------------------------------------------------------------------------------  
\# Step 1: Declare the library target using \`add\_mlir\_dialect\_library\`.  
\#  
\# This is the crucial registration step. We use the MLIR-specific macro to  
\# ensure the library is known to the MLIR build system (e.g., for tools like  
\# mlir-opt).  
\#  
\# CRITICAL: Provide NO source files in this call. Sources will be added later  
\# using \`target\_sources\`.  
\# \------------------------------------------------------------------------------  
add\_mlir\_dialect\_library(MLIRMyDialect

  \# DEPENDS: Specify dependencies on auto-generated TableGen files.  
  \# This ensures that the C++ headers (.h.inc) are generated before any  
  \# source file that includes them is compiled. The \`MLIRMyDialectOpsIncGen\`  
  \# target is typically defined in the corresponding \`include/MyDialect/CMakeLists.txt\`.  
  DEPENDS  
  MLIRMyDialectOpsIncGen

  \# LINK\_LIBS: Specify link-time dependencies on other MLIR libraries.  
  \# The PUBLIC keyword ensures that these link dependencies are propagated to  
  \# any target that links against MLIRMyDialect.  
  LINK\_LIBS  
  PUBLIC  
    \# Core MLIR libraries required by most dialects.  
    MLIRIR  
    MLIRDialect  
    MLIRSupport

    \# Other dialects this dialect might depend on.  
    \# For example, if your dialect uses operations from the 'arith' dialect:  
    \# MLIRArith  
)

\# \------------------------------------------------------------------------------  
\# Step 2: Add C++ source files to the target using \`target\_sources\`.  
\#  
\# This is the modern, decoupled way to specify sources. It is more readable  
\# and allows for more complex project structures where sources might be  
\# added from multiple subdirectories.  
\# \------------------------------------------------------------------------------  
target\_sources(MLIRMyDialect  
  \# The PRIVATE keyword specifies that these files are part of the  
  \# implementation of MLIRMyDialect. They are not part of its public API  
  \# that consumers of the library need to know about. This is almost  
  \# always the correct scope for.cpp files.  
  PRIVATE  
    MyDialect.cpp  
    MyOps.cpp  
    MyTypes.cpp  
    \# Add other.cpp files for passes, etc., here.  
    \# e.g., MyDialectCanonicalization.cpp  
)

\# \------------------------------------------------------------------------------  
\# Step 3 (Optional): Add public headers to the target's interface.  
\#  
\# If your dialect library has public C++ headers (other than the main  
\# dialect header which is usually in the top-level include directory), you can  
\# also add them using \`target\_sources\` with a PUBLIC scope. This is less  
\# common for the standard dialect structure but is a powerful feature.  
\#  
\# target\_sources(MLIRMyDialect  
\#   PUBLIC  
\#     \# This makes MyDialect/Utils.h part of the library's public interface.  
\#     \# Note: The path should be relative to the current CMakeLists.txt.  
\#     ${CMAKE\_CURRENT\_SOURCE\_DIR}/Utils.h  
\# )  
\# \------------------------------------------------------------------------------

\# \------------------------------------------------------------------------------  
\# Step 4: Link against external (non-MLIR, non-LLVM) libraries if needed.  
\#  
\# Use the standard \`target\_link\_libraries\` for any other dependencies.  
\# \------------------------------------------------------------------------------  
\# target\_link\_libraries(MLIRMyDialect  
\#   PRIVATE  
\#     SomeOtherLibrary::SomeOtherTarget  
\# )  
\# \------------------------------------------------------------------------------

#### **Referenzen**

1. Creating a Dialect \- MLIR \- LLVM, Zugriff am August 11, 2025, [https://mlir.llvm.org/docs/Tutorials/CreatingADialect/](https://mlir.llvm.org/docs/Tutorials/CreatingADialect/)  
2. Step-by-step Guide to Adding a New Dialect in MLIR \- Perry Gibson, Zugriff am August 11, 2025, [https://gibsonic.org/blog/2024/01/11/new\_mlir\_dialect.html](https://gibsonic.org/blog/2024/01/11/new_mlir_dialect.html)  
3. mlir/cmake/modules/AddMLIR.cmake \- toolchain/llvm-project \- Git at ..., Zugriff am August 11, 2025, [https://android.googlesource.com/toolchain/llvm-project/+/refs/heads/ndk-release-r22/mlir/cmake/modules/AddMLIR.cmake](https://android.googlesource.com/toolchain/llvm-project/+/refs/heads/ndk-release-r22/mlir/cmake/modules/AddMLIR.cmake)  
4. mlir/cmake/modules/AddMLIR.cmake · release/11.x \- Gricad-gitlab, Zugriff am August 11, 2025, [https://gricad-gitlab.univ-grenoble-alpes.fr/violetf/llvm-project/-/blob/release/11.x/mlir/cmake/modules/AddMLIR.cmake?ref\_type=heads](https://gricad-gitlab.univ-grenoble-alpes.fr/violetf/llvm-project/-/blob/release/11.x/mlir/cmake/modules/AddMLIR.cmake?ref_type=heads)  
5. D79067 \[MLIR\] Fix libMLIR.so and LLVM\_LINK\_LLVM\_DYLIB \- LLVM Phabricator archive, Zugriff am August 11, 2025, [https://reviews.llvm.org/D79067](https://reviews.llvm.org/D79067)  
6. D73130 \[MLIR\] Add support for libMLIR.so \- LLVM Phabricator archive, Zugriff am August 11, 2025, [https://reviews.llvm.org/D73130](https://reviews.llvm.org/D73130)  
7. cmake/modules/AddLLVM.cmake \- toolchain/llvm \- Git at Google \- Android GoogleSource, Zugriff am August 11, 2025, [https://android.googlesource.com/toolchain/llvm/+/release\_33/cmake/modules/AddLLVM.cmake](https://android.googlesource.com/toolchain/llvm/+/release_33/cmake/modules/AddLLVM.cmake)  
8. llvm-project/mlir/cmake/modules/AddMLIRPython.cmake at main \- GitHub, Zugriff am August 11, 2025, [https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake](https://github.com/llvm/llvm-project/blob/main/mlir/cmake/modules/AddMLIRPython.cmake)  
9. mlir/lib/Dialect/Transform/IR/CMakeLists.txt ... \- GitLab, Zugriff am August 11, 2025, [https://code.ornl.gov/llvm-doe/llvm-project/-/blob/89bb0cae46f85bdfb04075b24f75064864708e78/mlir/lib/Dialect/Transform/IR/CMakeLists.txt](https://code.ornl.gov/llvm-doe/llvm-project/-/blob/89bb0cae46f85bdfb04075b24f75064864708e78/mlir/lib/Dialect/Transform/IR/CMakeLists.txt)  
10. target\_sources — CMake 4.1.0 Documentation, Zugriff am August 11, 2025, [https://cmake.org/cmake/help/latest/command/target\_sources.html](https://cmake.org/cmake/help/latest/command/target_sources.html)  
11. Enhanced source file handling with target\_sources() \- Crascit, Zugriff am August 11, 2025, [https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target\_sources/](https://crascit.com/2016/01/31/enhanced-source-file-handling-with-target_sources/)  
12. MLIR Release Notes, Zugriff am August 11, 2025, [https://mlir.llvm.org/docs/ReleaseNotes/](https://mlir.llvm.org/docs/ReleaseNotes/)  
13. LLVM 18.X Release Milestone \- GitHub, Zugriff am August 11, 2025, [https://github.com/llvm/llvm-project/milestone/23?closed=1](https://github.com/llvm/llvm-project/milestone/23?closed=1)  
14. LLVM Weekly \- \#532, March 11th 2024, Zugriff am August 11, 2025, [https://llvmweekly.org/issue/532](https://llvmweekly.org/issue/532)  
15. Part 1: What Is New In LLVM 18? \- Tools, Software and IDEs blog \- Arm Community, Zugriff am August 11, 2025, [https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/p1-whats-new-in-llvm-18](https://community.arm.com/arm-community-blogs/b/tools-software-ides-blog/posts/p1-whats-new-in-llvm-18)  
16. 18.1.6 Released\! \- Announcements \- LLVM Discussion Forums, Zugriff am August 11, 2025, [https://discourse.llvm.org/t/18-1-6-released/79068](https://discourse.llvm.org/t/18-1-6-released/79068)  
17. Deprecations & Current Refactoring \- MLIR \- LLVM, Zugriff am August 11, 2025, [https://mlir.llvm.org/deprecation/](https://mlir.llvm.org/deprecation/)