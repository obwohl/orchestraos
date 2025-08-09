

# **Canonical CMake Configuration for Out-of-Tree MLIR Passes with TableGen-Generated Dialect Headers**

## **Executive Summary**

This report provides a definitive guide to correctly configuring CMake for out-of-tree MLIR passes that rely on custom, TableGen-generated dialect headers. The persistent C++ compilation error, specifically "error: ‘TaskOp’ is not a member of ‘orchestra’", arises from a fundamental misconfiguration within the CMake build system. This issue typically indicates that the C++ compiler cannot locate the necessary declarations for operations defined in the custom dialect, primarily due to incorrect include paths for TableGen-generated header files or an improper build order. The canonical solution involves the precise application of target\_include\_directories utilizing CMake generator expressions ($\<BUILD\_INTERFACE:...\>, $\<INSTALL\_INTERFACE:...\>) and the strategic use of the DEPENDS keyword. This ensures that generated headers are correctly located and available to the C++ compiler at the appropriate time, enabling robust, maintainable, and relocatable out-of-tree MLIR development.

## **1\. The Challenge of Out-of-Tree MLIR Development and TableGen Integration**

### **1.1. Rationale for Out-of-Tree Development in MLIR**

Developing MLIR components, such as custom dialects, transformation passes, and specialized tools, outside of the main LLVM/MLIR monorepository offers significant strategic advantages for developers. This "out-of-tree" approach facilitates independent versioning of custom components, which is critical for managing project lifecycles that are distinct from the upstream MLIR development cycle.1 For instance, a domain-specific compiler built on MLIR can evolve its dialect and passes without being tightly coupled to the release cadence of the broader LLVM project.

Furthermore, out-of-tree development leads to substantially reduced build times during iterative development. Developers are only required to compile their specific components rather than the entire, expansive LLVM/MLIR project, which can significantly accelerate the development feedback loop. This modularity also simplifies integration into existing projects that may not require the full LLVM infrastructure, promoting experimentation with new features or domain-specific optimizations without the overhead and rigorous review process associated with direct upstream contributions.1 MLIR's architectural design inherently supports this modular, out-of-tree development, positioning it as a highly extensible framework. The very existence of specialized CMake modules designed for MLIR component integration, rather than solely relying on manual, low-level CMake scripting, underscores this fundamental architectural choice.1 This design choice is intended to facilitate the growth of the MLIR ecosystem and foster independent innovation.

The problem encountered by the user, while specific to CMake, highlights a broader tension between the desire for modular, out-of-tree development and the inherent complexity of integrating a highly generative framework like MLIR. The user's choice to develop out-of-tree was motivated by the benefits of modularity and faster builds. However, this choice introduces the challenge of correctly configuring a complex build system, CMake, to interact seamlessly with a powerful code generation tool, TableGen, that is deeply integrated with MLIR's internal structure. The "fundamental disconnect" described by the user precisely captures this tension. The resolution is not merely a CMake configuration adjustment; it represents a restoration of the seamless integration that MLIR's in-tree build system provides by default. A canonical CMake pattern, therefore, is not simply a convenience; it is a necessity to fully realize the benefits of out-of-tree MLIR development without sacrificing build reliability. This ensures that the promise of modularity and independent development can be fully realized in practice.

### **1.2. MLIR's Build System Philosophy and TableGen's Role**

MLIR, as an integral part of the broader LLVM project, leverages CMake as its meta-build system. CMake does not directly build the project but rather generates the necessary build files (e.g., Makefiles, Ninja files) for the chosen underlying build tool.1 This cross-platform capability is a cornerstone of the LLVM ecosystem's build philosophy. Crucially, MLIR extends CMake's capabilities by providing a suite of custom CMake modules, such as

AddMLIR.cmake and TableGen.cmake.1 These modules are specifically designed to simplify the configuration and integration of MLIR-specific components, particularly those that involve TableGen for code generation. These specialized modules are indispensable for correctly integrating out-of-tree projects, as they abstract away much of the underlying complexity of managing TableGen dependencies and MLIR-specific build rules.

The find\_package(MLIR REQUIRED CONFIG) command, a standard practice in out-of-tree MLIR projects, is not merely about locating MLIR libraries; it is fundamentally about discovering and integrating the build system infrastructure provided by MLIR itself.1 This command sets crucial variables like

MLIR\_DIR, MLIR\_INCLUDE\_DIRS, and MLIR\_CMAKE\_DIR, which are then used to include MLIR's specialized CMake macros. This mechanism is vital because it enables developers to build against a locally compiled MLIR (from a build tree) during development and seamlessly transition to a system-wide installed MLIR (from an install tree) for deployment or distribution.1 This dual-tree support is a foundational aspect of CMake and LLVM/MLIR's design, ensuring that out-of-tree components remain functional across different development and deployment environments. A common pitfall in out-of-tree MLIR projects is not fully appreciating the transitive setup required by

find\_package(MLIR), which goes beyond just setting include and library paths to enabling the entire MLIR-specific CMake ecosystem. This understanding is crucial for robust project initialization.

TableGen serves as a pivotal tool in the MLIR ecosystem, providing a generic language and associated tooling for maintaining and processing domain-specific information.1 Its role is central to MLIR's declarative approach to defining compiler infrastructure. MLIR extensively utilizes TableGen's Operation Definition Specification (ODS) to declaratively define core IR entities such as operations, types, attributes, and passes.1 This declarative methodology significantly reduces the amount of boilerplate C++ code that developers would otherwise need to write and maintain manually. From these

.td (TableGen Definition) files, mlir-tblgen generates substantial C++ code, including template specializations, accessor methods for operation parameters, parsing and printing logic for IR entities, and verification methods.1 This automated code generation streamlines development, enhances consistency, and minimizes the risk of errors associated with manual implementation of repetitive patterns. When

mlir-tblgen processes .td files, it produces a variety of C++ .inc files, such as \<Dialect\>Ops.h.inc, \<Dialect\>Ops.cpp.inc, \<Dialect\>Types.h.inc, and \<Dialect\>AttrDefs.h.inc.1 These files are not standalone compilation units but are specifically designed to be included within hand-written C++ source and header files.

The "single source of truth" philosophy, which is fundamental to MLIR's IR definition, extends implicitly to the build system. MLIR's core design principle is to define IR declaratively via TableGen, making the .td files the authoritative source of information. This principle implicitly extends to the build system: CMake, via MLIR's custom modules, is designed to seamlessly translate these declarative definitions into executable C++ code. The user's "‘TaskOp’ is not a member of ‘orchestra’" error signifies that this translation chain is broken at the build layer. The C++ compiler cannot find the generated declarations for TaskOp, which are derived directly from the .td files. This indicates a failure in the build system's ability to correctly consume the output of this "single source of truth." The causal relationship can be understood as: OrchestraOps.td (single source of truth) → mlir-tblgen (code generation) → OrchestraOps.h.inc (generated C++ declarations) → CMake (build system configuration) → C++ compiler (consumer). The error points to a breakdown between the generated .h.inc and the C++ compiler's ability to find and interpret it, indicating a CMake misconfiguration.

### **1.3. Understanding the "Not a Member Of" Error: A Deep Dive into the Disconnect**

The error: ‘TaskOp’ is not a member of ‘orchestra’ fundamentally means that the C++ compiler cannot locate the declaration for TaskOp within the orchestra namespace. This type of error almost invariably points to one of two primary issues within the build system:

1. **Missing Include Path:** The directory containing the TableGen-generated header file (OrchestraOps.h.inc), which holds the TaskOp declaration, is not correctly added to the C++ compiler's include path.1  
2. **Incorrect Build Order or Generation Failure:** The OrchestraOps.h.inc file was either not generated at all, or the C++ compilation was attempted before TableGen had completed its generation process.1

The orchestra namespace itself originates from the let cppNamespace \= "::mlir::orchestra"; declaration within the dialect's TableGen definition file (e.g., OrchestraDialect.td or OrchestraOps.td).1 This declaration instructs

mlir-tblgen to generate C++ code with the operations encapsulated within the specified namespace. The error message confirms that the compiler is correctly looking within this namespace, but the TaskOp declaration is absent from the files it has processed. This indicates that while the compiler understands the namespace structure, it lacks access to the actual definition of TaskOp within that structure.

This "not a member of" error is a precise diagnostic symptom of a deeper build system misconfiguration, rather than a C++ coding error in DivergenceToSpeculation.cpp itself. The user has correctly identified the C++ namespace (orchestra) and the operation name (TaskOp). The error is not undeclared identifier TaskOp, which might suggest a missing using namespace directive or an incorrect scope. Instead, it explicitly states ‘TaskOp’ is not a member of ‘orchestra’. This implies that the compiler successfully located the orchestra namespace, but it could not find TaskOp *inside* that namespace. This strongly suggests that the header file containing the definition of TaskOp within that namespace was either not included in the compilation unit or was not present or complete when the compiler attempted to process it. The user's prior attempts with INTERFACE libraries and direct target\_include\_directories were a correct line of inquiry, but they likely missed the crucial details of *which path* to add (specifically, the path relative to CMAKE\_CURRENT\_BINARY\_DIR from the consumer's perspective) and *when* to add it (enforced by explicit DEPENDS relationships). The error message serves as a precise indicator of where the build system chain broke, guiding the expert to the exact points of failure.

## **2\. Core MLIR CMake Concepts for Dialect and Pass Integration**

### **2.1. Essential MLIR CMake Macros: add\_mlir\_dialect(), mlir\_tablegen(), add\_mlir\_dialect\_library()**

MLIR's CMake modules provide several high-level macros that streamline the integration of TableGen-defined components, abstracting away much of the underlying complexity of managing TableGen dependencies and MLIR-specific build rules.1

* **add\_mlir\_dialect(name)**: This macro serves as the primary entry point for declaring an MLIR dialect within the CMake build system. It automatically sets up the necessary TableGen rules to generate the core operation and dialect declarations (e.g., \<name\>Ops.h.inc, \<name\>Ops.cpp.inc). A significant aspect of this macro is its creation of a public dependency target, typically named MLIR\<name\>IncGen.1 This target is crucial for managing build order, ensuring that the TableGen generation step is completed before any C++ compilation that relies on these generated headers. This explicit dependency prevents common "file not found" or "undeclared identifier" errors during compilation.  
* **mlir\_tablegen(output\_file\_base\_name command\_args...)**: This macro acts as a specialized wrapper around LLVM's generic tablegen macro, tailored for MLIR-specific code generation.1 While  
  add\_mlir\_dialect handles the default operation and dialect declarations, mlir\_tablegen is used for generating more specific components, such as attribute definitions (using flags like \-gen-attrdef-decls and \-gen-attrdef-defs) or pass declarations.1 This layered approach to TableGen integration allows for common patterns to be abstracted by  
  add\_mlir\_dialect, while providing fine-grained control over specific code generation needs via mlir\_tablegen.  
* **add\_public\_tablegen\_target(name)**: This macro is used when mlir\_tablegen is invoked directly for specific generated files (e.g., attributes, types) to create a public CMake target for their output dependencies.1 This allows other targets to correctly specify their build order requirements.  
* **add\_mlir\_dialect\_library(name)**: This macro is specifically intended for libraries associated with a custom dialect. Beyond defining the library, it appends the library's name to the global MLIR\_DIALECT\_LIBS property.1 This global list is particularly useful for tools like  
  mlir-opt, which can then automatically discover and link against all available dialects, simplifying the process of building a comprehensive MLIR tool.1 This demonstrates a higher-level abstraction provided by MLIR's CMake system for managing the ecosystem of dialects.

MLIR's CMake macros are not merely syntactic sugar; they embody specific build-system best practices and dependency management logic. These macros abstract away the complex, low-level details of add\_custom\_command, add\_custom\_target, and precise dependency ordering that would otherwise be required. The user's query indicates attempts at manual CMake configuration, which often leads to overlooking these implicit behaviors. Deviating from these canonical macros or misunderstanding their implicit behaviors is a common source of build issues. The solution lies in understanding and correctly leveraging these high-level abstractions rather than attempting to re-implement their underlying logic manually. This approach significantly promotes maintainability and reduces the likelihood of subtle build errors.

The following table summarizes these key MLIR CMake functions and their purposes:

| CMake Function | Purpose |
| :---- | :---- |
| add\_mlir\_dialect(name) | Declares an MLIR dialect, setting up TableGen rules for operations and dialect declarations. Creates a dependency target (e.g., MLIR\<name\>IncGen) to ensure generated files are available before compilation. 1 |

### **2.2. Navigating CMake Path Variables: CMAKE\_SOURCE\_DIR, CMAKE\_BINARY\_DIR, CMAKE\_CURRENT\_SOURCE\_DIR, CMAKE\_CURRENT\_BINARY\_DIR**

Understanding the various CMake path variables is crucial for correctly configuring an out-of-tree MLIR project, especially when dealing with generated files. These variables indicate different locations within the source and build trees, and their values dynamically change depending on the CMakeLists.txt file currently being processed.1 TableGen-generated headers are typically placed in the binary directory corresponding to where their

add\_mlir\_dialect or mlir\_tablegen command was invoked.1 Misunderstanding these paths can lead to "file not found" errors.

* **CMAKE\_CURRENT\_BINARY\_DIR**: This variable points to the full path of the build directory that corresponds to the CMAKE\_CURRENT\_SOURCE\_DIR.1 This is where build artifacts, including TableGen-generated files, for the current subdirectory are placed.1 Its value also changes with  
  add\_subdirectory().  
* **add\_subdirectory() Impact**: When add\_subdirectory(source\_dir \[binary\_dir\]) is invoked, CMake processes the CMakeLists.txt file located in source\_dir.1 A critical aspect for MLIR projects is its handling of the  
  binary\_dir argument. If binary\_dir is not explicitly specified, it defaults to a path that mirrors the source\_dir within the current output directory.1 For instance, if the top-level  
  CMakeLists.txt is in orchestra-compiler/ and the build directory is orchestra-compiler/build/, then add\_subdirectory(include/Orchestra) will cause the CMakeLists.txt in orchestra-compiler/include/Orchestra/ to be processed, and any generated files (such as those from add\_mlir\_dialect or mlir\_tablegen) will be placed in orchestra-compiler/build/include/Orchestra/.1 This implicit mapping is predictable but necessitates careful path management when specifying include directories for compilation units that depend on these generated headers.

The user's "fundamental disconnect" is highly likely a miscalculation of relative paths within CMake's hierarchical build structure, particularly concerning the dynamic nature of CMAKE\_CURRENT\_BINARY\_DIR when add\_subdirectory() is used. In the user's project, DivergenceToSpeculation.cpp resides in lib/Orchestra/, while OrchestraOps.h.inc is generated into build/include/Orchestra/. If the lib/Orchestra/CMakeLists.txt file is being processed, CMAKE\_CURRENT\_BINARY\_DIR would point to build/lib/Orchestra/. To correctly specify the include path to build/include/Orchestra/ from the context of build/lib/Orchestra/, a relative path like ../include/Orchestra is required.1 This subtle but critical relative path calculation is frequently overlooked. Debugging CMake path issues often requires a mental map (or actual inspection) of the

build directory structure and a clear understanding of how CMAKE\_CURRENT\_BINARY\_DIR shifts with add\_subdirectory calls. This is a common hurdle in modular CMake projects, especially when dealing with generated files.

The following table provides an overview of essential CMake variables for paths:

| CMake Variable | Description |  
| :--- | :--- | | CMAKE\_SOURCE\_DIR | The full path to the root of the top-level source tree of the entire project. This variable remains constant throughout the CMake configuration process for a given project. 1 | |  
CMAKE\_BINARY\_DIR | The full path to the root of the top-level build tree. This is where all generated build artifacts for the entire project are stored. 1 | |

CMAKE\_CURRENT\_SOURCE\_DIR | The full path to the source directory containing the CMakeLists.txt file currently being processed by CMake. Its value changes as add\_subdirectory() commands are processed. 1 | |

CMAKE\_CURRENT\_BINARY\_DIR | The full path to the build directory that corresponds to the CMAKE\_CURRENT\_SOURCE\_DIR. This is where build artifacts (including TableGen-generated files) for the current subdirectory are placed. Its value also changes with add\_subdirectory(). 1 | |

MLIR\_DIR | The path to the MLIR installation or build directory that find\_package(MLIR) successfully located. 1 | |

MLIR\_INCLUDE\_DIRS | A CMake list variable containing paths to the public include directories of MLIR. 1 | |

MLIR\_CMAKE\_DIR | The path to the directory containing MLIR's custom CMake modules (e.g., AddMLIR.cmake). 1 |

## **3\. The Canonical CMake Pattern for TableGen-Generated Header Consumption**

This section outlines the definitive CMake configuration pattern that directly addresses the user's core problem by ensuring TableGen-generated headers are correctly found and consumed by dependent C++ compilation units.

### **3.1. Top-Level Project Configuration (orchestra-compiler/CMakeLists.txt)**

The foundation of any out-of-tree MLIR project begins with a well-structured top-level CMakeLists.txt file. This file establishes the project's basic configuration and locates the necessary MLIR infrastructure.1

The file should begin by specifying the minimum required CMake version, with 3.20.0 or higher recommended for compatibility with modern MLIR builds.1 The project is then defined, specifying C++ and C as the required languages. It is also essential to explicitly set the C++ standard to C++17, which MLIR generally requires, and enable position-independent code for shared libraries.1

A critical step for out-of-tree development is locating the MLIR installation. This is achieved using find\_package(MLIR REQUIRED CONFIG).1 This command searches for

MLIRConfig.cmake and sets a series of variables, including MLIR\_DIR (the path to the MLIR installation or build directory), MLIR\_INCLUDE\_DIRS (a list of include directories for MLIR headers), and MLIR\_CMAKE\_DIR (the path to MLIR's CMake modules).1 Once MLIR has been located, its custom CMake modules must be made available to the project by appending

MLIR\_CMAKE\_DIR to CMAKE\_MODULE\_PATH.1 This step is critical because it allows CMake to find and include MLIR's specialized macros, without which the entire TableGen integration and MLIR-specific build processes would be significantly more complex or impossible.1 Following this, core LLVM and MLIR CMake utilities are included:

include(AddLLVM), include(TableGen), and include(AddMLIR).1

While target\_include\_directories is generally preferred for specifying include paths on a per-target basis, it is common practice in top-level MLIR projects to include global directories for core MLIR and LLVM headers. This ensures that all source files can readily locate fundamental headers without requiring explicit per-file path specifications.1 Finally,

add\_subdirectory() commands are used to include the custom dialect and pass components, promoting a modular build system.1

### **3.2. Configuring Dialect Header Generation (include/Orchestra/CMakeLists.txt)**

Within the include/Orchestra/CMakeLists.txt file, MLIR CMake macros are used to trigger the TableGen code generation. The add\_mlir\_dialect() macro is used to process the primary operations TableGen file, OrchestraOps.td. This command automatically sets up the generation of OrchestraOps.h.inc (and other .inc files) into the binary directory corresponding to this CMakeLists.txt file, which will be build/include/Orchestra/ (i.e., ${CMAKE\_CURRENT\_BINARY\_DIR}).1 Crucially, it also creates the

MLIROrchestraOpsIncGen target, which serves as a dependency for any C++ compilation that requires these generated files.1

To ensure that other parts of the project can correctly find and include these generated headers, a canonical and robust approach is to define an INTERFACE library specifically for the dialect's public headers. This INTERFACE library (e.g., OrchestraHeaders) is then used to propagate the necessary include paths. This is a powerful CMake idiom for managing usage requirements and promoting modularity, going beyond simple include paths. It decouples the header generation location from its consumption, making the build system more maintainable, less prone to errors when project structure changes, and inherently more scalable for larger projects with multiple components.

The target\_include\_directories() command for this INTERFACE library must employ CMake Generator Expressions ($\<BUILD\_INTERFACE:...\> and $\<INSTALL\_INTERFACE:...\>).1 This is fundamental for creating relocatable packages that function correctly in both the build tree (during development) and after installation (for deployment). For the

OrchestraHeaders INTERFACE library, the path ${CMAKE\_CURRENT\_BINARY\_DIR} is used for the build interface because, from the perspective of include/Orchestra/CMakeLists.txt, CMAKE\_CURRENT\_BINARY\_DIR points directly to build/include/Orchestra/, where OrchestraOps.h.inc is generated.1 The

INSTALL\_INTERFACE path include/Orchestra assumes that during installation, the headers will be installed under ${CMAKE\_INSTALL\_PREFIX}/include/Orchestra.1

CMake

\# orchestra-compiler/include/Orchestra/CMakeLists.txt  
\# This CMakeLists.txt is processed by add\_subdirectory(include/Orchestra)

\# Define the dialect and generate operation/dialect declarations  
\# This creates MLIROrchestraOpsIncGen target and OrchestraOps.h.inc etc.  
add\_mlir\_dialect(OrchestraOps orchestra)

\# Define a target for the public headers of the dialect  
\# This ensures that other targets can link against this and get the correct include paths  
add\_library(OrchestraHeaders INTERFACE)  
target\_include\_directories(OrchestraHeaders INTERFACE  
    $\<BUILD\_INTERFACE:${CMAKE\_CURRENT\_BINARY\_DIR}\> \# Generated headers in build tree  
    $\<INSTALL\_INTERFACE:include/Orchestra\> \# Installed headers relative to prefix  
)

### **3.3. Consuming Generated Headers in Your Library (lib/Orchestra/CMakeLists.txt)**

This section addresses the most critical aspects for resolving the user's "‘TaskOp’ is not a member of ‘orchestra’" error by correctly consuming the generated headers. The implementation of the custom dialect and MLIR pass typically resides within a library target.

The add\_mlir\_dialect\_library(MLIROrchestra...) macro is used to define the library containing OrchestraDialect.cpp and DivergenceToSpeculation.cpp.1 This macro is preferred for dialect-specific code as it automatically registers the library with the global

MLIR\_DIALECT\_LIBS property, making it discoverable by MLIR tools like mlir-opt.1

Crucial: Establishing Build Dependencies on TableGen Targets (DEPENDS)  
The DEPENDS keyword is paramount. It ensures that mlir-tblgen runs and produces OrchestraOps.h.inc before DivergenceToSpeculation.cpp is compiled.1 This explicitly prevents "file not found" or "undeclared identifier" errors that arise from incorrect build order. The C++ library or executable target must explicitly declare a dependency on the TableGen generation target (e.g.,  
MLIROrchestraOpsIncGen), which is created by the add\_mlir\_dialect macro in include/Orchestra/CMakeLists.txt.1 Without this explicit dependency, CMake might attempt to compile C++ files before

mlir-tblgen has produced the required headers, leading to compilation failures. Relying on implicit build order or manual verification is insufficient for complex projects; explicit CMake DEPENDS declarations are essential for build system correctness, reliability, and reproducibility across different build environments and configurations.

Crucial: Linking the Header Interface Library  
Instead of directly using target\_include\_directories with complex relative paths, the canonical approach is to link the OrchestraHeaders INTERFACE library defined in include/Orchestra/CMakeLists.txt. This automatically propagates the correct include paths, including the necessary generator expressions for build and install trees. This use of the OrchestraHeaders INTERFACE library with generator expressions is the canonical solution for managing generated header paths, promoting relocatability and reducing boilerplate in consuming targets. This pattern is a fundamental aspect of creating shareable, out-of-tree MLIR components that function correctly regardless of their installation location, encapsulating complex path logic and making the consuming CMakeLists.txt cleaner and less error-prone.  
Finally, the library definition includes linking against necessary MLIR core libraries using LINK\_COMPONENTS (e.g., Core) and LINK\_LIBS PUBLIC (e.g., MLIRIR, MLIRPass, MLIRTransforms, MLIRSupport).1 The

PUBLIC keyword ensures that these linking requirements are propagated transitively to any other target that links against MLIROrchestra.1

CMake

\# orchestra-compiler/lib/Orchestra/CMakeLists.txt  
\# This CMakeLists.txt is processed by add\_subdirectory(lib/Orchestra)

add\_mlir\_dialect\_library(MLIROrchestra  
    SOURCES  
        OrchestraDialect.cpp  
        DivergenceToSpeculation.cpp  
    DEPENDS  
        MLIROrchestraOpsIncGen \# Ensure ops and dialect headers are generated before compilation  
        \# If you have custom attributes/types, add their IncGen targets here too:  
        \# MLIROrchestraAttrDefsIncGen  
        \# MLIROrchestraTypesIncGen  
    LINK\_COMPONENTS  
        Core \# Link against core MLIR components  
    LINK\_LIBS  
        PUBLIC MLIRIR \# Core MLIR infrastructure  
        PUBLIC MLIRPass \# Pass management framework  
        PUBLIC MLIRTransforms \# Common MLIR transformations  
        PUBLIC MLIRSupport \# General MLIR support utilities  
        PUBLIC OrchestraHeaders \# Propagate include paths for hand-written and generated headers  
)

Understanding which TableGen files are generated and where they should be included in C++ source files is a common point of confusion. The following table clarifies the typical inclusion patterns for dialect-related generated files:

| Generated File (.inc suffix) | C++ Inclusion Point | Purpose |  
| :--- | :--- | :--- | | MyDialect.h.inc | include/MyDialect/MyDialect.h | Contains the C++ class declaration for the dialect itself. 1 | |  
MyDialect.cpp.inc | lib/MyDialect/MyDialect.cpp | Provides C++ method definitions for the dialect class. 1 | |

MyOps.h.inc | include/MyDialect/MyDialect.h or include/MyDialect/MyOps.h | Contains C++ class declarations for all operations defined in MyOps.td. 1 | |

MyOps.cpp.inc | lib/MyDialect/MyDialect.cpp (with \#define GET\_OP\_LIST and \#define GET\_OP\_CLASSES) | Contains C++ method definitions for operations, used to register operations with the dialect and instantiate their classes. 1 | |

MyTypes.h.inc | include/MyDialect/MyDialect.h or include/MyDialect/MyTypes.h | Contains C++ class declarations for custom types. 1 | |

MyTypes.cpp.inc | lib/MyDialect/MyDialect.cpp or lib/MyDialect/MyTypes.cpp | Provides C++ method definitions for custom types (e.g., parsing, printing). 1 | |

MyAttrDefs.h.inc | include/MyDialect/MyDialect.h or include/MyDialect/MyAttrs.h | Contains C++ class declarations for custom attributes. 1 | |

MyAttrDefs.cpp.inc | lib/MyDialect/MyDialect.cpp or lib/MyDialect/MyAttrs.cpp | Provides C++ method definitions for custom attributes. 1 | |

MyOpsInterfaces.h.inc | include/MyDialect/MyDialect.h or relevant interface headers | Contains C++ class declarations for operation interfaces. 1 | |

MyOpsInterfaces.cpp.inc | lib/MyDialect/MyDialect.cpp or relevant interface source files | Provides C++ method definitions for operation interfaces. 1 |

## **4\. Step-by-Step Application to the Orchestra Project**

This section provides concrete modifications to the user's existing project structure, applying the canonical CMake patterns discussed previously.

### **4.1. Modifying orchestra-compiler/CMakeLists.txt**

The top-level CMakeLists.txt file needs to establish the project, find the MLIR installation, and include the subdirectories for the custom dialect and its library.

CMake

\# orchestra-compiler/CMakeLists.txt  
cmake\_minimum\_required(VERSION 3.20.0)  
project(OrchestraCompiler LANGUAGES CXX C)

set(CMAKE\_CXX\_STANDARD 17 CACHE STRING "C++ standard to conform to")  
set(CMAKE\_POSITION\_INDEPENDENT\_CODE ON) \# Often required for shared libraries

\# Find MLIR installation  
find\_package(MLIR REQUIRED CONFIG)  
message(STATUS "Using MLIRConfig.cmake in: ${MLIR\_DIR}")  
message(STATUS "Using LLVMConfig.cmake in: ${LLVM\_DIR}") \# LLVM\_DIR is often set by MLIR's config

\# Add MLIR's CMake modules to the module path  
list(APPEND CMAKE\_MODULE\_PATH "${MLIR\_CMAKE\_DIR}")

\# Include essential LLVM/MLIR CMake utilities  
include(AddLLVM)  
include(TableGen)  
include(AddMLIR)

\# Globally include LLVM and MLIR headers (optional, but common)  
include\_directories(${LLVM\_INCLUDE\_DIRS})  
include\_directories(${MLIR\_INCLUDE\_DIRS})  
include\_directories(${PROJECT\_SOURCE\_DIR}/include) \# For hand-written public headers

\# Add subdirectories for the dialect and its library  
add\_subdirectory(include/Orchestra)  
add\_subdirectory(lib/Orchestra)  
\# add\_subdirectory(tools/orchestra-opt) \# Optional: if you have a custom tool  
\# add\_subdirectory(test/Orchestra) \# Optional: if you have tests

### **4.2. Modifying orchestra-compiler/include/Orchestra/CMakeLists.txt**

This file is responsible for defining the Orchestra dialect and triggering the TableGen generation of its operations. It also defines an INTERFACE library to expose the include path for the generated headers.

CMake

\# orchestra-compiler/include/Orchestra/CMakeLists.txt  
\# This CMakeLists.txt is processed by add\_subdirectory(include/Orchestra)

\# Define the dialect and generate operation/dialect declarations  
\# This creates MLIROrchestraOpsIncGen target and OrchestraOps.h.inc, OrchestraOps.cpp.inc etc.  
\# The 'orchestra' is the C++ namespace and mnemonic.  
add\_mlir\_dialect(OrchestraOps orchestra)

\# For custom attributes or types (if OrchestraAttrDefs.td or OrchestraTypes.td exist)  
\# These would be generated in ${CMAKE\_CURRENT\_BINARY\_DIR} (e.g., build/include/Orchestra)  
\# set(LLVM\_TARGET\_DEFINITIONS OrchestraAttrDefs.td)  
\# mlir\_tablegen(OrchestraAttrDefs.h.inc \-gen-attrdef-decls \-attrdefs-dialect=orchestra)  
\# mlir\_tablegen(OrchestraAttrDefs.cpp.inc \-gen-attrdef-defs \-attrdefs-dialect=orchestra)  
\# add\_public\_tablegen\_target(MLIROrchestraAttrDefsIncGen)

\# Define a target for the public headers of the dialect  
\# This ensures that other targets can link against this and get the correct include paths  
add\_library(OrchestraHeaders INTERFACE)  
target\_include\_directories(OrchestraHeaders INTERFACE  
    \# Path to generated headers in build tree, relative to this CMakeLists.txt's binary dir  
    $\<BUILD\_INTERFACE:${CMAKE\_CURRENT\_BINARY\_DIR}\>  
    \# Path to installed headers relative to CMAKE\_INSTALL\_PREFIX  
    $\<INSTALL\_INTERFACE:include/Orchestra\>  
)

### **4.3. Modifying orchestra-compiler/lib/Orchestra/CMakeLists.txt**

This file defines the library that contains the C++ implementations of the Orchestra dialect and the DivergenceToSpeculation pass. It is crucial for this file to correctly declare dependencies on the TableGen generation targets and link against the OrchestraHeaders INTERFACE library.

CMake

\# orchestra-compiler/lib/Orchestra/CMakeLists.txt  
\# This CMakeLists.txt is processed by add\_subdirectory(lib/Orchestra)

\# Define the dialect library, including its C++ sources  
\# and dependencies on the TableGen generated headers  
add\_mlir\_dialect\_library(MLIROrchestra  
    SOURCES  
        OrchestraDialect.cpp  
        DivergenceToSpeculation.cpp  
    DEPENDS  
        \# Ensure operations and dialect headers are generated before compilation.  
        \# This target is created by add\_mlir\_dialect(OrchestraOps orchestra) in include/Orchestra/CMakeLists.txt.  
        MLIROrchestraOpsIncGen  
        \# If you have custom attributes/types, add their IncGen targets here too:  
        \# MLIROrchestraAttrDefsIncGen  
        \# MLIROrchestraTypesIncGen  
    LINK\_COMPONENTS  
        Core \# Link against core MLIR components  
    LINK\_LIBS  
        PUBLIC MLIRIR \# Core MLIR infrastructure  
        PUBLIC MLIRPass \# Pass management framework  
        PUBLIC MLIRTransforms \# Common MLIR transformations  
        PUBLIC MLIRSupport \# General MLIR support utilities  
        \# Link the OrchestraHeaders INTERFACE library to propagate the correct include paths  
        \# for both hand-written and TableGen-generated headers.  
        PUBLIC OrchestraHeaders  
)

### **4.4. Correct C++ Inclusion in DivergenceToSpeculation.cpp**

The \#include path within DivergenceToSpeculation.cpp should be relative to the directories added to the include path by the OrchestraHeaders INTERFACE library. Given that OrchestraOps.h.inc is generated into build/include/Orchestra/ and OrchestraHeaders exposes build/include/Orchestra/ (or include/Orchestra for install), the C++ inclusion should use the path relative to that exposed directory.

The correct inclusion for using TaskOp and other operations from the Orchestra dialect is:

C++

// orchestra-compiler/lib/Orchestra/DivergenceToSpeculation.cpp

\#**include** "Orchestra/OrchestraOps.h.inc" // This header contains the C++ declarations for TaskOp

//... other includes...

// When using the operation, ensure the fully qualified name is used  
// as defined by the 'cppNamespace' in your Orchestra dialect's.td file.  
rewriter.create\<orchestra::TaskOp\>(/\*... arguments... \*/);

The step-by-step application of these CMake patterns provides a concrete, reusable template for future out-of-tree MLIR dialect and pass development. This approach directly translates the theoretical concepts of generator expressions, explicit DEPENDS relationships, and INTERFACE libraries into actionable CMakeLists.txt modifications for the user's specific project. This direct application is crucial for the user to understand and implement the solution. The pattern of defining an INTERFACE library for headers in the include subdirectory and then linking it in the lib subdirectory is a highly composable and canonical approach that can be reused for other dialects or components. This detailed walkthrough not only solves the immediate problem but also provides a reusable template for future out-of-tree MLIR dialect and pass development, significantly enhancing productivity and reducing future build-related frustrations. It solidifies the understanding of how MLIR's build system is designed to integrate modular components.

## **5\. Troubleshooting and Best Practices**

Developing with MLIR and CMake can present several common challenges, particularly when integrating out-of-tree components and TableGen-generated code. Understanding the root causes of these issues is key to efficient resolution. This section provides a diagnostic framework that mirrors the layered nature of compilation (pre-processing, compilation, linking, runtime), enabling systematic debugging of future build issues.

### **5.1. Diagnosing and Resolving "Not a member of" or "Undeclared Identifier" Errors**

These errors typically indicate that the C++ compiler cannot locate the necessary declarations for classes, functions, or members that are expected to be present, often originating from TableGen-generated files.1 The most frequent cause is that the include path to the TableGen-generated

.h.inc files is not correctly added to the target's compilation flags, or the generation step itself has not completed before compilation.1

To diagnose and resolve these issues:

* **Review target\_include\_directories**: Verify that the directory containing the generated .h.inc files (e.g., build/include/Orchestra/) is correctly added to the PUBLIC or PRIVATE include paths of the target that is failing to compile.1 The behavior of  
  add\_subdirectory() and CMAKE\_CURRENT\_BINARY\_DIR dictates where these files are placed, and the relative path from the *consumer's* CMAKE\_CURRENT\_BINARY\_DIR is critical.  
* **Check DEPENDS on TableGen Target**: Ensure that the C++ library or executable target (e.g., MLIROrchestra) explicitly declares a dependency on the TableGen generation target (e.g., MLIROrchestraOpsIncGen).1 Without this explicit dependency, CMake might attempt to compile C++ files before  
  mlir-tblgen has produced the required headers, leading to the observed errors.  
* **Verify \#include Path in C++ Source**: Double-check that the \#include directives in your C++ files use the correct relative path for the generated headers (e.g., \#include "Orchestra/OrchestraOps.h.inc").1 Incorrect paths, even if the directory is included, will lead to errors.  
* **TableGen Generation Failure**: Examine the CMake output for any errors or warnings during the mlir-tblgen execution.1 If TableGen itself fails, the  
  .h.inc files will not be created, leading to subsequent compilation errors. TableGen's diagnostics for unimplemented functions or contract violations can sometimes be subtle, requiring careful review of the TableGen documentation or source code.1

### **5.2. Addressing Linker Errors (e.g., "undefined reference to...")**

Linker errors occur when the compiled code requires symbols (functions, variables) from a library that is not being linked or is linked incorrectly.1

* **Missing LINK\_LIBS or LINK\_COMPONENTS**: Review the target\_link\_libraries() calls for your executable or library. Ensure all required MLIR components (e.g., MLIRIR, MLIRPass, MLIRTransforms, MLIRSupport) and your custom dialect library (MLIROrchestra) are explicitly listed.1  
* **Incorrect PUBLIC/PRIVATE/INTERFACE Scope**: If your custom pass library depends on your dialect library, and a tool links against your pass library, the dialect library must be linked with PUBLIC scope to ensure its symbols are transitively available to the tool.1  
* **Dialect/Pass Not Registered at Runtime**: Even if all libraries are correctly linked, the MLIR framework needs to be explicitly informed about your custom dialect and pass. Verify that registry.insert\<mlir::orchestra::OrchestraDialect\>(); and mlir::PassRegistration\<mlir::orchestra::DivergenceToSpeculationPass\>(); (or similar registration calls) are present and correctly executed in your tool's main function.1 This explicit registration is a crucial runtime step that complements the CMake build configuration.

### **5.3. General Best Practices for Robust MLIR CMake Configurations**

To ensure the long-term maintainability and scalability of out-of-tree MLIR projects, the following best practices are recommended:

* **Mirror MLIR's Internal Structure**: Adopting a directory layout similar to MLIR's in-tree organization (e.g., include/Orchestra, lib/Orchestra, tools/orchestra-opt) enhances familiarity for developers and improves overall project clarity and maintainability.1  
* **Leverage add\_subdirectory() for Modularity**: Break down the project into logical CMake subdirectories. This modular approach simplifies the management of build rules, dependencies, and source files, making the project easier to navigate and extend.1  
* **Consistently Use CMake Generator Expressions**: Always employ $\<BUILD\_INTERFACE\> and $\<INSTALL\_INTERFACE\> for specifying include directories and other usage requirements.1 This practice is fundamental for creating robust and relocatable packages that can be easily distributed and used across different systems without requiring specific directory structures.  
* **Be Explicit with Dependencies**: Clearly define DEPENDS for all TableGen-generated files and specify LINK\_LIBS/LINK\_COMPONENTS for all required libraries.1 This explicit declaration of dependencies makes the build graph transparent and helps prevent subtle build failures.  
* **Minimize Linked Libraries**: When building custom MLIR tools, avoid over-linking. Explicitly link only the MLIR components and other libraries that are strictly necessary for the tool's functionality.1 This practice reduces the size of the final executable and can significantly improve build times.  
* **Stay Updated with MLIR Documentation**: The MLIR API and CMake best practices can evolve. Regularly consult the official MLIR documentation and analyze upstream examples to stay informed of the latest recommendations and potential API breaking changes.1

## **Conclusion**

The problem of correctly configuring CMake for out-of-tree MLIR passes that depend on TableGen-generated dialect headers, as exemplified by the "error: ‘TaskOp’ is not a member of ‘orchestra’" compilation error, is a common challenge that stems from specific nuances in MLIR's build system and CMake's dependency management. The analysis presented in this report provides a canonical solution that addresses this fundamental disconnect.

The core of the resolution lies in a precise understanding and application of two critical CMake mechanisms:

1. **Correct Include Path Management with Generator Expressions**: Ensuring that the C++ compiler can find the TableGen-generated .h.inc files requires meticulously adding their binary directory to the include path of the consuming target. The use of an INTERFACE library (e.g., OrchestraHeaders) combined with CMake generator expressions ($\<BUILD\_INTERFACE:${CMAKE\_CURRENT\_BINARY\_DIR}\> and $\<INSTALL\_INTERFACE:include/Orchestra\>) is the robust and relocatable pattern. This approach correctly handles the dynamic nature of build paths in both development and installed environments.  
2. **Explicit Build Order Dependencies**: The C++ compilation of files like DivergenceToSpeculation.cpp must be guaranteed to occur *after* mlir-tblgen has successfully generated the necessary headers. This is achieved by explicitly declaring a DEPENDS relationship on the TableGen generation target (e.g., MLIROrchestraOpsIncGen) within the add\_mlir\_dialect\_library call. This prevents race conditions and ensures that the required files are present when the compiler needs them.

MLIR's custom CMake macros, such as add\_mlir\_dialect and add\_mlir\_dialect\_library, are not merely conveniences; they encapsulate complex build-system logic and best practices. Leveraging these macros, rather than attempting to manually replicate their underlying functionality, is crucial for building robust and maintainable out-of-tree MLIR components. The "single source of truth" philosophy, central to MLIR's declarative IR definition, extends to the build system, and correctly configuring CMake restores this seamless translation from declarative definition to executable code.

By adhering to the canonical CMake patterns outlined in this report—specifically, by correctly configuring the top-level project, setting up the dialect header generation with an INTERFACE library, and ensuring proper dependencies and include path propagation in the dialect's C++ library—developers can effectively resolve the described compilation errors. This structured approach not only provides a solution to the immediate problem but also offers a reusable template and diagnostic framework for future out-of-tree MLIR development, fostering a more productive and less error-prone workflow.

