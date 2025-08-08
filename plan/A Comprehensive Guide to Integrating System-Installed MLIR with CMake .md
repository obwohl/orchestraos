

# **A Comprehensive Guide to Integrating System-Installed MLIR with CMake on Debian/Ubuntu**

## **Section 1: The Canonical Solution for Debian/Ubuntu LLVM Installations**

When integrating a complex, multi-component framework like LLVM/MLIR into a custom project, the interaction between the project's build system (CMake), the framework's own build architecture, and the system's package manager can create non-obvious configuration challenges. The problem of find\_package(MLIR...) failing, despite the library being correctly installed via apt, is a classic example of this three-body problem. The solution is not to modify the project's CMakeLists.txt file, but rather to invoke CMake with a precise hint that respects the hierarchical nature of the LLVM project.

### **1.1 The Correct cmake Invocation**

For an LLVM/MLIR version 18 installation on Ubuntu/Debian located at /usr/lib/llvm-18/, the correct and sufficient command to configure the project is:

Bash

cmake.. \-DLLVM\_DIR=/usr/lib/llvm-18/lib/cmake/llvm

This command provides a single, targeted hint to CMake. It directly specifies the location of the directory containing the LLVMConfig.cmake file. This is the master configuration script for the entire LLVM project and its sub-projects. Critically, no corresponding hint for MLIR\_DIR is necessary or desirable. Providing hints for both can lead to conflicting or incomplete configuration states. By guiding CMake to the primary LLVM package, the build system can correctly resolve all sub-project dependencies, including MLIR, through the mechanisms established by LLVM's own CMake infrastructure.

### **1.2 The Corrected CMakeLists.txt**

The user's original CMakeLists.txt file is, in fact, already correct and follows modern CMake best practices. It does not require any modification. The order of the find\_package calls is the most important aspect.

CMake

cmake\_minimum\_required(VERSION 3.20)  
project(MyMLIRProject LANGUAGES CXX)

set(CMAKE\_CXX\_STANDARD 17)  
set(CMAKE\_CXX\_STANDARD\_REQUIRED ON)

\# Correct Order: Find the parent project (LLVM) first.  
find\_package(LLVM 18 REQUIRED CONFIG)  
\# Then, find the sub-project (MLIR).  
find\_package(MLIR 18 REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM\_PACKAGE\_VERSION}")  
message(STATUS "Found MLIR ${MLIR\_PACKAGE\_VERSION}")

add\_executable(my-tool main.cpp)

\# Link against the imported targets provided by MLIR's config script.  
target\_link\_libraries(my-tool PRIVATE  
  MLIRIR  
  MLIRSupport  
)

The success of this script is entirely dependent on the find\_package(LLVM...) call succeeding and properly configuring the environment for the subsequent find\_package(MLIR...) call. The failure reported by the user occurs because the second call cannot find its dependency (LLVM) in a properly configured state.

### **1.3 The Core Principle: The Primacy of the LLVM Package**

The foundational concept for resolving this issue is understanding that MLIR is not a peer of LLVM but a sub-project within it.1 When LLVM is built from source, MLIR is enabled via the

\-DLLVM\_ENABLE\_PROJECTS=mlir flag.1 This hierarchical relationship is preserved in the installed CMake package configuration files.

The LLVMConfig.cmake script is the designated entry point for any project wishing to consume LLVM libraries. It is responsible for two key actions:

1. Defining the imported targets and variables for the core LLVM components.  
2. Setting up the necessary CMake environment, including modifications to CMAKE\_MODULE\_PATH, so that the configuration files for its sub-projects (like MLIR and Clang) can be located and processed correctly.

Therefore, the idiomatic approach is to **find the parent, not the child**. A project developer's responsibility is to tell CMake where to find LLVM. LLVM's configuration scripts then take on the responsibility of telling CMake where to find MLIR. Attempting to locate MLIR directly, without first establishing the LLVM context, bypasses this essential setup phase and is the root cause of the failure.

## **Section 2: Deconstructing the find\_package Conundrum: CMake, LLVM, and Debian**

To fully grasp why the user's logical and well-informed attempts failed, it is necessary to examine the precise mechanics of three interacting systems: CMake's package search procedure, the specific structure of LLVM's CMake export architecture, and the non-standard installation layout used by the Debian/Ubuntu apt packages. The failure is not due to a single error but to a misalignment between these three systems.

### **2.1 The Anatomy of a find\_package Search in CONFIG Mode**

The find\_package command is a cornerstone of modern CMake, allowing projects to locate and use external dependencies.3 When invoked with the

CONFIG keyword (or when it fails to find a Find\<PackageName\>.cmake module), it enters "Config Mode" and searches for a package configuration file, typically named \<PackageName\>Config.cmake or \<lowercase-package-name\>-config.cmake.5

CMake follows a well-defined search procedure to locate this file, consulting a series of locations in a specific order of precedence.5 Understanding this hierarchy is crucial for debugging any package-finding issue.

1. **Package-Specific Root Variables:** CMake first checks for \<PackageName\>\_ROOT (and \<PACKAGENAME\>\_ROOT) as both a CMake variable and an environment variable. This has the highest precedence for a given package.5  
2. **CMake-Specific Cache Variables:** These are intended to be set on the command line via \-DVAR=VALUE. The most important is CMAKE\_PREFIX\_PATH, which can contain a semicolon-separated list of installation prefixes to search.7  
3. **CMake-Specific Environment Variables:** These are intended to be set in the user's shell. This includes \<PackageName\>\_DIR (which points directly to the directory containing the config file) and CMAKE\_PREFIX\_PATH.5  
4. **HINTS and PATHS:** Paths specified directly in the find\_package call with the HINTS or PATHS keywords.  
5. **Standard System Environment Variables:** The PATH environment variable is searched. Directories ending in /bin or /sbin have their parent directory added to the search prefixes.7  
6. **CMake Package Registries:** CMake maintains a User Package Registry (\~/.cmake/packages on Unix) and a System Package Registry where packages can register their locations.3  
7. **System Paths:** Finally, CMake searches standard system installation prefixes, such as /usr and /usr/local.

The following table summarizes the primary mechanisms for influencing this search, which is essential knowledge for any developer working with CMake dependencies.

| Mechanism | Type | Precedence | Best For |
| :---- | :---- | :---- | :---- |
| cmake \-D\<Pkg\>\_DIR=/path/to/config | CMake Cache Variable | Highest | Pinning a specific, non-standard build of a package, bypassing the entire search procedure. |
| cmake \-DCMAKE\_PREFIX\_PATH=/path/to/prefix | CMake Cache Variable | High | Pointing to a custom installation root (e.g., /opt/my\_libs, \~/local) that contains multiple packages. CMake will search standard subdirectories like lib/cmake/\<Pkg\> within this prefix. |
| export \<Pkg\>\_DIR=/path/to/config | Environment Variable | Medium | Configuring a build environment within a shell session without modifying the cmake command line. Overridden by the cache variable. |
| export CMAKE\_PREFIX\_PATH=/path/to/prefix | Environment Variable | Medium | Same as the cache variable, but for a shell session. |
| CMake Package Registry | Persistent Metadata | Low | Registering a custom-built library system-wide or user-wide so that find\_package can find it automatically without any hints. |
| Standard System Paths (/usr, /usr/local) | Default Behavior | Lowest | Finding packages installed by system package managers into standard locations. |

### **2.2 The LLVM Project's CMake Architecture: A Hierarchical Model**

The LLVM project's CMake architecture is not a flat collection of peer libraries; it is a sophisticated, hierarchical system designed to manage a core project (LLVM) and numerous sub-projects (like MLIR, Clang, LLD).1 This hierarchy is the key to understanding the configuration process.

When find\_package(LLVM...) is successfully executed, the LLVMConfig.cmake script does more than just define variables like LLVM\_INCLUDE\_DIRS and LLVM\_LIBRARIES. It performs a critical setup routine that prepares the CMake environment for all its sub-projects. A crucial part of this setup is populating the LLVM\_CMAKE\_DIR variable, which points to the directory containing helper CMake modules like AddLLVM.cmake and AddMLIR.cmake.8 Projects that build against LLVM/MLIR are expected to add this directory to their

CMAKE\_MODULE\_PATH:

CMake

\# This logic is what happens inside a project that correctly finds LLVM.  
list(APPEND CMAKE\_MODULE\_PATH "${LLVM\_CMAKE\_DIR}")  
list(APPEND CMAKE\_MODULE\_PATH "${MLIR\_CMAKE\_DIR}") \# MLIR\_DIR is set by LLVMConfig.cmake  
include(AddLLVM)  
include(AddMLIR)

Furthermore, the MLIRConfig.cmake file itself contains a non-negotiable dependency on LLVM, expressed through the find\_dependency macro. This macro is a wrapper around find\_package that propagates the REQUIRED and QUIET parameters from the parent call. Essentially, the MLIRConfig.cmake script begins with a statement equivalent to: find\_package(LLVM REQUIRED).

This creates a strict dependency chain: find\_package(MLIR) cannot succeed unless the CMake environment is already configured such that find\_package(LLVM) can succeed from within the MLIR configuration script. This is the "Find the Parent, Not the Child" principle in action. If find\_package(MLIR) is called in an environment where LLVM has not yet been found and configured, its internal find\_dependency(LLVM) call will fail, which in turn causes the top-level find\_package(MLIR) to fail with the "Could not find a package configuration file" error. The error message is about MLIR, but the root cause is the failure to find its dependency, LLVM.

### **2.3 The Debian/Ubuntu LLVM Package File Layout**

The final piece of the puzzle is the specific file layout chosen by the Debian/Ubuntu packagers for the official LLVM apt packages.11 Using

dpkg \-L llvm-18-dev and dpkg \-L libmlir-18-dev reveals the installation structure 13:

* **Installation Prefix:** /usr/lib/llvm-18/  
* **Libraries (.so):** /usr/lib/llvm-18/lib/  
* **Headers:** /usr/lib/llvm-18/include/  
* **CMake Config (LLVM):** /usr/lib/llvm-18/lib/cmake/llvm/LLVMConfig.cmake  
* **CMake Config (MLIR):** /usr/lib/llvm-18/lib/cmake/mlir/MLIRConfig.cmake

The most important observation is that the root installation prefix, /usr/lib/llvm-18/, is **not** a standard path that CMake searches by default. The default search paths are typically /usr, /usr/local, and other FHS-compliant locations. This non-standard layout is the fundamental reason why find\_package fails without any hints. CMake simply does not know to look inside /usr/lib/llvm-18/.

### **2.4 A Forensic Analysis of Failed Attempts**

With a clear understanding of the three interacting systems, it is now possible to perform a forensic analysis of each of the user's failed attempts.

| Attempt | Expected Behavior (by user) | Actual Outcome | Reason for Failure |  |
| :---- | :---- | :---- | :---- | :---- |
| cmake.. (no hints) | Fails, but for unknown reasons. | Failure. | CMake searches default system paths like /usr and /usr/local. It does not search the non-standard /usr/lib/llvm-18 directory, so it finds neither LLVMConfig.cmake nor MLIRConfig.cmake. |  |
| cmake.. \-DMLIR\_DIR=/usr/lib/llvm-18/lib/cmake/mlir | find\_package(MLIR) should directly find MLIRConfig.cmake in this directory. | Failure. | This correctly tells find\_package(MLIR) where to find MLIRConfig.cmake. However, that script immediately calls find\_dependency(LLVM). Since LLVM\_DIR has not been set and /usr/lib/llvm-18 is not a default search path, this internal dependency lookup fails. The find\_package(MLIR) call fails as a result of its own dependency failing. |  |
| cmake.. \-DMLIR\_DIR=... \-DLLVM\_DIR=... | Both packages should be found directly. | Failure. | This attempt seems logically sound, but it can fail due to the order of operations within CMake. The find\_package(LLVM) call must execute *before* the find\_package(MLIR) call. If the user's CMakeLists.txt has the correct order, this *might* work, but it is verbose and brittle. The canonical solution is simpler and more robust because it relies on the intended dependency mechanism. The failure here likely stems from find\_package(MLIR) being processed first or in a context where the effects of find\_package(LLVM) are not yet fully visible. |  |
| cmake.. \-DCMAKE\_PREFIX\_PATH=/usr/lib/llvm-18 | CMake should search this prefix and find both packages in lib/cmake/.... | Failure. | This is the most subtle failure. According to CMake's documentation 5, this | *should* work. CMake is expected to search \<prefix\>/lib/cmake/\<name\> for each package. The failure, as noted in the user's debug output, suggests a nuance in how the Debian-packaged scripts are configured or an edge case in CMake's search logic. The direct hint via \-DLLVM\_DIR is more explicit and bypasses this search procedure entirely, making it more reliable. The failure may also relate to the fact that find\_package(LLVM) must run first; setting CMAKE\_PREFIX\_PATH does not guarantee the order of discovery or the proper propagation of state between the two find\_package calls in the way that finding LLVM first does. |

The conclusion from this analysis is clear: the only robust and idiomatic method is to provide a single hint for the top-level LLVM project, allowing its own well-defined CMake infrastructure to handle the discovery of its sub-projects.

## **Section 3: Building a Robust Standalone MLIR Project**

Moving from diagnosis to prescription, this section provides the templates and best practices for creating a standalone C++ project that correctly consumes a system-installed MLIR.

### **3.1 A Gold-Standard CMakeLists.txt for MLIR Development**

The following CMakeLists.txt serves as a robust and well-commented template for a typical out-of-tree MLIR-based tool. It incorporates modern CMake practices and leverages the infrastructure provided by the LLVM/MLIR package configuration scripts.

CMake

\# Require a CMake version compatible with LLVM/MLIR's own requirements.  
\# LLVM 18 requires at least CMake 3.20. \[14\]  
cmake\_minimum\_required(VERSION 3.20)

\# Define the project.  
project(MyMLIRTool LANGUAGES CXX)

\# MLIR and LLVM require at least C++17. \[14, 15\]  
set(CMAKE\_CXX\_STANDARD 17)  
set(CMAKE\_CXX\_STANDARD\_REQUIRED ON)

\# \--- Package Discovery \---  
\# The order is critical. Find the top-level project (LLVM) first.  
\# The CONFIG keyword enforces "Config Mode" and is best practice. \[5, 6\]  
find\_package(LLVM 18 REQUIRED CONFIG)  
message(STATUS "Found LLVM ${LLVM\_PACKAGE\_VERSION} at ${LLVM\_DIR}")

\# Now that LLVM is found, its configuration script has prepared the  
\# environment for its sub-projects. Finding MLIR will now succeed.  
find\_package(MLIR 18 REQUIRED CONFIG)  
message(STATUS "Found MLIR ${MLIR\_PACKAGE\_VERSION} at ${MLIR\_DIR}")

\# \--- Configure Build Environment \---  
\# The find\_package scripts populate several important variables.  
\# Use these variables to configure the compiler.  
\# LLVM\_INCLUDE\_DIRS and MLIR\_INCLUDE\_DIRS contain the paths to the header files.  
include\_directories(  
  ${LLVM\_INCLUDE\_DIRS}  
  ${MLIR\_INCLUDE\_DIRS}  
)

\# LLVM\_DEFINITIONS contains necessary preprocessor defines, e.g., for  
\# controlling assertions or platform-specific features.  
add\_definitions(${LLVM\_DEFINITIONS})

\# \--- Define Executable Target \---  
add\_executable(my-tool main.cpp)

\# \--- Linking \---  
\# Use modern target\_link\_libraries with imported targets.  
\# This is the most important part of consuming the libraries correctly.  
\# The targets (e.g., MLIRIR) are defined by the MLIRConfig.cmake script.  
\# They automatically handle transitive dependencies.  
target\_link\_libraries(my-tool PRIVATE  
  \# Core MLIR libraries for IR manipulation and support utilities.  
  MLIRIR  
  MLIRSupport

  \# Library for parsing MLIR textual assembly format.  
  MLIRParser

  \# Library for the pass management infrastructure.  
  MLIRPass

  \# Add other MLIR components as needed, for example:  
  \# MLIRAnalysis  
  \# MLIRTransforms  
  \# MLIRConversion  
)

### **3.2 Linking MLIR Components Correctly**

The most significant advantage of using find\_package in CONFIG mode is its creation of IMPORTED targets.4 Instead of manually managing library paths (

\-L), library names (-l), include paths (-I), and preprocessor definitions (-D), an IMPORTED target encapsulates all of this information.

When the MLIRConfig.cmake script is processed, it defines a series of targets like MLIRIR, MLIRSupport, etc. These are not libraries that are built by the current project; they are aliases that represent the pre-built MLIR libraries on the system. The properties of these targets (e.g., INTERFACE\_INCLUDE\_DIRECTORIES, INTERFACE\_LINK\_LIBRARIES, INTERFACE\_COMPILE\_DEFINITIONS) contain all the necessary information for a consumer.

When a project uses target\_link\_libraries(my-tool PRIVATE MLIRIR), CMake automatically performs the following actions:

1. Adds the MLIRIR library to the link line.  
2. Reads the INTERFACE\_LINK\_LIBRARIES property of the MLIRIR target and transitively adds all of its dependencies to the link line. For example, MLIRIR depends on LLVMCore and LLVMSupport. These will be linked automatically without being explicitly mentioned.17  
3. Reads the INTERFACE\_INCLUDE\_DIRECTORIES property and adds the necessary include paths to the compilation command for my-tool.  
4. Reads the INTERFACE\_COMPILE\_DEFINITIONS property and adds the required preprocessor definitions.

This modern approach is vastly superior to older methods that involved manipulating global variables like CMAKE\_CXX\_FLAGS and LINK\_LIBRARIES. It provides target-specific, transitive dependency management that is robust and maintainable.

The following table provides a reference for some of the most common MLIR component libraries (which correspond to the imported target names) and their purpose.

| Imported Target Name | Purpose | Header Prefix |
| :---- | :---- | :---- |
| MLIRIR | Core IR data structures: Operation, Block, Region, Attribute, Type, MLIRContext. Essential for almost any MLIR tool. | mlir/IR/ |
| MLIRSupport | Core support libraries and data structures, often mirroring LLVM's. Includes mlir/Support/. | mlir/Support/ |
| MLIRParser | Functionality for parsing the MLIR textual assembly format (.mlir files). | mlir/Parser/ |
| MLIRPass | The pass management infrastructure: Pass, PassManager, OperationPass, etc. | mlir/Pass/ |
| MLIRTransforms | A collection of general-purpose transformation and optimization passes, like CSE and inlining. | mlir/Transforms/ |
| MLIRAnalysis | A collection of common analyses, such as alias analysis and dependence analysis. | mlir/Analysis/ |
| MLIRDialect | The base infrastructure for defining and using dialects. | mlir/Dialect/ |
| MLIRConversion | The infrastructure for dialect conversion passes. | mlir/Conversion/ |

To link against a specific dialect, such as the Func dialect, one would link against MLIRFuncDialect. For a conversion pass, one might link against MLIRArithToLLVM.

### **3.3 Managing Static vs. Shared LLVM/MLIR Builds**

The packages installed via apt from apt.llvm.org provide shared libraries (.so files) by default.11 This has several implications for a downstream project:

* **Smaller Executable Size:** The final my-tool binary will be relatively small, as it does not contain the LLVM and MLIR code itself.  
* **Runtime Dependency:** The executable requires the corresponding libLLVM-18.so, libMLIR-18.so, etc., to be present and findable by the system's dynamic linker at runtime. For a system where the libraries were installed via apt, this is generally not an issue.  
* **Deployment:** When deploying the tool to another machine, the LLVM/MLIR shared libraries must also be deployed or installed on the target system.

If a developer were to build LLVM from source, they could control the library type using the BUILD\_SHARED\_LIBS CMake option.19 Setting

\-DBUILD\_SHARED\_LIBS=OFF (the default) would produce static archives (.a files), leading to a large, self-contained executable with no runtime dependency on LLVM/MLIR libraries.

When *consuming* a pre-built LLVM/MLIR installation via find\_package, the choice of static versus shared has already been made. The IMPORTED targets created by LLVMConfig.cmake will point to either the .so files or the .a files, depending on how LLVM was built. The consuming project cannot change this; it simply uses the libraries as they were provided.

## **Section 4: Advanced Project Integration and Development Workflow**

Once the fundamental connection between a standalone project and the system-installed MLIR is established, the full power of the MLIR out-of-tree development ecosystem becomes available. This includes integrating custom dialects and passes using the same CMake infrastructure as MLIR itself, and setting up a professional-grade testing workflow with the LLVM Integrated Tester (lit).

### **4.1 Integrating Custom Dialects and Passes**

A primary use case for MLIR is the creation of custom dialects to represent domain-specific abstractions.22 The MLIR build system provides a suite of CMake functions to automate the boilerplate associated with this process, such as code generation from TableGen (

.td) files.8 Because the project has successfully found and included the MLIR CMake modules, these functions are now available for use.

This demonstrates a powerful aspect of the integration: the project is not merely linking against libraries, but is inheriting a domain-specific build system framework. This "out-of-tree development flywheel" allows external projects to behave as if they were part of the main llvm-project repository.

Consider a project structure for a custom "Foo" dialect:

my-mlir-tool/  
├── CMakeLists.txt  
├── main.cpp  
├── include/  
│   └── foo/  
│       ├── CMakeLists.txt  
│       ├── FooDialect.h  
│       └── FooOps.td  
└── lib/  
    └── foo/  
        ├── CMakeLists.txt  
        └── FooDialect.cpp

The CMakeLists.txt in include/foo/ would use add\_mlir\_dialect to handle the TableGen definitions 24:

CMake

\# include/foo/CMakeLists.txt  
add\_mlir\_dialect(FooOps foo)

This command automatically creates targets to invoke mlir-tblgen and generate files like FooOps.h.inc and FooOps.cpp.inc in the build directory.

The CMakeLists.txt in lib/foo/ would use add\_mlir\_dialect\_library to compile the C++ implementation of the dialect into a library 8:

CMake

\# lib/foo/CMakeLists.txt  
add\_mlir\_dialect\_library(MLIRFoo  
  FooDialect.cpp  
  DEPENDS  
  MLIRFooOpsIncGen \# Dependency on the generated headers  
  LINK\_LIBS  
  PUBLIC  
  MLIRIR  
)

This function is a wrapper around LLVM's add\_llvm\_library that registers the library as a dialect library. It creates a MLIRFoo library target.

Finally, the root CMakeLists.txt would add these subdirectories and link the main tool against the new custom dialect library:

CMake

\# Root CMakeLists.txt (additions)  
add\_subdirectory(include/foo)  
add\_subdirectory(lib/foo)

\#... in target\_link\_libraries for my-tool...  
target\_link\_libraries(my-tool PRIVATE  
  \#... other MLIR libraries...  
  MLIRFoo \# Link against our new custom dialect  
)

This structured, modular approach, enabled by the inherited CMake functions, is the standard and maintainable way to develop out-of-tree MLIR components.

### **4.2 Configuring Out-of-Tree Testing with lit**

Any serious compiler project requires an extensive test suite. The LLVM project, including MLIR, uses the LLVM Integrated Tester (lit) and FileCheck for its regression tests.26 Setting up

lit for an out-of-tree project can be challenging because it requires bridging the gap between the build environment (CMake) and the test execution environment (lit).28

The core mechanism is a two-stage configuration process. First, CMake processes a template configuration file (lit.site.cfg.py.in), injecting the paths to the just-built executables and the project source directory. Then, lit executes this generated Python script to configure its environment before discovering and running the tests.

**Step 1: Update Root CMakeLists.txt to Enable Testing**

CMake

\# Root CMakeLists.txt (additions)  
enable\_testing()

\# Find the lit executable. It is provided by the mlir-18-tools package.  
find\_program(LLVM\_LIT "llvm-lit-18")  
if(NOT LLVM\_LIT)  
  message(FATAL\_ERROR "llvm-lit-18 not found. Please install mlir-18-tools.")  
endif()

\# Configure the site configuration file for lit. This is the bridge.  
set(LIT\_TEST\_DIR ${CMAKE\_CURRENT\_BINARY\_DIR}/test)  
configure\_file(  
  ${CMAKE\_CURRENT\_SOURCE\_DIR}/test/lit.site.cfg.py.in  
  ${LIT\_TEST\_DIR}/lit.site.cfg.py  
  @ONLY  
)

\# Add a custom target to run the tests.  
add\_test(  
  NAME MyMLIRTool-test  
  COMMAND ${LLVM\_LIT} \-sv ${LIT\_TEST\_DIR}  
)  
\# Allow running tests with \`make check\` or \`ninja check\`  
add\_custom\_target(check  
  COMMAND ${CMAKE\_CTEST\_COMMAND} \--output-on\-failure  
  DEPENDS MyMLIRTool-test  
)

**Step 2: Create the lit Configuration Files**

Create a test/ directory in the project root.

test/lit.site.cfg.py.in (The CMake Template)  
This template contains placeholders (@VAR@) that CMake will replace.

Python

\# Autogenerated by CMake

import sys  
import lit.util

from lit.llvm import llvm\_config

\# name: The name of this test suite.  
config.name \= 'MyMLIRTool'

\# test\_source\_root: The root path where tests are located.  
config.test\_source\_root \= "@CMAKE\_CURRENT\_SOURCE\_DIR@/test"

\# test\_exec\_root: The root path where tests should be run.  
config.test\_exec\_root \= "@CMAKE\_CURRENT\_BINARY\_DIR@/test"

\# substitutions: A list of substitutions to make in test scripts.  
config.substitutions.append(('%my\_tool', "@CMAKE\_BINARY\_DIR@/my-tool"))

\# test\_format: The test format to use to interpret tests.  
config.test\_format \= lit.formats.ShTest(True)

\# Tweak the PATH to include the built tool directory.  
sys.path.insert(0, config.my\_tool\_obj\_root)

test/lit.local.cfg (The Local Test Config)  
This file tells lit how to run tests in this directory and its subdirectories.

Python

\# Tell lit how to run tests.  
config.test\_format \= lit.formats.ShTest(True)

\# Add substitutions for FileCheck.  
config.substitutions.append(('%FileCheck', '@LLVM\_FILECHECK\_EXECUTABLE@'))

*(Note: You would need to add find\_program(LLVM\_FILECHECK "FileCheck-18") to your CMakeLists.txt and pass it to the template)*

**Step 3: Write a Test Case**

Create a file like test/simple.mlir.

MLIR

// RUN: %my\_tool %s | FileCheck %s

// This is a dummy MLIR file for demonstration.  
// Suppose my-tool processes this and prints "Hello MLIR\!".

// CHECK: Hello MLIR\!

module {  
  func.func @main() {  
    return  
  }  
}

The RUN: line is a command for lit to execute. %my\_tool is a substitution defined in lit.site.cfg.py that expands to the full path of the executable. The output of this command is piped to FileCheck, which verifies that the output contains the string specified in the CHECK: line.27

With this setup, running cmake \--build. \--target check from the build directory will execute the entire test suite, providing a robust, automated, and maintainable workflow for developing MLIR-based tools.

## **Conclusions**

The challenge of configuring CMake to find system-installed MLIR on Debian/Ubuntu systems is not a result of user error or a flaw in CMake's standard logic, but rather a consequence of the specific interaction between three distinct systems: CMake's package search procedure, the LLVM project's hierarchical CMake architecture, and the non-standard installation layout of the official Debian apt packages.

The definitive solution is to provide a single, precise hint to CMake that respects this hierarchy: \-DLLVM\_DIR=/usr/lib/llvm-\<version\>/lib/cmake/llvm. This command correctly points find\_package to the master LLVM configuration script, which then assumes the responsibility of configuring the build environment for its sub-projects, including MLIR. Attempts to hint MLIR\_DIR directly or rely on CMAKE\_PREFIX\_PATH are brittle because they either bypass or fail to correctly trigger this essential, two-stage initialization process.

For developers building tools on top of MLIR, adopting this configuration approach is the gateway to the full out-of-tree development ecosystem. A correctly configured project not only gains the ability to link against MLIR libraries but also inherits the powerful, domain-specific CMake functions (add\_mlir\_dialect, add\_mlir\_dialect\_library) that automate and standardize the development of custom dialects and passes. Furthermore, by understanding how to bridge the build and test environments via a CMake-generated lit.site.cfg.py file, developers can implement a professional-grade regression testing workflow using the same tools (lit, FileCheck) as the LLVM project itself. Mastering this configuration is therefore a foundational step for any serious development effort within the MLIR ecosystem.
