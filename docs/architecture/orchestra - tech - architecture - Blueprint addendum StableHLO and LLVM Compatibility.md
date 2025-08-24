

# **Engineering Blueprint: Resolving the StableHLO Dependency Conflict in OrchestraOS**

## **1\. Analysis of the StableHLO Dependency Conflict in the OrchestraOS Context**

This report provides a comprehensive analysis of the versioning and dependency conflict between the OrchestraOS compiler's strategic architectural vision and the logistical challenges of integrating the StableHLO toolchain. The core problem, as articulated in the query, is the apparent incompatibility of building OrchestraOS with a modern MLIR/LLVM v20+ environment while simultaneously integrating StableHLO, a library that is explicitly pinned to an older, fixed LLVM commit. This analysis confirms that the stated technical conflict is both real and substantial, but argues that it stems from a misunderstanding of the correct integration model, rather than the choice of StableHLO itself.

### **1.1 Deconstructing the LLVM/MLIR Versioning Problem: A Foundational Incompatibility**

The observation that working with a StableHLO library built against an older MLIR/LLVM version (e.g., v18) alongside a project built on a newer version (e.g., v20+) is effectively "impossible" is a precise and accurate diagnosis of a fundamental C++ application binary interface (ABI) and application programming interface (API) conflict. The LLVM project, which serves as the foundational infrastructure for MLIR, operates on an aggressive, rapid development cycle where backward compatibility is not a primary concern for its C++ APIs.1 While the LLVM IR itself and the MLIR bytecode format have strong compatibility guarantees, the C++ interfaces that developers use to build and transform MLIR-based compilers are not guaranteed to be stable across major releases.2 This policy, while necessary to facilitate rapid innovation and architectural evolution, means that libraries cannot be dynamically linked across different major versions of LLVM without a high probability of runtime errors, symbol conflicts, and undefined behavior.

The StableHLO project's build process explicitly illustrates this dependency pinning. The repository's build scripts mandate a checkout of a specific, fixed commit hash from the LLVM monorepo (llvm\_version.txt).4 This is a necessary and standard practice for open-source projects that rely on a rapidly evolving upstream. By pinning to a known-good commit, the StableHLO team ensures that its codebase remains correct and that its own continuous integration (CI) tests for backward and forward compatibility remain valid.5 The conflict arises when a downstream consumer, such as OrchestraOS, attempts to link this version-locked StableHLO C++ library into a different, more modern LLVM environment. This is the root cause of the incompatibility and the source of the user's frustration.

### **1.2 The "Compiler-to-Compiler" Strategy and its Foundational Dependency on StableHLO**

The architectural vision of OrchestraOS is to function as a "meta-OS" for heterogeneous hardware, intelligently orchestrating workloads across different vendors' accelerators.6 To achieve this, the blueprint correctly identifies a "compiler-to-compiler" integration strategy for non-NVIDIA targets like Google TPUs and AWS Trainium.6 The core principle of this strategy is to perform high-level, hardware-agnostic optimizations within the OrchestraIR dialect, and then lower to a common, well-defined intermediate representation (IR) that is accepted as input by the vendor's own, highly-optimized compiler backend.

In this context, the selection of StableHLO as this common IR is not just a valid choice; it is the definitive, state-of-the-art solution. StableHLO is explicitly designed and maintained by the OpenXLA project to serve as a "portability layer between different ML frameworks and ML compilers".4 Its purpose is to unify the disparate world of machine learning frameworks (e.g., PyTorch, JAX) and hardware backends (e.g., XLA for TPUs, Neuron Compiler for Trainium) by providing a single, stable, and well-specified intermediate language.7 Therefore, the user's initial choice of StableHLO was strategically sound and aligns with the prevailing direction of the ML compiler ecosystem. The problem is not with the what, but with the how. The user's query about potentially abandoning StableHLO reveals a misunderstanding of the integration model. The solution is not to bypass StableHLO, but to leverage its design as a tool, not a library, thereby avoiding the C++ versioning problem entirely.

## **2\. Foundational Principles of MLIR and OpenXLA Versioning: The Path to a Solution**

A definitive solution to the versioning problem requires a deeper understanding of the MLIR and StableHLO compatibility frameworks. The problem, while appearing to be a direct C++ ABI conflict, is solved at a higher level of abstraction through the use of standardized serialization formats and dedicated compatibility dialects. The path forward is to abandon the assumption of a direct C++ library link and instead leverage the official, tool-based interoperability mechanism that the StableHLO project has already implemented.

### **2.1 A Primer on MLIR's Bytecode Format and Dialect Versioning**

At its core, MLIR provides a powerful mechanism for serializing its intermediate representation into a compact binary format known as MLIR bytecode.2 This format was designed to provide the benefits of a binary representation—namely, improved serialization speed, reduced file size, and memory mapping capabilities—while also supporting a stable, versioned schema.2 The foundational promise of the bytecode format is that it is backward-compatible, meaning an MLIR dialect should be able to deserialize any older bytecode representation of itself.2 This provides a stable base for the evolution of the IR.

However, a critical distinction must be made between the core MLIR bytecode infrastructure and the dialects built on top of it. While the underlying format is stable, the operations, types, and attributes of a specific dialect are not guaranteed to be immutable.2 A dialect can change its internal structure, its operation semantics, or its attribute schemas in ways that could break backward compatibility if a consumer were to rely on its in-memory C++ representation. This is precisely the issue faced by a project attempting to link C++ libraries from different MLIR versions. The MLIR bytecode format provides a necessary foundation for stability, but it is not a complete, out-of-the-box solution for dialect-level compatibility across major versions.

### **2.2 The StableHLO Compatibility Framework: Guarantees and Limitations**

Recognizing the limitations of MLIR's base C++ APIs and the need for long-term stability in the machine learning ecosystem, the StableHLO project implemented its own rigorous compatibility framework on top of the MLIR bytecode format.9 This framework is the explicit solution to the very problem the user is facing. The centerpiece of this framework is the concept of a "portable artifact".10 A portable artifact is an MLIR bytecode file that has been serialized in a very specific way, with a version stamp that provides strong compatibility guarantees. The StableHLO project guarantees

**five years of backward compatibility** and **two years of forward compatibility** for these artifacts.10 This means that a StableHLO program serialized today will be correctly deserialized by a StableHLO library from five years in the future, and a program serialized by a future library can be read by a current library for at least two years.

This robust versioning is made possible by the VHLO (Versioned HLO) dialect, which serves as a compatibility layer on top of StableHLO.5 The VHLO dialect provides a snapshot of the StableHLO dialect at a given point in time, versioning individual program elements to ensure that older artifacts can be correctly interpreted and upgraded by newer tools.5 This architecture enables the

stablehlo-translate tool to perform a version-aware serialization and deserialization, converting a program into a portable artifact and back again.10 The

stablehlo-translate tool is the official, supported mechanism for creating these artifacts.10 The existence of this battle-tested, formally specified compatibility framework means that the OrchestraOS team does not need to solve the C++ ABI problem from scratch. The solution is to leverage this existing, proven mechanism by treating the

stablehlo-translate tool as the official, version-aware bridge between the OrchestraOS compiler and the OpenXLA ecosystem.

## **3\. Proposed Architectural Solution: The StableHLO Bridge Library**

The definitive solution to the LLVM version conflict is to abandon the assumption of a direct C++ library link to StableHLO and to instead implement a "StableHLO Bridge Library" that serves as a thin, version-agnostic orchestrator. This architectural pivot not only solves the immediate technical problem but also strengthens the core "compiler-to-compiler" vision of OrchestraOS by treating the OpenXLA ecosystem as a robust, external toolchain.

### **3.1 Architectural Vision: A Decoupled, Version-Agnostic Interface**

The principle guiding this new architecture is decoupling. The OrchestraOS compiler, which is built on modern MLIR v20+, will perform all of its high-level, proprietary orchestration and optimization. When it is time to hand off the IR to the OpenXLA toolchain, it will do so not via a C++ function call but via an inter-process communication that relies on a stable, versioned artifact format. The "StableHLO Bridge" will be a lightweight component responsible for managing this handoff. It is not a C++ library that must be linked; rather, it is a wrapper around the stablehlo-translate tool. This approach elegantly pushes the LLVM versioning problem outside the OrchestraOS build, localizing it to a minimal, independent component that is easily managed.

### **3.2 The Core Component: The stablehlo-translate-based Bridge**

The primary workflow for this architectural solution involves a series of well-defined steps:

1. **Lowering to StableHLO Textual IR:** A new, dedicated pass, orchestra-lower-to-stablehlo, is introduced into the OrchestraOS pipeline. This pass will take the optimized, high-level linalg dialect and convert it into the StableHLO dialect's textual representation. This pass is part of the OrchestraOS compiler and is therefore built against MLIR v20+, which is not an issue since it is only generating text, not interacting with a foreign C++ library.  
2. **External Tool Invocation:** The OrchestraOS build system or compiler driver then invokes the stablehlo-translate command-line tool as a subprocess. This tool, crucially, is the one built and pinned against the older, fixed LLVM commit, which is a manageable, standalone dependency.10  
3. **Versioned Serialization:** The StableHLO textual IR is piped into the standard input of stablehlo-translate. The tool's primary function is to serialize the program into a portable MLIR bytecode artifact, explicitly using a target version string (e.g., stablehlo-translate \--serialize file.mlir \--target=1.0.0 \> portable\_artifact.mlir.bc).10  
4. **Handoff to Vendor Toolchain:** The resulting portable bytecode artifact is a self-contained, version-stable binary. It is this artifact that is then handed off to the final vendor compiler (e.g., Google's xla\_compile or AWS's neuronx-cc), which can consume it without any C++ linking or versioning issues.

A reverse workflow is also possible and highly valuable for debugging. The stablehlo-translate \--deserialize command allows a developer to convert a portable artifact back into a human-readable textual MLIR format.10 This allows the team to inspect the exact IR being passed to the vendor compiler, which is critical for diagnostics and performance tuning. This elegant architectural separation provides both correctness and debuggability.

### **3.3 Implementation Details: Workflows and Tooling for OrchestraOS**

Implementing the StableHLO Bridge requires a clear, well-defined workflow. First, the team must maintain a separate, version-locked build of the openxla/stablehlo repository. However, the build target is minimal: only the stablehlo-translate tool and its required dependencies need to be built, not the entire C++ library. This significantly reduces the build complexity and time. This small, isolated dependency can be managed via a separate CI job that is only triggered when the llvm\_version.txt file in the stablehlo repository changes.4 The OrchestraOS build system (e.g., CMake) can then be configured to locate and invoke this pre-built binary. The use of standard library functionality for subprocess management will ensure that the bridge is portable and does not introduce new dependencies or C++ version conflicts. This minimal, external dependency isolates the versioning risk and allows the main OrchestraOS compiler to continue its development and maintain its dependency on the latest MLIR/LLVM versions without friction.

### **3.4 Trade-off Analysis: The Bridge vs. Direct Integration**

The proposed architectural shift is a strategic choice with tangible benefits and trade-offs. The following table provides a formal comparison of the stablehlo-translate-based bridge against the naive approach of a direct C++ library link.

| Feature | stablehlo-translate-based Bridge | Direct C++ Library Link |
| :---- | :---- | :---- |
| **Build Complexity** | **Low.** The bridge is a thin, easily managed subprocess wrapper. The stablehlo-translate binary is a pre-built, version-locked artifact. | **High.** Requires managing multiple LLVM versions, complex linker flags, and high risk of ABI/API conflicts. Leads to significant "dependency hell." |
| **Versioning Risk** | **None.** The communication relies on a version-stable bytecode format with a 5-year backward compatibility guarantee. The version-locked dependency is external and isolated. | **Critical.** Inevitable C++ ABI conflicts and API breakages will occur with every major LLVM release, making the compiler brittle and difficult to maintain. |
| **Performance Overhead** | **Negligible.** Subprocess communication has a small fixed overhead, but the conversion to a portable artifact is a one-time operation per program. | **Zero.** The most performant method, but only if the versions align, which is a logistical impossibility for a long-term project. |
| **Maintainability** | **High.** The architecture separates concerns cleanly. The OrchestraOS team focuses on high-level passes, while the stablehlo-translate team handles versioning and serialization. The bridge itself is trivial to maintain. | **Low.** Every major LLVM update requires a non-trivial engineering effort to re-align C++ APIs and resolve conflicts, slowing down the development cycle. |
| **Alignment with Vision** | **High.** Aligns perfectly with the "compiler-to-compiler" strategy and the notion of OrchestraOS as a high-level orchestrator that delegates to best-in-class backends. | **Low.** Conflicts with the core principle of a clean, decoupled architecture. |

The comparative analysis clearly demonstrates that the proposed bridge architecture is the superior choice. It is the only path that ensures long-term architectural stability, minimizes the maintenance burden, and aligns with the strategic vision of the OrchestraOS platform.

## **4\. Strategic Evaluation of Alternative Integration Paths**

To fully address the user's inquiry, it is necessary to rigorously evaluate and formally reject the most plausible alternative integration paths. This analysis confirms that the proposed stablehlo-translate-based bridge is not just a viable solution, but the optimal one.

### **4.1 Option A: Bypassing StableHLO and Targeting a Different IR**

A potential alternative would be to bypass StableHLO entirely and lower the OrchestraIR to a different intermediate representation for handoff to the vendor-specific compilers. This is not a tenable strategy for several critical reasons. The OpenXLA ecosystem, which includes Google's XLA compiler and AWS's Neuron Compiler, has officially standardized on StableHLO as its primary and preferred input format.7 Bypassing StableHLO would require OrchestraOS to reverse-engineer or implement a proprietary, unstable, and undocumented IR. This would mean forgoing the significant benefits of StableHLO's formal specification, its battle-tested compatibility guarantees, and the collective engineering effort of the OpenXLA community.13 Such an approach would directly contradict the strategic goal of leveraging external, best-in-class toolchains and would place a monumental, and likely intractable, implementation and maintenance burden on the OrchestraOS team.

### **4.2 Option B: Maintaining a Separate, Locked-Version LLVM Build**

The most straightforward, but least effective, alternative is to maintain a second, complete, and version-locked build of the LLVM toolchain. This would involve a full build of LLVM v18 alongside the modern LLVM v20+ build. A developer could then attempt to link to the appropriate libraries at the correct stages of compilation. This approach, however, is a classic example of "dependency hell." It introduces an immense build and testing burden, creates a significant risk of obscure runtime errors due to library conflicts, and requires an independent and ongoing effort to manage two separate sets of dependencies. This logistical nightmare would consume valuable engineering resources, directly impede the rapid iteration required for an ambitious project like OrchestraOS, and ultimately fail to provide a robust, long-term solution. The proposed bridge architecture, which isolates the versioning problem to a single, external binary, is a far more elegant and pragmatic solution.

## **5\. Revised OrchestraOS Compiler Pipeline and Integration Roadmap**

The final step is to synthesize the preceding analysis into a concrete, actionable roadmap for the OrchestraOS engineering team. This involves updating the compiler's pass pipeline to explicitly incorporate the new stablehlo-translate-based bridge.

### **5.1 A New linalg-to-stablehlo Lowering and the Bridge Stage**

The existing orchestra-to-linalg pass will remain unchanged, continuing to perform the core high-level orchestration and lowering of the proprietary IR. A new, dedicated pass, orchestra-lower-to-stablehlo, will be introduced to handle the conversion from the generic linalg dialect to the stablehlo dialect. This pass will ensure that the IR conforms to the StableHLO specification and is ready for serialization. Crucially, this pass will not be responsible for the serialization itself. Instead, it will produce the IR in memory or as a textual file, which is then handed off to a new pipeline stage. The "StableHLO Bridge Stage" is an external, non-MLIR pass that is implemented as a simple build script. This script will perform the subprocess invocation of stablehlo-translate, pipe the IR to its input, and manage the output of the portable bytecode artifact. This clean separation of concerns ensures that the core compiler logic remains independent of the external toolchain.

### **5.2 Integrating the StableHLO Bridge into the End-to-End Pipeline**

The revised multi-target compiler pipeline, as outlined in the original blueprint addendum, is updated here to reflect the inclusion of the StableHLO Bridge and the explicit handling of the versioning problem. The new pipeline provides a definitive guide for implementation, clearly showing how the system dispatches to the correct vendor-specific backend while leveraging the StableHLO toolchain for Google TPU and AWS Trainium targets.

| Stage | Input Dialect(s) | Key Transformation Passes | Output Dialect(s) |
| :---- | :---- | :---- | :---- |
| **1\. Ingestion** | torch, tf, onnx | Framework-specific Normalization, Functionalization, Shape Inference | func, linalg |
| **2\. Scheduling** | func, linalg, scf | Proprietary Topology-Aware Scheduling Pass | orchestra, scf |
| **3\. High-Level Opt.** | orchestra, scf, cf | \-lift-cf-to-scf, divergence-to-speculation (PDL-driven) | orchestra |
| **4\. Structured Lowering** | orchestra | orchestra-to-linalg, orchestra-transfer-to-dma | linalg, memref, tensor |
| **5\. Hardware Opt.** | linalg, memref, tensor | \-transform-interpreter (with target-specific transform script) | scf, vector, memref, tensor |
| **6\. Bufferization** | tensor, scf, vector | \-one-shot-bufferize | memref, scf, vector |
| **7\. Accelerator Lowering** | scf, vector, memref | Target-dispatch based on orchestra.task arch attribute: |  |
|  |  | NVIDIA Path: linalg-to-nvgpu, gpu-lower-to-nvvm-pipeline | gpu, nvvm, llvm |
|  |  | AMD Path: linalg-to-rock, rock-to-amdgpu | gpu, amdgpu, rocdl, llvm |
|  |  | **Google/AWS Path:** linalg-to-stablehlo (pass) | stablehlo (text) |
| **8\. Executable Gen.** | gpu, nvvm, llvm, stablehlo | NVIDIA/AMD: \-gpu-to-llvm, gpu-module-to-binary | llvm, gpu.binary |
|  |  | **Google/AWS:** **StableHLO Bridge Stage** (external tool invocation) → Handoff to Vendor Compiler (xla\_compile, neuronx-cc) | Vendor-specific binary (e.g., NEFF) |

### **5.3 Long-Term Strategy for Managing a Multi-Compiler Ecosystem**

The solution presented in this blueprint is more than a tactical fix for a versioning problem; it is a strategic architectural decision that enhances the long-term viability and leadership of the OrchestraOS platform. The primary takeaway is that the problem was not with the choice of StableHLO, but with an incorrect assumption about how to integrate it. The correct path is to embrace StableHLO's explicit design as a versioned, portable artifact format and to use its accompanying command-line tools as a first-class compiler primitive. This approach ensures that OrchestraOS remains a high-level orchestrator, delegating the final, target-specific lowering to the best-in-class, vendor-provided backends.6

By isolating the LLVM version conflict to a single, external binary, the OrchestraOS team can continue its rapid development on the latest MLIR infrastructure, take advantage of the newest features like the Properties system and the Action framework, and avoid the brittleness and maintenance overhead of a direct C++ library link to a version-locked dependency. This architecture positions OrchestraOS to thrive in the increasingly heterogeneous world of AI hardware, as it can seamlessly integrate with a growing number of compiler-as-a-service platforms that standardize on formats like StableHLO. This is the path to long-term technological leadership and scalability.

#### **Referenzen**

1. LLVM Developer Policy \- Documentation, Zugriff am August 24, 2025, [https://llvm.org/docs/DeveloperPolicy.html](https://llvm.org/docs/DeveloperPolicy.html)  
2. MLIR Bytecode Format, Zugriff am August 24, 2025, [https://mlir.llvm.org/docs/BytecodeFormat/](https://mlir.llvm.org/docs/BytecodeFormat/)  
3. LLVM IR Target \- MLIR, Zugriff am August 24, 2025, [https://mlir.llvm.org/docs/TargetLLVMIR/](https://mlir.llvm.org/docs/TargetLLVMIR/)  
4. openxla/stablehlo: Backward compatible ML compute opset inspired by HLO/MHLO \- GitHub, Zugriff am August 24, 2025, [https://github.com/openxla/stablehlo](https://github.com/openxla/stablehlo)  
5. \[RFC\] Increase StableHLO Compatibility Guarantees. \- Google Groups, Zugriff am August 24, 2025, [https://groups.google.com/a/openxla.org/g/openxla-discuss/c/rfd30zKR9uU](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/rfd30zKR9uU)  
6. orchestra \- tech \- Blueprint addendum  
7. StableHLO | OpenXLA Project, Zugriff am August 24, 2025, [https://openxla.org/stablehlo](https://openxla.org/stablehlo)  
8. OpenXLA Project, Zugriff am August 24, 2025, [https://openxla.org/](https://openxla.org/)  
9. StableHLO Bytecode \- OpenXLA Project, Zugriff am August 24, 2025, [https://openxla.org/stablehlo/bytecode](https://openxla.org/stablehlo/bytecode)  
10. stablehlo/docs/compatibility.md at main \- GitHub, Zugriff am August 24, 2025, [https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md](https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md)  
11. XLA Terminology | OpenXLA Project, Zugriff am August 24, 2025, [https://openxla.org/xla/terminology](https://openxla.org/xla/terminology)  
12. OpenXLA overall architecture & components \- Google Groups, Zugriff am August 24, 2025, [https://groups.google.com/a/openxla.org/g/openxla-discuss/c/DnPUmpyk4y0](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/DnPUmpyk4y0)  
13. Releases · openxla/stablehlo \- GitHub, Zugriff am August 24, 2025, [https://github.com/openxla/stablehlo/releases](https://github.com/openxla/stablehlo/releases)