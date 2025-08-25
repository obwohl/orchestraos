

# **The OrchestraOS Compiler: An Architectural Specification and Implementation Guide**

**Document Control**

* **Version:** 1.0  
* **Status:** Final  
* **Last Updated:** August 25, 2025  
* **Author(s):** OrchestraOS Architecture Group  
* **Reviewer(s):** Core Compiler Team, Performance Engineering Team, Engineering Leadership

---

## **Part I: Vision and Architectural Mandates**

This part establishes the strategic "Why" behind the OrchestraOS compiler. It grounds the project's technical decisions in market realities and defines the core principles that govern all subsequent design choices. This section is intended to justify the architecture to all stakeholders, from new engineers to executive leadership.1

### **1\. An Orchestration Engine for the Heterogeneous AI Era**

#### **1.1. The "Meta-OS" for AI: A Compiler-Centric Vision**

The foundational architectural vision of the OrchestraOS compiler is to serve as a "meta-OS" for artificial intelligence. This vision is centered on a compiler-driven strategy for orchestrating complex, heterogeneous hardware systems.4 The central tenet of this approach is the elevation of scheduling, data movement, and execution strategy to first-class citizens of the Intermediate Representation (IR). This directly addresses the primary challenge in modern high-performance computing, moving beyond traditional compilation where the compiler is a mere code generator. Instead, it repositions the compiler as the central intelligence layer responsible for managing the entire computational landscape.4

The selection of MLIR as the foundational infrastructure is an unequivocal and validated choice for this ambitious undertaking. MLIR's inherent modularity and its multi-level representation capabilities provide the essential framework required to model complex hardware abstractions, perform optimizations at multiple levels of granularity, and manage the progressive lowering of high-level computational graphs to hardware-specific machine code. This infrastructure is not merely a convenience but a prerequisite for realizing the compiler's role as the primary orchestration engine for the OrchestraOS platform.4

#### **1.2. Strategic Market Analysis (2025-2030)**

The engineering decisions to support a complex, multi-vendor hardware landscape represent a significant investment. This investment is justified by a rigorous, data-driven analysis of the AI training accelerator market, which reveals a clear trajectory from single-vendor incumbency to a multi-polar, ecosystem-driven competitive environment. The AI hardware market is projected to experience explosive growth, expanding from approximately $60-67 billion in 2025 to nearly $300 billion by 2034\.4 This immense market expansion fuels the competitive dynamics and validates the strategic importance of a multi-target compiler.

The competitive landscape is characterized by NVIDIA's entrenched dominance, with a market share consistently estimated between 80% and 95% for data center GPUs, largely due to the maturity of its CUDA software ecosystem.4 However, viable challengers are making significant inroads. AMD has emerged as a direct competitor with its Instinct accelerator series and open-source ROCm software stack. Concurrently, vertically integrated hyperscalers have created a new class of compilation target.4

The market data shows a clear trend away from a single-vendor monopoly towards a multi-polar ecosystem. This is not just about customer choice; it is a fundamental shift driven by hyperscalers seeking to control their own technological destiny, optimize for their specific workloads, and reduce dependency on any single supplier. Therefore, a compiler that only supports one vendor—even the dominant one—is strategically vulnerable, its relevance tied to the fortunes of that single vendor. A truly heterogeneous compiler like OrchestraOS, however, becomes an enabling technology for the entire ecosystem. It provides portability and performance across platforms, making it a more durable and strategically valuable asset. The existence of OrchestraOS is not a bet on a single hardware vendor, but on the continued growth and diversification of the AI hardware market itself. This multi-vendor support is not merely a feature; it is the central pillar of the project's long-term strategic viability and a key differentiator.

| Vendor | Estimated 2025 Market Share (%) | Projected 2030 Market Share (%) | Key Growth Drivers / Strategic Rationale |
| :---- | :---- | :---- | :---- |
| **NVIDIA** | 75 \- 85 | 60 \- 70 | Entrenched CUDA software ecosystem moat; Aggressive product roadmap (Blackwell, Rubin); Strong relationships with all major cloud providers and enterprises. 4 |
| **AMD** | 5 \- 10 | 10 \- 15 | Strong EPYC CPU momentum creating data center inroads; Open-source ROCm software stack as a CUDA alternative; Gaining adoption with hyperscalers for cost/performance. 4 |
| **Google (TPU)** | 5 \- 10 | 10 \- 15 | Vertical integration with Google Cloud Platform; Co-design of hardware and software (XLA compiler) for optimal performance on key workloads (LLMs, GenAI); Captive market of Google's internal services. 4 |
| **Amazon (Trainium)** | 3 \- 8 | 8 \- 12 | Dominant cloud market share (AWS); Focus on price-performance to drive adoption within the AWS ecosystem; Significant internal investment and adoption by key partners like Anthropic. 4 |
| **Other ASICs/FPGAs** | \< 5 | \< 5 | Niche applications, edge computing, and specialized inference workloads; Unlikely to capture significant training market share against vertically integrated or ecosystem-driven players. 4 |
| **Table 1: AI Training Accelerator Market Analysis & Strategic Rationale.** Note: Market share represents the portion of AI training workloads, not necessarily discrete unit sales. |  |  |  |

#### **1.3. The "Platform-as-a-Target" Paradigm**

A critical insight from the market analysis is the emergence of a new class of compilation target: the "Platform-as-a-Target".6 Unlike NVIDIA and AMD, which sell discrete hardware components, Google and Amazon offer access to an entire platform (GCP and AWS) where their custom silicon is a key, but integrated, element. They control the entire stack, from the physical chip design to the high-level ML framework bindings.6

Consequently, supporting AWS Trainium or Google TPUs is not merely a matter of generating machine code for a specific instruction set architecture. It is a matter of integrating with the AWS Neuron SDK and OpenXLA ecosystems, which serve as the designated, and in most cases exclusive, entry points to these platforms.6 This fundamentally alters the integration model for OrchestraOS. Instead of functioning as a traditional backend code generator that emits low-level machine code, the compiler must adopt a "compiler-to-compiler" integration strategy. In this model, OrchestraOS performs its high-level, proprietary orchestration and optimization, and then lowers its representation to the high-level IR accepted by the platform's native compiler (e.g., StableHLO), effectively using the vendor's compiler as its backend. This approach leverages the immense engineering investment the vendors have made in their own toolchains while allowing OrchestraOS to focus on its unique value proposition in cross-system orchestration.6

### **2\. Core Architectural Principles and Decision Records**

This section serves as a formal, high-level record of the foundational design philosophies and the non-negotiable architectural decisions that govern the OrchestraOS compiler. It is intended to be a stable reference for preventing architectural drift and ensuring future development is aligned with the core vision.8

#### **2.1. Principle 1: Declarative Control**

The architecture pivots from a traditional, imperative C++-driven pass pipeline to a more flexible and agile model orchestrated by the MLIR transform dialect. This paradigm shift decouples the *policy* of optimization (i.e., *what* transformations to apply and in *what* order) from the *mechanism* of transformation (the C++ code that implements the logic). This empowers performance engineers to script, tune, and rapidly iterate on complex optimization strategies without requiring compiler rebuilds, dramatically accelerating the performance tuning cycle.4

#### **2.2. Principle 2: Dynamic, Adaptive Optimization**

Recognizing that static, ahead-of-time (AOT) compilation is insufficient for the dynamic workloads of modern AI, the architecture elevates dynamic optimization to a first-class citizen. The proposed Feedback-Driven Optimization (FDO) loop is deeply integrated with MLIR's native "Action" tracing framework. By modeling the Just-In-Time (JIT) recompilation process as a formal Action, it becomes a transparent, traceable, and debuggable component of the compiler infrastructure, ensuring the dynamic behavior of the compiler is as robust and introspectable as its static components.4

#### **2.3. Principle 3: Integration via Stable Contracts**

A recurring pattern in the OrchestraOS architecture is the management of complexity through stable, versioned interfaces, replacing brittle, implementation-level dependencies. The co-design with AMD's rocMLIR relies on an IR pattern "contract".6 The integration with hyperscaler platforms relies on a "portable artifact" contract via StableHLO.11 The FDO loop relies on a Protobuf data contract (

DivergenceProfile) to communicate between the runtime and the JIT compiler.5

These are not isolated solutions but manifestations of a single, powerful architectural principle: **Integration via Stable Contracts**. By defining and adhering to well-specified, abstract interfaces, the compiler can decouple independently evolving systems. This principle provides a unifying theory for the design, helps engineers understand the rationale behind key decisions, and offers a guiding framework for future integration challenges.

#### **2.4. Decision Record: The Imperative of the StableHLO Bridge**

A critical architectural decision is the method of integration with the OpenXLA ecosystem for targeting Google TPUs and AWS Trainium. A direct C++ library link between OrchestraOS (built on modern MLIR v20+) and the StableHLO library (which is explicitly pinned to an older, fixed LLVM commit) is **effectively impossible**.7 This is not a temporary inconvenience but a fundamental C++ Application Binary Interface (ABI) and Application Programming Interface (API) instability inherent in the LLVM project's rapid development cycle, where backward compatibility of C++ APIs is not guaranteed across major releases.11

Attempting such a link would lead to intractable "dependency hell," symbol conflicts, and undefined behavior.11 Therefore, the definitive architectural solution is the implementation of a "StableHLO Bridge Library." This is not a linked C++ library but a thin, version-agnostic wrapper that invokes the standalone

stablehlo-translate command-line tool as a subprocess. This approach elegantly pushes the LLVM versioning problem outside the OrchestraOS build, localizing it to a minimal, independent component. It leverages StableHLO's explicit design as a tool for creating versioned, portable artifacts, which is the only architecturally sound and maintainable solution for long-term integration.11 This decision is final and serves to prevent future architectural debates on this topic.

### **3\. The Organizational Imperative: A Framework for Scalable Engineering**

Key technical choices within the compiler are not made in a vacuum; they have profound implications for the structure and agility of the engineering organization. The adoption of the transform dialect is a prime example of a technical decision that provides a strategic organizational advantage.5

#### **3.1. Decoupling Policy from Mechanism**

The transform dialect creates a clean separation between the stable *mechanism* of a transformation (the C++ implementation of a primitive like tiling, exposed as a transform dialect operation) and the rapidly evolving *policy* of its application (a target-specific MLIR script that orchestrates a complex optimization sequence).4

#### **3.2. Democratizing Performance Engineering**

This decoupling empowers performance engineers, who are experts in hardware architecture but not necessarily in C++ compiler internals, to rapidly iterate on optimization strategies. They can author new fusion strategies, adjust tile sizes, and experiment with new patterns simply by editing a text-based MLIR script. This breaks the dependency on the core compiler engineering team's release cycle, which is a common bottleneck in monolithic compiler architectures.5

#### **3.3. The Organizational Moat: Agility as a Competitive Advantage**

This agility is not just a convenience but a significant competitive advantage. In the fast-moving AI accelerator market, performance is a key differentiator. The ability to be the first to demonstrate state-of-the-art performance on a new chip is a major business win. A competitor architected around monolithic C++ passes has a fundamentally slower iteration cycle, as their organizational structure creates a bottleneck where performance experts must file tickets and wait for compiler engineers to implement and release new strategies.

The OrchestraOS architecture, by contrast, creates an "organizational moat." The ability to adapt is structurally superior because it decouples the lifecycles of the compiler's core infrastructure and its performance-tuning policies. This directly impacts time-to-market for achieving peak performance on new targets and represents a defensible, long-term advantage that is difficult for competitors to replicate without a similar architectural overhaul.5

---

## **Part II: The OrchestraIR Dialect: A Formal Specification**

This part provides the definitive specification for the core OrchestraIR dialect. It moves beyond a conceptual overview to provide the precise, versioned schemas and contracts necessary for implementation, directly addressing the critique that prior documentation was insufficient as an implementation guide.7

### **4\. Core Operations and Semantics**

The OrchestraIR dialect is the semantic core of the compiler, serving as the language through which system orchestration decisions are expressed and optimized. The following subsections provide a formal specification for each operation. The complete TableGen Operation Definition Specification (ODS) is available in the Appendix.

#### **4.1. orchestra.schedule**

* **Summary:** A top-level container for a scheduled Directed Acyclic Graph (DAG) of tasks.  
* **Syntax:**  
  MLIR  
  orchestra.schedule {  
    //... body region with orchestra.task operations...  
  }

* **Semantics:** Encapsulates a complete execution plan for a computational unit. The operations within its region form a DAG defined by SSA use-def chains.

#### **4.2. orchestra.task**

* **Summary:** An atomic unit of computation assigned to a specific hardware resource.  
* **Syntax:**  
  MLIR  
  %results \= orchestra.task \-\> (type,...) target \= {... } {  
    //... body region with computation...  
    orchestra.yield %values : type,...  
  }

* **Properties:**  
  * target (DictionaryAttr): A dictionary of attributes specifying placement constraints and hardware features. The formal schema is defined in Section 5\.  
* **Semantics:** Encapsulates a computation to be executed on a resource defined by its target property. The body region contains the operations to be executed.  
* **Modernization:** The target attribute is implemented using the MLIR Properties system for superior performance and type safety, eliminating the runtime overhead of generic dictionary lookups.4

#### **4.3. orchestra.transfer**

* **Summary:** A first-class operation representing the movement of data between two symbolic memory spaces.  
* **Syntax:**  
  MLIR  
  orchestra.transfer %source\_buffer to %dest\_buffer from @source\_mem to @dest\_mem

* **Operands:**  
  * source\_buffer, dest\_buffer (MemRefType or TensorType): The data being moved.  
* **Attributes:**  
  * from, to (SymbolRefAttr): References to symbols defining the logical memory spaces (e.g., @host\_dram, @gpu0\_hbm).  
* **Semantics:** Elevates data movement to an optimizable, first-class citizen. It abstracts the physical transfer, deferring its resolution to a later lowering stage. This provides a concrete handle for hardware-aware data movement optimizations.4

#### **4.4. orchestra.commit**

* **Summary:** A conditional selection operation that chooses one set of values from a combined list based on a condition.  
* **Syntax:**  
  MLIR  
  %results \= orchestra.commit %condition, %values

* **Operands:**  
  * condition (i1): The boolean condition.  
  * values (Variadic): A list containing values for both the 'true' and 'false' branches.  
* **Properties:**  
  * num\_true (IntegerAttr): The number of values from the start of the values list that belong to the 'true' branch.  
* **Semantics:** If the condition is true, the first num\_true values are returned; otherwise, the remaining values are returned. This structure is a well-established pattern for working around MLIR parsing limitations.4  
* **Modernization:** The num\_true attribute is implemented as a Property for efficiency and type safety.4

#### **4.5. orchestra.yield**

* **Summary:** A standard terminator for regions within OrchestraIR operations.  
* **Syntax:**  
  MLIR  
  orchestra.yield %values : type,...

* **Semantics:** Terminates a region and yields values to the parent operation, consistent with standard MLIR conventions.4

### **5\. A Formalized Schema for Multi-Vendor Targeting**

#### **5.1. Schema Rationale**

The target property of the orchestra.task operation serves a critical architectural function beyond simple metadata. It acts as the central dispatch mechanism that connects the hardware-agnostic OrchestraIR to the entire suite of hardware-specific backends. It is the polymorphic handle that drives the compiler's heterogeneous strategy, enabling the transform dialect framework to select the correct optimization script and the multi-path lowering pipeline to route the IR to the correct backend passes. This property operationalizes the compiler's core value proposition.4

#### **5.2. Base Schema (Version 1.0)**

The target property is a DictionaryAttr with the following mandatory keys:

* arch (StringAttr): The primary discriminator for dispatching to the correct lowering pipeline. Its value is a standardized string identifying the target architecture family (e.g., "nvidia\_blackwell", "amd\_cdna3", "google\_tpu\_v5e", "aws\_trainium2").  
* device\_id (IntegerAttr): The integer identifier for the specific device within the system.

#### **5.3. Architecture-Specific Extensions**

In addition to the base schema, the dictionary can contain optional, architecture-specific keys that provide fine-grained information to optimization and code generation passes. This extensible design allows transform scripts and lowering patterns to query specific hardware features or constraints.4

MLIR

// Example for an AMD Instinct MI300X GPU  
%result \= orchestra.task \-\> (tensor\<256x256xf32\>)  
  target \= {arch \= "amd\_cdna3", device\_id \= 0, mfma \= true, lds\_size \= 65536} {  
  //... linalg operations...  
  orchestra.yield %some\_value : tensor\<256x256xf32\>  
}

// Example for a Google Cloud TPU v5e  
%result \= orchestra.task \-\> (tensor\<1024x1024xbf16\>)  
  target \= {arch \= "google\_tpu\_v5e", device\_id \= 0, logical\_core\_count \= 4} {  
  //... linalg operations...  
  orchestra.yield %another\_value : tensor\<1024x1024xbf16\>  
}

### **6\. API Contracts and Data Schemas**

This section provides the formal, versioned data contracts required for inter-component communication, a critical missing piece for an implementation guide.7 These contracts are designed following best practices for API design, ensuring clarity, consistency, and stability.12

#### **6.1. The DivergenceProfile Contract (Version 1.0)**

* **Rationale:** This data contract is essential for decoupling the runtime profiling infrastructure from the JIT compiler. It allows the two components to evolve independently, as long as they both adhere to the agreed-upon schema.4 Furthermore, the schema is designed not only for the immediate needs of JIT recompilation but also for the long-term strategic goal of offline data collection. The aggregated  
  DivergenceProfile data from thousands of deployments forms a unique, proprietary dataset linking IR patterns to real-world performance. This dataset can be used to train a "meta-optimizer" to improve the AOT compiler's heuristics, creating a powerful feedback loop and a compounding performance advantage over time. The schema therefore includes fields that are critical for this future ML-based AOT optimization.4  
* **Format:** Protocol Buffers (Protobuf) is specified as the serialization format for its efficiency, strong typing, and schema evolution capabilities.  
* **Schema Definition:** The complete .proto schema definition is provided in the Appendix. A summary is provided below.

| Field | Type | Description | Rationale/Usage |
| :---- | :---- | :---- | :---- |
| profile\_version | string | Version of the profile schema for forward/backward compatibility. | Ensures graceful evolution of the contract. |
| kernel\_name | string | The mangled name of the gpu.func containing the branch. | JIT: Identifies the kernel to recompile. |
| branch\_id | uint32 | A unique, compiler-generated ID for the specific branch op. | JIT: Identifies the specific branch to transform. |
| invocation\_count | uint64 | Total times this branch was encountered by any thread. | JIT/AOT: Provides statistical significance. |
| divergence\_rate | float | Percentage of warps that experienced divergence at this branch. | JIT/AOT: Primary signal for optimization profitability. |
| path\_outcome\_counts | map\<string, uint64\> | Counts for each path, e.g., {"then": 5.2M, "else": 4.8M}. | JIT: Informs which rewrite pattern to apply. |
| hardware\_id | string | Identifier for the specific hardware SKU (e.g., "NVIDIA\_A100"). | AOT: Essential for training hardware-specific meta-optimizers. |
| input\_shape\_hash | uint64 | Hash of the input tensor shapes for this invocation. | AOT: Correlates divergence with input data characteristics. |
| **Table 2: DivergenceProfile Data Contract Schema (Summary).** |  |  |  |

#### **6.2. The FDO JIT Service RPC Contract (Version 1.0)**

* **Rationale:** This contract defines the precise remote procedure call (RPC) interface for the runtime to request a JIT re-compilation from the compiler service.  
* **Signatures:** The interface is defined using gRPC. The service definition includes the RecompileKernel RPC, its request message (containing the original MLIR module as a string and the serialized DivergenceProfile payload), and its response message (containing the compiled CUBIN as bytes and a status code).

Protocol Buffers

// In FdoJitService.proto  
syntax \= "proto3";  
import "DivergenceProfile.proto";

service FdoJitService {  
  rpc RecompileKernel(RecompileRequest) returns (RecompileResponse);  
}

message RecompileRequest {  
  string original\_mlir\_module \= 1;  
  DivergenceProfile profile \= 2;  
}

message RecompileResponse {  
  enum StatusCode {  
    SUCCESS \= 0;  
    COMPILATION\_ERROR \= 1;  
    INTERNAL\_ERROR \= 2;  
  }  
  StatusCode status \= 1;  
  bytes compiled\_cubin \= 2;  
  string error\_message \= 3;  
}

---

## **Part III: Optimization Frameworks: From Static Policy to Dynamic Adaptation**

This part details the "What" and "How" of the compiler's optimization capabilities, contrasting the declarative, static framework with the dynamic, adaptive one.

### **7\. The Declarative Optimization Engine**

#### **7.1. Rationale: Agility and Composability**

The hardware-aware optimization framework is built entirely on the MLIR transform dialect. This architectural choice replaces a rigid, monolithic C++ pass with a flexible, script-driven system. The rationale is to achieve agility and composability. Performance engineers can now write new fusion strategies, adjust tile sizes, and change the order of transformations simply by editing a text file, accelerating the performance tuning cycle by orders of magnitude compared to a C++-based approach.4

#### **7.2. Authoring Target-Specific Strategies**

The power of the transform dialect lies in its ability to declaratively script complex, multi-step optimization sequences. The following example script implements a fusion strategy for a common pattern (Matmul \-\> Add \-\> ReLU). This script is a simple text file, editable by performance engineers without recompiling the compiler. The core concepts involve:

1. **Matching Operations:** Using transform.structured.match to find the root of a pattern in the payload IR.  
2. **Tracing Dataflow:** Using transform.get\_op\_operand and transform.get\_defining\_op to walk the use-def chain backwards from a consumer to its producers.  
3. **Applying Transformations:** Using operations like transform.structured.tile\_using\_forall to create a fusion loop and transform.structured.fuse\_into\_containing\_op to move producer operations inside that loop.

MLIR

// In nvidia\_ampere\_fusion\_strategy.mlir  
module attributes {transform.with\_named\_sequence} {  
  transform.named\_sequence @\_\_transform\_main(  
    %func:\!transform.op\<"func.func"\>  
  ) {  
    // Stage 1: Find the root of the desired fusion pattern (the final consumer).  
    %relu \= transform.structured.match ops{\["linalg.generic"\]}  
      attributes{iterator\_types \= \["parallel", "parallel"\]} in %func

    // Stage 2: Tile the root op to create the outer fusion loop.  
    %tiled\_relu, %fusion\_loop:2 \=  
      transform.structured.tile\_using\_forall %relu tile\_sizes   
      \-\> (\!transform.op\<"linalg.generic"\>,\!transform.op\<"scf.forall"\>)

    // Stage 3: Trace backwards to find the producer (Add).  
    %relu\_input \= transform.get\_op\_operand %tiled\_relu at 0  
    %add \= transform.get\_defining\_op %relu\_input  
      : (\!transform.value) \-\>\!transform.op\<"linalg.generic"\>

    // Stage 4: Fuse the producer (Add) into the fusion loop.  
    %fused\_add, %fusion\_loop\_2:2 \=  
      transform.structured.fuse\_into\_containing\_op %add into %fusion\_loop  
      \-\> (\!transform.op\<"linalg.generic"\>,\!transform.op\<"scf.forall"\>)

    //... repeat for Matmul...  
    transform.yield  
  }  
}

#### **7.3. Integrated Phase-Ordering**

A critical consideration in compiler design is the phase-ordering problem, where the optimal sequence of transformations is context-dependent. Operator fusion and memory layout transformation are inherently interdependent. Fusing first might prevent an optimal layout change; changing layout first might prevent a profitable fusion.5

The transform dialect script is the ideal mechanism to solve this problem by making the sequence of operations explicit and programmable. For example, fusing a matmul and add operation for an Intel XMX engine is only highly profitable *because* the resulting fused operation can be lowered to a single, powerful xevm.mma instruction. This lowering, in turn, is only possible if the VNNI packing layout transformation is performed on one of the operands.5 The script can be authored to first insert the necessary

tensor.pack operation and then apply the fusion logic to the newly layout-transformed IR, ensuring that decisions are made holistically for an optimal outcome.

### **8\. The Adaptive Optimization Engine**

#### **8.1. Static Mitigation: PDL-Driven Speculative Execution**

The Single Instruction, Multiple Thread (SIMT) execution model of modern GPUs incurs a significant "Penalty of Divergence" when threads within a warp follow different paths through conditional logic, forcing serialization.5 The

divergence-to-speculation pass provides a static transformation to mitigate this by converting control-flow divergence into data-parallel speculation. An scf.if operation is rewritten into a data-parallel DAG where the then and else regions are cloned into two separate, non-divergent orchestra.task operations executed in parallel, with an orchestra.commit operation selecting the correct result.4

The correctness of this transformation hinges on a non-negotiable safety check: the then and else regions must be free of side effects, a condition rigorously verified by querying the MemoryEffectOpInterface for Write effects.4 The matching portion of this pattern is expressed declaratively using the Pattern Description Language (PDL), which cleanly separates the "what" of the match (the IR structure) from the "how" of the rewrite (the C++ logic) for improved clarity and maintainability.4

#### **8.2. Dynamic Mitigation: The FDO Closed-Loop Architecture**

For divergence patterns that are highly data-dependent, the Feedback-Driven Optimization (FDO) loop transforms the compiler into a "learning" system that adapts to observed runtime behavior. The closed-loop architecture consists of five stages 4:

1. **AOT Instrumentation:** The OrchestraBranchProfiler pass inserts lightweight profiling code (arith.atomic\_rmw) into gpu.func operations to track branch outcomes.  
2. **Execution and Profiling:** As the application executes, the embedded primitives collect statistics, which are aggregated into a structured DivergenceProfile.  
3. **Analysis and Triggering:** A runtime service analyzes the profiles and triggers an adaptive response when it detects a stable but suboptimal divergence pattern.  
4. **JIT Recompilation:** The runtime service invokes a JIT compilation service via RPC, passing the original MLIR and the DivergenceProfile. The JIT service applies the feedback-driven-remap pass and uses NVRTC to generate a CUBIN on-the-fly.  
5. **Dynamic Loading and Hot-Swapping:** The new CUBIN is dynamically loaded back into the running application using CUDA runtime APIs, and the dispatch table is updated to route subsequent calls to the optimized version.

#### **8.3. Introspection and Control: The OrchestraJITAction**

Implementing the FDO loop as a disconnected external process makes it opaque and difficult to debug. A key development in MLIR is the "Action" framework, a mechanism to encapsulate any transformation in a way that can be intercepted for debugging and control.5 The JIT recompilation step is therefore modeled as a custom

OrchestraJITAction. This integration elevates FDO from a powerful but opaque feature into a core, debuggable, and controllable architectural capability, providing profound benefits 4:

* **Debuggability and Traceability:** The entire JIT recompilation process becomes traceable using standard MLIR debugging tools like \-mlir-print-ir-before-action.  
* **Programmatic Control:** A developer can use a flag like \--mlir-elide-actions=OrchestraJITAction to globally disable all JIT recompilations, a powerful tool for isolating performance issues.

#### **8.4. Profile-Guided Rewrite Patterns**

The feedback-driven-remap pass, invoked by the JIT service, uses the DivergenceProfile to select and apply one of two powerful rewrite patterns 4:

* **Pattern A (Inter-Kernel): Profile-Guided Data Re-layout:** Ideal when the profile shows a strong correlation between an input value and the branch path. The transformation inserts a high-level orchestra.transfer op on the host side to re-sort the kernel's input data buffer, grouping elements that cause convergent branching.  
* **Pattern B (Intra-Kernel): Profile-Guided Thread-to-Data Remapping:** A more aggressive pattern used when divergence is stable but non-sortable. The pass modifies the kernel's internal logic, using shared memory and atomic operations to dynamically re-assign data items to threads immediately before a divergent branch, effectively forming new, convergent warps on-the-fly.

#### **8.5. The Criticality of the FDO Cost Model**

The intra-kernel thread remapping pattern (Pattern B) introduces significant overhead from shared memory accesses, atomic operations, and synchronization. It **must not** be applied without a robust cost model.5 This is a critical implementation detail that was under-emphasized in prior documentation. The cost model, informed by the quantitative data in the

DivergenceProfile, must rigorously determine that the performance penalty from the original divergent branch is greater than the estimated cost of the remapping logic. The absence of this check can easily lead to performance regressions, turning a potential optimization into a pessimization. This check is a non-negotiable requirement for the feedback-driven-remap pass.

---

## **Part IV: Multi-Target Integration and Lowering Playbooks**

This part provides the detailed, target-specific implementation guides—the "playbooks"—that are necessary for an implementation-focused document. Each section is a self-contained guide for an engineer tasked with working on a specific backend.7

| Feature | NVIDIA Blackwell (sm\_100+) | Google TPU (v5e+) | AMD Instinct (CDNA3+) | AWS Trainium (Trn2+) |
| :---- | :---- | :---- | :---- | :---- |
| **High-Level Entry IR** | linalg, tensor | linalg, tensor | linalg, tensor | linalg, tensor |
| **Vendor Handoff IR** | N/A (End-to-End) | stablehlo | N/A (End-to-End) | stablehlo |
| **Kernel Gen. Dialect** | scf, vector | Handled by XLA Compiler | rock (from rocMLIR) | Handled by Neuron Compiler |
| **HW Primitives Dialect** | nvgpu, nvvm | Proprietary (Internal to XLA) | amdgpu, rocdl | Proprietary (Internal to Neuron) |
| **Matrix Acceleration** | nvvm.tcgen05 family | XLA-generated systolic array ops | amdgpu.mfma | Neuron-generated systolic ops |
| **Synchronization** | nvgpu.mbarrier family | Handled by XLA Compiler | gpu.barrier | Handled by Neuron Compiler |
| **Key Memory Abstraction** | memref in shared memory | tensor (in stablehlo) | memref in LDS | tensor (in stablehlo) |
| **Expert Escape Hatch** | PTX Assembly Inline | N/A | GCN Assembly Inline | Neuron Kernel Interface (NKI) |
| **Table 3: Multi-Target Lowering Strategy Matrix.** This serves as the definitive, at-a-glance technical reference for the entire backend. |  |  |  |  |

### **9\. Foundational Lowering Infrastructure**

The foundational methodology for all lowering paths is MLIR's DialectConversion framework, the standard and correct approach for this task.4 This framework is governed by a

ConversionTarget that defines the legality of the final IR, a RewritePatternSet that contains the transformation logic, and a TypeConverter that manages the translation of types. A central, non-trivial component is the custom OrchestraTypeConverter. This class is the architectural linchpin that bridges the abstraction gap between logical and physical memory models. It is responsible for the critical task of translating the symbolic, logical memory spaces from OrchestraIR (e.g., a StringAttr like @gpu0\_hbm) into the concrete, integer-based gpu.address\_space attributes required by the hardware dialects (e.g., \#gpu.address\_space\<global\>). Without this custom, stateful type converter, the lowering process would fail due to type mismatches, making it an essential element for correctness.4

### **10\. Playbook: End-to-End Lowering for NVIDIA Architectures (Hopper, Blackwell)**

The lowering strategy for NVIDIA GPUs follows a "white-box" model, where OrchestraOS maintains end-to-end control of code generation. This strategy must be aware of the target architecture's specific capabilities to generate the most efficient code.4

* **For NVIDIA Hopper-class GPUs (sm\_90):** The lowering of orchestra.transfer requires a stateful pass that rewrites the operation to an nvgpu.device\_async\_copy, which returns a \!nvgpu.device\_async\_token. A second walk of the IR then inserts an nvgpu.device\_async\_wait operation immediately before the first use of the destination buffer, maximizing the window for overlapping data transfer with computation.  
* **For NVIDIA Blackwell architecture (sm\_100+):** To generate state-of-the-art code, the compiler must target the latest architectural features. When targeting Blackwell, the lowering strategy for orchestra.transfer operations that move data between global and shared memory is updated. Instead of nvgpu.device\_async\_copy, the pass must generate a sequence based on the more powerful and flexible Tensor Memory Accelerator (TMA). The new lowering sequence is:  
  1. A memory transfer descriptor is created using nvgpu.tma.create.descriptor.  
  2. The asynchronous copy is initiated with nvgpu.tma.async.load (for global-to-shared) or nvgpu.tma.async.store (for shared-to-global).  
  3. Synchronization is handled via the more advanced nvgpu.mbarrier family of operations, allowing for more fine-grained, transactional synchronization.  
* **Matrix Acceleration:** The lowering path targets nvgpu.mma.sync for Hopper and the nvvm.tcgen05 family of intrinsics for Blackwell.

### **11\. Playbook: Co-Designed Lowering for AMD Architectures (CDNA-class)**

The "white-box" lowering path for AMD GPUs integrates with the open-source ROCm toolchain and follows a multi-dialect progressive lowering strategy: linalg \-\> rock (from rocMLIR) \-\> amdgpu and rocdl.4

* **Data Transfer:** The lowering of orchestra.transfer for AMD's CDNA-class architectures targets the rocdl.global.load.lds operation, which is designed to efficiently load data from global memory into the fast, on-chip Local Data Store (LDS). Correctness is ensured by inserting a gpu.barrier after the transfer to synchronize all threads in the workgroup.4  
* **The rocMLIR "Contract" in Practice:** The amd\_instinct\_cdna3\_strategy.mlir transform script cannot be authored in isolation. The rocMLIR tool is a highly specialized generator tuned to recognize and optimize specific linalg.generic patterns.4 To achieve peak performance, the transform script must produce IR that precisely matches this "contract." For example, to trigger rocMLIR's highest-performance GEMM kernels, the  
  linalg.generic op must be structured with specific iterator types and operand permutations that rocMLIR's LinalgToRock pass is designed to match. An example of such a target pattern is shown below:

MLIR

// Target linalg.generic pattern for rocMLIR's high-performance GEMM path  
\#map\_a \= affine\_map\<(d0, d1, d2) \-\> (d0, d2)\>  
\#map\_b \= affine\_map\<(d0, d1, d2) \-\> (d2, d1)\>  
\#map\_c \= affine\_map\<(d0, d1, d2) \-\> (d0, d1)\>  
linalg.generic {  
  indexing\_maps \= \[\#map\_a, \#map\_b, \#map\_c\],  
  iterator\_types \= \["parallel", "parallel", "reduction"\]  
}...

The transform script must ensure that tiling and packing operations result in a linalg.generic that conforms to this structure. This provides the implementation-level detail required to guide performance engineers.7

### **12\. Playbook: Compiler-to-Compiler Integration for Hyperscalers (Google TPU, AWS Trainium)**

#### **12.1. Handoff via StableHLO**

For Google TPUs and AWS Trainium, the compilation target for OrchestraOS is not machine code, but a StableHLO program. StableHLO is an MLIR-based operation set explicitly designed by the OpenXLA project to serve as a portability layer between high-level machine learning frameworks and the diverse ecosystem of ML compilers and hardware backends.4 It is the official, standardized, and framework-agnostic entry point to both the XLA compiler for TPUs and the Neuron Compiler for Trainium. By targeting

StableHLO, OrchestraOS leverages the billions of dollars of engineering investment that Google and Amazon have poured into their respective compiler backends.11

#### **12.2. The StableHLO Bridge in Practice**

As established in Section 2.4, a direct C++ link is impossible. The StableHLO Bridge architecture, which invokes the stablehlo-translate tool as a subprocess, is the definitive solution.11 The workflow is as follows 4:

1. The OrchestraOS compiler lowers the optimized linalg dialect into the StableHLO dialect's **textual** representation.  
2. The StableHLO Bridge invokes the pre-built stablehlo-translate tool as a subprocess, piping the textual IR to its standard input.  
3. The tool serializes the program into a portable MLIR bytecode artifact, using a specific target version (e.g., stablehlo-translate \--serialize file.mlir \--target=1.0.0).  
4. This self-contained, version-stable binary artifact is the final handoff to the vendor compiler (e.g., Google's xla\_compile or AWS's neuronx-cc).

### **13\. Playbook: Leveraging Expert Escape Hatches (AWS Neuron Kernel Interface)**

While the StableHLO-based approach provides a robust, general-purpose integration path, achieving peak performance for critical kernels often requires more direct hardware control. To address this, the AWS Neuron SDK provides a powerful "escape hatch": the Neuron Kernel Interface (NKI). NKI is a Python-based programming environment, similar to Triton, that allows developers to write custom, low-level compute kernels with direct access to Trainium hardware primitives.6

To leverage this capability, OrchestraOS provides a mechanism to bypass the standard compiler and use a custom NKI kernel for performance-critical hotspots. A new, optional string attribute, nki\_source, is added to the orchestra.task operation to hold the Python-based NKI source code. A new compiler pass, orchestra-lower-nki-tasks, identifies these tasks, extracts the source code, invokes the NKI compiler offline, and links the resulting custom kernel into the final executable. This two-tiered strategy provides both the breadth of coverage necessary for general-purpose use and the depth of control required by performance experts to achieve state-of-the-art results.4

---

## **Part V: Synthesis and Implementation Roadmap**

This final part provides a holistic view of the entire system, a definitive map of the compiler's structure, and a summary of the strategic mandates for the engineering team.

### **14\. The Unified Compiler Pass Pipeline**

The successful implementation of a complex compiler depends not only on the quality of its individual components but also on their precise integration into a coherent pipeline. The following table serves as the operational map of the entire compiler, defining the precise order of transformation passes and making pass dependencies explicit. It visually represents the core architectural strategy: a "funnel" from multiple frontends to a common hardware-agnostic IR (linalg), followed by a "fan-out" to distinct, hardware-specific backends at Stage 7\. This clarity is invaluable for understanding the compiler's structure and for debugging its flow.4

| Stage | Input Dialect(s) | Key Transformation Passes | Output Dialect(s) |
| :---- | :---- | :---- | :---- |
| **1\. Ingestion** | torch, tf, onnx | Framework-specific Normalization, Functionalization, Shape Inference | func, linalg |
| **2\. Scheduling** | func, linalg, scf | Proprietary Topology-Aware Scheduling Pass | orchestra, scf |
| **3\. High-Level Opt.** | orchestra, scf, cf | \-lift-cf-to-scf, divergence-to-speculation (PDL-driven) | orchestra |
| **4\. Structured Lowering** | orchestra | orchestra-to-linalg, orchestra-transfer-to-dma | linalg, memref, tensor |
| **5\. Hardware Opt.** | linalg, memref, tensor | \-transform-interpreter (with target-specific transform script) | scf, vector, memref, tensor |
| **6\. Bufferization** | tensor, scf, vector | \-one-shot-bufferize | memref, scf, vector |
| **7\. Accelerator Lowering** | scf, vector, memref | **Target-dispatch based on orchestra.task arch attribute:** |  |
|  |  | **NVIDIA Path:** linalg-to-nvgpu, gpu-lower-to-nvvm-pipeline | gpu, nvvm, llvm |
|  |  | **AMD Path:** linalg-to-rock, rock-to-amdgpu | gpu, amdgpu, rocdl, llvm |
|  |  | **Google/AWS Path:** linalg-to-stablehlo (pass) | stablehlo (text) |
| **8\. Executable Gen.** | gpu, nvvm, llvm, stablehlo | **NVIDIA/AMD:** \-gpu-to-llvm, gpu-module-to-binary | llvm, gpu.binary |
|  |  | **Google/AWS:** **StableHLO Bridge Stage** (external tool invocation) → Handoff to Vendor Compiler (xla\_compile, neuronx-cc) | Vendor-specific binary (e.g., NEFF) |
| **Table 4: The Unified End-to-End Compiler Pass Pipeline.** |  |  |  |

### **15\. Glossary of Terms**

* **AOT (Ahead-of-Time):** Compilation that occurs before the program is executed.  
* **ABI (Application Binary Interface):** The low-level interface between an application and the operating system or another application.  
* **FDO (Feedback-Driven Optimization):** A compilation technique that uses runtime profile data to guide optimizations.  
* **IR (Intermediate Representation):** A data structure used internally by a compiler to represent source code.  
* **JIT (Just-in-Time):** Compilation that occurs during program execution.  
* **MLIR (Multi-Level Intermediate Representation):** A compiler infrastructure project that provides a flexible, extensible framework for building compilers.  
* **NKI (Neuron Kernel Interface):** A Python-based environment for writing custom, low-level kernels for AWS Trainium accelerators.  
* **ODS (Operation Definition Specification):** The TableGen-based system used in MLIR to define the properties of operations.  
* **PDL (Pattern Description Language):** A declarative language within MLIR for specifying the matching portion of a rewrite pattern.  
* **Platform-as-a-Target:** A compilation target, such as a hyperscaler cloud platform, where integration requires interfacing with a complete software ecosystem rather than just generating machine code for a specific ISA.  
* **SIMT (Single Instruction, Multiple Thread):** An execution model used by GPUs where a single instruction is executed by multiple threads in parallel on different data.  
* **StableHLO:** An MLIR-based operation set designed by the OpenXLA project to serve as a portable IR between ML frameworks and backends.

### **16\. Appendix: Full OrchestraIR ODS and DivergenceProfile Schema**

#### **OrchestraOps.td**

Code-Snippet

// In OrchestraOps.td  
\#ifndef ORCHESTRA\_OPS  
\#define ORCHESTRA\_OPS

include "mlir/Interfaces/SideEffectInterfaces.td"

def Orchestra\_Dialect : Dialect {  
  let name \= "orchestra";  
  let cppNamespace \= "::mlir::orchestra";  
  let summary \= "A dialect for high-level orchestration of heterogeneous systems.";  
  let usePropertiesForAttributes \= 1;  
}

class Orchestra\_Op\<string mnemonic, list\<Trait\> traits \=\> :  
    Op\<Orchestra\_Dialect, mnemonic, traits\>;

//... (Full ODS for orchestra.schedule, orchestra.task, etc. as specified in Section 4)...

def Orchestra\_TaskOp : Orchestra\_Op\<"task",\> {  
  let summary \= "An asynchronous unit of computation assigned to a resource.";  
  let arguments \= (ins Variadic\<AnyType\>:$operands);  
  let results \= (outs Variadic\<AnyType\>:$results);  
  let regions \= (region AnyRegion:$body);  
  let properties \=;  
  let hasVerifier \= 1;  
}

def Orchestra\_CommitOp : Orchestra\_Op\<"commit",\> {  
  let summary \= "Selects between two sets of values based on a condition.";  
  let arguments \= (ins I1:$condition, Variadic\<AnyType\>:$values);  
  let results \= (outs Variadic\<AnyType\>:$results);  
  let properties \= \[  
    Property\<"num\_true", "::mlir::IntegerAttr",  
             "Number of values belonging to the 'true' branch."\>  
  \];  
  let hasVerifier \= 1;  
  let hasCanonicalizer \= 1;  
}

\#endif // ORCHESTRA\_OPS

#### **DivergenceProfile.proto**

Protocol Buffers

syntax \= "proto3";

package orchestra.profiling;

message DivergenceProfile {  
  // Version of the profile schema for forward/backward compatibility.  
  string profile\_version \= 1;

  // The mangled name of the gpu.func containing the branch.  
  string kernel\_name \= 2;

  // A unique, compiler-generated ID for the specific scf.if or cf.cond\_br.  
  uint32 branch\_id \= 3;

  // Total times this branch was encountered by any thread.  
  uint64 invocation\_count \= 4;

  // Percentage of warps that experienced divergence at this branch.  
  float divergence\_rate \= 5;

  // Counts for each path, e.g., {"then": 5200000, "else": 4800000}.  
  map\<string, uint64\> path\_outcome\_counts \= 6;

  // Identifier for the specific hardware SKU (e.g., "NVIDIA\_A100").  
  // Essential for training hardware-specific meta-optimizers.  
  string hardware\_id \= 7;

  // Hash of the input tensor shapes for this invocation.  
  // Correlates divergence with input data characteristics.  
  uint64 input\_shape\_hash \= 8;  
}  
