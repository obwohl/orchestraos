

# **An Authoritative Engineering Blueprint for Multi-Vendor AI Accelerator Support in the OrchestraOS MLIR 20 Compiler**

## **Introduction**

The foundational architectural vision of the OrchestraOS compiler, articulated in the 'orchestra \- tech \- MLIR 20 Blueprint,' establishes a robust and forward-looking strategy centered on the compiler-driven orchestration of complex, heterogeneous hardware systems.1 The core principle—elevating scheduling, data movement, and execution strategy to first-class citizens of the Intermediate Representation (IR)—remains the definitive approach to mastering the challenges of modern high-performance computing. The selection of MLIR as the foundational infrastructure is unequivocally validated, providing the essential modularity and multi-level representation capabilities required to realize this vision.1

However, the landscape of high-performance AI acceleration is undergoing a period of unprecedented diversification and growth. While the initial blueprint correctly establishes a state-of-the-art architecture for a single-vendor ecosystem, the strategic imperative for OrchestraOS as a "meta-OS" is to provide seamless, performant orchestration across a multi-vendor hardware environment. The rapid maturation of custom silicon from hyperscalers and the emergence of viable competitors in the discrete GPU market are not peripheral trends but central forces reshaping the future of AI computation. To maintain its technological leadership and fulfill its architectural promise, OrchestraOS must proactively expand its hardware support beyond the incumbent.

This document serves as a strategic and technical addendum to the 'orchestra \- tech \- MLIR 20 Blueprint'. Its purpose is threefold: first, to conduct a rigorous, data-driven analysis of the AI training accelerator market to identify and prioritize the next set of strategic hardware targets; second, to perform a deep technical investigation into the state-of-the-art MLIR-based compilation strategies for these new targets; and third, to synthesize these findings into a series of concrete, actionable, and architecturally sound enhancements to the blueprint. The outcome is a definitive engineering guide for evolving the OrchestraOS compiler into a truly heterogeneous orchestration platform, securing its long-term market relevance and competitive advantage.

## **Section 1: Strategic Target Analysis for AI Training Accelerators (2025-2030)**

This section provides the strategic justification for the subsequent technical deep dives. A thorough analysis of the AI training hardware market, encompassing both the current landscape and a five-year forecast, is essential to identify the most relevant and strategically critical targets for the OrchestraOS compiler. This data-driven approach ensures that engineering investments are aligned with market realities and future growth trajectories.

### **1.1 The Competitive Landscape: From Incumbency to Hyperscale Competition**

The market for AI training hardware is characterized by the entrenched dominance of a single incumbent, challenged by a growing cohort of well-capitalized and technologically sophisticated competitors. The overall market is experiencing explosive growth, with the AI hardware market projected to grow from approximately $60-67 billion in 2025 to nearly $300 billion by 2034, at a Compound Annual Growth Rate (CAGR) of around 18%.4 The closely related data center accelerator market shows a similar trajectory, expected to expand from over $17 billion in 2024 to more than $63 billion by 2030, with a CAGR of 24.7%.5 This immense market expansion validates the strategic importance of the entire sector and fuels the competitive dynamics.

#### **1.1.1 NVIDIA's Entrenched Dominance**

NVIDIA remains the undisputed leader in the AI training hardware market, particularly for data center GPUs. Multiple market analyses consistently place its market share in the range of 80% to 95%.7 This commanding position is not merely a function of superior hardware performance but is deeply rooted in the maturity and widespread adoption of its CUDA software ecosystem. CUDA provides a full-stack, vertically integrated platform that includes libraries, compilers, and a vast developer community, creating a significant competitive moat that is difficult for rivals to overcome.11 The company's relentless product cadence, with the new Blackwell platform being rapidly deployed by major cloud providers in 2025, ensures that demand continues to outstrip supply and solidifies its role as the primary, must-support target for any serious AI software stack.13

#### **1.1.2 The Rise of Viable Challengers**

Despite NVIDIA's dominance, several powerful challengers are making significant inroads, driven by the dual imperatives of reducing dependency on a single supplier and optimizing performance for specific workloads and environments.

* **Advanced Micro Devices (AMD):** AMD has emerged as NVIDIA's most direct competitor in the discrete GPU market. The company is experiencing significant momentum, with its server CPU market share nearing 40% in Q1 2025, up from 25% in 2023, driven by the success of its EPYC processors.14 This success is translating to its data center GPU segment, where the Instinct accelerator series (MI300X, MI350, MI400) is gaining traction with hyperscalers and national-level AI projects.14 AMD's strategy is centered on providing high-performance hardware coupled with an open-source software stack, ROCm, as a direct alternative to the closed CUDA ecosystem.14 The company's data center revenue has shown strong year-over-year growth, signaling its increasing relevance as a key player in the AI training market.14  
* **Google (Tensor Processing Units \- TPUs):** Google's strategy with its custom-designed TPUs is fundamentally different from that of NVIDIA and AMD. As a vertically integrated hyperscaler, Google does not sell TPUs as discrete hardware. Instead, TPUs are the foundational accelerators within the Google Cloud Platform (GCP), designed to provide optimal performance and cost-efficiency for workloads running on Google's infrastructure.19 The success of TPUs is therefore measured by the adoption and scale of AI workloads on GCP. The total addressable market captured by these workloads is substantial and growing rapidly, driven by both internal Google services and external customers.22 The continuous innovation, with the recent introduction of the Trillium architecture, demonstrates Google's long-term commitment to this platform.20  
* **Amazon Web Services (AWS) (Trainium):** Similar to Google, Amazon's development of custom silicon, specifically the Trainium family of accelerators, is a strategic initiative to enhance the capabilities of the AWS ecosystem. The primary goals are to reduce reliance on third-party hardware, lower operational costs, and provide customers with a highly optimized platform for large-scale AI model training.24 AWS's massive investment in this area, including the $100 billion "Project Rainier" for purpose-built AI data centers, and the adoption of Trainium by major AI companies like Anthropic, underscore the strategic importance of this hardware.4 Trainium is positioned to deliver significant price-performance advantages over comparable GPU-based instances, making it a compelling option for AWS customers.26

The following table summarizes the competitive positioning and strategic rationale for each major player in the AI training accelerator market.

| Vendor | Estimated 2025 Market Share (%) | Projected 2030 Market Share (%) | Key Growth Drivers / Strategic Rationale |
| :---- | :---- | :---- | :---- |
| **NVIDIA** | 75 \- 85 | 60 \- 70 | Entrenched CUDA software ecosystem moat; Aggressive product roadmap (Blackwell, Rubin); Strong relationships with all major cloud providers and enterprises.10 |
| **AMD** | 5 \- 10 | 10 \- 15 | Strong EPYC CPU momentum creating data center inroads; Open-source ROCm software stack as a CUDA alternative; Gaining adoption with hyperscalers for cost/performance.14 |
| **Google (TPU)** | 5 \- 10 | 10 \- 15 | Vertical integration with Google Cloud Platform; Co-design of hardware and software (XLA compiler) for optimal performance on key workloads (LLMs, GenAI); Captive market of Google's internal services.22 |
| **Amazon (Trainium)** | 3 \- 8 | 8 \- 12 | Dominant cloud market share (AWS); Focus on price-performance to drive adoption within the AWS ecosystem; Significant internal investment and adoption by key partners like Anthropic.25 |
| **Other ASICs/FPGAs** | \< 5 | \< 5 | Niche applications, edge computing, and specialized inference workloads; Unlikely to capture significant training market share against vertically integrated or ecosystem-driven players.4 |

Table 1: AI Training Accelerator Market Share & 5-Year Forecast (Synthesized from 4). Note: Market share represents the portion of AI training workloads, not necessarily discrete unit sales.

### **1.2 Relevance Assessment and Prioritization for OrchestraOS**

The market analysis provides a clear directive for the strategic evolution of the OrchestraOS compiler. To remain relevant and achieve its vision as a true "meta-OS" for heterogeneous computing, the compiler must support the hardware platforms where the vast majority of AI training workloads will be executed over the next five years.

Based on this analysis, the following prioritization is established:

* **Tier 1 (Incumbent): NVIDIA.** Support for NVIDIA GPUs is non-negotiable and must remain the highest priority. The existing lowering paths to NVIDIA architectures, as detailed in the blueprint, must be continuously updated to target the latest hardware features, such as those in the Blackwell architecture.  
* **Tier 2 (Strategic New Targets): Google (TPU), AMD (Instinct), and Amazon (Trainium).** These three platforms represent the most significant and viable challengers to NVIDIA's dominance. Their growing market share, substantial backing from parent companies, and distinct technological approaches make them critical targets. Achieving robust, performant support for this trio is the central strategic goal for the next phase of compiler development.

This prioritization is informed by two critical observations that must shape the architectural approach to multi-vendor support. First, the primary axis of competition in the AI accelerator market has decisively shifted from raw hardware specifications to the maturity, openness, and integration of the accompanying software ecosystems. NVIDIA's enduring lead is a testament to the power of its CUDA platform.11 AMD's principal challenge lies in maturing its ROCm software stack to rival CUDA's breadth and stability.14 Concurrently, Google and Amazon are constructing their entire value propositions around deeply integrated software stacks—OpenXLA and the Neuron SDK, respectively—that are inseparable from their cloud platforms.28 This reality dictates that a generic, "one-size-fits-all" lowering strategy for OrchestraOS is non-viable. To unlock competitive performance on any of these targets, the compiler must integrate deeply and intelligently with the native compiler toolchain of each vendor. This understanding provides the core justification for the detailed technical investigations in the subsequent sections of this report; mastering these vendor-specific toolchains is paramount.

Second, the emergence of vertically integrated hyperscalers has created a new class of compilation target: the "Platform-as-a-Target." Unlike NVIDIA and AMD, which sell discrete hardware components, Google and Amazon offer access to an entire platform (GCP and AWS) where their custom silicon is a key, but integrated, element.21 They control the entire stack, from the physical chip design to the high-level ML framework bindings. Consequently, supporting Trainium or TPUs is not merely a matter of generating machine code for a specific instruction set architecture. It is a matter of integrating with the OpenXLA and Neuron SDK ecosystems, which serve as the designated, and in most cases exclusive, entry points to these platforms. This fundamentally alters the integration model for OrchestraOS. Instead of functioning as a traditional backend code generator that emits low-level machine code, the compiler must adopt a "compiler-to-compiler" integration strategy. In this model, OrchestraOS will perform its high-level, proprietary orchestration and optimization, and then lower its representation to the high-level IR accepted by the platform's native compiler (e.g., StableHLO), effectively using the vendor's compiler as its backend. This approach leverages the immense engineering investment the vendors have made in their own toolchains while allowing OrchestraOS to focus on its unique value proposition in cross-system orchestration.

## **Section 2: State-of-the-Art MLIR Compilation for Google Cloud TPUs**

Supporting Google's Tensor Processing Units (TPUs) requires a compilation strategy that integrates with Google's proprietary but powerful XLA (Accelerated Linear Algebra) compiler ecosystem. While the innermost workings of the TPU backend are not open-source, Google provides a well-defined, modern, and framework-agnostic entry point through the OpenXLA project. This section details the technical strategy for targeting TPUs, proposing an integration path that leverages this open interface and draws upon established MLIR best practices for its conceptual architecture.

### **2.1 The OpenXLA and StableHLO Entry Point**

The primary and state-of-the-art entry point into the XLA compiler stack for targeting TPUs and other accelerators is the **StableHLO** dialect.29 StableHLO is an MLIR-based operation set designed explicitly to serve as a portability layer between high-level machine learning frameworks (like PyTorch, JAX, and TensorFlow) and the diverse ecosystem of ML compilers and hardware backends.29 By standardizing on StableHLO as an input format, frameworks can produce a single representation that can be consumed by any compatible backend, including Google's XLA compiler for TPUs.34

This ecosystem is complemented by the **PJRT (Pretty much Just another RunTime)** interface, which provides a uniform, hardware-independent API for loading and executing compiled programs on various devices.32 For OrchestraOS, this means that the compilation target is a StableHLO program, and the execution interface is PJRT. This clear separation of concerns, enabled by the OpenXLA project, provides a robust and well-supported integration path.

### **2.2 A Conceptual Model for Progressive Lowering to TPU Architectures**

A significant challenge in developing a third-party compiler for Google TPUs is that the final lowering stages of the XLA compiler—from its internal HLO representation to the specific machine code for the TPU systolic array—are proprietary and not publicly documented.37 To construct a coherent and state-of-the-art architecture in the absence of this direct information, it is necessary to rely on a conceptual model that embodies best practices for compiling to similar accelerator architectures.

The research paper "TPU-MLIR: A Compiler For TPU Using MLIR" provides such a model.38 Although this paper describes a compiler for a different TPU (from Sophgo), its architectural principles are highly relevant and represent a state-of-the-art approach for an end-to-end MLIR-based ASIC compiler. The proposed architecture is based on a two-dialect progressive lowering pipeline:

1. **A high-level TOP (Tensor Operation) dialect:** This dialect is designed to be both framework- and hardware-agnostic. It captures the pure semantics of the neural network graph, representing operations on logical tensors without exposing hardware-specific details. This level of abstraction is conceptually analogous to the standard linalg dialect in the upstream MLIR ecosystem, which serves as a common representation for linear algebra operations before target-specific lowering.38  
2. **A low-level TPU dialect:** This dialect is hardware-dependent and serves as the abstraction layer for the target accelerator. It contains operations that map more directly to the capabilities of a TPU-like systolic array architecture. Operations in this dialect carry device-specific attributes to control quantization, specify memory layouts, and manage on-chip memory. This dialect is the final stage before the generation of low-level hardware commands.38

This two-stage approach—from a generic, high-level representation to a target-aware, low-level representation—is a fundamental pattern in modern MLIR-based compilers and provides a sound conceptual basis for the OrchestraOS TPU compilation strategy.

### **2.3 Proposed orchestra.tpu Integration and Lowering Pipeline**

Synthesizing the open-source entry point provided by OpenXLA with the conceptual model from the TPU-MLIR paper yields a clear and pragmatic implementation strategy for OrchestraOS. The strategy is not to replicate the entire XLA backend but to intelligently interface with it at the appropriate level of abstraction.

The proposed pipeline consists of two primary stages:

1. **Lowering to a Common Compiler IR:** The main OrchestraOS pass pipeline will first lower the high-level OrchestraIR dialect to the standard linalg dialect on tensors. This aligns the TPU compilation path with the rest of the OrchestraOS ecosystem, as linalg serves as the universal, hardware-agnostic representation for structured tensor computations. This stage is where hardware-independent optimizations, such as high-level operator fusion and algebraic simplification, are performed.42  
2. **Targeting the XLA Ecosystem via StableHLO:** Following the linalg stage, a dedicated pass pipeline, orchestra-lower-to-stablehlo, will be responsible for converting the linalg operations into their stablehlo equivalents. The resulting stablehlo module becomes the definitive "handoff" artifact that is passed to the XLA compiler. At this point, the OrchestraOS compiler's role is complete, and the XLA compiler takes full responsibility for all subsequent TPU-specific optimizations, tiling, layout assignment, and final machine code generation.43

This "gray-box" integration strategy is a deliberate architectural choice. A full "white-box" approach, which would involve creating a proprietary orchestra.tpu dialect and a complete code generation backend from scratch (as described in the Sophgo paper 38), represents a monumental engineering effort that would require access to Google's proprietary hardware documentation. Conversely, a simple "black-box" approach that relies on high-level Python APIs would forfeit the fine-grained control and optimization opportunities that a compiler-level integration provides.

The proposed strategy of targeting StableHLO strikes the optimal balance. It leverages the billions of dollars of engineering investment that Google has poured into the XLA compiler backend for TPUs, which is highly optimized for Google's internal workloads.34 This dramatically reduces the implementation cost and risk for OrchestraOS. The primary engineering challenge is shifted from the intractable problem of reverse-engineering a TPU backend to the well-defined and tractable problem of building a robust and comprehensive

linalg-to-stablehlo lowering pipeline. This is a well-understood domain within the MLIR community, with existing patterns and infrastructure to build upon.43 This approach allows OrchestraOS to focus its proprietary engineering resources on its core value proposition: high-level, multi-node, heterogeneous orchestration and optimization within

OrchestraIR, while delegating the final, target-specific code generation to the best-in-class vendor toolchain.

## **Section 3: State-of-the-Art MLIR Compilation for AMD Instinct GPUs**

The compilation strategy for AMD's Instinct family of data center GPUs must be designed to integrate with the ROCm (Radeon Open Compute) platform, AMD's open-source software stack for GPU computation. Unlike the more opaque compilation path for Google TPUs, the MLIR-based toolchain for ROCm is open-source, providing a transparent, multi-level dialect stack that allows for deep, "white-box" integration and optimization.

### **3.1 The ROCm Ecosystem and the rocMLIR Kernel Generator**

The **ROCm** software platform is the comprehensive, open-source ecosystem for programming AMD GPUs, providing everything from kernel drivers and runtimes to compilers and high-level libraries.44 A key component of this modern stack is

**rocMLIR**, an official, MLIR-based project specifically designed to function as a high-performance kernel generator for convolution and General Matrix Multiplication (GEMM) operations—the computational heart of most deep learning models.45

The primary role of rocMLIR is to take a high-level representation of a tensor operation and compile it down to a highly optimized GCN (Graphics Core Next) assembly kernel. To achieve this, it is designed to leverage the specific architectural features of modern AMD GPUs, such as the Matrix-Fused-Multiply-Add (MFMA) instructions on CDNA-class architectures or the Wavefront Matrix-Multiply-Accumulate (WMMA) instructions on newer RDNA architectures. The rocMLIR toolchain provides command-line flags (e.g., \-mfma=on, \-wmma=on) to explicitly control the use of these hardware acceleration features during kernel generation.45

### **3.2 A Multi-Dialect Lowering Strategy: From linalg to rock and amdgpu**

The compilation path for AMD GPUs within the MLIR ecosystem follows a well-defined progressive lowering strategy, moving from hardware-agnostic dialects to increasingly hardware-specific ones. This transparent pipeline provides multiple opportunities for optimization and integration.

1. **linalg Dialect:** As with other targets, the linalg dialect serves as the common, hardware-agnostic entry point. High-level optimizations that are not specific to AMD's architecture, such as certain types of operator fusion or algebraic simplifications, are performed at this level. This ensures consistency with the broader OrchestraOS compiler architecture.46  
2. **rock Dialect:** The rock dialect is a high-level, domain-specific dialect internal to the rocMLIR project. Its purpose is to represent and optimize GEMM and convolution-like operations in a way that is amenable to the rocMLIR kernel generation heuristics. The lowering from linalg to rock is a critical step where high-level architectural decisions are made. This includes selecting tiling strategies, choosing data layouts for optimal memory access, and preparing the operation for mapping to the GPU's hierarchical memory and compute units. While not part of upstream LLVM, this dialect is the primary interface to the rocMLIR generator's powerful optimization capabilities.45  
3. **amdgpu and rocdl Dialects:** These are the low-level, hardware-specific dialects that are part of the upstream MLIR project. The rocMLIR compiler lowers the abstract rock dialect operations into a combination of operations from these dialects.  
   * The **amdgpu dialect** provides direct MLIR wrappers for AMD-specific hardware intrinsics and functionality. This includes operations like amdgpu.mfma and amdgpu.wmma, which map directly to the matrix acceleration instructions on the hardware, as well as operations for managing low-level hardware features like the Local Data Share (LDS) and wavefront-level data permutations.48  
   * The **rocdl dialect** models the ROCm device-side LLVM IR. It provides the necessary components to represent a full GPU kernel that can be compiled by the LLVM backend into the final GCN instruction set architecture (ISA).49

This full pipeline, from linalg down to amdgpu and rocdl, represents a complete, transparent, and extensible compilation path for generating high-performance kernels for AMD GPUs.

### **3.3 Proposed Integration with the OrchestraOS transform Dialect Framework**

The integration of AMD GPU support into OrchestraOS will directly leverage the powerful, declarative optimization framework established in the 'orchestra \- tech \- MLIR 20 Blueprint'.1 This approach avoids hardcoding optimization strategies in C++ and instead uses target-specific MLIR scripts to control the compilation process.

A new, target-specific transform script, for example amd\_instinct\_mi300\_strategy.mlir, will be authored by performance engineers. This script will contain a sequence of transform dialect operations that are executed by the \-transform-interpreter pass. These operations will be responsible for applying the optimal sequence of tiling, fusion, vectorization, and data layout packing transformations at the linalg dialect level, preparing the IR for the subsequent lowering to rocMLIR.

A crucial aspect of this strategy is that the transform dialect script for AMD targets must be meticulously co-designed with the internal lowering patterns of the rocMLIR kernel generator. The rocMLIR tool is not a generic linalg compiler; it is a highly specialized generator tuned to recognize and optimize specific linalg.generic patterns that correspond to high-performance GEMM and convolution kernels. The performance of the final generated assembly is critically dependent on the precise structure of the linalg IR it receives as input. Factors such as the chosen tile sizes, the permutation of operands, and the application of tensor packing transformations at the linalg level directly influence which lowering path is selected within rocMLIR, how effectively the problem is mapped to the rock dialect, and ultimately, how efficiently the final amdgpu.mfma instructions can be scheduled.

Therefore, performance engineers authoring the amd\_instinct\_mi300\_strategy.mlir script cannot operate in isolation. They must treat the rocMLIR source code, particularly its linalg-to-rock conversion passes, as a form of "contract" that specifies the optimal input format. By analyzing this contract, they can author a transform script that precisely manipulates the linalg IR to match the exact patterns that rocMLIR is designed to accelerate most effectively. This creates a powerful, "white-box" optimization opportunity. It allows OrchestraOS to use its high-level, declarative framework to intelligently guide the downstream vendor compiler toward its most performant code generation paths, ensuring that the generated kernels achieve near-optimal performance without requiring any modification to the downstream compiler itself.

## **Section 4: State-of-the-Art MLIR Compilation for AWS Trainium Accelerators**

The strategy for supporting Amazon's custom AI accelerators, AWS Trainium, requires integration with the AWS Neuron SDK. This SDK provides the complete software stack necessary to compile, execute, and debug machine learning models on AWS's purpose-built hardware. The Neuron compiler's architecture, which is based on MLIR and embraces open standards like OpenXLA, provides a clear and robust path for integration.

### **4.1 The AWS Neuron SDK and its MLIR-based Compiler Architecture**

The **AWS Neuron SDK** is the exclusive and mandatory software development kit for targeting AWS Trainium and Inferentia chips.31 It is a comprehensive suite of tools that includes a compiler, runtime libraries, and debugging and profiling utilities.31

A critical feature of the Neuron SDK is that its compiler is explicitly built on MLIR and modern compiler technologies.30 This architectural choice enables it to support a variety of high-level ML frameworks, including PyTorch and JAX. Most importantly for OrchestraOS, the Neuron Compiler officially supports

**OpenXLA**, and specifically the **StableHLO** dialect, as a primary input format.31 This provides a standardized, compiler-level interface for third-party tools.

The Neuron Compiler is responsible for a wide range of sophisticated optimizations tailored for the Trainium architecture. These include hardware-aware operator fusion, automatic mixed-precision casting (quantization), and optimizations for parallel execution across multiple NeuronCore accelerators.51 The final output of the compilation process is a proprietary binary artifact known as a

**NEFF** (Neuron Executable File Format), which is loaded by the Neuron runtime for execution on the hardware.51

### **4.2 A StableHLO-Targeted "Black-Box" Integration Strategy**

Given the Neuron Compiler's support for StableHLO, the primary integration strategy for OrchestraOS will mirror the approach proposed for Google TPUs. This strategy leverages the common open standard as a clean handoff point between the two compiler systems.

The end-to-end pipeline will be as follows:

1. The high-level OrchestraIR representation, containing the system-wide orchestration plan, is lowered to the hardware-agnostic linalg dialect.  
2. A dedicated pass, orchestra-lower-to-stablehlo, converts the linalg operations into their stablehlo equivalents.  
3. The resulting StableHLO module is then passed as input to the Neuron Compiler, which is invoked via its command-line interface (neuronx-cc).54

In this model, the Neuron Compiler is treated as a "black-box" backend. OrchestraOS is responsible for all high-level, hardware-agnostic optimizations and for producing a valid StableHLO program. The Neuron Compiler is then responsible for all subsequent Trainium-specific optimizations, scheduling, and machine code generation, culminating in the final NEFF executable. This approach maximizes leverage of AWS's significant investment in their compiler technology, minimizes the implementation burden on the OrchestraOS team, and ensures that the generated code benefits from the latest hardware-specific optimizations developed by AWS.

### **4.3 Advanced Integration via the Neuron Kernel Interface (NKI)**

While the StableHLO-based approach provides a robust and general-purpose integration path, achieving peak performance for certain critical computational kernels often requires more direct control over the hardware. To address this need, the AWS Neuron SDK provides a powerful "escape hatch" for expert developers: the **Neuron Kernel Interface (NKI)**.31

NKI is a Python-based programming environment, with a syntax and semantics similar to the popular Triton language, that allows developers to write custom, low-level compute kernels. It provides direct access to the hardware primitives and instruction set of the Trainium and Inferentia accelerators, enabling performance engineers to build and tune kernels for optimal performance in a way that may not be achievable through the general-purpose compiler alone.31

This interface represents a strategic opportunity for OrchestraOS to offer a differentiated, best-in-class performance solution. No general-purpose compiler can be expected to generate the absolute optimal code for every possible algorithm. High-performance libraries have historically relied on hand-tuned assembly or low-level intrinsics to achieve their results. The NKI provides a modern, structured, and safer way to achieve the same outcome on AWS hardware.

To leverage this capability, a two-tiered integration strategy is proposed:

1. **General Path (StableHLO):** The vast majority of operations will be compiled via the standard StableHLO "black-box" path, providing broad coverage and ease of use.  
2. **Expert Path (NKI):** For performance-critical hotspots, OrchestraOS will provide a mechanism to bypass the standard compiler and use a custom NKI kernel.

This requires a direct enhancement to the OrchestraIR dialect. A new, optional string attribute, nki\_source, will be added to the orchestra.task operation. This attribute will be used to hold the Python-based NKI source code for a custom kernel. A new compiler pass, orchestra-lower-nki-tasks, will be implemented. This pass will identify orchestra.task operations that have the nki\_source attribute, extract the source code, invoke the NKI compiler offline to produce a kernel object, and then link this custom kernel into the final NEFF executable. This two-tiered strategy provides both the breadth of coverage necessary for general-purpose use and the depth of control required by performance experts to achieve state-of-the-art results on the Trainium platform.

## **Section 5: Enhancements to the 'orchestra \- tech \- MLIR 20 Blueprint'**

The preceding analysis of the AI accelerator market and the deep dive into the state-of-the-art compilation strategies for the new strategic targets—Google TPU, AMD Instinct, and AWS Trainium—necessitate a series of concrete enhancements to the 'orchestra \- tech \- MLIR 20 Blueprint'. This section synthesizes these findings into actionable modifications, providing a clear and updated roadmap for the engineering team. These enhancements are designed to integrate seamlessly with the existing architecture, extending its capabilities to support a true multi-vendor, heterogeneous environment.

### **5.1 Extending the orchestra.task Target Schema for New Devices**

The existing schema for the target attribute on the orchestra.task operation, which relies on a simple string or a generic dictionary, is insufficient for the demands of a multi-vendor environment. To enable the compiler to make intelligent, target-specific decisions, the schema must be formalized and enriched.

The target attribute of orchestra.task will be a DictionaryAttr with a formally defined schema. A mandatory arch key of type StringAttr will serve as the primary discriminator for dispatching to the correct lowering pipeline. Its value will be a standardized string identifying the target architecture family (e.g., "nvidia\_blackwell", "amd\_cdna3", "google\_tpu\_v5e", "aws\_trainium2").

In addition to the arch key, the dictionary can contain optional, architecture-specific keys that provide fine-grained information to the optimization and code generation passes. This extensible design allows the transform dialect scripts and lowering patterns to query specific hardware features or constraints.

**Example orchestra.task with Enhanced Target Schema:**

MLIR

// Example for an AMD Instinct MI300X GPU  
%result \= orchestra.task \-\> (tensor\<256x256xf32\>)  
    target \= {arch \= "amd\_cdna3", device\_id \= 0, mfma \= true, lds\_size \= 65536} {  
  //... linalg operations...  
  orchestra.yield %some\_value : tensor\<256x256xf32\>  
}

// Example for a Google Cloud TPU v5e  
%result \= orchestra.task \-\> (tensor\<1024x1024xbf16\>)  
    target \= {arch \= "google\_tpu\_v5e", logical\_core\_count \= 4} {  
  //... linalg operations...  
  orchestra.yield %another\_value : tensor\<1024x1024xbf16\>  
}

This enhanced schema makes the hardware target's capabilities a first-class citizen of the IR, enabling more powerful and precise target-aware optimizations throughout the compilation process.

### **5.2 Authoring Target-Specific transform Dialect Optimization Scripts**

The blueprint's strategic pivot to a declarative optimization framework orchestrated by the transform dialect is the key to managing the complexity of multi-vendor support.1 This approach separates the stable

*mechanism* of transformation (the C++ implementation of transform ops) from the rapidly evolving *policy* of optimization (the target-specific MLIR scripts).

To support the new hardware, the engineering team is directed to create and maintain a library of target-specific transform scripts. These scripts will be invoked by the \-transform-interpreter pass based on the arch key of the orchestra.task being compiled. The initial set of scripts will include:

* **nvidia\_blackwell\_strategy.mlir:** An evolution of the existing Ampere script, updated to leverage new instructions and memory hierarchies of the Blackwell architecture.  
* **amd\_instinct\_cdna3\_strategy.mlir:** This script will be meticulously co-designed with the rocMLIR kernel generator. Its primary goal will be to apply tiling, packing, and fusion patterns at the linalg level that produce IR structures perfectly matching the "contract" for the most performant linalg-to-rock lowering paths, as detailed in Section 3.3.  
* **google\_tpu\_strategy.mlir:** This script will focus on linalg transformations that have a clean and efficient lowering to stablehlo operations. The strategy will be to perform high-level fusions and canonicalizations that simplify the graph before handing it off to the XLA compiler.  
* **aws\_trainium\_strategy.mlir:** The initial strategy for Trainium will be similar to the TPU strategy, focusing on producing clean stablehlo for consumption by the Neuron Compiler. Future iterations may incorporate more advanced logic to decide when to outline a region for compilation via the NKI.

This script-based architecture ensures that the compiler remains agile, allowing performance engineers to tune and adapt optimization strategies for new hardware or models without requiring a full compiler rebuild.

### **5.3 Revised State-of-the-Art Lowering Strategy Matrix**

To provide a definitive, at-a-glance reference for the entire code generation backend, Table 3 from the original blueprint is hereby superseded and replaced with the following expanded matrix. This table codifies the core architectural decisions for lowering from high-level dialects to the specific primitives of each supported vendor. It serves as the primary technical guide for the implementation and future maintenance of the compiler's backend.

| Feature | NVIDIA Blackwell (sm\_100+) | Google TPU (v5e+) | AMD Instinct (CDNA3+) | AWS Trainium (Trn2+) |
| :---- | :---- | :---- | :---- | :---- |
| **High-Level Entry IR** | linalg, tensor | linalg, tensor | linalg, tensor | linalg, tensor |
| **Vendor Handoff IR** | N/A (End-to-End) | stablehlo | N/A (End-to-End) | stablehlo |
| **Kernel Gen. Dialect** | scf, vector | Handled by XLA Compiler | rock (from rocMLIR) | Handled by Neuron Compiler |
| **HW Primitives Dialect** | nvgpu, nvvm | Proprietary (Internal to XLA) | amdgpu, rocdl | Proprietary (Internal to Neuron) |
| **Matrix Acceleration** | nvgpu.tma, nvvm.wmma | XLA-generated systolic array ops | amdgpu.mfma | Neuron-generated systolic array ops |
| **Synchronization** | nvgpu.mbarrier family | Handled by XLA Compiler | gpu.barrier | Handled by Neuron Compiler |
| **Key Memory Abstraction** | memref in shared memory | tensor (in stablehlo) | memref in LDS | tensor (in stablehlo) |
| **Expert Escape Hatch** | PTX Assembly Inline | N/A | GCN Assembly Inline | Neuron Kernel Interface (NKI) |

Table 2: Revised State-of-the-Art Lowering Strategies for Key AI Accelerator Architectures (Supersedes Blueprint Table 3).

### **5.4 Revised End-to-End Compiler Pass Pipeline**

The integration of multiple, distinct lowering paths requires a modification to the overall compiler pipeline. The progressive lowering philosophy remains unchanged, but the final accelerator-specific stage must now incorporate a dispatch mechanism to invoke the correct target-specific pipeline. Table 4 from the original blueprint is updated to reflect this multi-target architecture. The key modification is in Stage 7, "Accelerator Lowering," which now explicitly shows the conditional paths based on the target architecture.

| Stage | Input Dialect(s) | Key Transformation Passes | Output Dialect(s) |
| :---- | :---- | :---- | :---- |
| 1\. Ingestion | torch, tf, onnx | Framework-specific Normalization, Functionalization, Shape Inference | func, linalg |
| 2\. Scheduling | func, linalg, scf | Proprietary Topology-Aware Scheduling Pass | orchestra, scf |
| 3\. High-Level Opt. | orchestra, scf, cf | \-lift-cf-to-scf, divergence-to-speculation (PDL-driven) | orchestra |
| 4\. Structured Lowering | orchestra | orchestra-to-linalg, orchestra-transfer-to-dma | linalg, memref, tensor |
| 5\. Hardware Opt. | linalg, memref, tensor | \-transform-interpreter (with target-specific transform script) | scf, vector, memref, tensor |
| 6\. Bufferization | tensor, scf, vector | \-one-shot-bufferize | memref, scf, vector |
| 7\. **Accelerator Lowering** | scf, vector, memref | **Target-dispatch based on orchestra.task arch attribute:** |  |
|  |  | **NVIDIA Path:** linalg-to-nvgpu, gpu-lower-to-nvvm-pipeline | gpu, nvvm, llvm |
|  |  | **AMD Path:** linalg-to-rock, rock-to-amdgpu | gpu, amdgpu, rocdl, llvm |
|  |  | **Google/AWS Path:** linalg-to-stablehlo | stablehlo |
| 8\. Executable Gen. | gpu, nvvm, llvm, stablehlo | **NVIDIA/AMD:** \-gpu-to-llvm, gpu-module-to-binary | llvm, gpu.binary |
|  |  | **Google/AWS:** Handoff to Vendor Compiler (xla\_compile, neuronx-cc) | Vendor-specific binary (e.g., NEFF) |

Table 3: The Revised, End-to-End Multi-Target Compiler Pass Pipeline for OrchestraOS (Supersedes Blueprint Table 4).

## **Conclusion and Strategic Imperatives**

This blueprint addendum outlines a comprehensive strategy for evolving the OrchestraOS compiler into a state-of-the-art, multi-vendor orchestration platform for AI accelerators. The analysis of the market landscape confirms that while NVIDIA remains the incumbent, the growing influence of AMD in the discrete GPU market and the strategic importance of custom silicon from Google and Amazon make them essential targets for support. The successful implementation of this expanded vision hinges on the adherence to several cross-cutting strategic imperatives that build upon the foundation of the original blueprint.

First, the architecture must **embrace open standards for portability and interoperability.** The decision to target StableHLO as the handoff point for both Google TPUs and AWS Trainium is a prime example of this principle. By leveraging a common, vendor-supported IR, OrchestraOS can integrate with these powerful ecosystems efficiently, reducing engineering costs and future-proofing the architecture against platform-specific changes.

Second, the compiler must enable **deep, co-designed optimization for peak performance.** The "white-box" integration strategy with AMD's rocMLIR exemplifies this imperative. True performance is achieved not by treating vendor compilers as opaque black boxes, but by using high-level declarative frameworks like the transform dialect to intelligently guide them to their most efficient code generation paths. This requires a deep understanding of the downstream compiler's internal "contract" and represents a significant source of competitive differentiation.

Third, the system must provide **expert-level control and extensibility.** The proposed integration with the AWS Neuron Kernel Interface (NKI) provides a critical "escape hatch" for performance engineers to achieve results beyond the capabilities of any general-purpose compiler. By making such advanced features a first-class part of the OrchestraIR dialect, the compiler empowers expert users to solve the most demanding performance challenges.

By executing on the technical directives outlined in this document—extending the core IR, authoring target-specific optimization policies, and implementing the multi-path lowering pipelines—OrchestraOS will not merely be adding support for new devices. It will be realizing its core architectural vision as a true "meta-OS" for the next generation of artificial intelligence, capable of intelligently and performantly orchestrating workloads across the entire landscape of high-performance AI hardware. This capability represents a profound and defensible technological advantage in an increasingly heterogeneous computing world.

#### **Referenzen**

1. orchestra \- tech \- MLIR 20 Blueprint  
2. MLIR (software) \- Wikipedia, Zugriff am August 23, 2025, [https://en.wikipedia.org/wiki/MLIR\_(software)](https://en.wikipedia.org/wiki/MLIR_\(software\))  
3. MLIR, Zugriff am August 23, 2025, [https://mlir.llvm.org/](https://mlir.llvm.org/)  
4. AI Hardware Market Size & Share, Statistics Report 2025-2034, Zugriff am August 23, 2025, [https://www.gminsights.com/industry-analysis/ai-hardware-market](https://www.gminsights.com/industry-analysis/ai-hardware-market)  
5. Data Center Accelerator Market Size | Industry Report, 2030 \- Grand View Research, Zugriff am August 23, 2025, [https://www.grandviewresearch.com/industry-analysis/data-center-accelerator-market-report](https://www.grandviewresearch.com/industry-analysis/data-center-accelerator-market-report)  
6. Data Center Accelerator Market To Reach $63.22Bn By 2030, Zugriff am August 23, 2025, [https://www.grandviewresearch.com/press-release/global-data-center-accelerator-market](https://www.grandviewresearch.com/press-release/global-data-center-accelerator-market)  
7. www.fool.com, Zugriff am August 23, 2025, [https://www.fool.com/investing/2025/08/22/what-are-3-great-tech-stocks-to-buy-right-now/\#:\~:text=Nvidia%20(NVDA%20%2D0.40%25),%2C%20was%2092%25%20in%20Q1.](https://www.fool.com/investing/2025/08/22/what-are-3-great-tech-stocks-to-buy-right-now/#:~:text=Nvidia%20\(NVDA%20%2D0.40%25\),%2C%20was%2092%25%20in%20Q1.)  
8. Prediction: This Artificial Intelligence (AI) Stock Will Outperform Nvidia Through 2030, Zugriff am August 23, 2025, [https://www.nasdaq.com/articles/prediction-artificial-intelligence-ai-stock-will-outperform-nvidia-through-2030](https://www.nasdaq.com/articles/prediction-artificial-intelligence-ai-stock-will-outperform-nvidia-through-2030)  
9. “Nvidia holds about 80% of the AI chip market.” \- YouTube, Zugriff am August 23, 2025, [https://www.youtube.com/watch?v=XJDcb-R373A](https://www.youtube.com/watch?v=XJDcb-R373A)  
10. How Nvidia's AI Made It the World's Most Valuable Firm | Technology Magazine, Zugriff am August 23, 2025, [https://technologymagazine.com/articles/how-nvidias-ai-made-it-the-worlds-most-valuable-firm](https://technologymagazine.com/articles/how-nvidias-ai-made-it-the-worlds-most-valuable-firm)  
11. AI compute: Nvidia's Grip and AMD's Chance \- UncoverAlpha, Zugriff am August 23, 2025, [https://www.uncoveralpha.com/p/ai-compute-nvidias-grip-and-amds](https://www.uncoveralpha.com/p/ai-compute-nvidias-grip-and-amds)  
12. Is NVIDIA's 95% Market Share Here to Stay? The Future of AI Hardware: Training vs Inference \- YouTube, Zugriff am August 23, 2025, [https://www.youtube.com/watch?v=XzCX5HLkQ\_A](https://www.youtube.com/watch?v=XzCX5HLkQ_A)  
13. Best AI Stocks for 2025: Artificial Intelligence Investing | The Motley Fool, Zugriff am August 23, 2025, [https://www.fool.com/investing/stock-market/market-sectors/information-technology/ai-stocks/](https://www.fool.com/investing/stock-market/market-sectors/information-technology/ai-stocks/)  
14. AMD's Stealth Advance in AI Data Centers: Why Market Share Growth Is Underestimated, Zugriff am August 23, 2025, [https://www.ainvest.com/news/amd-stealth-advance-ai-data-centers-market-share-growth-underestimated-2506/](https://www.ainvest.com/news/amd-stealth-advance-ai-data-centers-market-share-growth-underestimated-2506/)  
15. Advancing AI 2025: AMD's Latest Products Focused on Openness \- Counterpoint Research, Zugriff am August 23, 2025, [https://www.counterpointresearch.com/insight/advancing-ai-2025-amds-latest-products-focused-on-openness](https://www.counterpointresearch.com/insight/advancing-ai-2025-amds-latest-products-focused-on-openness)  
16. AMD Stock Forecast: NASDAQ:AMD Targets $200 With 24% Upside on AI Momentum, Zugriff am August 23, 2025, [https://www.tradingnews.com/news/amd-stock-forecast-nasdsaq-amd-targets-200-usd-with-24-percent-upside-on-ai-momentum](https://www.tradingnews.com/news/amd-stock-forecast-nasdsaq-amd-targets-200-usd-with-24-percent-upside-on-ai-momentum)  
17. AMD vs. APH: Which Tech Supply Chain Stock Is a Better Buy Now? \- Nasdaq, Zugriff am August 23, 2025, [https://www.nasdaq.com/articles/amd-vs-aph-which-tech-supply-chain-stock-better-buy-now](https://www.nasdaq.com/articles/amd-vs-aph-which-tech-supply-chain-stock-better-buy-now)  
18. AMD's Strategic Position in the AI and HPC Market: Balancing Long-Term Growth with Near-Term Realities \- AInvest, Zugriff am August 23, 2025, [https://www.ainvest.com/news/amd-strategic-position-ai-hpc-market-balancing-long-term-growth-term-realities-2508/](https://www.ainvest.com/news/amd-strategic-position-ai-hpc-market-balancing-long-term-growth-term-realities-2508/)  
19. Google's Decade Long Bet on Custom Chips Is the AI Wild Card Wall Street Hasn't Priced In, Profitable Businesses Will Run on ASICs, Not GPUs, GPT-5 making it move? : r/investing \- Reddit, Zugriff am August 23, 2025, [https://www.reddit.com/r/investing/comments/1mlui74/googles\_decade\_long\_bet\_on\_custom\_chips\_is\_the\_ai/](https://www.reddit.com/r/investing/comments/1mlui74/googles_decade_long_bet_on_custom_chips_is_the_ai/)  
20. Tensor Processing Units (TPUs) \- Google Cloud, Zugriff am August 23, 2025, [https://cloud.google.com/tpu](https://cloud.google.com/tpu)  
21. \[D\] Why does it seem like Google's TPU isn't a threat to nVidia's GPU? \- Reddit, Zugriff am August 23, 2025, [https://www.reddit.com/r/MachineLearning/comments/1g1okem/d\_why\_does\_it\_seem\_like\_googles\_tpu\_isnt\_a\_threat/](https://www.reddit.com/r/MachineLearning/comments/1g1okem/d_why_does_it_seem_like_googles_tpu_isnt_a_threat/)  
22. Tensor Processing Unit Market Size & Share Report, 2030 \- Grand View Research, Zugriff am August 23, 2025, [https://www.grandviewresearch.com/industry-analysis/tensor-processing-unit-market-report](https://www.grandviewresearch.com/industry-analysis/tensor-processing-unit-market-report)  
23. Cloud TPU release notes | Google Cloud, Zugriff am August 23, 2025, [https://cloud.google.com/tpu/docs/release-notes](https://cloud.google.com/tpu/docs/release-notes)  
24. Amazon Steps Up Effort to Rival Nvidia in AI Chip Market \- Nasdaq, Zugriff am August 23, 2025, [https://www.nasdaq.com/articles/amazon-steps-effort-rival-nvidia-ai-chip-market](https://www.nasdaq.com/articles/amazon-steps-effort-rival-nvidia-ai-chip-market)  
25. Amazon's AI chip ambitions: challenging Nvidia's dominance \- Vested Finance, Zugriff am August 23, 2025, [https://vestedfinance.com/blog/us-stocks/amazons-ai-chip-ambitions-challenging-nvidias-dominance/](https://vestedfinance.com/blog/us-stocks/amazons-ai-chip-ambitions-challenging-nvidias-dominance/)  
26. Amazon's AI Ambition: Can AWS Power a $3 Trillion Valuation by 2030? \- AInvest, Zugriff am August 23, 2025, [https://www.ainvest.com/news/amazon-ai-ambition-aws-power-3-trillion-valuation-2030-2506/](https://www.ainvest.com/news/amazon-ai-ambition-aws-power-3-trillion-valuation-2030-2506/)  
27. AI Accelerator \- AWS Trainium, Zugriff am August 23, 2025, [https://aws.amazon.com/ai/machine-learning/trainium/](https://aws.amazon.com/ai/machine-learning/trainium/)  
28. Introduction to Cloud TPU \- Google Cloud, Zugriff am August 23, 2025, [https://cloud.google.com/tpu/docs/intro-to-tpu](https://cloud.google.com/tpu/docs/intro-to-tpu)  
29. OpenXLA is available now to accelerate and simplify machine learning | Google Open Source Blog, Zugriff am August 23, 2025, [https://opensource.googleblog.com/2023/03/openxla-is-ready-to-accelerate-and-simplify-ml-development.html](https://opensource.googleblog.com/2023/03/openxla-is-ready-to-accelerate-and-simplify-ml-development.html)  
30. What about TVM, XLA, and AI compilers? (Democratizing AI Compute, Part 6\) \- Modular, Zugriff am August 23, 2025, [https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers](https://www.modular.com/blog/democratizing-ai-compute-part-6-what-about-ai-compilers)  
31. SDK for Gen AI and Deep Learning \- AWS Neuron, Zugriff am August 23, 2025, [https://aws.amazon.com/ai/machine-learning/neuron/](https://aws.amazon.com/ai/machine-learning/neuron/)  
32. XLA Terminology | OpenXLA Project, Zugriff am August 23, 2025, [https://openxla.org/xla/terminology](https://openxla.org/xla/terminology)  
33. StableHLO Specification | OpenXLA Project, Zugriff am August 23, 2025, [https://openxla.org/stablehlo/spec](https://openxla.org/stablehlo/spec)  
34. XLA architecture \- OpenXLA Project, Zugriff am August 23, 2025, [https://openxla.org/xla/architecture](https://openxla.org/xla/architecture)  
35. Compilers: Talking to The Hardware \- Unify, Zugriff am August 23, 2025, [https://unify.ai/blog/deep-learning-compilers](https://unify.ai/blog/deep-learning-compilers)  
36. OpenXLA Dev Lab 2024: Building Groundbreaking ML Systems Together, Zugriff am August 23, 2025, [https://opensource.googleblog.com/2024/05/openxla-dev-lab-2024-building-grouundbreaking-systems-together.html](https://opensource.googleblog.com/2024/05/openxla-dev-lab-2024-building-grouundbreaking-systems-together.html)  
37. What is the difference between Tensorflow XLA and Tensorflow Lite / Android NNAPI?, Zugriff am August 23, 2025, [https://stackoverflow.com/questions/53656693/what-is-the-difference-between-tensorflow-xla-and-tensorflow-lite-android-nnap](https://stackoverflow.com/questions/53656693/what-is-the-difference-between-tensorflow-xla-and-tensorflow-lite-android-nnap)  
38. \[2210.15016\] TPU-MLIR: A Compiler For TPU Using MLIR \- ar5iv, Zugriff am August 23, 2025, [https://ar5iv.labs.arxiv.org/html/2210.15016](https://ar5iv.labs.arxiv.org/html/2210.15016)  
39. TPU-MLIR: A Compiler For TPU Using MLIR \- ResearchGate, Zugriff am August 23, 2025, [https://www.researchgate.net/publication/364814226\_TPU-MLIR\_A\_Compiler\_For\_TPU\_Using\_MLIR](https://www.researchgate.net/publication/364814226_TPU-MLIR_A_Compiler_For_TPU_Using_MLIR)  
40. TPU-MLIR: A Compiler For TPU Using MLIR, Zugriff am August 23, 2025, [https://arxiv.org/abs/2210.15016](https://arxiv.org/abs/2210.15016)  
41. arXiv:2210.15016v2 \[cs.PL\] 9 Feb 2023, Zugriff am August 23, 2025, [https://arxiv.org/pdf/2210.15016](https://arxiv.org/pdf/2210.15016)  
42. Towards a high-performance AI compiler with upstream MLIR \- arXiv, Zugriff am August 23, 2025, [https://arxiv.org/html/2404.15204v1](https://arxiv.org/html/2404.15204v1)  
43. \[RFC\] Add StableHLO \=\> Linalg lowering to openxla/stablehlo \- Google Groups, Zugriff am August 23, 2025, [https://groups.google.com/a/openxla.org/g/openxla-discuss/c/KsRp9euuuB0](https://groups.google.com/a/openxla.org/g/openxla-discuss/c/KsRp9euuuB0)  
44. ROCm \- Wikipedia, Zugriff am August 23, 2025, [https://en.wikipedia.org/wiki/ROCm](https://en.wikipedia.org/wiki/ROCm)  
45. ROCm/rocMLIR \- GitHub, Zugriff am August 23, 2025, [https://github.com/ROCm/rocMLIR](https://github.com/ROCm/rocMLIR)  
46. GPU Compilation with MLIR \- Reddit, Zugriff am August 23, 2025, [https://www.reddit.com/r/Compilers/comments/1k5vra3/gpu\_compilation\_with\_mlir/](https://www.reddit.com/r/Compilers/comments/1k5vra3/gpu_compilation_with_mlir/)  
47. rocMLIR techniques vs Linalg \#1289 \- GitHub, Zugriff am August 23, 2025, [https://github.com/ROCm/rocMLIR/discussions/1289](https://github.com/ROCm/rocMLIR/discussions/1289)  
48. 'amdgpu' Dialect \- MLIR, Zugriff am August 23, 2025, [https://mlir.llvm.org/docs/Dialects/AMDGPU/](https://mlir.llvm.org/docs/Dialects/AMDGPU/)  
49. Dialects \- MLIR, Zugriff am August 23, 2025, [https://mlir.llvm.org/docs/Dialects/](https://mlir.llvm.org/docs/Dialects/)  
50. mlir::ROCDL Namespace Reference \- LLVM, Zugriff am August 23, 2025, [https://mlir.llvm.org/doxygen/namespacemlir\_1\_1ROCDL.html](https://mlir.llvm.org/doxygen/namespacemlir_1_1ROCDL.html)  
51. Deploying Multimodal Models at Scale with AWS Neuron SDK \- XenonStack, Zugriff am August 23, 2025, [https://www.xenonstack.com/blog/multimodal-models-aws-neuron-sdk](https://www.xenonstack.com/blog/multimodal-models-aws-neuron-sdk)  
52. Machine Learning \- Compiler Engineer II, AWS Neuron, Annapurna Labs \- Amazon.jobs, Zugriff am August 23, 2025, [https://amazon.jobs/en/jobs/2996426/machine-learning-compiler-engineer-ii-aws-neuron-annapurna-labs](https://amazon.jobs/en/jobs/2996426/machine-learning-compiler-engineer-ii-aws-neuron-annapurna-labs)  
53. Sr. ML Compiler Engineer, AWS Neuron, Annapurna Labs \- Job ID: 2991510 | Amazon.jobs, Zugriff am August 23, 2025, [https://www.amazon.jobs/en-gb/jobs/2991510/sr-ml-compiler-engineer-aws-neuron-annapurna-labs](https://www.amazon.jobs/en-gb/jobs/2991510/sr-ml-compiler-engineer-aws-neuron-annapurna-labs)  
54. Neuron Architecture — AWS Neuron Documentation, Zugriff am August 23, 2025, [https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/index.html](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/index.html)