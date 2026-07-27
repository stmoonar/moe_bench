---
title: "ParallelKittens: Systematic and Practical Simplification of Multi-GPU AI Kernels"
source: "https://arxiv.org/html/2511.13940v1"
author:
published:
created: 2026-07-05
description:
tags:
  - "clippings"
---
Stuart H. Sul Simran Arora Benjamin F. Spector Christopher Ré

###### Abstract

Inter-GPU communication has become a major bottleneck for modern AI workloads as models scale and improvements in hardware compute throughput outpace improvements in interconnect bandwidth. Existing systems mitigate this through compute-communication overlap but often fail to meet theoretical peak performance across heterogeneous workloads and new accelerators. Instead of operator-specific techniques, we ask whether a small set of simple, reusable principles can systematically guide the design of optimal multi-GPU kernels. We present ParallelKittens (PK), a minimal CUDA framework that drastically simplifies the development of overlapped multi-GPU kernels. PK extends the ThunderKittens framework and embodies the principles of multi-GPU kernel design through eight core primitives and a unified programming template, derived from a comprehensive analysis of the factors that govern multi-GPU performance—data-transfer mechanisms, resource scheduling, and design overheads. We validate PK on both Hopper and Blackwell architectures. With fewer than 50 lines of device code, PK achieves up to $2.33\times$ speedup for data- and tensor-parallel workloads, $4.08\times$ for sequence-parallel workloads, and $1.22\times$ for expert-parallel workloads.

## 1 Introduction

![Refer to caption](https://arxiv.org/html/2511.13940v1/figures/pull-figure.png)

Figure 1: We study the principles for high performance multi-GPU kernels and introduce ParallelKittens ( PK ), an opinionated collection of programming primitives to encapsulate these principles. The GPU memory hierarchy and corresponding abstractions are shown on the left (Section 3.2.1 ), and the program template with its key multi-GPU kernel components is shown on the right (Section 3.2.3 ).

A few years ago, GPU compute utilization was often limited by intra-GPU memory access. However, IO-aware algorithms like FlashAttention [^4], domain-specific languages (DSLs) that support efficient mapping of operators to hardware [^29] [^18] [^27], and the continued scaling of AI models have have left inter-GPU communication as the primary remaining bottleneck. Even with high-speed interconnects like NVLink [^15] and compute-friendly phases like prefill, communication can occupy over $50\%$ of execution time in large language model (LLM) workloads, leaving GPU compute idle [^3]. The problem is compounded by the relatively slow improvements in communication hardware: from the Nvidia A100 [^17] to the B200 [^20], BF16 tensor core performance improved by $7.2\times$ and High Bandwidth Memory (HBM) bandwidth by $5.1\times$, while intra-node communication (NVLink) improved by only $3\times$ and inter-node (PCIe/InfiniBand) by just $2\times$.

To mitigate communication overhead, prior methods overlap inter-GPU communication with intra-GPU computation for common operators like General Matrix Multiplication (GEMM), attention, and Mixture-of-Experts (MoE) layers [^13] [^3] [^32] [^35] [^31] [^1]. These approaches reduce non-overlapped communication time in data, tensor, sequence, and expert parallelism [^25] [^12], which are common strategies for distributing industry-scale training and inference across many GPUs. However, prior works either (i) rely on bespoke kernels for specific AI operators and depend on complex low-level primitives (e.g., CUTLASS, NVSHMEM, Linux IPC), (ii) employ compiler-based approaches that fail to adapt to new accelerators—occasionally generating kernels slower than non-overlapped baselines—or (iii) utilize off-the-shelf libraries, resulting in up to $4.08\times$ slower performance than hand-tuned implementations.

As hardware shifts toward unified multi-GPU systems—illustrated by Nvidia’s roadmap from NVL72 to NVL144 (2026) and NVL576 (2027) [^21] —we would need simple, general principles and programming primitives that enable peak-performance multi-GPU operations. In this work, we identify three key principles for designing efficient multi-GPU kernels and analyze each in detail (Section 3.1).

1. Transfer mechanism. Inter-GPU networking relies on three mechanisms—copy engines, tensor memory accelerators (TMA), and register-level instructions—that differ in maximum bandwidth, effective message granularity, supported functionality, and compute occupancy. Understanding these trade-offs and choosing the right mechanism is crucial for peak performance. For instance, copy engines achieve the highest efficiency (81% of theoretical maximum) but require large messages ($\geq$ 256 MB) for saturation. TMA attains near-peak throughput (74%) with only 2 KB messages (Figure 2). Register-level instructions operate efficiently at a 128 B granularity but need about 76 streaming multiprocessors (SMs) to saturate bandwidth (70%), whereas TMA needs 15 (Figure 3). However, only register-level instructions support in-network reduction. Existing systems do not capture these trade-offs; for instance, Triton Distributed, Flux, and CUTLASS rely on the copy engine for intra-node all-gather GEMM, becoming slower than the non-overlapped baseline on smaller matrix sizes (Figure 7).
2. Scheduling. The distribution of compute and communication work across SMs must be chosen based on workload characteristics. We identify inter-SM and intra-SM overlapping as the two primary scheduling strategies, trading off compute utilization and communication versatility. Intra-SM overlapping is preferred when computation and communication granularities align; for example, in GEMM reduce-scatter, intra-SM overlapping outperforms inter-SM by $1.2\times$. In contrast, inter-SM overlapping enables communication patterns that can significantly reduce transfer size. For instance, leveraging in-network reduction through inter-SM overlapping achieves a $3.62\times$ performance improvement for GEMM all-reduce (Figure 5) and $1.57\times$ for all-gather GEMM. No prior work explores both scheduling strategies; existing methods either rely on a single type or omit device-side overlapping altogether, thereby failing to generalize (e.g., applying the Flux intra-SM overlapping design to GEMM all-reduce would lead to the slowdown above).
3. Design overheads. Widely used communication libraries (e.g., NCCL, NVSHMEM) encapsulate design choices—specifically in synchronization and buffering—that favor simplicity over performance. We show that the choices in prior libraries can cause over $1.7\times$ performance loss in pure communication kernels (e.g., all-reduce) and up to $4.5\times$ higher communication latency. By adopting a design that enables explicit user control over memory allocation and synchronization, these overheads can be substantially reduced.

Building on these insights, we introduce ParallelKittens (PK), an opinionated collection of C++ embedded programming primitives that extends the ThunderKittens (TK) framework [^27] (Section 3.2). PK exposes only the most efficient transfer mechanisms for each functionality (e.g., TMA for point-wise communication, register operations for in-network acceleration), provides minimal synchronization primitives and a general program template that simplifies achieving both inter- and intra-SM overlapping scheduling, and offers full control over performance-critical components (e.g., NVLink transfers) while abstracting away non-essential multi-GPU complexities (e.g., inter-process communication and virtual memory exchange).

We validate PK across diverse parallel AI workloads on both Hopper and Blackwell architectures, including data, tensor, sequence, and expert parallelism (i.e., fused parallel GEMMs, distributed attention variants, and MoEs). Compared with the strongest baselines, PK achieves up to $2.33\times$ higher compute throughput (FLOP/s) for data- and tensor-parallel workloads, $4.08\times$ for sequence-parallel workloads, and $1.22\times$ for expert-parallel workloads, effectively reducing non-overlapped communication time down to $1\%$, $9\%$, and $15\%$, respectively. PK matches the performance of the strongest hand-optimized kernels (Flux, Comet, CUTLASS), outperforms compiler-based approaches (Triton Distributed) by $1.07$ – $5.63\times$, and surpasses communication library-based approaches (xDiT, YunChang) by $1.01$ – $4.08\times$ across varying problem sizes.

Each PK kernel required fewer than 50 lines of additional device code beyond the original single-GPU GEMM or attention kernels. The complete implementation of PK, including its kernels, is fully open-sourced and is currently being adopted at Cursor for large-scale in-house training.

To summarize, our contributions are:

- A detailed analysis of multi-GPU programming that decomposes performance into interpretable factors (transfer mechanisms, scheduling strategies, and design overheads) and validates each with microbenchmarks.
- ParallelKittens, a minimal collection of multi-GPU primitives and a unified programming template that extends the familiar ThunderKittens framework.
- Kernels built with ParallelKittens that match or surpass hand-optimized kernel performance while substantially reducing code complexity.

## 2 Background

In this section, we provide background on modern datacenter-grade GPUs and review prior efforts on optimizing multi-GPU AI kernels.<sup>1</sup>

### 2.1 GPU Architecture

A GPU kernel loads data from HBM, performs computation, and writes the results back to HBM. Multi-GPU kernels distribute the workload across multiple GPUs and access the HBMs of all devices.

##### GPU hierarchy.

GPU kernels execute tens of thousands of hardware threads in parallel across over a hundred streaming multiprocessors (SMs). Memory farther from the SM provides greater capacity at higher latency. Each SM contains 64 KB of registers private to individual threads and accessible every clock cycle. Threads are organized into thread blocks, each executing on a single assigned SM. Threads in a thread block communicate via 227 KB of shared memory (SMEM), a per-SM on-chip SRAM offering up to 33 TB/s of bandwidth. All threads share a 50 MB L2 cache ($\approx$ 12 TB/s) connected to 80 GB HBM (3 TB/s). Threads can also access peer GPU HBM over NVLink (450 GB/s unidirectional), enabling multi-GPU kernel development.

##### GPU networking.

Multi-GPU systems rely on a hierarchy of interconnects. PCIe (64 GB/s) is the channel for CPU-to-GPU (e.g., kernel launches, host-initiated transfers) and multinode communication over InfiniBand/TCP. NVLink (450 GB/s) provides point-to-point connections between GPUs and the NVSwitch; NVSwitch interconnects all NVLink endpoints into a non-blocking fabric for full GPU-to-GPU communication. NVSwitch also supports in-network, off-device acceleration for multicast and reduction. Unless otherwise noted, all inter-GPU communication in this paper occurs via NVLink/NVSwitch.

##### Execution overlap.

GPUs contain various execution units specialized for different compute, memory, and communication operations. For compute, Tensor Cores perform tiled matrix multiplications, while CUDA Cores handle element-wise arithmetic. For memory, the Tensor Memory Accelerator (TMA) performs bulk data transfers between SMEM and HBM and can be invoked asynchronously by a single thread. Alternatively, a per-GPU copy engine (dedicated DMA unit) moves large contiguous regions of device memory independently of the SMs and is invoked from the host.

Within an SM, threads can concurrently issue instructions to different execution units. Achieving optimal performance therefore depends on effectively overlapping their use to hide non-critical operations and maximize the throughput of critical ones. We distinguish inter-SM overlapping, where entire SMs are dedicated almost exclusively to compute, memory, or communication tasks, from intra-SM overlapping, where different warps or threads within the same SM concurrently drive compute, memory, or inter-GPU traffic. These resources saturate at different rates, creating opportunities for various overlapping strategies.

### 2.2 Related Works

We are inspired by the extensive amount of work that accelerates multi-GPU AI workloads.

##### Operator-specific kernels.

Many prior works hand-tune particular AI operators by overlapping computation and communication, e.g., TP-Async [^9], Flux [^3], Ring Attention [^13], DeepEP [^5], Comet [^31], FlashDMoE [^1], and several distributed GEMM kernels from CUTLASS [^28]. These approaches employ techniques ranging from overlapping host-triggered copies with device kernels, to highly optimized on-device schedulers and device-initiated communication. While these systems deliver strong performance for specific targets, they demand complex implementations and offer limited reusable abstractions. For instance, FlashDMoE is optimized only for TF32 precision, with BF16/FP16 support still under development five months after its release. In contrast, PK distills general principles applicable across diverse workloads, achieving speedups comparable to hand-optimized kernels while simplifying implementation.

##### Scheduling frameworks.

Frameworks such as Megatron-LM [^25], FlexFlow [^11], and NanoFlow [^37] automate parallelization and scheduling, and are complementary to PK. These systems primarily orchestrate bulk collective operations (e.g., NCCL), which require synchronization before and after data transfers, and employ stream-level overlap. NanoFlow offers finer-grained scheduling by partitioning SMs among compute, memory, and network operations to saturate available bandwidth without full occupancy (i.e., inter-SM overlapping). However, achieving peak kernel performance also requires intra-SM warp specialization with device-initiated, tile-level transfers; PK provides that layer.

##### Multi-GPU programming primitives.

DSLs and libraries have been proposed to simplify multi-GPU kernel development [^22] [^16] [^35] [^2] [^8]. Triton Distributed [^35] and TileLink [^36] extend Triton [^29] with OpenSHMEM-style one-sided operations, enabling compiler-based generation of multi-GPU kernels. However, these approaches lack explicit workload distribution control (e.g., warp or SM specialization) needed for optimal overlap. Also, our benchmarks show that Triton Distributed, originally tuned for H800 GPUs, fails to adapt efficiently to other architectures such as H100s (Section 4). In contrast, PK provides a lightweight C++ layer that enables direct control over communication workload distribution, enabling arbitrary scheduling and optimization across Hopper and Blackwell GPUs. NCCLX [^26] complements PK by accelerating inter-node collectives for large clusters ($\geq 100$ k GPUs), but does not exploit device-initiated asynchronous overlapping (via TMA) or in-network acceleration, both critical for fine-grained overlap with peak bandwidth utilization.

## 3 ParallelKittens

We present our analysis of the design tradeoffs of multi-GPU kernels and present ParallelKittens.

### 3.1 Analysis

We start with a general, high-level cost model that provides a roadmap for the analysis.

#### 3.1.1 Cost Model

The objective of designing a multi-GPU kernel is to minimize its total wall-clock time $T_{\text{kernel}}$, which reflects the combined cost of compute, memory, and communication operations. The key contributors include:

$$
\begin{split}T_{\text{kernel}}=T_{\text{launch}}+\max(T_{\text{comp}},\,T_{\text{mem}},\,T_{\text{comm}})+\,T_{\text{non-overlap}}+T_{\text{sync}}\end{split}
$$

In this simple model, $T_{\text{launch}}$ denotes the per-kernel launch cost, including host-side latency and per-thread block setup and teardown (e.g., tensor memory allocation and pipeline fill/drain phases). $T_{\text{comp}}$, $T_{\text{mem}}$, and $T_{\text{comm}}$ represent the full-pipeline time spent on computation, memory access, and communication, respectively. Ideally, these components overlap so that the total time equals the maximum of the three, but $T_{\text{non-overlap}}$ accounts for operations that cannot be overlapped. The cost of each component (e.g., $T_{\text{comm}}$) depends on the work size ($S_{\text{comm}}$) and achievable bandwidth ($B_{\text{comm}}$), i.e., $T_{\text{comm}}=S_{\text{comm}}/B_{\text{comm}}$. Finally, $T_{\text{sync}}$ captures the synchronization overhead across SMs or devices.

These costs are controlled by three design decisions: first, the specific transfer mechanism that we select to move data between GPUs (Section 3.1.2); second, the kernel scheduling strategy for overlapping computation and communication (Section 3.1.3); and third, the communication abstraction’s design choices, including peer-memory allocation, management, and access (Section 3.1.4).

#### 3.1.2 Transfer Mechanism

We now discuss the choice of communication mechanism.

Table 1: The observed NVLink bandwidth utilization (GB/s) when using all SMs to transfer 1GB of data, and its ratio to the theoretical maximum (450 GB/s for H100s, 900 GB/s for B200s).

| Method | H100 BW (Ratio) | B200 BW (Ratio) |
| --- | --- | --- |
| Copy Engine | 368.82 (82%) | 726.13 (81%) |
| TMA Op | 350.01 (78%) | 669.12 (74%) |
| Register Op | 342.68 (76%) | 628.35 (70%) |

##### Host versus device-initiated communication.

The per-GPU copy engine is host-initiated and supports only contiguous memory transfers. As shown in Table 1, it achieves the highest throughput for large, all-at-once data movements. However, when fine-grained communication is required (e.g., all-to-all communication in MoEs), performance degrades significantly because additional overhead is incurred for data rearrangement or repeated transfer invocations. Figure 2 illustrates this behavior. To sustain over 80% bandwidth utilization, the transfer granularity must be at least 256 MB when using the copy engine, whereas device-side methods achieve comparable utilization with only 2 KB.

Consequently, PK relies exclusively on device-side communication for the following reasons. First, host-initiated transfers are suitable primarily for large contiguous data blocks (e.g., weight movements in fully sharded data parallelism [^33]). In such cases, overlapping computation and communication is often trivial: the host transfer and device kernel can be launched on separate streams without kernel modifications. Second, although the copy engine has the advantage of not occupying SM resources, only a small number of SMs are needed to saturate the interconnect bandwidth using device-initiated communication, as shown in Figure 3. Moreover, intra-SM overlapping enables computation to proceed concurrently with that communication.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x1.png)

Figure 2: Observed memory bandwidth utilization for a 1 GB peer-to-peer transfer over NVLink. For device-initiated (TMA) transfers, the maximum supported message size is 227 KB; throughput values beyond this limit are held constant for visual comparison.

##### Device-initiated communication mechanisms.

There are two main mechanisms for device-initiated communication in modern datacenter-grade GPUs:

1. The first is via the Tensor Memory Accelerator (TMA), which supports NVLink transfers and NVSwitch-accelerated broadcasts. A key advantage of TMA is that it can be launched asynchronously by a single thread without increasing register pressure, allowing other threads in the same SM to overlap the execution of compute or memory work (intra-SM overlap).
2. The second is via plain register-level instructions (e.g., ld, st). As shown in Table 1, they are relatively inefficient, achieving about 70% of the peak bandwidth on B200 GPUs. Because these instructions are synchronous and operate at the register level, saturating NVLink bandwidth requires full SM occupancy—thousands of threads issuing instructions concurrently—as well as higher register pressure and manual memory coalescing.

Table 2: Different multi-GPU transfer mechanisms (copy engine, TMA, and register operations) and supported functionalities.

| Functionality | CE | TMA | Reg |
| --- | --- | --- | --- |
| P2P Transfer | ✓ | ✓ | ✓ |
| In-fabric Broadcast | ✓ | ✓ | ✓ |
| P2P Reduction | $\times$ | ✓ | ✓ |
| In-fabric Reduction | $\times$ | $\times$ | ✓ |
| Elementwise Transfer | $\times$ | $\times$ | ✓ |

We find that these mechanisms excel in different scenarios. As illustrated in Figure 3, register-level operations require $3.2$ – $5.1\times$ more SMs than TMA to saturate NVLink bandwidth, leaving little opportunity for intra-SM overlap. Register instructions are therefore useful when neither the copy engine nor TMA provides the required functionality. A representative case is NVSwitch in-network reduction (e.g., multimem.ld\_reduce and multimem.red), which can substantially speed up workloads like all-reduce. Existing communication libraries do not exploit this design space; for instance, NVSHMEM relies exclusively on register-level operations for intra-node data transfers. Table 2 summarizes the functionalities supported by each mechanism.

#### 3.1.3 Scheduling

We now examine workload scheduling strategies for multi-GPU kernels. There are two main ways to overlap compute and communication within a kernel:

1. Intra-SM overlapping partitions the threads within an SM into two pools: one issuing compute/memory instructions and the other issuing communication instructions.
2. Inter-SM overlapping partitions the SMs into two pools: one for computation and the other for communication.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x2.png)

Figure 3: The number of SMs it takes to saturate NVLink Bandwidth, using different communication mechanisms.

While prior work primarily uses inter-SM overlapping [^37] [^31], we find that each provides different compute-communication trade-offs depending on workload characteristics.<sup>2</sup>

##### Intra-SM overlapping.

Intra-SM overlapping is effective when the ideal communication pattern aligns with that of computation, allowing communication to be naturally embedded within the computation pipeline. In such cases, it is superior to inter-SM overlapping for two main reasons:

1. Unlike in inter-SM overlapping, all compute units (i.e., tensor cores) across all SMs are busy in an intra-SM overlapping scheme. This is crucial because, unlike communication bandwidth, compute throughput scales linearly with the number of SMs that perform computation.
2. Inter-SM communication incurs additional synchronization overhead $T_{\text{sync}}$, as it must traverse the HBM. Our microbenchmarks show that a single intra-SM synchronization using mbarrier objects incurs approximately 64 ns of latency, whereas inter-SM synchronization through the HBM takes about 832 ns.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x3.png)

Figure 4: GEMM reduce-scatter (RS) and all-reduce (AR) performance across overlapping schedules. Measured on 8 × 8\\times H100 GPUs with local GEMM shape N / N\\times N\\times N/8 ( = 32768 N=32768 ) and element type BF16.

We illustrate these effects using a kernel that fuses a GEMM with a reduce-scatter (RS). Figure 4 (left) shows that the GEMM+RS kernel achieves higher compute throughput under an intra-SM overlapping schedule, due to higher compute utilization and lower synchronization overhead.

We further show that intra-SM overlapping can almost completely hide communication overhead in certain regimes. Consider an $M\times N\times K$ GEMM+RS fused kernel with per-iteration tiles of size $m\times n\times k$. In a typical GEMM kernel, an output tile region is selected, and the $m\times n\times k$ sub-GEMM is executed $K/k$ times before the result is stored.

Given the per-element size $s$, sustained tensor core throughput $R$ (in FLOP/s), and per-GPU NVLink bandwidth $B$ (in bytes/s), the compute and communication times for producing a single output tile of size $m\times n$ are given by:

$$
T_{\text{comp\_tile}}=\frac{2mnk}{R}\times\frac{K}{k}=\frac{2mnK}{R}
$$
 
$$
T_{\text{comm\_tile}}=\frac{smn}{B}
$$

From this, communication can be completely hidden by computation when $T_{\text{comp\_tile}}\geq T_{\text{comm\_tile}}$, i.e.,

$$
K\geq\frac{sR}{2B}
$$

For BF16 GEMM on H100 GPUs, $s=2$, $R=989\times 10^{12}$, and $B=450\times 10^{9}$, implying that communication is hidden when $K\gtrsim 2197$. We verify this empirically in Table 3, where we ablate our fused GEMM+RS kernel against a standalone GEMM kernel. The results show that at $K=2048$, the non-overlapped communication ratio drops by roughly half, and beyond that, communication becomes nearly fully hidden. The residual communication time near $K=2048$ arises from atomic additions required for output tile accumulation, which prevent complete overlap.

Table 3: Measured BF16 GEMM and GEMM+RS performance (ms).

| $M$ & $N$ | $K$ | GEMM | GEMM+RS | Comm Ratio |
| --- | --- | --- | --- | --- |
| 32768 | 512 | 2.071 | 6.483 | 68% |
| 32768 | 1024 | 2.918 | 6.613 | 56% |
| 32768 | 2048 | 5.567 | 7.531 | 26% |
| 32768 | 4096 | 11.78 | 11.828 | $<$ 1% |
| 32768 | 8192 | 23.285 | 25.325 | 8% |

##### Inter-SM overlapping.

While intra-SM overlapping fully utilizes GPU compute, it constrains communication to follow the computation pattern. This leads to two potential drawbacks: the inability to exploit in-network acceleration and suboptimal L2 caching behavior. Inter-SM overlapping mitigates these issues but introduces a partitioning trade-off: deciding how many SMs to allocate to communication versus computation.

In-network acceleration. Recent networking hardware integrates compute directly into the interconnect fabric, enabling in-network reductions and collective offload within switches and link controllers [^23] [^19]. This transforms the interconnect from a passive data mover into an active participant in collectives. For communication-heavy kernels such as fused GEMM all-reduce (AR), in-network reduction can significantly reduce bandwidth usage. However, performing it within the same SM is impractical due to register pressure, limited occupancy, and inter-GPU synchronization costs. A more effective approach is to accumulate partial results in HBM, signal completion after each local write, and delegate a few specialized SMs to execute a single in-network all-reduce once all devices finish.

This tradeoff is shown in Figure 4 (right). Intra-SM overlapping issues $N$ atomic writes to $N$ destinations for each output tile, where $N$ is the number of GPUs. Even with a fully interconnected NVSwitch fabric, each GPU is limited by its 450 GB/s per-port bandwidth, causing concurrent peer writes to serialize at the destination. Inter-SM overlapping reduces $T_{\text{comm}}$ by roughly a factor of $N$, typically outweighing the cost of dedicating a few SMs to communication.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x4.png)

Figure 5: Comparison of different inter-SM scheduling performance on all-gather (AG) GEMM ( N × / 8 N\\times N/8\\times N ).

Remote cache reuse. Another limitation of intra-SM overlapping arises from the far-sided nature of L2 caching for peer HBM accesses. Data fetched from a peer GPU is cached only on the source device, not on the requester. Consequently, every remote access is bottlenecked by NVLink bandwidth. A representative case appears in Ring Attention [^13], where key and value (KV) tensors are reused across multiple attention blocks. Letting each thread block independently load them from remote GPUs leads to redundant transfers and rapid interconnect saturation. Instead, performing bulk transfers of the next block’s $K$ and $V$ tensors to local HBM using communication-dedicated SMs, while the remaining SMs compute, substantially reduces $T_{\text{comm}}$ and improves L2 reuse, as shown in Section 4.2.

SM partitioning. Inter-SM overlapping requires balancing SMs between communication and computation. As shown in Figure 5, the optimal split depends on input size: larger workloads favor more compute SMs, while smaller ones need proportionally more SMs for communication. PK allows users to automatically search for the optimal SM allocation at runtime through a unified program template.

#### 3.1.4 Minimizing design overheads

Ideally, abstractions should preserve the developer’s ability to achieve peak hardware performance. In practice, however, certain design choices in widely used communication libraries like NCCL and NVSHMEM—particularly in synchronization and buffering—constrain this ability.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x5.png)

Figure 6: All-reduce sum kernel comparison (BF16).

##### Two-way synchronization and intermediate buffering.

Many multi-GPU communication libraries impose synchronization and buffering constraints. For example, NCCL enforces two-way synchronization for every operation: both sender and receiver must be ready and acknowledge each other before data transfer begins, even for point-to-point communication. In addition, to reduce peer memory exchange overhead, NCCL employs small pre-allocated intermediate buffers (communication channels), introducing extra data movement. While such overheads are masked for large inputs, they become significant in fine-grained communication. PK avoids these issues by using pre-allocated destination buffers, enabling direct, one-way transfers without intermediate staging. As shown in Figure 6, this design improves the performance of pure communication kernels such as all-reduce by up to $1.79\times$.

##### Peer-memory access and synchronization.

NVSHMEM, the de facto low-level standard for multi-GPU communication, also introduces additional overhead in its public API functions. Each remote peer access performs a global memory load (\_\_ldg) to retrieve the peer address and enforces a group synchronization (e.g., \_\_syncthreads). By keeping peer addresses in registers and removing unnecessary synchronizations, PK eliminates these costs, achieving up to $4.5\times$ lower element-wise NVLink access latency and about 20 GB/s higher bandwidth utilization.

### 3.2 Abstractions

We introduce ParallelKittens (PK), a collection of abstractions that generalizes the tile-based programming principles proposed in ThunderKittens—and successor systems such as CuTe DSL and TileLang—to the multi-GPU setting. PK provides a minimal and complementary set of primitives for efficient multi-GPU communication. These abstractions expose high-performance communication mechanisms, enable the workload scheduling patterns described earlier, and minimize performance overheads by design. PK hides low-level complexity that does not impact performance, while preserving full user control through its CUDA/C++ embedded design.

#### 3.2.1 Data Structure

PK defines a data structure for each level of the GPU memory hierarchy, as illustrated in Figure 1 (left). At the register level, the minimum unit of execution is a $16\times 16$ tile, consistent with the original TK design. At the shared memory level, users operate on shared tiles that enable asynchronous, tile-granularity loads from and stores to peer HBM by a single thread. Store operations optionally support atomic reductions on peer memory and multicast to multiple devices via in-network broadcast. These operations preserve tensor-core–friendly layouts to remain efficient within local compute pipelines. At the HBM level, ParallelKittens introduces the Parallel Global Layout (PGL), which represents identically shaped and sized memory regions allocated across all devices. PGL serves as the central data structure enabling asynchronous P2P transfers, broadcasts, and synchronous in-fabric multicasts and reductions over tile-indexed regions. All data abstractions enforce essential principles such as coalesced interconnect access, swizzling to minimize bank conflicts and match tensor-core layouts, and fully device-initiated communication.

#### 3.2.2 Multi-GPU Operations

We introduce eight new primitives, which suffice to implement all kernels demonstrated in Section 4. The original ThunderKittens operators are also extended to remain fully compatible with the aforementioned data structures.

##### P2P communication primitives

- store\_async(dst, src, coord) // Store a shared tile to multicast memory.
- store\_add\_async(dst, src, coord) // Atomically add a shared tile to multicast memory.

##### Network-accelerated communication primitives

- reduce(dst, dst\_coord, src, src\_coord) // Reduce data from multicast memory to local HBM.
- all\_reduce(dst\_and\_src, coord) // Reduce data from multicast memory and write back to it.

##### Inter-device and inter-SM synchronization primitives

- signal(bar, coord, dev\_idx, val) // Signal a specific device’s barrier.
- signal\_all(bar, coord, val) // Signal all devices’ barriers simultaneously.
- wait(bar, coord, dev\_idx, expected) // Wait until a device’s barrier reaches a value.
- barrier(bar, coord, dev\_idx) // Wait for all devices to reach this point.

Because all PK data structures are tile-based, the new primitives also operate at tile granularity, ranging from $16\times 16$ (the minimum tile) up to the shared-memory limit (about $256\times 256$). All operations are device-initiated and use coordinates (coord) represented as int4 values specifying tile indices in local or remote HBM. P2P primitives are asynchronous and single-threaded, enabling fusion with other operations (e.g., tensor-core compute), whereas network-accelerated primitives require at least warp-level participation for optimal throughput. Synchronization primitives provide simple signaling and waiting mechanisms, enabling users to design arbitrary workload scheduling schemes. A complete API description is provided in Appendix C.

#### 3.2.3 Program Template

We provide a unified program template for implementing a wide range of multi-GPU kernels. As shown in Figure 1 (right), the template defines four worker components—loader, storer, consumer, and communicator—each encapsulating a common warp/SM specialization. The loader performs local or peer HBM reads, while the storer handles local or peer HBM writes. When either component accesses peer HBM, intra-SM overlapping is employed. The communicator occupies one or more SMs exclusively to perform dedicated communication, enabling inter-SM overlapping. Finally, the consumer issues tensor- or CUDA-core–based local compute. Beyond providing a structural pattern, the template automates common low-level tasks, including kernel configuration, shared memory and TMA setup, barrier and synchronization management, and tuning for optimal SM/warp partitioning. This allows users to focus solely on the per-tile compute and communication logic. A detailed description of the template and an example kernel are provided in Appendix D.

#### 3.2.4 Utilities

We provide inter-process communication (IPC) and PyTorch utilities for seamless integration with multi-process execution (e.g., via torchrun). These utilities manage low-level OS driver interactions and support pre-allocation of multi-GPU memory, enabling direct P2P communication without intermediate staging overheads. Appendices E and F provide further implementation details.

## 4 Experiments

![Refer to caption](https://arxiv.org/html/2511.13940v1/x6.png)

Figure 7: AG + GEMM performance. Local GEMM size is N × / 8 N\\times N/8\\times N, with given in the X-axis.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x7.png)

Figure 8: GEMM + RS performance. Local GEMM size is N × / 8 N\\times N\\times N/8, with given in the X-axis.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x8.png)

Figure 9: GEMM + AR performance. Local GEMM size is N × / 8 N\\times N\\times N/8, with given in the X-axis.

We demonstrate that PK generalizes across a diverse range of multi-GPU AI workloads by implementing representative kernels with its abstractions and comparing them against existing frameworks and hand-optimized baselines.

All experiments were conducted using $8\times$ Nvidia H100 80GB SXM GPUs, interconnected via 4th-generation NVLink and NVSwitch, using CUDA 12.6 and PyTorch 2.8.0. All matrix multiplications use BF16 as the element type and FP32 as the tensor core accumulator type. For brevity, we denote the GEMM shape as $M\times N\times K$, where the first operand has dimensions $M\times K$ and the second has dimensions $K\times N$. We report the observed average compute throughput.

Although the experiments in this section use H100 GPUs, PK is fully compatible with B200 GPUs and exhibits similar performance characteristics. We present results on Blackwell GPUs in Appendices A and B.

### 4.1 Data and Tensor Parallelism

To efficiently scale large models, weights are often sharded across multiple devices using tensor parallelism [^25] [^34], which partitions weight matrices along the row or column dimension. A common strategy combines this with data parallelism [^14]: inputs sharded by rows are first all-gathered (AG), followed by a GEMM with column-sharded weights, a non-linear activation, and a second GEMM with row-sharded weights, after which a reduce-scatter (RS) or all-reduce (AR) is applied. Communication and computation are overlapped by pairing AG with the first GEMM (AG+GEMM) and RS or AR with the second (GEMM+RS, GEMM+AR).

For these workloads, we compare against the cuBLAS GEMM combined with NCCL as the non-overlapped baseline, compiler-based approaches (Triton Distributed), and hand-optimized kernels (Flux and CUTLASS). Flux and CUTLASS do not provide GEMM–AR kernels and are therefore omitted in those cases. Figures 7, 8, and 9 show the results. Overall, PK achieves a $1.06$ – $1.68\times$ speedup over the non-overlapped baseline and outperforms compiler-based approaches by $1.07$ – $5.63\times$. Compared to hand-optimized kernels, PK matches or surpasses their performance, achieving $0.97$ – $2.33\times$ speedup over Flux and $0.90$ – $7.39\times$ over CUTLASS. We also note that AG+GEMM and GEMM+RS are often used back-to-back in practice, and no single baseline outperforms PK when both are combined.

We further observe that compiler-based approaches can exhibit inconsistent performance across diverse hardware platforms. For instance, Triton Distributed, originally developed for H800 GPUs, sometimes performs below the non-overlapped baseline on H100s. Hand-tuned kernels also show reduced efficiency on certain problem shapes.

Under sufficiently large reduction axes, the non-overlapped portion of communication time in PK falls below 1%. The communication component of our kernels (excluding GEMM) is implemented in fewer than 50 lines of device code, using the primitives introduced in Section 3.2.

### 4.2 Sequence Parallelism

![Refer to caption](https://arxiv.org/html/2511.13940v1/x9.png)

Figure 10: Ring Attention performance across sequence lengths ( B = 16 B=16, H H=16 D 128 D=128 ).

Modern AI workloads increasingly involve inputs with long sequence lengths, requiring a single sequence to be distributed across multiple devices. While sharding along the sequence dimension has minimal impact on MLP or MoE layers, attention layers require each token to attend to all others within the same sequence. This necessitates sequence-parallel approaches such as Ring Attention [^13] and DeepSpeed-Ulysses [^10]. In our evaluation, we compare against the state-of-the-art implementations: xDiT [^7] for Ring Attention and YunChang [^6] for DeepSpeed-Ulysses.

##### Ring Attention.

In Ring Attention, key–value (KV) tensors are partitioned across devices, with each GPU computing blockwise attention on its local shard while concurrently transmitting it to a peer. The baseline xDiT implementation overlaps computation and KV exchange coarsely by launching NCCL P2P sends and FlashAttention-3 kernels on separate CUDA streams. In contrast, PK can fuse these into a single kernel with explicit inter-SM overlap, precisely allocating SMs between computation and communication, deciding how they synchronize, and auto-tuning this partitioning for optimal performance. As shown in Figure 10, this yields a $1.07\times$ – $4.08\times$ speedup over the baseline—evaluated at total sequence lengths (shown on the X-axis) <sup>3</sup> evenly partitioned across 8 devices—and reduces the non-overlapped communication fraction down to 9%.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x10.png)

Figure 11: DeepSpeed-Ulysses attention layer performance across sequence lengths ( B = 16 B=16, H 128 H=128 D D=128 ).

![Refer to caption](https://arxiv.org/html/2511.13940v1/x11.png)

Figure 12: Expert-parallel token dispatch + GEMM performance ( TopK = 8 \\text{TopK}=8, N experts 256 N\_{\\text{experts}}=256 H 7168 H=7168 expert 2048 H\_{\\text{expert}}=2048 ).

##### DeepSpeed-Ulysses.

In DeepSpeed-Ulysses, an all-to-all exchange occurs before and after self-attention. Everything except self-attention is sequence-sharded, while self-attention remains head-sharded. The main bottleneck is the fine-grained all-to-all; as NCCL does not natively support this along the inner dimension, the baseline relies on tensor reshaping before and after communication. Using PK, we implement a fine-grained all-to-all kernel that removes this overhead. As shown in Figure 11, this yields a $1.01\times$ – $1.39\times$ speedup, evaluated at total sequence lengths (shown on the X-axis) evenly split across 8 devices. The complete kernel remains under 50 lines of device code.

### 4.3 Expert Parallelism

To scale architectures with MoE layers [^24], multiple experts are distributed evenly across devices, a strategy known as expert parallelism. However, this approach requires costly scattering and gathering of tokens before and after the expert MLP layers. Several approaches mitigate this by overlapping token communication with GEMM computation [^32] [^31] [^1]. We compare against Comet [^31], the state-of-the-art fine-grained overlapping strategy for expert parallelism. For demonstration, we evaluate the first half of the MoE layer: overlapping token dispatch with the first expert MLP. As shown in Figure 12, where the total set of input tokens (shown on the X-axis) is initially partitioned evenly across devices, PK matches or surpasses the hand-tuned baseline, achieving $0.92$ – $1.22\times$ the performance of Comet, with fewer than 40 lines of device code added to a grouped GEMM kernel.

## 5 Conclusion

This work presents ParallelKittens, a minimal and systematic framework for building high-performance multi-GPU kernels. By formalizing the design space through three key principles—transfer mechanisms, scheduling strategies, and design overheads—we demonstrate that a small set of primitives can match or surpass the performance of hand-optimized kernels while greatly simplifying implementation. As this work focuses on intra-node execution, extending these abstractions to inter-node communication remains an important direction for future work. At the same time, intra-node systems are rapidly scaling, as shown by Nvidia’s NVL72 and upcoming NVL144, NVL576 architectures, which makes the study of efficient intra-node kernel design increasingly critical for distributed AI workloads.

Our framework and kernels are open sourced at: [https://github.com/HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens).

## Acknowledgements

We are grateful to Cursor and Together AI for making this work possible. We thank Dylan Lim for his assistance with the initial implementation of PGL operations. We thank Yasa Baig, Kelly Buchanan, Francois Chaubard, Mayee Chen, Catherine Deng, Andy Dimnaku, Owen Dugan, Daniel Y. Fu, Roberto Garcia, Ronny Junkins, Ishane Khare, Hermann Kumbong, Jerry Liu, Avanika Narayan, Jon Saad-Falcon, and Alex Waitz for helpful feedback and discussions during this work. We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF2247015 (Hardware-Aware), CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); US DEVCOM ARL under Nos. W911NF-23-2-0184 (Long-context) and W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under Nos. N000142312633 (Deep Signal Processing); Stanford HAI under No. 247183; NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), and members of the Stanford DAWN project: Meta, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government.

## References

## Appendix

We present ParallelKittens performance on Blackwell GPUs (Appendix A), additional collective performance results (Appendix B), ParallelKittens API specification (Appendix C), program template and example kernels (Appendix D), multi-GPU setup process (Appendix E), and in-network acceleration setup process (Appendix F).

## Appendix A Blackwell GPU Performance

In this section, we demonstrate that PK generalizes across different hardware architectures by presenting representative kernel performance on Blackwell GPUs and comparing against available baselines that also support this architecture.

All experiments were conducted using $8\times$ Nvidia B200 GPUs, interconnected via 5th-generation NVLink and NVSwitch (900 GB/s unidirectional bandwidth), using CUDA 12.8 and PyTorch 2.8.0. All matrix multiplications use BF16 as the element type and FP32 as the tensor core accumulator type. For brevity, we denote the GEMM shape as $M\times N\times K$, where the first operand has dimensions $M\times K$ and the second has dimensions $K\times N$. We report the observed average compute throughput.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x12.png)

Figure 13: GEMM + RS performance. Local GEMM size is N × / 8 N\\times N\\times N/8, with given in the X-axis.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x13.png)

Figure 14: DeepSpeed-Ulysses attention layer performance across sequence lengths ( B = 16 B=16, H 128 H=128 D D=128 ).

## Appendix B Additional Collective Performance

In this section, we report additional results on pure collective kernel performance and compare them against NCCL. We particularly examine how performance can improve significantly when the communication pattern is fine-grained: for example, when performing all-gather or reduce-scatter along the tensor dimension (the last dimension) instead of the batch dimension (the first dimension), or when performing all-to-all operations across head and sequence dimensions. In such cases, the memory layout becomes discontiguous, which makes NCCL inefficient, as it supports collectives only on contiguous partitions and thus requires extra reshaping and copying. In contrast, PK can execute these collectives directly on the original layout. The results below illustrate this advantage.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x14.png)

Figure 15: Tensor dimension all-gather performance comparison (BF16). The gathered matrix size is N × N\\times N, with given in the X-axis.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x15.png)

Figure 16: Tensor dimension reduce-scatter performance comparison (BF16). The scattered matrix size is N × / 8 N\\times N/8, with given in the X-axis.

![Refer to caption](https://arxiv.org/html/2511.13940v1/x16.png)

Figure 17: 4-dimensional ( B, S H D ) all-to-all performance comparison (BF16), with = 1 B=1 128 H=128 D=128, and varying given in the X-axis. The dimension is gathered and the dimension is evenly scattered across 8 GPUs.

## Appendix C ParallelKittens API Specification

We provide the full specification of PK primitives, including each function’s name, signature, parameters, and description.

```
template <int axis, cache_policy policy, kittens::ducks::st::all ST,
          kittens::ducks::pgl::all PGL, kittens::ducks::coord::tile COORD>
__device__ void store_async(const PGL &dst, const ST &src, const COORD &idx)
```

Template Parameters:

- axis: Tensor axis for the operation (0-3).
- policy: Cache policy (NORMAL or cache hint).
- ST: Shared tile type.
- PGL: Parallel global layout type.
- COORD: Coordinate type for indexing.

Parameters:

- dst: Destination parallel global layout.
- src: Source shared memory tile.
- idx: Coordinate specifying the destination position.

Description: Asynchronously stores a shared memory tile to multicast memory using the Tensor Memory Accelerator (TMA). Launched by a single thread.

```
template <int axis, cache_policy policy, kittens::ducks::st::all ST,
          kittens::ducks::pgl::all PGL, kittens::ducks::coord::tile COORD>
__device__ void store_add_async(const PGL &dst, const ST &src, const COORD &idx)
```

Template Parameters:

- axis: Tensor axis for the operation (0-3).
- policy: Cache policy (NORMAL or cache hint).
- ST: Shared tile type.
- PGL: Parallel global layout type.
- COORD: Coordinate type for indexing.

Parameters:

- dst: Destination parallel global layout.
- src: Source shared memory tile.
- idx: Coordinate specifying the destination position.

Description: Asynchronously performs an atomic add reduction from a shared memory tile to multicast memory via TMA. The operation atomically adds the source tile values to the existing values at the destination. Launched by a single thread.

```
template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP,
          kittens::ducks::pgl::all PGL, kittens::ducks::gl::all GL>
__device__ void reduce(GL &dst, const coord &dst_idx, PGL &src, const coord &src_idx)
```

Template Parameters:

- TILE\_ROWS: Number of rows in the tile.
- TILE\_COLS: Number of columns in the tile.
- OP: Reduction operation to apply (sum, max, or min).
- PGL: Parallel global layout type.
- GL: Global layout type.

Parameters:

- dst: Reference to the destination global layout.
- dst\_idx: Coordinate specifying the destination tile’s position.
- src: Reference to the source parallel global layout.
- src\_idx: Coordinate specifying the source tile’s position.

Description: Performs a reduction from multicast memory to device-local global memory. The function loads data from the source parallel global layout using in-network reduction operations and stores the result to the destination global layout. Collectively launched by one or more warps. Each warp processes multiple rows of the tile, performing the specified reduction operation during the multicast load and then writing the reduced values to the destination global memory.

```
template <int TILE_ROWS, int TILE_COLS, kittens::reduce_op OP,
          kittens::ducks::pgl::all PGL>
__device__ void all_reduce(PGL &dst_and_src, const coord &idx)
```

Template Parameters:

- TILE\_ROWS: Number of rows in the tile.
- TILE\_COLS: Number of columns in the tile.
- OP: Reduction operation to apply (sum, max, or min).
- PGL: Parallel global layout type.

Parameters:

- dst\_and\_src: Reference to the parallel global layout object.
- idx: Coordinate specifying the tile’s position with batch (b), depth (d), row (r), and column (c) indices.

Description: Performs an all-reduce collective operation on a tile of data on multicast memory. The function reduces data across all participating GPUs for the specified tile. Collectively launched by one or more warps. Each warp processes multiple rows, loading data from multicast memory with the specified reduction operation, then writing the result back to the same multicast location. The operation leverages in-network acceleration hardware to efficiently perform the reduction without explicit peer-to-peer copies.

```
__device__ void signal(const barrier_t &barrier, const coord &idx,
                       const int dst_dev_idx, const int val)
```

Parameters:

- barrier: Reference to the barrier object (parallel global layout of integers).
- idx: Element-wise coordinate specifying the barrier location.
- dst\_dev\_idx: Target device index to signal.
- val: Value to add to the barrier counter.

Description: Signals a specific device’s barrier by atomically adding a value to its counter. This primitive is used to coordinate synchronization between thread blocks and GPUs.

```
__device__ void signal_all(const barrier_t &barrier, const coord &idx, const int val)
```

Parameters:

- barrier: Reference to the barrier object.
- idx: Element-wise coordinate specifying the barrier location.
- val: Value to add to all devices’ barrier counters.

Description: Signals all devices simultaneously by performing a multicast atomic add operation. Uses in-network multicast hardware to efficiently update barrier counters across all participating devices with a single operation.

```
__device__ void wait(const barrier_t &barrier, const coord &idx,
                     const int dev_idx, const int expected)
```

Parameters:

- barrier: Reference to the barrier object.
- idx: Element-wise coordinate specifying the barrier location.
- dev\_idx: Device index to wait on.
- expected: Expected barrier value to wait for.

Description: Waits until a device’s barrier counter reaches the expected value. Continuously polls the barrier location using relaxed memory ordering loads until the expected value is observed. This provides a spinning wait mechanism for inter-SM and inter-GPU synchronization.

```
__device__ void barrier(const barrier_t &barrier, const coord &idx, const int dev_idx)
```

Parameters:

- barrier: Reference to the barrier object.
- idx: Element-wise coordinate specifying the barrier location.
- dev\_idx: Current device index.

Description: Implements a complete barrier synchronization across all devices. This ensures all participating GPUs reach the same synchronization point before proceeding.

## Appendix D ParallelKittens Program Template and Example Kernels

##### Load-Compute-Store-Communicate (LCSC) Template.

The LCSC template provides a structured approach for implementing multi-GPU kernels with specialized worker components. The template enables flexible warp/SM specialization and overlapping strategies for compute, memory, and communication operations.

High-level Template Structure:

```
struct lcsc_template {
    static void loader(globals, comp_sem, comp_smem, comp_regs);
    static void storer(globals, comp_sem, comp_smem, comp_regs);
    static void consumer(globals, comp_sem, comp_smem, comp_regs);
    static void communicator(globals, comm_sem, comm_smem, comm_regs);
};
```

Required Components:

- comp\_sem: struct of semaphores for synchronization within compute SMs.
- comm\_sem: struct of semaphores for synchronization within communication SMs.
- comp\_smem: struct of shared memory layouts for compute SMs.
- comm\_smem: struct of shared memory layouts for communication SMs.

Workers:

- loader: Performs memory loads from local or peer HBM using TMA.
- storer: Performs memory stores to local or peer HBM.
- consumer: Performs tensor/CUDA core operations on loaded data.
- communicator: Performs dedicated inter-GPU communication. Executes on separate communication SMs.

Execution Model: The template automatically distributes SMs between computation and communication roles based on num\_comm\_sms, passed in to the host entry function. Compute SMs execute loader, storer, and consumer functions with producer-consumer synchronization through semaphores. Communication SMs execute the communicator function independently. The framework handles warpgroup specialization, register allocation, and task distribution across workers. Programmers can utilize this template by defining the above struct, and passing it to the launch interface:

```
lcsc::launch_kernel<config, globals, lcsc_template>(G, stream);
```

Where the parameters are:

- config: Compile-time configuration struct defining SM and thread counts.
- globals: Runtime globals struct containing device memory pointers and parameters.
- lcsc\_template: User-defined LCSC template implementation.
- G: Instance of globals struct.
- stream: CUDA stream for kernel execution.

We present a fused GEMM + all-reduce (AR) kernel implemented using the LCSC template in Figure 18. We highlight that the kernel contains both a fully optimized GEMM and fused all-reduce logic, with the communication-relevant code comprising only about 10 lines of device code. We also open-source all remaining kernels evaluated in this paper through our GitHub repository.

[⬇](data:text/plain;base64,X19kZXZpY2VfXyBpbmxpbmUgdm9pZCBsb2FkZXIoY29uc3QgZ2xvYmFscyAmRywgY29tcF9zZW0gJnNlbSwgY29tcF9zbWVtICZzbWVtLCBjb21wX3JlZ3MgJnJlZ3MpIHsKICAgIGludDIgaWR4ID0gaW50ZXJwcmV0X3Rhc2socmVncy50YXNrX2lkKTsKICAgIGZvciAoaW50IHJlZF9pZHggPSAwOyByZWRfaWR4IDwgcmVncy5udW1faXRlcnM7IHJlZF9pZHgrKykgewogICAgICAgIHdhaXQoc2VtLmlucHV0c19maW5pc2hlZFtyZWdzLnN0YWdlXSwgZ2V0X3BoYXNlYml0PDE+KHJlZ3MucGhhc2ViaXRzLCByZWdzLnN0YWdlKSk7CiAgICAgICAgdXBkYXRlX3BoYXNlYml0PDE+KHJlZ3MucGhhc2ViaXRzLCByZWdzLnN0YWdlKTsKICAgICAgICB0bWE6OmV4cGVjdF9ieXRlcyhzZW0uaW5wdXRzX2Fycml2ZWRbcmVncy5zdGFnZV0sIHNpemVvZihBX3RpbGUpICogMiArIHNpemVvZihCX3RpbGUpKTsKICAgICAgICBpZiAocmVkX2lkeCA9PSBQSVBFTElORV9TVEFHRVMgLSAxKSB7CiAgICAgICAgICAgIHdhaXQoc2VtLm91dHB1dHNfZmluaXNoZWQsIGdldF9waGFzZWJpdDwxPihyZWdzLnBoYXNlYml0cywgUElQRUxJTkVfU1RBR0VTKSk7CiAgICAgICAgICAgIHVwZGF0ZV9waGFzZWJpdDwxPihyZWdzLnBoYXNlYml0cywgUElQRUxJTkVfU1RBR0VTKTsKICAgICAgICB9CiAgICAgICAgZm9yIChpbnQgaSA9IDA7IGkgPCAyOyBpKyspCiAgICAgICAgICAgIHRtYTo6bG9hZF9hc3luYyhzbWVtLmlucHV0c1tyZWdzLnN0YWdlXS5BW2ldLCBHLkEsIHtpZHgueCAqIDIgKyBpLCByZWRfaWR4fSwgc2VtLmlucHV0c19hcnJpdmVkW3JlZ3Muc3RhZ2VdKTsKICAgICAgICB0bWE6OmxvYWRfYXN5bmMoc21lbS5pbnB1dHNbcmVncy5zdGFnZV0uQiwgRy5CLCB7cmVkX2lkeCwgaWR4Lnl9LCBzZW0uaW5wdXRzX2Fycml2ZWRbcmVncy5zdGFnZV0pOwogICAgICAgIHJlZ3Muc3RhZ2UgPSAocmVncy5zdGFnZSArIDEpICUgUElQRUxJTkVfU1RBR0VTOwogICAgfQp9CgpfX2RldmljZV9fIGlubGluZSB2b2lkIHN0b3Jlcihjb25zdCBnbG9iYWxzICZHLCBjb21wX3NlbSAmc2VtLCBjb21wX3NtZW0gJnNtZW0sIGNvbXBfcmVncyAmcmVncykgewogICAgaW50MiBpZHggPSBpbnRlcnByZXRfdGFzayhyZWdzLnRhc2tfaWQpOwogICAgd2FpdChzZW0ub3V0cHV0c19hcnJpdmVkLCBnZXRfcGhhc2ViaXQ8MD4ocmVncy5waGFzZWJpdHMsIDApKTsKICAgIHVwZGF0ZV9waGFzZWJpdDwwPihyZWdzLnBoYXNlYml0cywgMCk7CiAgICBmb3IgKGludCBpID0gMDsgaSA8IDI7IGkrKykKICAgICAgICB0bWE6OnN0b3JlX2FzeW5jKEcuQ1tHLmRldl9pZHhdLCByZWdzLkNbaV0sIHtpZHgueCAqIDIgKyBpLCBpZHgueX0pOwogICAgdG1hOjpzdG9yZV9hc3luY19yZWFkX3dhaXQoKTsKICAgIGFycml2ZShzZW0ub3V0cHV0c19maW5pc2hlZCk7CiAgICBpbnQgc2lnbmFsX2Rldl9pZHggPSByZWdzLnRhc2tfaWQgJSBOVU1fREVWSUNFUzsKICAgIGRldmljZTxOVU1fREVWSUNFUz46OnNpZ25hbChHLmJhcnJpZXIsIHtpZHgueCwgaWR4Lnl9LCBzaWduYWxfZGV2X2lkeCwgMSk7Cn0KCl9fZGV2aWNlX18gaW5saW5lIHZvaWQgY29uc3VtZXIoY29uc3QgZ2xvYmFscyAmRywgY29tcF9zZW0gJnNlbSwgY29tcF9zbWVtICZzbWVtLCBjb21wX3JlZ3MgJnJlZ3MpIHsKICAgIHJ0X2ZsPFJPV19CTE9DSyAvIDgsIENPTF9CTE9DSz4gQ19hY2N1bTsKICAgIHdhcnA6Onplcm8oQ19hY2N1bSk7CiAgICBmb3IgKGludCByZWRfaWR4ID0gMDsgcmVkX2lkeCA8IHJlZ3MubnVtX2l0ZXJzOyByZWRfaWR4KyspIHsKICAgICAgICB3YWl0KHNlbS5pbnB1dHNfYXJyaXZlZFtyZWdzLnN0YWdlXSwgZ2V0X3BoYXNlYml0PDA+KHJlZ3MucGhhc2ViaXRzLCByZWdzLnN0YWdlKSk7CiAgICAgICAgdXBkYXRlX3BoYXNlYml0PDA+KHJlZ3MucGhhc2ViaXRzLCByZWdzLnN0YWdlKTsKICAgICAgICB3YXJwZ3JvdXA6Om1tYV9BQihDX2FjY3VtLCBzbWVtLmlucHV0c1tyZWdzLnN0YWdlXS5BW3JlZ3Mud2FycGdyb3VwX2lkXSwgc21lbS5pbnB1dHNbcmVncy5zdGFnZV0uQik7CiAgICAgICAgd2FycGdyb3VwOjptbWFfYXN5bmNfd2FpdCgpOwogICAgICAgIHdhcnA6OmFycml2ZShzZW0uaW5wdXRzX2ZpbmlzaGVkW3JlZ3Muc3RhZ2VdKTsKICAgICAgICByZWdzLnN0YWdlID0gKHJlZ3Muc3RhZ2UgKyAxKSAlIFBJUEVMSU5FX1NUQUdFUzsKICAgIH0KICAgIGdyb3VwPDg+OjpzeW5jKDMpOwogICAgd2FycGdyb3VwOjpzdG9yZShyZWdzLkNbcmVncy53YXJwZ3JvdXBfaWRdLCBDX2FjY3VtKTsKICAgIHdhcnBncm91cDo6c3luYyhyZWdzLndhcnBncm91cF9pZCArIDEpOwogICAgd2FycGdyb3VwOjphcnJpdmUoc2VtLm91dHB1dHNfYXJyaXZlZCk7Cn0KCl9fZGV2aWNlX18gaW5saW5lIHZvaWQgY29tbXVuaWNhdG9yKGNvbnN0IGdsb2JhbHMgJkcsIGNvbW1fc2VtICZzZW0sIGNvbW1fc21lbSAmc21lbSwgY29tbV9yZWdzICZyZWdzKSB7CiAgICBpbnQyIGlkeCA9IGludGVycHJldF90YXNrKHJlZ3MudGFza19pZCk7CiAgICBpZiAodGhyZWFkSWR4LnggPT0gMCkKICAgICAgICBkZXZpY2U8TlVNX0RFVklDRVM+Ojp3YWl0KEcuYmFycmllciwge2lkeC54LCBpZHgueX0sIEcuZGV2X2lkeCwgTlVNX0RFVklDRVMpOwogICAgX19zeW5jdGhyZWFkcygpOwogICAgZ3JvdXA8TlVNX1dBUlBTPjo6YWxsX3JlZHVjZTxST1dfQkxPQ0ssIENPTF9CTE9DSywgcmVkdWNlX29wOjpBREQ+KEcuQywge2lkeC54LCBpZHgueX0pOwp9)

\_\_device\_\_ inline void loader(const globals &G, comp\_sem &sem, comp\_smem &smem, comp\_regs &regs) {

int2 idx = interpret\_task(regs.task\_id);

for (int red\_idx = 0; red\_idx < regs.num\_iters; red\_idx++) {

wait(sem.inputs\_finished\[regs.stage\], get\_phasebit<1>(regs.phasebits, regs.stage));

update\_phasebit<1>(regs.phasebits, regs.stage);

tma::expect\_bytes(sem.inputs\_arrived\[regs.stage\], sizeof(A\_tile) \* 2 + sizeof(B\_tile));

if (red\_idx == PIPELINE\_STAGES - 1) {

wait(sem.outputs\_finished, get\_phasebit<1>(regs.phasebits, PIPELINE\_STAGES));

update\_phasebit<1>(regs.phasebits, PIPELINE\_STAGES);

}

for (int i = 0; i < 2; i++)

tma::load\_async(smem.inputs\[regs.stage\].A\[i\], G.A, {idx.x \* 2 + i, red\_idx}, sem.inputs\_arrived\[regs.stage\]);

tma::load\_async(smem.inputs\[regs.stage\].B, G.B, {red\_idx, idx.y}, sem.inputs\_arrived\[regs.stage\]);

regs.stage = (regs.stage + 1) % PIPELINE\_STAGES;

}

}

\_\_device\_\_ inline void storer(const globals &G, comp\_sem &sem, comp\_smem &smem, comp\_regs &regs) {

int2 idx = interpret\_task(regs.task\_id);

wait(sem.outputs\_arrived, get\_phasebit<0>(regs.phasebits, 0));

update\_phasebit<0>(regs.phasebits, 0);

for (int i = 0; i < 2; i++)

tma::store\_async(G.C\[G.dev\_idx\], regs.C\[i\], {idx.x \* 2 + i, idx.y});

tma::store\_async\_read\_wait();

arrive(sem.outputs\_finished);

int signal\_dev\_idx = regs.task\_id % NUM\_DEVICES;

device<NUM\_DEVICES>::signal(G.barrier, {idx.x, idx.y}, signal\_dev\_idx, 1);

}

\_\_device\_\_ inline void consumer(const globals &G, comp\_sem &sem, comp\_smem &smem, comp\_regs &regs) {

rt\_fl<ROW\_BLOCK / 8, COL\_BLOCK> C\_accum;

warp::zero(C\_accum);

for (int red\_idx = 0; red\_idx < regs.num\_iters; red\_idx++) {

wait(sem.inputs\_arrived\[regs.stage\], get\_phasebit<0>(regs.phasebits, regs.stage));

update\_phasebit<0>(regs.phasebits, regs.stage);

warpgroup::mma\_AB(C\_accum, smem.inputs\[regs.stage\].A\[regs.warpgroup\_id\], smem.inputs\[regs.stage\].B);

warpgroup::mma\_async\_wait();

warp::arrive(sem.inputs\_finished\[regs.stage\]);

regs.stage = (regs.stage + 1) % PIPELINE\_STAGES;

}

group<8>::sync(3);

warpgroup::store(regs.C\[regs.warpgroup\_id\], C\_accum);

warpgroup::sync(regs.warpgroup\_id + 1);

warpgroup::arrive(sem.outputs\_arrived);

}

\_\_device\_\_ inline void communicator(const globals &G, comm\_sem &sem, comm\_smem &smem, comm\_regs &regs) {

int2 idx = interpret\_task(regs.task\_id);

if (threadIdx.x == 0)

device<NUM\_DEVICES>::wait(G.barrier, {idx.x, idx.y}, G.dev\_idx, NUM\_DEVICES);

\_\_syncthreads();

group<NUM\_WARPS>::all\_reduce<ROW\_BLOCK, COL\_BLOCK, reduce\_op::ADD>(G.C, {idx.x, idx.y});

}

Figure 18: Fused GEMM + AR kernel implemented with the LCSC template

## Appendix E Multi-GPU Memory Setup Process

We describe the low-level multi-GPU memory setup process, a major complexity in multi-GPU programming, which PK abstracts away from programmers.

The basic requirement of multi-GPU programming is that kernels must be able to access memory (HBM) on peer devices. To enable this, we need to create a new mapping in the current device’s virtual address space that points to the peer device’s physical memory. After such, the kernel can simply dereference the address, and the NVLink and NVSwitch fabric handle the underlying transfer.

There are three ways to create such mappings: (1) CUDA Unified Virtual Addressing, (2) CUDA Inter-Process Communication, and (3) manual Virtual Memory Management.

### E.1 CUDA Unified Virtual Addressing (UVA)

UVA provides a single unified virtual address space across GPUs, but with the limitation that it applies only within a single process. That is, if we avoid using multiple processes altogether, there exists no heterogeneous virtual address spaces.

![Refer to caption](https://arxiv.org/html/2511.13940v1/figures/cuda-ipc-flow.png)

Figure 19: CUDA IPC flow.

However, we note that modern production distributed training and inference are built around a multi-processing model. Distributed runners like torchrun assume 1 GPU device per rank (process), and working around this is quite complicated. Thus, multi-processing is the preferred model of launching multi-GPU workloads, which brings us to the next two methods.

### E.2 CUDA Inter-Process Communication (IPC)

Calling cudaIpcGetMemHandle on the address in the current virtual address space returns a 64-byte stub that can be shared across processes through standard IPC mechanisms like shared memory or Unix domain sockets. The receiving process then can call cudaIpcOpenMemHandle, which maps the given stub into its own address space. Figure 19 visualizes this flow.

While this method is straightforward and works on pre-allocated device memory (e.g., existing PyTorch tensors), its drawback is that it cannot use the NVSwitch accelerator for faster reduction and broadcast operations.

![Refer to caption](https://arxiv.org/html/2511.13940v1/figures/vmm-flow.png)

Figure 20: CUDA VMM flow.

![Refer to caption](https://arxiv.org/html/2511.13940v1/figures/multicast-creation.png)

Figure 21: CUDA multicast object creation process.

### E.3 Manual Virtual Memory Management (VMM)

For VMM, we start by manually allocating the GPU physical memory with cuMemCreate. This allows setting the CU\_MEM\_HANDLE\_TYPE\_POSIX\_FILE\_DESCRIPTOR property on this physical memory, which then lets us export the physical memory reference as a Linux file descriptor by calling cuMemExportToShareableHandle.

Because file descriptors are tied to a specific process in Linux, they cannot be shared directly. The standard way to transfer a file descriptor in Linux is to send it as a control message over a Unix domain socket. Once we send the file descriptor over to the destination process, it can then import the physical memory reference using cuMemImportFromShareableHandle and map it into its own virtual address space using the VMM API. The overall flow is illustrated in Figure 20.

A downside of this approach is that the given memory must be allocated with VMM and is subject to size granularity requirements, typically at 2MB for H100s and B200s. As a result, a PyTorch-allocated tensor, which is usually allocated by the standard cudaMalloc without size alignment, cannot be shared directly across processes. Instead, we need a custom tensor class that manages device memory allocation and deallocation with custom VMM logic. The main advantage, however, is that this method enables the use of NVSwitch in-network accelerators.

## Appendix F In-network Acceleration Setup Process

In order to utilize NVSwitch acceleration, we first allocate local memory on each participating device with VMM. Then we create a multicast object, which is an abstraction over multiple physical locations in multiple devices. To do this, we create a 8-byte stub that represents the multicast object with cuMulticastCreate, register all devices as participants, and map each device’s physical memory region to it.

A multicast object behaves just like VMM-allocated physical memory: we can share it with other processes and map a virtual address to it using the same mechanism described in the VMM setup process. That is, we export the multicast object as a POSIX file descriptor, open them on each device, and map them into each process’s virtual address space. The overall setup process and the exact names of the CUDA functions called are shown in Figure 21.

After completing the above, each process has two addresses: one mapping to the current device’s physical memory (local address) and another mapping to the multicast object (multicast address). Writing to and reading from the local address is a standard global memory access. Writing to the multicast address triggers a broadcast across all participating devices, multicasted in the NVSwitch fabric. Reading from the multicast address causes undefined behavior. Finally, in-fabric reduction operations can be invoked on the multicast address using the PTX instructions multimem.red and multimem.ld\_reduce. This is illustrated in Figure 22.

![Refer to caption](https://arxiv.org/html/2511.13940v1/figures/multicast-hierarchy.png)

Figure 22: CUDA multicast object hierarchy.

[^1]: Osayamen Jonathan Aimuyo, Byungsoo Oh, and Rachee Singh. FlashDMoE: Fast Distributed MoE in a Single Kernel. *arXiv preprint arXiv:2506.04667*, June 2025.

[^2]: AMD. Iris: First-class multi-gpu programming experience in triton. [https://github.com/ROCm/iris](https://github.com/ROCm/iris), 2025.

[^3]: Liwen Chang, Wenlei Bao, Qi Hou, Chengquan Jiang, Ningxin Zheng, Xuanrun Zhang, Zuquan Song, Ziheng Jiang, Haibin Lin, and Xin Liu. FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion. *arXiv preprint arXiv:2406.06858v1*, June 2024.

[^4]: Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In *Advances in Neural Information Processing Systems (NeurIPS)*, 2022.

[^5]: DeepSeek-AI, Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Dongjie Ji, Erhang Li, Fangyun Lin, Fucong Dai, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Han Bao, Hanwei Xu, Haocheng Wang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Li, Hui Qu, J.L. Cai, Jian Liang, Jianzhong Guo, Jiaqi Ni, Jiashi Li, Jiawei Wang, Jin Chen, Jingchang Chen, Jingyang Yuan, Junjie Qiu, Junlong Li, Junxiao Song, Kai Dong, Kai Hu, Kaige Gao, Kang Guan, Kexin Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Xu, Leyi Xia, Liang Zhao, Litong Wang, Liyue Zhang, Meng Li, Miaojun Wang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingming Li, Ning Tian, Panpan Huang, Peiyi Wang, Peng Zhang, Qiancheng Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, R.J. Chen, R.L. Jin, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runxin Xu, Ruoyu Zhang, Ruyi Chen, S.S. Li, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaoqing Wu, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuang Zhou, Shuiping Yu, Shunfeng Zhou, Shuting Pan, T. Wang, Tao Yun, Tian Pei, Tianyu Sun, W.L. Xiao, Wangding Zeng, Wanjia Zhao, Wei An, Wen Liu, Wenfeng Liang, Wenjun Gao, Wenqin Yu, Wentao Zhang, X.Q. Li, Xiangyue Jin, Xianzu Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaojin Shen, Xiaokang Chen, Xiaokang Zhang, Xiaosha Chen, Xiaotao Nie, Xiaowen Sun, Xiaoxiang Wang, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xinnan Song, Xinxia Shan, Xinyi Zhou, Xinyu Yang, Xinyuan Li, Xuecheng Su, Xuheng Lin, Y.K. Li, Y.Q. Wang, Y.X. Wei, Y.X. Zhu, Yang Zhang, Yanhong Xu, Yanping Huang, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Li, Yaohui Wang, Yi Yu, Yi Zheng, Yichao Zhang, Yifan Shi, Yiliang Xiong, Ying He, Ying Tang, Yishi Piao, Yisong Wang, Yixuan Tan, Yiyang Ma, Yiyuan Liu, Yongqiang Guo, Yu Wu, Yuan Ou, Yuchen Zhu, Yuduan Wang, Yue Gong, Yuheng Zou, Yujia He, Yukun Zha, Yunfan Xiong, Yunxian Ma, Yuting Yan, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z.F. Wu, Z.Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhen Huang, Zhen Zhang, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhipeng Xu, Zhiyu Wu, Zhongyu Zhang, Zhuoshu Li, Zihui Gu, Zijia Zhu, Zijun Liu, Zilin Li, Ziwei Xie, Ziyang Song, Ziyi Gao, and Zizheng Pan. DeepSeek-V3 Technical Report. *arXiv preprint arXiv:2412.19437*, December 2024.

[^6]: Jiarui Fang and Shangchun Zhao. A unified sequence parallelism approach for long context generative ai. *arXiv preprint arXiv:2405.07719*, 2024.

[^7]: Jiarui Fang, Jinzhe Pan, Xibo Sun, Aoyu Li, and Jiannan Wang. xdit: an inference engine for diffusion transformers (dits) with massive parallelism. *arXiv preprint arXiv:2411.01738*, 2024.

[^8]: Google. Pallas: a jax kernel language, 2025. URL [https://docs.jax.dev/en/latest/pallas/index.html](https://docs.jax.dev/en/latest/pallas/index.html).

[^9]: Horace He, Less Wright, Luca Wehrstedt, Tianyu Liu, and Wanchao Liang. Introducing async tensor parallelism in pytorch. https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487/1, September 2024.

[^10]: Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. System optimizations for enabling training of extreme long sequence transformer models. In *Proceedings of the 43rd ACM Symposium on Principles of Distributed Computing (PODC ’24)*, pp. 121–130, New York, NY, USA, 2024. Association for Computing Machinery. doi: 10.1145/3662158.3662806. URL [https://doi.org/10.1145/3662158.3662806](https://doi.org/10.1145/3662158.3662806).

[^11]: Zhihao Jia, Matei Zaharia, and Alex Aiken. Beyond Data and Model Parallelism for Deep Neural Networks. *Proceedings of the 2nd SysML Conference*, 2019.

[^12]: Wanchao Liang, Tianyu Liu, Less Wright, Will Constable, Andrew Gu, Chien-Chin Huang, Iris Zhang, Wei Feng, Howard Huang, Junjie Wang, Sanket Purandare, Gokul Nadathur, and Stratos Idreos. Torchtitan: One-stop pytorch native solution for production ready LLM pretraining. In *The Thirteenth International Conference on Learning Representations*, 2025. URL [https://openreview.net/forum?id=SFN6Wm7YBI](https://openreview.net/forum?id=SFN6Wm7YBI).

[^13]: Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In *The Twelfth International Conference on Learning Representations*, 2024. URL [https://openreview.net/forum?id=WsRHpHH4s0](https://openreview.net/forum?id=WsRHpHH4s0).

[^14]: Deepak Narayanan, Mohammad Shoeybi, Jared Casper, Patrick LeGresley, Mostofa Patwary, Vijay Korthikanti, Dmitri Vainbrand, Prethvi Kashinkunti, Julie Bernauer, Bryan Catanzaro, Amar Phanishayee, and Matei Zaharia. Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM. *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*, 2021.

[^15]: Nvidia. Nvidia NVLink and NVLink Switch. [https://www.nvidia.com/en-us/data-center/nvlink/](https://www.nvidia.com/en-us/data-center/nvlink/).

[^16]: Nvidia. Nvshmem. [https://developer.nvidia.com/nvshmem](https://developer.nvidia.com/nvshmem).

[^17]: Nvidia. Nvidia ampere architecture in-depth. [https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/), May 2020.

[^18]: Nvidia. Nvidia CuTe. [https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00\_quickstart.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md), 2024.

[^19]: Nvidia. Advancing performance with nvidia sharp in-network computing. [https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/](https://developer.nvidia.com/blog/advancing-performance-with-nvidia-sharp-in-network-computing/), 2024.

[^20]: Nvidia. Nvidia blackwell architecture technical brief. [https://resources.nvidia.com/en-us-blackwell-architecture](https://resources.nvidia.com/en-us-blackwell-architecture), 2025.

[^21]: Nvidia. Company Overview. [https://s201.q4cdn.com/141608511/files/doc\_presentations/2025/08/Q226-NVDA-Company-Overview-Final.pdf](https://s201.q4cdn.com/141608511/files/doc_presentations/2025/08/Q226-NVDA-Company-Overview-Final.pdf), August 2025.

[^22]: Nvidia. Nvidia collective communications library (nccl). [https://developer.nvidia.com/nccl](https://developer.nvidia.com/nccl), 2025a.

[^23]: Nvidia. Nvidia nvlink and nvlink switch. [https://www.nvidia.com/en-us/data-center/nvlink/](https://www.nvidia.com/en-us/data-center/nvlink/), 2025b.

[^24]: Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In *International Conference on Learning Representations*, 2017. doi: 1701.06538. URL [https://arxiv.org/abs/1701.06538](https://arxiv.org/abs/1701.06538).

[^25]: Mohammad Shoeybi, Mostofa Patwary, Raul Puri, Patrick LeGresley, Jared Casper, and Bryan Catanzaro. Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism. *arXiv preprint arXiv:1909.08053*, September 2019.

[^26]: Min Si, Pavan Balaji, Yongzhou Chen, Ching-Hsiang Chu, Adi Gangidi, Saif Hasan, Subodh Iyengar, Dan Johnson, Bingzhe Liu, Regina Ren, Ashmitha Jeevaraj Shetty, Greg Steinbrecher, Yulun Wang, Bruce Wu, Xinfeng Xie, Jingyi Yang, Mingran Yang, Kenny Yu, Minlan Yu, Cen Zhao, Wes Bland, Denis Boyda, Suman Gumudavelli, Prashanth Kannan, Cristian Lumezanu, Rui Miao, Zhe Qu, Venkat Ramesh, Maxim Samoylov, Jan Seidel, Srikanth Sundaresan, Feng Tian, Qiye Tan, Shuqiang Zhang, Yimeng Zhao, Shengbao Zheng, Art Zhu, and Hongyi Zeng. Collective communication for 100k+ gpus. *arXiv preprint arXiv:2510.20171*, October 2025.

[^27]: Benjamin F. Spector, Simran Arora, Aaryan Singhal, Arjun Parthasarathy, Daniel Y. Fu, and Christopher Ré. Thunderkittens: Simple, fast, and adorable kernels. In *The Thirteenth International Conference on Learning Representations*, April 2025. URL [https://openreview.net/forum?id=0fJfVOSUra](https://openreview.net/forum?id=0fJfVOSUra).

[^28]: Vijay Thakkar, Pradeep Ramani, Cris Cecka, Aniket Shivam, Honghao Lu, Ethan Yan, Jack Kosaian, Mark Hoemmen, Haicheng Wu, Andrew Kerr, Matt Nicely, Duane Merrill, Dustyn Blasig, Fengqi Qiao, Piotr Majcher, Paul Springer, Markus Hohnerbach, Jin Wang, and Manish Gupta. Cutlass: Cuda templates for linear algebra subroutines. [https://github.com/NVIDIA/cutlass](https://github.com/NVIDIA/cutlass).

[^29]: Philippe Tillet, H. T. Kung, and David Cox. Triton: an intermediate language and compiler for tiled neural network computations. In *Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages*, 2019.

[^30]: William Tsu. Introducing Nvidia HGX H100: An Accelerated Server Platform for AI and High-Performance Computing. [https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/](https://developer.nvidia.com/blog/introducing-nvidia-hgx-h100-an-accelerated-server-platform-for-ai-and-high-performance-computing/), April 2022.

[^31]: Shulai Zhang, Ningxin Zheng, Haibin Lin, Ziheng Jiang, Wenlei Bao, Chengquan Jiang, Qi Hou, Weihao Cui, Size Zheng, Li-Wen Chang, Quan Chen, and Xin Liu. Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts. *Proceedings of the 8th MLSys Conference*, March 2025.

[^32]: Chenggang Zhao, Shangyan Zhou, Liyue Zhang, Chengqi Deng, Zhean Xu, Yuxuan Liu, Kuai Yu, Jiashi Li, and Liang Zhao. Deepep: an efficient expert-parallel communication library. [https://github.com/deepseek-ai/DeepEP](https://github.com/deepseek-ai/DeepEP), 2025.

[^33]: Yanli Zhao, Andrew Gu, Rohan Varma, Liang Luo, Chien-Chin Huang, Min Xu, Less Wright, Hamid Shojanazeri, Myle Ott, Sam Shleifer, Alban Desmaison, Can Balioglu, Pritam Damania, Bernard Nguyen, Geeta Chauhan, Yuchen Hao, Ajit Mathews, and Shen Li. Pytorch fsdp: Experiences on scaling fully sharded data parallel, 2023.

[^34]: Lianmin Zheng, Zhuohan Li, Hao Zhang Zhang, Yonghao Zhuang, Zhifeng Chen, Yanping Huang, Yida Wang, Yuanzhong Xu Xu, Danyang Zhuo, and Eric P Xing. Alpa: Automating inter-and Intra-Operator parallelism for distributed deep learning. *16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)*, 2022.

[^35]: Size Zheng, Wenlei Bao, Qi Hou, Xuegui Zheng, Jin Fang, Chenhui Huang, Tianqi Li, Haojie Duanmu, Renze Chen, Ruifan Xu, Yifan Guo, Ningxin Zheng, Ziheng Jiang, Xinyi Di, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, Liqiang Lu, Yun Liang, Jidong Zhai, and Xin Liu. Triton-distributed: Programming Overlapping Kernels on Distributed AI Systems with the Triton Compiler. *arXiv preprint arXiv:2504.19442*, June 2025a.

[^36]: Size Zheng, Jin Fang, Xuegui Zheng, Qi Hou, Wenlei Bao, Ningxin Zheng, Ziheng Jiang, Dongyang Wang, Jianxi Ye, Haibin Lin, Li-Wen Chang, and Xin Liu. TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives. *arXiv preprint arXiv:2503.20313*, March 2025b.

[^37]: Kan Zhu, Yufei Gao, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Tian Tang, Qinyu Xu, Zihao Ye, Keisuke Kamahori, Chien-Yu Lin, Ziren Wang, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. NanoFlow: Towards Optimal Large Language Model Serving Throughput. *arXiv preprint arXiv:2408.12757*, May 2025.