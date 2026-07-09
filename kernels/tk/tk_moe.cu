/**
 * @file tk_moe.cu
 * @brief Combined TK MoE extension for the moe_bench DistributedScheme (bf16 EP).
 *
 * Exposes, in one module, everything the TKFusedEP scheme needs:
 *   - TKParallelTensor bindings (IPC symmetric buffers + broker)
 *   - grouped_gemm(inputs, weights, outputs, padded_counts)      [layer0 up / aux]
 *   - moe_dispatch_gemm(...)                                     [layer0 dispatch ⊕ gate GEMM]
 *   - moe_gemm_combine_fused(...)                                [layer1 W2 GEMM ⊕ combine]
 *   - pcie_device_barrier(barrier, seq)
 *
 * All kernels are the ones verified standalone in tileoverlap/{01,02,03}. This
 * file just re-hosts their entrypoints under one PYBIND11_MODULE so the Python
 * scheme imports a single .so. Compiled per world size (-DTK_NUM_DEVICES=N).
 */

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

#include "sm120_common.cuh"   // copied next to this file at build time

using namespace kittens;
using namespace tileoverlap;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 4
#endif
#ifndef TK_HIDDEN
#define TK_HIDDEN 7168
#endif

/* ===================================================================== *
 * 1. Plain grouped GEMM (local): outputs[rows(e)] = inputs[rows(e)] @ W[e]
 * ===================================================================== */
namespace gg {
struct globals {
    using cfg = gemm_config;
    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl     = gl<bf16, 1, 1, -1, -1>;
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    activations_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
};
struct no_gate { __device__ inline void operator()(int) const {} };
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    grouped_gemm_sm120(G, no_gate{}, blockIdx.x, gridDim.x);
}
void entry(const at::Tensor &inputs, const at::Tensor &weights, at::Tensor &outputs,
           const at::Tensor &padded_tokens_per_expert, const int expert_offset) {
    TORCH_CHECK(inputs.size(0) % gemm_config::ROW_BLOCK == 0, "tokens % 128");
    TORCH_CHECK(inputs.size(1) % gemm_config::RED_BLOCK == 0, "K % 64");
    TORCH_CHECK(weights.size(2) % gemm_config::COL_BLOCK == 0, "N % 128");
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(inputs),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .expert_offset = expert_offset
    };
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, inputs.device().index()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<sm, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
// DEBUG (docs/11 T1): identical to entry() but launches an explicit num_blocks
// grid instead of the full SM count. Used to isolate the "16 SMs yielded to comm"
// cost — running the same W2 GEMM at 94 blocks vs 110 blocks with nothing else
// changed. No comm, no epilogue: pure GEMM SM-yield measurement.
void entry_nb(const at::Tensor &inputs, const at::Tensor &weights, at::Tensor &outputs,
              const at::Tensor &padded_tokens_per_expert, const int expert_offset,
              const int num_blocks) {
    TORCH_CHECK(inputs.size(0) % gemm_config::ROW_BLOCK == 0, "tokens % 128");
    TORCH_CHECK(inputs.size(1) % gemm_config::RED_BLOCK == 0, "K % 64");
    TORCH_CHECK(weights.size(2) % gemm_config::COL_BLOCK == 0, "N % 128");
    TORCH_CHECK(num_blocks >= 1, "num_blocks must be >= 1");
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(inputs),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .expert_offset = expert_offset
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace gg

/* ===================================================================== *
 * 2. Dispatch ⊕ grouped GEMM (layer0): pull tokens from peers to local,
 *    fuse the gate (or up) GEMM. Port of tileoverlap/02.
 * ===================================================================== */
namespace disp {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec);
    using pre_tokens_pgl  = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using indices_gl      = gl<int, 1, 1, -1, 2>;
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_pgl pre_tokens;
    post_tokens_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    indices_gl pull_dispatch_indices;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;
    const int expert_offset;
    const int num_padded_local_tokens;
    const int num_comp_sms;
};
__device__ inline void dispatch(const globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int token_idx = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (token_idx < G.num_padded_local_tokens) {
            const int src_dev_idx = G.pull_dispatch_indices[{token_idx, 0}];
            const int src_token_idx = G.pull_dispatch_indices[{token_idx, 1}];
            if (src_dev_idx >= 0 && src_token_idx >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.pre_tokens[src_dev_idx], {src_token_idx, 0}, token_arrived[lane_id]);
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.activations, token[lane_id], {token_idx, 0});
                tma::store_async_wait();
            }
            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                         :: "l"(&G.barrier[G.dev_idx][{token_idx / gemm_config::ROW_BLOCK}]), "r"(1) : "memory");
        }
    }
}
struct dispatch_gate {
    const globals &G;
    __device__ inline void operator()(int row_idx) const {
        int v;
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        while (v != gemm_config::ROW_BLOCK) {
            __nanosleep(32);
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        }
    }
};
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, dispatch_gate{G}, blockIdx.x, G.num_comp_sms);
    else
        dispatch(G, blockIdx.x - G.num_comp_sms);
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = 0;
}
void entry(kittens::py::TKParallelTensor &pre_tokens, at::Tensor &post_tokens,
           at::Tensor &weights, at::Tensor &outputs, at::Tensor &padded_tokens_per_expert,
           at::Tensor &pull_dispatch_indices, kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "weights first dim must equal local expert count (NUM_GPUS mismatch?)");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .pull_dispatch_indices = kittens::py::tensor_to_gl<globals::indices_gl>(pull_dispatch_indices),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = dev_idx * num_local_experts,
        .num_padded_local_tokens = num_padded_local_tokens, .num_comp_sms = num_comp_sms
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int dispatch_blocks = (num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_comp_sms + dispatch_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
__global__ __launch_bounds__(32)
void barrier_kernel(const __grid_constant__ globals::barrier_pgl bar, const int dev_idx, const int seq) {
    pcie_sync::pcie_barrier_all(bar, dev_idx, seq);
}
void barrier_entry(kittens::py::TKParallelTensor &barrier, const int seq) {
    auto bar = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    barrier_kernel<<<1, 32, 0, stream>>>(bar, barrier.local_rank_, seq);
    CUDACHECK(cudaGetLastError());
}
} // namespace disp

/* ===================================================================== *
 * 2b. Dispatch via source-side PUSH ⊕ grouped GEMM (layer0, optimization #1).
 *
 * Reverses 2's pull. Each card is both a SOURCE (pushes its own tokens out to
 * the expert cards that need them) and an EXPERT (runs grouped GEMM on tokens
 * pushed into its gathered buffer). Pushing uses the machine's strong path
 * (~51 GB/s) instead of the weak SM pull (~20 GB/s, probe [C2]); remote
 * red.release.sys is legal here (probe [B] confirmed native atomics work).
 *
 * gathered is a pgl now (peers write it). The GEMM producer's gate spins on a
 * LOCAL row-block counter that peers increment remotely. Padding is handled by
 * pre-seeding each row-block counter with its padding slack, so real pushes
 * bring it exactly to ROW_BLOCK.
 *
 * push schedule (per source assignment, this card's view):
 *   push_indices[i] = (dst_dev, dst_slot) for this card's i-th outgoing token,
 *   push_src[i]     = local source token index to read from pre_tokens.
 * i ranges over this card's own (token,expert) assignments to remote+local experts.
 * ===================================================================== */
namespace dpush {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec);
    using pre_tokens_gl   = gl<bf16, 1, 1, -1, H, token_vec>;                       // local source tokens
    using gathered_pgl    = pgl<gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>, NUM_DEVICES, false>; // peers push here
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;          // local GEMM A-tile source
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using push_idx_gl     = gl<int, 1, 1, -1, 2>;    // (dst_dev, dst_slot) per outgoing assignment
    using push_src_gl     = gl<int, 1, 1, -1, 1>;    // local src token idx per outgoing assignment
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_gl pre_tokens;
    gathered_pgl gathered;         // this card's gathered view + peers'
    post_tokens_gl activations;    // == gathered[dev_idx], the local GEMM A-tile source
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    push_idx_gl push_indices;
    push_src_gl push_src;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;
    const int expert_offset;
    const int num_padded_local_tokens;   // rows of THIS card's gathered
    const int num_push;                  // number of outgoing assignments from this card
    const int num_comp_sms;
};
// Push block: each thread pushes one outgoing token to its (dst_dev, dst_slot),
// then remotely bumps that dst card's row-block counter.
__device__ inline void push(const globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int i = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (i < G.num_push) {
            const int dst_dev  = G.push_indices[{i, 0}];
            const int dst_slot = G.push_indices[{i, 1}];
            const int src_tok  = G.push_src[{i, 0}];
            if (dst_dev >= 0 && dst_slot >= 0 && src_tok >= 0) {
                // read local source token into smem, then TMA-push to dst card's gathered slot
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.pre_tokens, {src_tok, 0}, token_arrived[lane_id]);
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.gathered[dst_dev], token[lane_id], {dst_slot, 0});
                tma::store_async_wait();
                // remote release-add on dst card's row-block counter (native atomics OK here)
                asm volatile("{red.release.sys.global.add.s32 [%0], %1;}"
                             :: "l"(&G.barrier[dst_dev][{dst_slot / gemm_config::ROW_BLOCK}]), "r"(1) : "memory");
            }
        }
    }
}
// Atomic-free push: same TMA data plane as push() but NO remote red.add. Used by
// the "push2" path (push_data -> pcie_barrier_all -> plain grouped_gemm), which
// avoids the PCIe remote-atomic increment loss (docs/08) while still using the
// strong push bandwidth. Completion is a single cross-device barrier, not counters.
__device__ inline void push_data(const globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int i = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (i < G.num_push) {
            const int dst_dev  = G.push_indices[{i, 0}];
            const int dst_slot = G.push_indices[{i, 1}];
            const int src_tok  = G.push_src[{i, 0}];
            if (dst_dev >= 0 && dst_slot >= 0 && src_tok >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.pre_tokens, {src_tok, 0}, token_arrived[lane_id]);
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.gathered[dst_dev], token[lane_id], {dst_slot, 0});
                tma::store_async_wait();
            }
        }
    }
}
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void push_data_kernel(const __grid_constant__ globals G) {
    push_data(G, blockIdx.x);
}
// Standalone data-only push (no GEMM, no atomics). Caller sequences:
//   push_data_entry (all cards) -> pcie_device_barrier -> grouped_gemm.
void push_data_entry(kittens::py::TKParallelTensor &pre_tokens,
           kittens::py::TKParallelTensor &gathered, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &push_indices, at::Tensor &push_src,
           kittens::py::TKParallelTensor &barrier, const int num_push) {
    globals G {
        .pre_tokens = kittens::py::tensor_to_gl<globals::pre_tokens_gl>(pre_tokens.data_),
        .gathered = kittens::py::parallel_tensor_to_pgl<globals::gathered_pgl>(gathered),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(gathered.data_),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .push_indices = kittens::py::tensor_to_gl<globals::push_idx_gl>(push_indices),
        .push_src = kittens::py::tensor_to_gl<globals::push_src_gl>(push_src),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_, .num_local_experts = 0, .expert_offset = 0,
        .num_padded_local_tokens = 0, .num_push = num_push, .num_comp_sms = 0
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int push_blocks = (num_push + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(push_data_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    push_data_kernel<<<push_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
// Gate: producer warp spins on the LOCAL row-block counter (peers increment it
// remotely) until it reaches ROW_BLOCK. Counter is pre-seeded with padding slack.
struct push_gate {
    const globals &G;
    __device__ inline void operator()(int row_idx) const {
        int v;
        asm volatile("{ld.acquire.sys.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        while (v != gemm_config::ROW_BLOCK) {
            __nanosleep(64);
            asm volatile("{ld.acquire.sys.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        }
    }
};
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, push_gate{G}, blockIdx.x, G.num_comp_sms);
    else
        push(G, blockIdx.x - G.num_comp_sms);
}
// Padding slack seeding is done host-side by the scheme (it knows real vs padded
// per-expert counts): before launch it writes (ROW_BLOCK - real_in_block) into
// each row-block counter, so real pushes bring it exactly to ROW_BLOCK. No
// device-side counter reset (see entry): re-seeding + a scheme-side barrier
// handles the epoch boundary without racing in-flight remote adds.
void entry(kittens::py::TKParallelTensor &pre_tokens,
           kittens::py::TKParallelTensor &gathered, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &push_indices, at::Tensor &push_src,
           kittens::py::TKParallelTensor &barrier, const int num_comm_sms,
           const int num_padded_local_tokens, const int num_push) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "weights first dim must equal local expert count (NUM_GPUS mismatch?)");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .pre_tokens = kittens::py::tensor_to_gl<globals::pre_tokens_gl>(pre_tokens.data_),
        .gathered = kittens::py::parallel_tensor_to_pgl<globals::gathered_pgl>(gathered),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(gathered.data_),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .push_indices = kittens::py::tensor_to_gl<globals::push_idx_gl>(push_indices),
        .push_src = kittens::py::tensor_to_gl<globals::push_src_gl>(push_src),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = dev_idx * num_local_experts,
        .num_padded_local_tokens = num_padded_local_tokens, .num_push = num_push,
        .num_comp_sms = num_comp_sms
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int push_blocks = (num_push + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_comp_sms + push_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    // NO device-side reset here: row-0 counters are re-seeded from host every
    // iteration (scheme writes slack), and a cross-device barrier in the scheme
    // (after this call) ensures all peer pushes land before the next seed. A
    // reset here would race a slow peer's still-in-flight remote red.add.
}

// DEBUG: run ONLY the push blocks (no GEMM/gate) so the kernel can't hang; the
// caller inspects the dst counters afterward to isolate counter-update bugs from
// gate visibility bugs.
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void push_only_kernel(const __grid_constant__ globals G) {
    push(G, blockIdx.x);
}
void push_only_entry(kittens::py::TKParallelTensor &pre_tokens,
           kittens::py::TKParallelTensor &gathered, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &push_indices, at::Tensor &push_src,
           kittens::py::TKParallelTensor &barrier, const int num_push) {
    const int dev_idx = barrier.local_rank_;
    globals G {
        .pre_tokens = kittens::py::tensor_to_gl<globals::pre_tokens_gl>(pre_tokens.data_),
        .gathered = kittens::py::parallel_tensor_to_pgl<globals::gathered_pgl>(gathered),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(gathered.data_),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .push_indices = kittens::py::tensor_to_gl<globals::push_idx_gl>(push_indices),
        .push_src = kittens::py::tensor_to_gl<globals::push_src_gl>(push_src),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = 0, .expert_offset = 0,
        .num_padded_local_tokens = 0, .num_push = num_push, .num_comp_sms = 0
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int push_blocks = (num_push + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(push_only_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    push_only_kernel<<<push_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace dpush

/* ===================================================================== *
 * 2c. Dispatch via source-side PUSH ⊕ grouped GEMM, SLOT-signal completion
 *     (layer0, optimization #1c — docs/09). Fixes push's frozen remote-atomic
 *     loss (docs/08) WITHOUT giving up fusion (unlike push2).
 *
 * Protocol (docs/09 §3): source TMA-pushes each token to its host-fixed dst
 * slot (same schedule as dpush), then bumps a LOCAL per-(dst,row block) counter
 * with a GPU-scope atom.acq_rel (legal — no remote atomics). The thread that
 * brings a counter to its host-precomputed expected value is the SOLE elected
 * signaler for that (src card, dst row block): it fences and writes a monotonic
 * seq into the dst card's barrier[2 + my_rank][row block] via st.release.sys
 * (single writer, PCIe-safe). The GEMM producer's gate spins locally, waiting
 * only the source cards that actually contribute to each row block (gate_expected
 * > 0). Zero remote atomics; fusion preserved.
 *
 * Signal layout: barrier_l0 rows [2, 2+NUM_DEVICES) — row (2+s) = source s's
 * completion signals, col = dst-local row block. Rows 0/1 keep pull/barrier use.
 * ===================================================================== */
namespace dpush3 {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec);
    using pre_tokens_gl   = gl<bf16, 1, 1, -1, H, token_vec>;                       // local source tokens
    using gathered_pgl    = pgl<gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>, NUM_DEVICES, false>; // peers push here
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;          // local GEMM A-tile source
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using push_idx_gl     = gl<int, 1, 1, -1, 2>;    // (dst_dev, dst_slot) per outgoing assignment
    using push_src_gl     = gl<int, 1, 1, -1, 1>;    // local src token idx per outgoing assignment
    using push_cnt_gl     = gl<int, 1, 1, -1, 1>;    // flat local counter idx per outgoing assignment
    using cnt1d_gl        = gl<int, 1, 1, 1, -1>;    // 1D int arrays (local_cnt, push_expected)
    using gate_exp_gl     = gl<int, 1, 1, -1, -1>;   // (nblk_local, NUM_DEVICES) expected arrivals
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_gl pre_tokens;
    gathered_pgl gathered;         // this card's gathered view + peers'
    post_tokens_gl activations;    // == gathered[dev_idx], the local GEMM A-tile source
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    push_idx_gl push_indices;
    push_src_gl push_src;
    push_cnt_gl push_cnt_idx;      // per-assignment local counter index
    cnt1d_gl local_cnt;            // (total_dst_blocks,) local counters, zeroed each iter
    cnt1d_gl push_expected;        // (total_dst_blocks,) satisfaction count per counter
    gate_exp_gl gate_expected;     // (nblk_local, NUM_DEVICES) — gate uses >0 only
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;
    const int expert_offset;
    const int num_padded_local_tokens;   // rows of THIS card's gathered
    const int num_push;                  // number of outgoing assignments from this card
    const int num_comp_sms;
    const int seq;                       // monotonic signal seq (immune to reset)
};
// Push block: TMA-push one token, local-count, elected signaler writes dst slot.
__device__ inline void push3(const globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int i = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (i < G.num_push) {
            const int dst_dev  = G.push_indices[{i, 0}];
            const int dst_slot = G.push_indices[{i, 1}];
            const int src_tok  = G.push_src[{i, 0}];
            if (dst_dev >= 0 && dst_slot >= 0 && src_tok >= 0) {
                // read local source token into smem, then TMA-push to dst gathered slot
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.pre_tokens, {src_tok, 0}, token_arrived[lane_id]);
                wait(token_arrived[lane_id], 0);
                tma::store_async(G.gathered[dst_dev], token[lane_id], {dst_slot, 0});
                tma::store_async_wait();                    // my remote bulk write committed
                // LOCAL count (gpu scope, legal on PCIe — no remote atomics)
                const int c = G.push_cnt_idx[{i, 0}];
                int old;
                asm volatile("atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;"
                             : "=r"(old) : "l"(&G.local_cnt[{c}]), "r"(1) : "memory");
                if (old + 1 == G.push_expected[{c}]) {      // I'm the last (sole) completer
                    __threadfence_system();                 // §3.4: data write visible before signal
                    pcie_sync::signal_slot(G.barrier, dst_dev, 2 + G.dev_idx,
                                           dst_slot / gemm_config::ROW_BLOCK, G.seq);
                }
            }
        }
    }
}
// Gate: producer warp waits, per local row block, every source card that
// actually contributes (gate_expected > 0). Local acquire spin, no remote atomics.
struct push3_gate {
    const globals &G;
    __device__ inline void operator()(int row_idx) const {
        #pragma unroll 1
        for (int s = 0; s < globals::NUM_DEVICES; s++)
            if (G.gate_expected[{row_idx, s}] > 0)
                pcie_sync::wait_slot(G.barrier, G.dev_idx, 2 + s, row_idx, G.seq);
    }
};
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, push3_gate{G}, blockIdx.x, G.num_comp_sms);
    else
        push3(G, blockIdx.x - G.num_comp_sms);
}
void entry(kittens::py::TKParallelTensor &pre_tokens,
           kittens::py::TKParallelTensor &gathered, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &push_indices, at::Tensor &push_src,
           at::Tensor &push_cnt_idx, at::Tensor &local_cnt, at::Tensor &push_expected,
           at::Tensor &gate_expected, kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens, const int num_push,
           const int seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "weights first dim must equal local expert count (NUM_GPUS mismatch?)");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .pre_tokens = kittens::py::tensor_to_gl<globals::pre_tokens_gl>(pre_tokens.data_),
        .gathered = kittens::py::parallel_tensor_to_pgl<globals::gathered_pgl>(gathered),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(gathered.data_),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .push_indices = kittens::py::tensor_to_gl<globals::push_idx_gl>(push_indices),
        .push_src = kittens::py::tensor_to_gl<globals::push_src_gl>(push_src),
        .push_cnt_idx = kittens::py::tensor_to_gl<globals::push_cnt_gl>(push_cnt_idx),
        .local_cnt = kittens::py::tensor_to_gl<globals::cnt1d_gl>(local_cnt),
        .push_expected = kittens::py::tensor_to_gl<globals::cnt1d_gl>(push_expected),
        .gate_expected = kittens::py::tensor_to_gl<globals::gate_exp_gl>(gate_expected),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = dev_idx * num_local_experts,
        .num_padded_local_tokens = num_padded_local_tokens, .num_push = num_push,
        .num_comp_sms = num_comp_sms, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int push_blocks = (num_push + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_comp_sms + push_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

// DEBUG (docs/09 §5.4 / R1 decision point): run ONLY the push blocks (no GEMM,
// no gate) so the kernel can't hang; caller reconciles dst signal slots against
// gate_expected + compares gathered checksum vs the pull-path gathered.
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void push3_only_kernel(const __grid_constant__ globals G) {
    push3(G, blockIdx.x);
}
void push3_only_entry(kittens::py::TKParallelTensor &pre_tokens,
           kittens::py::TKParallelTensor &gathered, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &push_indices, at::Tensor &push_src,
           at::Tensor &push_cnt_idx, at::Tensor &local_cnt, at::Tensor &push_expected,
           at::Tensor &gate_expected, kittens::py::TKParallelTensor &barrier,
           const int num_push, const int seq) {
    const int dev_idx = barrier.local_rank_;
    globals G {
        .pre_tokens = kittens::py::tensor_to_gl<globals::pre_tokens_gl>(pre_tokens.data_),
        .gathered = kittens::py::parallel_tensor_to_pgl<globals::gathered_pgl>(gathered),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(gathered.data_),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .push_indices = kittens::py::tensor_to_gl<globals::push_idx_gl>(push_indices),
        .push_src = kittens::py::tensor_to_gl<globals::push_src_gl>(push_src),
        .push_cnt_idx = kittens::py::tensor_to_gl<globals::push_cnt_gl>(push_cnt_idx),
        .local_cnt = kittens::py::tensor_to_gl<globals::cnt1d_gl>(local_cnt),
        .push_expected = kittens::py::tensor_to_gl<globals::cnt1d_gl>(push_expected),
        .gate_expected = kittens::py::tensor_to_gl<globals::gate_exp_gl>(gate_expected),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = 0, .expert_offset = 0,
        .num_padded_local_tokens = 0, .num_push = num_push, .num_comp_sms = 0, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int push_blocks = (num_push + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(push3_only_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    push3_only_kernel<<<push_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace dpush3

/* ===================================================================== *
 * 3. W2 grouped GEMM ⊕ combine (layer1). Port of tileoverlap/03 fused.
 * ===================================================================== */
namespace comb {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using output_gl      = gl<bf16, 1, 1, -1, H>;
    using expert_out_pgl = pgl<gl<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    using combine_out_gl = gl<bf16, 1, 1, -1, H>;
    using combine_idx_gl = gl<int,  1, 1, -1, 2>;
    using combine_w_gl   = gl<float,1, 1, -1, 1>;
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    activations_gl activations;
    weights_gl weights;
    output_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
    expert_out_pgl expert_outputs;
    combine_out_gl combine_out;
    combine_idx_gl combine_indices;
    combine_w_gl combine_weights;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_source_tokens;
    const int num_comp_sms;
    const int combine_seq;
};
struct combine_signal_epilogue {
    const globals &G;
    const int col_blocks;
    __device__ inline void operator()(int row_idx, int) const {
        __threadfence_system();
        kittens::group<gemm_config::CONSUMER_WARPS>::sync(1);
        if (kittens::laneid() != 0 || kittens::warpid() != 0) return;
        int done;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(done) : "l"(&G.barrier[G.dev_idx][{0, row_idx}]) : "memory");
        if (done + 1 == col_blocks) {
            #pragma unroll 1
            for (int d = 0; d < globals::NUM_DEVICES; d++)
                pcie_sync::signal_slot(G.barrier, d, 1 + G.dev_idx, row_idx, G.combine_seq);
        }
    }
};
__device__ inline void combine(const globals &G, const int t) {
    if (t >= G.num_source_tokens) return;
    constexpr int H = globals::H, VEC = 8, HVEC = H / VEC;
    __shared__ int s_erank[globals::TOP_K], s_slot[globals::TOP_K];
    __shared__ float s_w[globals::TOP_K];
    if (threadIdx.x < globals::TOP_K) {
        const int k = threadIdx.x;
        const int erank = G.combine_indices[{t * globals::TOP_K + k, 0}];
        const int slot  = G.combine_indices[{t * globals::TOP_K + k, 1}];
        s_erank[k] = erank; s_slot[k] = slot;
        s_w[k] = G.combine_weights[{t * globals::TOP_K + k, 0}];
        if (erank >= 0 && slot >= 0)
            pcie_sync::wait_slot(G.barrier, G.dev_idx, 1 + erank, slot / gemm_config::ROW_BLOCK, G.combine_seq);
    }
    __syncthreads();
    bf16 *out_row = &G.combine_out[{t, 0}];
    float4 *out_v = reinterpret_cast<float4 *>(out_row);
    for (int c = threadIdx.x; c < HVEC; c += blockDim.x) {
        float acc[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) acc[i] = 0.0f;
        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) {
            const int e_rank = s_erank[k], slot = s_slot[k];
            if (e_rank < 0 || slot < 0) continue;
            const float w = s_w[k];
            const bf16 *peer_row = &G.expert_outputs[e_rank][{slot, 0}];
            const float4 packed = reinterpret_cast<const float4 *>(peer_row)[c];
            const bf16_2 *pv = reinterpret_cast<const bf16_2 *>(&packed);
            #pragma unroll
            for (int j = 0; j < VEC / 2; j++) {
                float2 f = __bfloat1622float2(pv[j]);
                acc[2*j] += w * f.x; acc[2*j+1] += w * f.y;
            }
        }
        bf16_2 res[VEC / 2];
        #pragma unroll
        for (int j = 0; j < VEC / 2; j++) res[j] = __floats2bfloat162_rn(acc[2*j], acc[2*j+1]);
        out_v[c] = *reinterpret_cast<const float4 *>(res);
    }
}
struct no_gate { __device__ inline void operator()(int) const {} };
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G) {
    const int col_blocks = static_cast<int>(G.weights.cols()) / gemm_config::COL_BLOCK;
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, no_gate{}, combine_signal_epilogue{G, col_blocks}, blockIdx.x, G.num_comp_sms);
    else
        combine(G, blockIdx.x - G.num_comp_sms);
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{0, i}] = 0;
}
void entry(at::Tensor &activations, at::Tensor &weights,
           kittens::py::TKParallelTensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           at::Tensor &combine_out, at::Tensor &combine_indices, at::Tensor &combine_weights,
           kittens::py::TKParallelTensor &barrier, const int num_comm_sms,
           const int num_padded_local_tokens, const int num_source_tokens, const int combine_seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "weights first dim must equal local expert count (NUM_GPUS mismatch?)");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::output_gl>(expert_outputs.data_),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = dev_idx * num_local_experts,
        .expert_outputs = kittens::py::parallel_tensor_to_pgl<globals::expert_out_pgl>(expert_outputs),
        .combine_out = kittens::py::tensor_to_gl<globals::combine_out_gl>(combine_out),
        .combine_indices = kittens::py::tensor_to_gl<globals::combine_idx_gl>(combine_indices),
        .combine_weights = kittens::py::tensor_to_gl<globals::combine_w_gl>(combine_weights),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens, .num_comp_sms = num_comp_sms, .combine_seq = combine_seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_comp_sms + num_source_tokens, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

// DEBUG (docs/11 T1): run ONLY the combine (gather+reduce) blocks, one per source
// token, with NO GEMM. Callers pass a combine_seq that the barrier ALREADY holds
// (from a prior full fused run on the same expert_outputs), so every wait_slot
// returns immediately and this measures the PURE cross-card gather+reduce cost
// (8 peer rows × 14KB per token) with zero GEMM and zero signal-wait — isolating
// combine bandwidth from HOL-wait in the layer1 tail.
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void combine_only_kernel(const __grid_constant__ globals G) {
    combine(G, blockIdx.x);
}
void combine_only_entry(at::Tensor &activations, at::Tensor &weights,
           kittens::py::TKParallelTensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           at::Tensor &combine_out, at::Tensor &combine_indices, at::Tensor &combine_weights,
           kittens::py::TKParallelTensor &barrier, const int num_comm_sms,
           const int num_padded_local, const int num_source_tokens, const int combine_seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::output_gl>(expert_outputs.data_),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = dev_idx * num_local_experts,
        .expert_outputs = kittens::py::parallel_tensor_to_pgl<globals::expert_out_pgl>(expert_outputs),
        .combine_out = kittens::py::tensor_to_gl<globals::combine_out_gl>(combine_out),
        .combine_indices = kittens::py::tensor_to_gl<globals::combine_idx_gl>(combine_indices),
        .combine_weights = kittens::py::tensor_to_gl<globals::combine_w_gl>(combine_weights),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local,
        .num_source_tokens = num_source_tokens, .num_comp_sms = 0, .combine_seq = combine_seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(combine_only_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    combine_only_kernel<<<num_source_tokens, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace comb

/* ===================================================================== *
 * 4. W2 GEMM ⊕ LOCAL pre-reduction (layer1, T6-v0; docs/11 §T6, docs/13).
 *    Regroups the combine sum BY EXPERT CARD. This kernel does the expert-card
 *    half: W2 GEMM writes expert_out locally, then (same grid) "job" blocks
 *    FP32-weighted-sum this card's slots that hit a given (src_dev, src_tok)
 *    into ONE partial row on a peer-readable buffer. The GEMM completion signal
 *    is LOCAL-only (jobs are same-card) — cheaper than combine's cross-card
 *    fence.sys broadcast. A separate pcie_device_barrier (verified push2
 *    pattern) then makes all cards' partials system-visible; moe_final_reduce
 *    (source card) pulls the contributing cards' partial rows and sums them.
 *    Zero NEW cross-device protocol — the v0 safe checkpoint before v1's push.
 * ===================================================================== */
namespace prered {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl     = gl<bf16, 1, 1, -1, H>;              // local W2 output (expert_out)
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    using partial_pgl    = pgl<gl<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;  // peer-readable partials
    using dst_gl         = gl<int, 1, 1, -1, 2>;              // (J,2) -> (src_dev, src_tok)
    using slots_gl       = gl<int, 1, 1, -1, TOP_K>;          // (J,TOP_K) local slots (-1 pad)
    using w_gl           = gl<float, 1, 1, -1, TOP_K>;        // (J,TOP_K) weights (0 pad)
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    activations_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
    partial_pgl partials;
    dst_gl prered_dst;
    slots_gl prered_slots;
    w_gl prered_w;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_source_tokens;
    const int num_jobs;
    const int num_comp_sms;
    const int seq;
};
// source-card final-reduce globals (own struct so no dummy gl init needed).
struct final_globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    using partial_pgl    = pgl<gl<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;
    using contrib_gl     = gl<int, 1, 1, -1, -1>;             // (num_tokens, world)
    using combine_out_gl = gl<bf16, 1, 1, -1, H>;
    partial_pgl partials;
    contrib_gl final_contrib;
    combine_out_gl combine_out;
    const int dev_idx;
    const int num_source_tokens;
};
// LOCAL completion signal: elect once per row block (gpu-scope atom), single
// writer stamps seq into local barrier row 1. Only a gpu-scope threadfence is
// needed — job blocks are on the same device (combine needed fence.sys because
// it signalled cross-card; here the cross-card step is the separate barrier).
struct prered_signal_epilogue {
    const globals &G;
    const int col_blocks;
    __device__ inline void operator()(int row_idx, int) const {
        __threadfence();
        kittens::group<gemm_config::CONSUMER_WARPS>::sync(1);
        if (kittens::laneid() != 0 || kittens::warpid() != 0) return;
        int done;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(done) : "l"(&G.barrier[G.dev_idx][{0, row_idx}]) : "memory");
        if (done + 1 == col_blocks)
            asm volatile("st.release.gpu.global.s32 [%0], %1;"
                         :: "l"(&G.barrier[G.dev_idx][{1, row_idx}]), "r"(G.seq) : "memory");
    }
};
struct no_gate { __device__ inline void operator()(int) const {} };
// one job: FP32 weighted-sum this card's TOP_K-or-fewer local slots that hit
// (src_dev, src_tok) into partial row [src_dev*num_source_tokens + src_tok].
__device__ inline void prered_job(const globals &G, const int j) {
    if (j >= G.num_jobs) return;
    constexpr int H = globals::H, VEC = 8, HVEC = H / VEC;
    __shared__ int s_slot[globals::TOP_K];
    __shared__ float s_w[globals::TOP_K];
    __shared__ int s_dst_row;
    if (threadIdx.x < globals::TOP_K) {
        const int k = threadIdx.x;
        const int slot = G.prered_slots[{j, k}];
        s_slot[k] = slot;
        s_w[k] = G.prered_w[{j, k}];
        if (slot >= 0)  // wait local completion of the row block holding this slot
            pcie_sync::wait_slot(G.barrier, G.dev_idx, 1, slot / gemm_config::ROW_BLOCK, G.seq);
    }
    if (threadIdx.x == 0) {
        const int src_dev = G.prered_dst[{j, 0}];
        const int src_tok = G.prered_dst[{j, 1}];
        s_dst_row = src_dev * G.num_source_tokens + src_tok;
    }
    __syncthreads();
    bf16 *out_row = &G.partials[G.dev_idx][{s_dst_row, 0}];
    float4 *out_v = reinterpret_cast<float4 *>(out_row);
    for (int c = threadIdx.x; c < HVEC; c += blockDim.x) {
        float acc[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) acc[i] = 0.0f;
        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) {
            const int slot = s_slot[k];
            if (slot < 0) continue;
            const float w = s_w[k];
            const bf16 *e_row = &G.outputs[{slot, 0}];
            const float4 packed = reinterpret_cast<const float4 *>(e_row)[c];
            const bf16_2 *pv = reinterpret_cast<const bf16_2 *>(&packed);
            #pragma unroll
            for (int jj = 0; jj < VEC / 2; jj++) {
                float2 f = __bfloat1622float2(pv[jj]);
                acc[2*jj] += w * f.x; acc[2*jj+1] += w * f.y;
            }
        }
        bf16_2 res[VEC / 2];
        #pragma unroll
        for (int jj = 0; jj < VEC / 2; jj++) res[jj] = __floats2bfloat162_rn(acc[2*jj], acc[2*jj+1]);
        out_v[c] = *reinterpret_cast<const float4 *>(res);
    }
}
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void gemm_prered_kernel(const __grid_constant__ globals G) {
    const int col_blocks = static_cast<int>(G.weights.cols()) / gemm_config::COL_BLOCK;
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, no_gate{}, prered_signal_epilogue{G, col_blocks}, blockIdx.x, G.num_comp_sms);
    else
        prered_job(G, blockIdx.x - G.num_comp_sms);
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{0, i}] = 0;
}
// source-card final reduce: out[t] = Σ_{d: contrib[t][d]} partial_d[my_rank*nt + t].
// one block per source token; plain sum (weights already applied expert-side).
__global__ void final_reduce_kernel(const __grid_constant__ final_globals G) {
    const int t = blockIdx.x;
    if (t >= G.num_source_tokens) return;
    constexpr int H = final_globals::H, VEC = 8, HVEC = H / VEC;
    const int base = G.dev_idx * G.num_source_tokens + t;
    __shared__ int s_contrib[final_globals::NUM_DEVICES];
    if (threadIdx.x < final_globals::NUM_DEVICES)
        s_contrib[threadIdx.x] = G.final_contrib[{t, static_cast<int>(threadIdx.x)}];
    __syncthreads();
    bf16 *out_row = &G.combine_out[{t, 0}];
    float4 *out_v = reinterpret_cast<float4 *>(out_row);
    for (int c = threadIdx.x; c < HVEC; c += blockDim.x) {
        float acc[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) acc[i] = 0.0f;
        #pragma unroll 1
        for (int d = 0; d < final_globals::NUM_DEVICES; d++) {
            if (!s_contrib[d]) continue;
            const bf16 *p_row = &G.partials[d][{base, 0}];
            const float4 packed = reinterpret_cast<const float4 *>(p_row)[c];
            const bf16_2 *pv = reinterpret_cast<const bf16_2 *>(&packed);
            #pragma unroll
            for (int jj = 0; jj < VEC / 2; jj++) {
                float2 f = __bfloat1622float2(pv[jj]);
                acc[2*jj] += f.x; acc[2*jj+1] += f.y;
            }
        }
        bf16_2 res[VEC / 2];
        #pragma unroll
        for (int jj = 0; jj < VEC / 2; jj++) res[jj] = __floats2bfloat162_rn(acc[2*jj], acc[2*jj+1]);
        out_v[c] = *reinterpret_cast<const float4 *>(res);
    }
}
void gemm_prered_entry(at::Tensor &activations, at::Tensor &weights,
           kittens::py::TKParallelTensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           kittens::py::TKParallelTensor &partials, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w,
           kittens::py::TKParallelTensor &barrier, const int num_comm_sms,
           const int num_padded_local_tokens, const int num_source_tokens,
           const int num_jobs, const int seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "weights first dim must equal local expert count (NUM_GPUS mismatch?)");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(expert_outputs.data_),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = dev_idx * num_local_experts,
        .partials = kittens::py::parallel_tensor_to_pgl<globals::partial_pgl>(partials),
        .prered_dst = kittens::py::tensor_to_gl<globals::dst_gl>(prered_dst),
        .prered_slots = kittens::py::tensor_to_gl<globals::slots_gl>(prered_slots),
        .prered_w = kittens::py::tensor_to_gl<globals::w_gl>(prered_w),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens, .num_jobs = num_jobs,
        .num_comp_sms = num_comp_sms, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(gemm_prered_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    gemm_prered_kernel<<<num_comp_sms + num_jobs, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
void final_reduce_entry(kittens::py::TKParallelTensor &partials, at::Tensor &final_contrib,
           at::Tensor &combine_out, kittens::py::TKParallelTensor &barrier,
           const int num_source_tokens) {
    const int dev_idx = barrier.local_rank_;
    final_globals G {
        .partials = kittens::py::parallel_tensor_to_pgl<final_globals::partial_pgl>(partials),
        .final_contrib = kittens::py::tensor_to_gl<final_globals::contrib_gl>(final_contrib),
        .combine_out = kittens::py::tensor_to_gl<final_globals::combine_out_gl>(combine_out),
        .dev_idx = dev_idx, .num_source_tokens = num_source_tokens
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    final_reduce_kernel<<<num_source_tokens, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace prered

#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("grouped_gemm", &gg::entry);
    m.def("grouped_gemm_nb", &gg::entry_nb);
    m.def("moe_dispatch_gemm", &disp::entry);
    m.def("moe_dispatch_push", &dpush::entry);
    m.def("moe_dispatch_push_only", &dpush::push_only_entry);
    m.def("moe_push_data", &dpush::push_data_entry);
    m.def("moe_dispatch_push3", &dpush3::entry);
    m.def("moe_dispatch_push3_only", &dpush3::push3_only_entry);
    m.def("pcie_device_barrier", &disp::barrier_entry);
    m.def("moe_gemm_combine_fused", &comb::entry);
    m.def("moe_combine_only", &comb::combine_only_entry);
    m.def("moe_gemm_prered_fused", &prered::gemm_prered_entry);
    m.def("moe_final_reduce", &prered::final_reduce_entry);
}
