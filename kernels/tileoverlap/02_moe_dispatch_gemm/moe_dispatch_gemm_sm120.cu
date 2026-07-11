/**
 * @file moe_dispatch_gemm_sm120.cu
 * @brief MoE EP dispatch fused into a grouped GEMM, tile-granularity overlap,
 *        for SM120 + PCIe. Port of kernels/parallel/moe_dispatch_gemm (H100)
 *        with the SM120/PCIe adaptations from experience/12:
 *
 *   - GEMM body: warp-level mma.sync pipeline (no wgmma), 99KB smem budget.
 *   - Communication: pull-only. Dispatch blocks pull whole token vectors from
 *     peer GPUs via TMA over PCIe P2P, land them in LOCAL memory, and signal
 *     with a LOCAL device-scope atomic (legal everywhere). No remote atomics,
 *     no multimem anywhere.
 *   - The GEMM kernel is NOT split: the producer warp simply spins on the
 *     local per-row-block counter before loading each 128-token row block.
 *
 * One kernel launch: blockIdx < num_comp_sms -> grouped GEMM (persistent);
 * remaining blocks -> dispatch (short-lived, queued on the leftover SMs).
 * DEADLOCK RULE: num_comp_sms < SM count, i.e. num_comm_sms >= 1.
 */

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

#include "../common/sm120_common.cuh"

using namespace kittens;
using namespace tileoverlap;

#ifndef TK_NUM_DEVICES
#define TK_NUM_DEVICES 2
#endif

#ifndef TK_HIDDEN
#define TK_HIDDEN 7168
#endif

struct globals {
    using cfg = gemm_config;

    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;

    // sv_bf<H> = H*2 bytes; TOKENS_PER_BLOCK * token size must fit in 96KB smem
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec); // 6 for H=7168

    // Local tokens before dispatch, replicated view of every device (unicast pgl, no multicast on PCIe)
    using pre_tokens_pgl  = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>; // local tokens after dispatch
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;          // (E_local, H, I)
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;                        // (num_padded_local_tokens, I)
    using counts_gl       = gl<int, 1, 1, 1, -1>;                          // (E_global,)
    using indices_gl      = gl<int, 1, 1, -1, 2>;                          // (num_padded_local_tokens, 2)
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>; // row 0: row-block counters; row 1: pcie_barrier slots

    pre_tokens_pgl pre_tokens;
    post_tokens_gl activations; // named for grouped_gemm_sm120; this is "post_tokens"
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

/* ------------------------------------------------------------------------
 * Dispatch: pull tokens from peer GPUs into local post-dispatch layout.
 * All barrier writes are LOCAL (we pulled the data here ourselves), so a
 * device-scope atomic is sufficient and PCIe-legal.
 * ---------------------------------------------------------------------- */
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

            // local signal: +1 on this 128-token row block's counter (row 0 of barrier)
            asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                         :: "l"(&G.barrier[G.dev_idx][{token_idx / gemm_config::ROW_BLOCK}]), "r"(1) : "memory");
        }
    }
}

/* ------------------------------------------------------------------------
 * Gate: producer warp spins on the local row-block counter until all 128
 * tokens of the row block have landed. This is the entire "fusion".
 * ---------------------------------------------------------------------- */
struct dispatch_gate {
    const globals &G;
    __device__ inline void operator()(int row_idx) const {
        int bar_val;
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                     : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        while (bar_val != gemm_config::ROW_BLOCK) {
            __nanosleep(32);
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}"
                         : "=r"(bar_val) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        }
    }
};

__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void moe_dispatch_gemm_kernel(const __grid_constant__ globals G) {
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, dispatch_gate{G}, blockIdx.x, G.num_comp_sms);
    else
        dispatch(G, blockIdx.x - G.num_comp_sms);
}

/* ------------------------------------------------------------------------
 * Epilogue: reset the LOCAL row-block counters (nobody else writes them).
 * ---------------------------------------------------------------------- */
__global__ __launch_bounds__(256)
void epilogue_kernel(const __grid_constant__ globals G) {
    const int num_blocks = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = 0;
}

/* ------------------------------------------------------------------------
 * Standalone PCIe device barrier (slot + sequence based, never resets).
 * Call between iterations / layers when a producer buffer is about to be
 * overwritten while peers might still be pulling from it.
 * ---------------------------------------------------------------------- */
struct barrier_globals {
    globals::barrier_pgl barrier;
    const int dev_idx;
    const int seq;
};

__global__ __launch_bounds__(32)
void pcie_barrier_kernel(const __grid_constant__ barrier_globals G) {
    pcie_sync::pcie_barrier_all(G.barrier, G.dev_idx, G.seq);
}

/* ------------------------------------------------------------------------ */

void moe_dispatch_gemm_entry(
    kittens::py::TKParallelTensor &pre_tokens,
    at::Tensor &post_tokens,
    at::Tensor &weights,
    at::Tensor &outputs,
    at::Tensor &padded_tokens_per_expert,
    at::Tensor &pull_dispatch_indices,
    kittens::py::TKParallelTensor &barrier,
    const int num_comm_sms,
    const int num_padded_local_tokens
) {
    TORCH_CHECK(pre_tokens.data_.size(1) == globals::H, "H mismatch with compiled TK_HIDDEN");
    TORCH_CHECK(post_tokens.size(1) == globals::H, "H mismatch with compiled TK_HIDDEN");
    TORCH_CHECK(num_padded_local_tokens % gemm_config::ROW_BLOCK == 0, "padded tokens must be a multiple of 128");
    TORCH_CHECK(weights.size(2) % gemm_config::COL_BLOCK == 0, "I must be a multiple of 128");
    TORCH_CHECK(padded_tokens_per_expert.size(0) % globals::NUM_DEVICES == 0, "num experts not divisible by world size");

    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(num_local_experts <= gemm_config::MAX_LOCAL_EXPERTS, "too many local experts");
    TORCH_CHECK(weights.size(0) == num_local_experts, "weights first dim must be local expert count");

    int sm_count;
    CUDACHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise: GEMM blocks spin)");
    TORCH_CHECK(num_comm_sms < sm_count, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm_count - num_comm_sms;

    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .pull_dispatch_indices = kittens::py::tensor_to_gl<globals::indices_gl>(pull_dispatch_indices),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_local_experts = num_local_experts,
        .expert_offset = dev_idx * num_local_experts,
        .num_padded_local_tokens = num_padded_local_tokens,
        .num_comp_sms = num_comp_sms
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int dispatch_blocks = (num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    const int total_blocks = num_comp_sms + dispatch_blocks;

    CUDACHECK(cudaFuncSetAttribute(moe_dispatch_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_config::DYNAMIC_SHARED_MEMORY + 1024));
    moe_dispatch_gemm_kernel<<<total_blocks, gemm_config::NUM_THREADS, gemm_config::DYNAMIC_SHARED_MEMORY + 1024, stream>>>(G);
    CUDACHECK(cudaGetLastError());

    const int epilogue_blocks = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    epilogue_kernel<<<epilogue_blocks, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

void pcie_device_barrier_entry(kittens::py::TKParallelTensor &barrier, const int seq) {
    TORCH_CHECK(seq >= 1, "seq must be a monotonically increasing positive integer");
    barrier_globals G {
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = barrier.local_rank_,
        .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    pcie_barrier_kernel<<<1, 32, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("moe_dispatch_gemm", &moe_dispatch_gemm_entry);
    m.def("pcie_device_barrier", &pcie_device_barrier_entry);
}
