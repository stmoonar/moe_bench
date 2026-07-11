/**
 * @file moe_gemm_combine_fused_sm120.cu
 * @brief Phase 3b — MoE layer1 GEMM(W2) fused with combine, one launch.
 *
 * Expert cards (comp SMs) run grouped GEMM over local experts producing
 * expert_outputs locally. An output-epilogue counts the col-blocks finished per
 * row-block; when a 128-token row block is fully written, one thread fires a
 * slot-based release-store to EVERY source card announcing "(this expert card,
 * this row block) is ready" (monotone sequence number, PCIe-safe, single writer).
 *
 * Source cards (combine blocks, blockIdx >= num_comp_sms) run one block per
 * source token: for each of its top-k experts they wait on the (e_rank, slot/128)
 * ready slot, then P2P-pull the row and FP32 weighted-accumulate — exactly the
 * 03a combine, now gated by the ready signal instead of a global barrier.
 *
 * Route B' from experience/12 §6: all pull + local FP32 reduce + one-directional
 * slot signals. No remote atomics, no multimem.
 *
 * Verified against the 03a non-fused anchor.
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
    static constexpr int H = TK_HIDDEN;      // W2 output dim = model hidden
    static constexpr int TOP_K = 8;
    static constexpr int COMBINE_THREADS = 256;

    // GEMM (W2): activations h (padded_local_tokens, I) @ weights (E_local, I, H)
    // -> expert_outputs (padded_local_tokens, H).
    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;   // (padded_local_tokens, I)
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;  // (E_local, I, H)
    using output_gl      = gl<bf16, 1, 1, -1, H>;                 // grouped_gemm store target (local expert out)
    using expert_out_pgl = pgl<gl<bf16, 1, 1, -1, H>, NUM_DEVICES, false>; // same buffer, peer-readable
    using counts_gl      = gl<int, 1, 1, 1, -1>;                  // (E_global,)
    using combine_out_gl = gl<bf16, 1, 1, -1, H>;                 // (num_source_tokens, H) local combine result
    using combine_idx_gl = gl<int,  1, 1, -1, 2>;                 // (num_source_tokens*TOP_K, 2)
    using combine_w_gl   = gl<float,1, 1, -1, 1>;                 // (num_source_tokens*TOP_K, 1)
    // barrier row 0: local col-block completion counters per row-block (device scope);
    // rows 1+e_rank: ready slots — bar[dst_dev][{1 + e_rank, row_block}] release-store seq.
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;

    // ---- GEMM side (this card as expert card); field names match grouped_gemm_sm120 ----
    activations_gl activations;
    weights_gl weights;
    output_gl outputs;             // == expert_outputs[dev_idx], the grouped_gemm store target
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;

    // ---- combine side (this card as source card) ----
    expert_out_pgl expert_outputs; // peers' expert outputs, pulled during combine
    combine_out_gl combine_out;
    combine_idx_gl combine_indices;
    combine_w_gl combine_weights;
    barrier_pgl barrier;

    const int dev_idx;
    const int num_padded_local_tokens;   // rows of expert_outputs on this card
    const int num_source_tokens;         // this card's source tokens (combine output rows)
    const int num_comp_sms;
    const int combine_seq;               // this layer's ready-signal sequence number
};

/* ------------------------------------------------------------------------
 * Output epilogue (expert card): after a (row_idx, col_idx) tile is stored,
 * make the tile's global stores visible SYSTEM-WIDE (the source card reads
 * expert_outputs over PCIe), then one thread bumps this row-block's local
 * col-completion counter. When it reaches col_blocks, the whole 128-token row
 * block is written AND flushed, so announce "(this expert card, row_idx) ready"
 * to every source card via a single-writer release-store of `combine_seq`.
 *
 * The __threadfence_system() is executed by ALL consumer threads: st.release.sys
 * on the signaling thread alone would only order ITS OWN writes, not the other
 * 7 warps' output strips. Every thread flushes its own strip, then a barrier
 * ensures all flushes retire before the signal.
 * ---------------------------------------------------------------------- */
struct combine_signal_epilogue {
    const globals &G;
    const int col_blocks;
    __device__ inline void operator()(int row_idx, int /*col_idx*/) const {
        __threadfence_system(); // every consumer thread makes its output strip peer-visible
        kittens::group<gemm_config::CONSUMER_WARPS>::sync(1); // all flushes retired
        if (kittens::laneid() != 0 || kittens::warpid() != 0) return;
        // local device-scope counter on barrier row 0
        int done;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(done) : "l"(&G.barrier[G.dev_idx][{0, row_idx}]) : "memory");
        if (done + 1 == col_blocks) {
            // this row block is fully written & flushed; signal all source cards
            #pragma unroll 1
            for (int d = 0; d < globals::NUM_DEVICES; d++)
                pcie_sync::signal_slot(G.barrier, d, 1 + G.dev_idx, row_idx, G.combine_seq);
        }
    }
};

/* ------------------------------------------------------------------------
 * Combine (source card): one block per source token. Each thread strips the H
 * columns; for each top-k expert, wait on the (e_rank, slot/128) ready slot,
 * then P2P-pull the row and FP32 weighted-accumulate. Same math as 03a.
 * ---------------------------------------------------------------------- */
__device__ inline void combine(const globals &G, const int t) {
    if (t >= G.num_source_tokens) return;
    constexpr int H = globals::H;
    constexpr int VEC = 8;
    constexpr int HVEC = H / VEC;
    static_assert(H % VEC == 0, "H must be a multiple of 8");

    __shared__ int   s_erank[globals::TOP_K];
    __shared__ int   s_slot [globals::TOP_K];
    __shared__ float s_w    [globals::TOP_K];
    if (threadIdx.x < globals::TOP_K) {
        const int k = threadIdx.x;
        const int erank = G.combine_indices[{t * globals::TOP_K + k, 0}];
        const int slot  = G.combine_indices[{t * globals::TOP_K + k, 1}];
        s_erank[k] = erank;
        s_slot [k] = slot;
        s_w    [k] = G.combine_weights[{t * globals::TOP_K + k, 0}];
        // wait for (expert card erank, row block slot/128) to be ready
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
            const int e_rank = s_erank[k];
            const int slot   = s_slot[k];
            if (e_rank < 0 || slot < 0) continue;
            const float w = s_w[k];
            const bf16 *peer_row = &G.expert_outputs[e_rank][{slot, 0}];
            const float4 packed = reinterpret_cast<const float4 *>(peer_row)[c];
            const bf16_2 *pv = reinterpret_cast<const bf16_2 *>(&packed);
            #pragma unroll
            for (int j = 0; j < VEC / 2; j++) {
                float2 f = __bfloat1622float2(pv[j]);
                acc[2 * j]     += w * f.x;
                acc[2 * j + 1] += w * f.y;
            }
        }
        bf16_2 res[VEC / 2];
        #pragma unroll
        for (int j = 0; j < VEC / 2; j++)
            res[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        out_v[c] = *reinterpret_cast<const float4 *>(res);
    }
}

/* ------------------------------------------------------------------------
 * Fused kernel: comp SMs run W2 grouped GEMM + signal epilogue; the remaining
 * blocks run combine (one block per source token, queued on leftover SMs).
 * DEADLOCK RULE: comp blocks spin in the (no-op) gate never, but combine blocks
 * spin in wait_slot — so comp SMs must finish and signal. num_comp_sms < grid.
 * ---------------------------------------------------------------------- */
struct no_gate { __device__ inline void operator()(int) const {} };

__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void fused_kernel(const __grid_constant__ globals G) {
    const int col_blocks = static_cast<int>(G.weights.cols()) / gemm_config::COL_BLOCK;
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, no_gate{}, combine_signal_epilogue{G, col_blocks}, blockIdx.x, G.num_comp_sms);
    else
        combine(G, blockIdx.x - G.num_comp_sms);
}

/* ------------------------------------------------------------------------
 * Epilogue: reset the LOCAL col-completion counters (barrier row 0). The ready
 * slots (rows 1+) use monotone seq numbers and never reset.
 * ---------------------------------------------------------------------- */
__global__ __launch_bounds__(256)
void reset_counters_kernel(const __grid_constant__ globals G) {
    const int num_blocks = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_blocks; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{0, i}] = 0;
}

/* ------------------------------------------------------------------------ */

void moe_gemm_combine_fused_entry(
    at::Tensor &activations,               // (num_padded_local_tokens, I) local h
    at::Tensor &weights,                   // (E_local, I, H) W2
    kittens::py::TKParallelTensor &expert_outputs, // (num_padded_max, H) peer-readable
    at::Tensor &padded_tokens_per_expert,  // (E_global,)
    at::Tensor &combine_out,               // (num_source_tokens, H) local result
    at::Tensor &combine_indices,           // (num_source_tokens*TOP_K, 2)
    at::Tensor &combine_weights,           // (num_source_tokens*TOP_K, 1)
    kittens::py::TKParallelTensor &barrier,
    const int num_comm_sms,
    const int num_padded_local_tokens,
    const int num_source_tokens,
    const int combine_seq
) {
    TORCH_CHECK(weights.size(2) == globals::H, "W2 output dim must be H (compiled TK_HIDDEN)");
    TORCH_CHECK(activations.size(1) % gemm_config::RED_BLOCK == 0, "I must be a multiple of 64");
    TORCH_CHECK(globals::H % gemm_config::COL_BLOCK == 0, "H must be a multiple of 128");
    TORCH_CHECK(num_padded_local_tokens % gemm_config::ROW_BLOCK == 0, "padded tokens must be a multiple of 128");
    TORCH_CHECK(padded_tokens_per_expert.size(0) % globals::NUM_DEVICES == 0, "num experts not divisible by world size");

    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts, "weights first dim must be local expert count");

    int sm_count;
    CUDACHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms >= 1");
    TORCH_CHECK(num_comm_sms < sm_count, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm_count - num_comm_sms;

    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::output_gl>(expert_outputs.data_),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts,
        .expert_offset = dev_idx * num_local_experts,
        .expert_outputs = kittens::py::parallel_tensor_to_pgl<globals::expert_out_pgl>(expert_outputs),
        .combine_out = kittens::py::tensor_to_gl<globals::combine_out_gl>(combine_out),
        .combine_indices = kittens::py::tensor_to_gl<globals::combine_idx_gl>(combine_indices),
        .combine_weights = kittens::py::tensor_to_gl<globals::combine_w_gl>(combine_weights),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens,
        .num_comp_sms = num_comp_sms,
        .combine_seq = combine_seq
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // Combine blocks queue on the leftover SMs; comp (GEMM) blocks are the lowest
    // blockIdx and persistent, so they stay resident and always produce signals.
    const int total_blocks = num_comp_sms + num_source_tokens;
    CUDACHECK(cudaFuncSetAttribute(fused_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_config::DYNAMIC_SHARED_MEMORY + 1024));
    fused_kernel<<<total_blocks, gemm_config::NUM_THREADS, gemm_config::DYNAMIC_SHARED_MEMORY + 1024, stream>>>(G);
    CUDACHECK(cudaGetLastError());

    const int reset_blocks = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_counters_kernel<<<reset_blocks, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("moe_gemm_combine_fused", &moe_gemm_combine_fused_entry);
}



