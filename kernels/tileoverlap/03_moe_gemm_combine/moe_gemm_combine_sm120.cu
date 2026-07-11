/**
 * @file moe_gemm_combine_sm120.cu
 * @brief Phase 3a — MoE layer1 combine, standalone correctness anchor.
 *
 *   For each local (source) token t and each of its top-k experts:
 *     out[t, :] += combine_weight[t,k] * expert_outputs[e_rank(t,k)][slot(t,k), :]
 *   with FP32 accumulation across the top-k terms.
 *
 * Route B' (experience/12 §6): the SOURCE card PULLs the finished expert-output
 * rows from the expert cards and reduces locally in FP32. No remote atomics, no
 * multimem — PCIe-safe.
 *
 * This standalone version is the NON-fused anchor: it assumes every expert card
 * has already finished its W2 GEMM (the caller does one device_barrier first),
 * so it just P2P-reads peer rows directly through the pgl unicast view. The
 * fused version (03b) will pull per-col-block under a slot ready-signal.
 *
 * Layout: one block per source token, blockDim.x threads stripe the H columns.
 * Peer reads are vectorized (float4 = 8 bf16) for bandwidth.
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
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    static constexpr int THREADS = 256;

    // Expert outputs after W2, replicated pgl view of every device (unicast, PCIe).
    // Each device holds its own (num_padded_local_tokens_that_dev, H) slab; peers
    // read it through the pgl unicast pointer.
    using expert_out_pgl = pgl<gl<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;
    using output_gl      = gl<bf16, 1, 1, -1, H>;              // (num_source_tokens, H) local
    using combine_idx_gl = gl<int,  1, 1, -1, 2>;              // (num_source_tokens*TOP_K, 2): (e_rank, remote_slot)
    using combine_w_gl   = gl<float,1, 1, -1, 1>;              // (num_source_tokens*TOP_K, 1)
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;

    expert_out_pgl expert_outputs;
    output_gl outputs;
    combine_idx_gl combine_indices;
    combine_w_gl combine_weights;
    barrier_pgl barrier;

    const int dev_idx;
    const int num_source_tokens;
};

/* ------------------------------------------------------------------------
 * Combine: one block per source token. Each thread owns a strip of H columns,
 * accumulates the top-k weighted expert rows in FP32, writes bf16.
 * Vectorized by float4 (8 bf16 per thread per step).
 * ---------------------------------------------------------------------- */
__global__ __launch_bounds__(globals::THREADS, 1)
void combine_kernel(const __grid_constant__ globals G) {
    const int t = blockIdx.x;               // source token
    if (t >= G.num_source_tokens) return;

    constexpr int H = globals::H;
    constexpr int VEC = 8;                  // bf16 per float4
    constexpr int HVEC = H / VEC;           // 896 for H=7168
    static_assert(H % VEC == 0, "H must be a multiple of 8");

    // Cache this token's top-k (e_rank, slot, weight) in registers/smem.
    __shared__ int   s_erank[globals::TOP_K];
    __shared__ int   s_slot [globals::TOP_K];
    __shared__ float s_w    [globals::TOP_K];
    if (threadIdx.x < globals::TOP_K) {
        const int k = threadIdx.x;
        s_erank[k] = G.combine_indices[{t * globals::TOP_K + k, 0}];
        s_slot [k] = G.combine_indices[{t * globals::TOP_K + k, 1}];
        s_w    [k] = G.combine_weights[{t * globals::TOP_K + k, 0}];
    }
    __syncthreads();

    bf16 *out_row = &G.outputs[{t, 0}];
    float4 *out_v = reinterpret_cast<float4 *>(out_row);

    for (int c = threadIdx.x; c < HVEC; c += blockDim.x) {
        float acc[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) acc[i] = 0.0f;

        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) {
            const int e_rank = s_erank[k];
            const int slot   = s_slot[k];
            if (e_rank < 0 || slot < 0) continue;   // padding / unused top-k slot
            const float w = s_w[k];
            // Peer read (P2P): expert card e_rank's row `slot`, columns [c*VEC, +VEC).
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

        // FP32 -> bf16, pack back into one float4 and store.
        bf16_2 res[VEC / 2];
        #pragma unroll
        for (int j = 0; j < VEC / 2; j++)
            res[j] = __floats2bfloat162_rn(acc[2 * j], acc[2 * j + 1]);
        out_v[c] = *reinterpret_cast<const float4 *>(res);
    }
}

/* ------------------------------------------------------------------------ */

void moe_combine_entry(
    kittens::py::TKParallelTensor &expert_outputs,
    at::Tensor &outputs,
    at::Tensor &combine_indices,
    at::Tensor &combine_weights,
    kittens::py::TKParallelTensor &barrier,
    const int num_source_tokens
) {
    TORCH_CHECK(expert_outputs.data_.size(1) == globals::H, "H mismatch with compiled TK_HIDDEN");
    TORCH_CHECK(outputs.size(1) == globals::H, "output H mismatch");
    TORCH_CHECK(outputs.size(0) == num_source_tokens, "output rows must equal num_source_tokens");
    TORCH_CHECK(combine_indices.size(0) == num_source_tokens * globals::TOP_K, "combine_indices rows");
    TORCH_CHECK(combine_weights.size(0) == num_source_tokens * globals::TOP_K, "combine_weights rows");

    const int dev_idx = barrier.local_rank_;

    globals G {
        .expert_outputs = kittens::py::parallel_tensor_to_pgl<globals::expert_out_pgl>(expert_outputs),
        .outputs = kittens::py::tensor_to_gl<globals::output_gl>(outputs),
        .combine_indices = kittens::py::tensor_to_gl<globals::combine_idx_gl>(combine_indices),
        .combine_weights = kittens::py::tensor_to_gl<globals::combine_w_gl>(combine_weights),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_source_tokens = num_source_tokens
    };

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    combine_kernel<<<num_source_tokens, globals::THREADS, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("moe_combine", &moe_combine_entry);
}
