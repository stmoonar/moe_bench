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
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
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
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
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
} // namespace comb

#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("grouped_gemm", &gg::entry);
    m.def("moe_dispatch_gemm", &disp::entry);
    m.def("pcie_device_barrier", &disp::barrier_entry);
    m.def("moe_gemm_combine_fused", &comb::entry);
}
