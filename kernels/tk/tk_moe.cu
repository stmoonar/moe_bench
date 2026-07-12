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
// GLU compute-only reference (docs/30): the dispenser GEMM with the fused
// SwiGLU store and NO gate — the apples-to-apples "L0 GEMM alone" baseline
// for time_tp_stages when v2+GLU is active (outputs = act (P, inter),
// weights column-interleaved).
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel_glu(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
                int *__restrict__ task_next, const int num_tasks) {
    grouped_gemm_sm120_dispenser(G, no_gate{}, noop_epilogue{},
                                 glu_store_policy<globals::outputs_gl>{G.outputs},
                                 blk_expert, task_next, num_tasks);
}
// 列外层 dispenser GEMM 的纯算参考(docs/33): 隔离 "列外层换序对 GEMM 本身
// 的代价"(L2 复用模式变化)与 "推送协议的代价" —— L1 v2 归因的对照组。
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel_cm(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
               int *__restrict__ task_next, const int num_tasks) {
    grouped_gemm_sm120_dispenser<true>(G, no_gate{}, noop_epilogue{},
                                       plain_store_policy<globals::outputs_gl>{G.outputs},
                                       blk_expert, task_next, num_tasks);
}
void entry_cm(const at::Tensor &inputs, const at::Tensor &weights, at::Tensor &outputs,
              const at::Tensor &padded_tokens_per_expert, const at::Tensor &blk_expert,
              at::Tensor &task_next, const int expert_offset) {
    TORCH_CHECK(inputs.size(0) % gemm_config::ROW_BLOCK == 0, "tokens % 128");
    TORCH_CHECK(inputs.size(1) % gemm_config::RED_BLOCK == 0, "K % 64");
    TORCH_CHECK(weights.size(2) % gemm_config::COL_BLOCK == 0, "N % 128");
    TORCH_CHECK(task_next.numel() == 1, "task_next must be a single int counter");
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(inputs),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .expert_offset = expert_offset
    };
    const int nblk = static_cast<int>(inputs.size(0)) / gemm_config::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert must have one entry per row block");
    const int num_tasks = nblk * (static_cast<int>(weights.size(2)) / gemm_config::COL_BLOCK);
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, inputs.device().index()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel_cm, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_cm<<<sm, gemm_config::NUM_THREADS, smem, stream>>>(
        G, blk_expert.data_ptr<int>(), task_next.data_ptr<int>(), num_tasks);
    CUDACHECK(cudaGetLastError());
}
void entry_glu(const at::Tensor &inputs, const at::Tensor &weights, at::Tensor &outputs,
               const at::Tensor &padded_tokens_per_expert, const at::Tensor &blk_expert,
               at::Tensor &task_next, const int expert_offset) {
    TORCH_CHECK(inputs.size(0) % gemm_config::ROW_BLOCK == 0, "tokens % 128");
    TORCH_CHECK(inputs.size(1) % gemm_config::RED_BLOCK == 0, "K % 64");
    TORCH_CHECK(weights.size(2) % gemm_config::COL_BLOCK == 0, "N % 128");
    TORCH_CHECK(outputs.size(1) == weights.size(2) / 2, "GLU outputs width must be N/2");
    TORCH_CHECK(task_next.numel() == 1, "task_next must be a single int counter");
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(inputs),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .expert_offset = expert_offset
    };
    const int nblk = static_cast<int>(inputs.size(0)) / gemm_config::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert must have one entry per row block");
    const int num_tasks = nblk * (static_cast<int>(weights.size(2)) / gemm_config::COL_BLOCK);
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, inputs.device().index()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel_glu, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_glu<<<sm, gemm_config::NUM_THREADS, smem, stream>>>(
        G, blk_expert.data_ptr<int>(), task_next.data_ptr<int>(), num_tasks);
    CUDACHECK(cudaGetLastError());
}
} // namespace gg

/* ===================================================================== *
 * 1b. FP8 grouped GEMM 单卡入口(docs/37 P1): A 1×128 group 量化 +
 *     W 128×128 block 量化, dispenser 结构, 输出 bf16。
 *     tools/verify_fp8_gemm.py 用它对拍 fp32 反量化参考并计时。
 * ===================================================================== */
namespace gg8 {
struct globals {
    using cfg = gemm_config_fp8;
    using activations_gl = gl<fp8e4m3, 1, 1, -1, -1, cfg::A_tile>;
    using weights_gl     = gl<fp8e4m3, 1, -1, -1, -1, cfg::B_tile>;
    using a_scales_gl    = gl<float, 1, 1, -1, -1>;
    using w_scales_gl    = gl<float, 1, -1, -1, -1>;
    using outputs_gl     = gl<bf16, 1, 1, -1, -1>;
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    activations_gl activations;
    weights_gl weights;
    a_scales_gl a_scales;
    w_scales_gl w_scales;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
};
struct no_gate { __device__ inline void operator()(int) const {} };
__global__ __launch_bounds__(gemm_config_fp8::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
            int *__restrict__ task_next, const int num_tasks) {
    grouped_gemm_sm120_fp8_dispenser(G, no_gate{}, noop_epilogue{},
                                     plain_store_policy<globals::outputs_gl>{G.outputs},
                                     blk_expert, task_next, num_tasks);
}
// 裸 mma 吞吐探针(docs/38): 跳过重标定, 结果错, 只测 fp8+f32acc 硬上限
__global__ __launch_bounds__(gemm_config_fp8::NUM_THREADS, 1)
void kernel_raw(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
                int *__restrict__ task_next, const int num_tasks) {
    grouped_gemm_sm120_fp8_dispenser<false, false>(
        G, no_gate{}, noop_epilogue{},
        plain_store_policy<globals::outputs_gl>{G.outputs},
        blk_expert, task_next, num_tasks);
}
void entry(const at::Tensor &inputs, const at::Tensor &a_scales,
           const at::Tensor &weights, const at::Tensor &w_scales, at::Tensor &outputs,
           const at::Tensor &padded_tokens_per_expert, const at::Tensor &blk_expert,
           at::Tensor &task_next, const int expert_offset, const bool raw) {
    using cfg = gemm_config_fp8;
    // 布局(docs/38): weights = B^T (E, N, K)(w1 原始布局, 免转置),
    // w_scales (E, N/128, K/128); mma_ABt + row-layout 加载。
    TORCH_CHECK(inputs.size(0) % cfg::ROW_BLOCK == 0, "tokens % ROW_BLOCK");
    TORCH_CHECK(inputs.size(1) % cfg::SCALE_K == 0, "K % 128 (scale blocks)");
    TORCH_CHECK(weights.size(2) == inputs.size(1), "weights must be (E, N, K), K match");
    TORCH_CHECK(weights.size(1) % cfg::COL_BLOCK == 0, "N % 128");
    TORCH_CHECK(a_scales.size(0) == inputs.size(0) &&
                a_scales.size(1) == inputs.size(1) / cfg::SCALE_K,
                "a_scales must be (rows, K/128)");
    TORCH_CHECK(w_scales.size(1) == weights.size(1) / cfg::COL_BLOCK &&
                w_scales.size(2) == weights.size(2) / cfg::SCALE_K,
                "w_scales must be (E, N/128, K/128)");
    TORCH_CHECK(task_next.numel() == 1, "task_next must be a single int counter");
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(const_cast<at::Tensor&>(inputs)),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(const_cast<at::Tensor&>(weights)),
        .a_scales = kittens::py::tensor_to_gl<globals::a_scales_gl>(const_cast<at::Tensor&>(a_scales)),
        .w_scales = kittens::py::tensor_to_gl<globals::w_scales_gl>(const_cast<at::Tensor&>(w_scales)),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(const_cast<at::Tensor&>(padded_tokens_per_expert)),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .expert_offset = expert_offset
    };
    const int nblk = static_cast<int>(inputs.size(0)) / cfg::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert must have one entry per row block");
    const int num_tasks = nblk * (static_cast<int>(weights.size(1)) / cfg::COL_BLOCK);
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, inputs.device().index()));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = cfg::DYNAMIC_SHARED_MEMORY + 1024;
    if (raw) {
        CUDACHECK(cudaFuncSetAttribute(kernel_raw, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel_raw<<<sm, cfg::NUM_THREADS, smem, stream>>>(
            G, blk_expert.data_ptr<int>(), task_next.data_ptr<int>(), num_tasks);
    } else {
        CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel<<<sm, cfg::NUM_THREADS, smem, stream>>>(
            G, blk_expert.data_ptr<int>(), task_next.data_ptr<int>(), num_tasks);
    }
    CUDACHECK(cudaGetLastError());
}
// 1×128 row-group 量化(docs/41): bf16 (rows, groups*128) -> fp8 + scales
// (rows, groups)。torch 的五连发小 kernel 链要 ~80us(docs/40 tok_copy 112),
// 单 kernel 版 ~10-15us。token 量化(groups=32)与 act 量化(P3, groups=6)
// 共用。block = 一行, 8 warps 按 group 跨步, warp 内 shfl 归约 amax。
__global__ __launch_bounds__(256)
void rowgroup_quant_kernel(const __nv_bfloat16 *__restrict__ in,
                           __nv_fp8_e4m3 *__restrict__ out,
                           float *__restrict__ scales,
                           const int rows, const int groups) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    for (int g = warp; g < groups; g += 8) {
        const __nv_bfloat16 *p = in + (size_t)row * groups * 128 + g * 128;
        float m = 0.f;
        #pragma unroll
        for (int i = lane; i < 128; i += 32)
            m = fmaxf(m, fabsf(__bfloat162float(p[i])));
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, off));
        m = fmaxf(m, 1e-8f);
        if (lane == 0) scales[(size_t)row * groups + g] = m / 448.f;
        const float inv = 448.f / m;
        __nv_fp8_e4m3 *q = out + (size_t)row * groups * 128 + g * 128;
        #pragma unroll
        for (int i = lane; i < 128; i += 32)
            q[i] = __nv_fp8_e4m3(__bfloat162float(p[i]) * inv);
    }
}
void rowgroup_quant_entry(const at::Tensor &in, at::Tensor &out, at::Tensor &scales) {
    TORCH_CHECK(in.dtype() == at::ScalarType::BFloat16 && in.is_contiguous(), "in: contiguous bf16");
    TORCH_CHECK(out.dtype() == at::ScalarType::Float8_e4m3fn, "out: fp8e4m3");
    TORCH_CHECK(in.size(1) % 128 == 0, "cols % 128");
    const int rows = static_cast<int>(in.size(0));
    const int groups = static_cast<int>(in.size(1)) / 128;
    TORCH_CHECK(scales.size(0) == rows && scales.size(1) == groups, "scales (rows, cols/128)");
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    rowgroup_quant_kernel<<<rows, 256, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16 *>(in.data_ptr()),
        reinterpret_cast<__nv_fp8_e4m3 *>(out.data_ptr()),
        scales.data_ptr<float>(), rows, groups);
    CUDACHECK(cudaGetLastError());
}
} // namespace gg8

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
 * 2b. Dedup dispatch ⊕ gate GEMM (layer0, T7-v0; docs/15). Same result as
 *     disp:: pull, but the cross-card TMA happens ONCE per unique source token
 *     (measured ~7.2× fewer at NE=256) instead of once per gathered slot:
 *       kernel 1 (pull_unique): pull the ~S unique (src_dev,src_tok) rows into
 *         a LOCAL staging buffer (the only cross-card traffic);
 *       [same-stream kernel boundary = the publish barrier]
 *       kernel 2 (kernel): local staging->gathered scatter per slot + the SAME
 *         per-row-block red.add counter, fused with the gate GEMM exactly as
 *         disp::. gathered ends byte-identical to the pull path.
 * ===================================================================== */
namespace ddisp {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec);
    using pre_tokens_pgl  = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using staging_gl      = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using s2s_gl          = gl<int, 1, 1, 1, -1>;   // slot_to_staging (P,)
    using need_gl         = gl<int, 1, 1, 1, -1>;   // staging_needed (S_max,)
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_pgl pre_tokens;
    staging_gl staging;
    post_tokens_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    s2s_gl slot_to_staging;
    need_gl staging_needed;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;
    const int expert_offset;
    const int num_padded_local_tokens;
    const int num_tokens;      // per-card source tokens (dense staging stride)
    const int s_max;           // world * num_tokens
    const int num_comp_sms;
};
// kernel 1: pull each NEEDED unique dense row (src_dev*num_tokens + src_tok)
// cross-card into local staging. One thread per row-block lane, like disp.
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void pull_unique_kernel(const __grid_constant__ globals G) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int d = blockIdx.x * globals::TOKENS_PER_BLOCK + lane_id;
        if (d < G.s_max && G.staging_needed[{d}] != 0) {
            const int src_dev = d / G.num_tokens;
            const int src_tok = d % G.num_tokens;
            init_semaphore(token_arrived[lane_id], 0, 1);
            tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
            tma::load_async(token[lane_id], G.pre_tokens[src_dev], {src_tok, 0}, token_arrived[lane_id]);
            wait(token_arrived[lane_id], 0);
            tma::store_async(G.staging, token[lane_id], {d, 0});
            tma::store_async_wait();  // staging row complete before kernel exit
        }
    }
}
// kernel 2 comm block: local staging->gathered copy + row-block counter bump.
__device__ inline void scatter(const globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int token_idx = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (token_idx < G.num_padded_local_tokens) {
            const int s = G.slot_to_staging[{token_idx}];
            if (s >= 0) {
                init_semaphore(token_arrived[lane_id], 0, 1);
                tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
                tma::load_async(token[lane_id], G.staging, {s, 0}, token_arrived[lane_id]);
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
        scatter(G, blockIdx.x - G.num_comp_sms);
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = 0;
}
void entry(kittens::py::TKParallelTensor &pre_tokens, at::Tensor &staging,
           at::Tensor &post_tokens, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &slot_to_staging,
           at::Tensor &staging_needed, kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens,
           const int num_tokens) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "weights first dim must equal local expert count (NUM_GPUS mismatch?)");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    const int s_max = static_cast<int>(staging_needed.size(0));
    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .staging = kittens::py::tensor_to_gl<globals::staging_gl>(staging),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .slot_to_staging = kittens::py::tensor_to_gl<globals::s2s_gl>(slot_to_staging),
        .staging_needed = kittens::py::tensor_to_gl<globals::need_gl>(staging_needed),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = dev_idx * num_local_experts,
        .num_padded_local_tokens = num_padded_local_tokens, .num_tokens = num_tokens,
        .s_max = s_max, .num_comp_sms = num_comp_sms
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    const int pull_blocks = (s_max + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    CUDACHECK(cudaFuncSetAttribute(pull_unique_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    pull_unique_kernel<<<pull_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int scatter_blocks = (num_padded_local_tokens + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_comp_sms + scatter_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace ddisp

/* ===================================================================== *
 * 2c. TP AllGather-dispatch ⊕ grouped GEMM (layer0, TP mode).
 *
 * TP shards the INTERMEDIATE dim, not the experts: every card holds all E
 * experts (thin), and needs ALL world*T tokens. The cross-card communication
 * is therefore a plain dense AllGather of the token shards — and because all
 * top-k experts of a token live on this card, every unique token is used
 * exactly TOP_K times locally. So the dedup insight (T7, docs/15) is not an
 * option here but the NATURAL form: pull each unique (src_dev, src_tok) row
 * ONCE cross-card, then scatter it to its TOP_K expert-sorted gathered slots
 * locally, bumping each slot's row-block counter (same local red.release.gpu
 * + spin==ROW_BLOCK gate as disp::, PCIe-safe).
 *
 * Unlike ddisp:: (two launches: pull-all THEN scatter⊕GEMM), this is ONE
 * launch: comm blocks pull+scatter per token while the GEMM chews row blocks
 * as their counters fill — true AG⊕GEMM overlap. Pull order is the host/GPU-
 * built `pull_order` permutation: unique tokens sorted by their MIN gathered
 * slot, i.e. expert-major. First-round finding (docs/20): a ring-by-source
 * order drains one source at a time, and since every expert's row blocks mix
 * all sources, NO row block completes until the last ring stage — the GEMM
 * stalls behind the whole AllGather. Min-slot order instead pulls each
 * expert's tokens (all sources interleaved -> all PCIe links busy at once)
 * before the next expert's, so row blocks become ready progressively and the
 * GEMM streams right behind the pull front.
 *
 * Padding: tp_slots only covers REAL assignments, so row-block counters are
 * PRE-SEEDED with their padding slack (dpush's proven trick — but purely
 * local here: the Python side seeds once, reset_kernel restores slack, not
 * zero, after each iteration). counter: slack + real bumps == ROW_BLOCK.
 * ===================================================================== */
namespace tpdisp {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec);
    using pre_tokens_pgl  = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using slots_gl        = gl<int, 1, 1, -1, TOP_K>;   // (world*T, TOP_K) gathered slot per assignment
    using slack_gl        = gl<int, 1, 1, 1, -1>;       // (nblk,) per-row-block padding slack
    using order_gl        = gl<int, 1, 1, 1, -1>;       // (world*T,) pull order (min-slot sorted)
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_pgl pre_tokens;
    post_tokens_gl activations;    // gathered (expert-sorted, ROW_BLOCK-padded), LOCAL
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    slots_gl tp_slots;
    slack_gl slack;
    order_gl pull_order;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;   // == E (TP: every expert on every card)
    const int expert_offset;       // == 0
    const int num_padded_local_tokens;
    const int num_tokens;          // per-card source tokens T
    const int s_max;               // world * T
    const int num_comp_sms;
};
// comm block: one thread per unique source token — pull the row once (ring
// order, own shard first), TMA-scatter it to its TOP_K gathered slots, then
// bump each slot's row-block counter (local gpu-scope release add, disp:: style).
__device__ inline void dispatch(const globals &G, const int sm_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK) {
        const int i = sm_idx * globals::TOKENS_PER_BLOCK + lane_id;
        if (i < G.s_max) {
            // expert-major pull order (docs/20): consecutive i are one expert's
            // tokens across ALL sources -> links concurrent, row blocks ready
            // progressively behind the pull front.
            const int d = G.pull_order[{i}];
            const int src_dev = d / G.num_tokens;
            const int src_tok = d % G.num_tokens;
            init_semaphore(token_arrived[lane_id], 0, 1);
            tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
            tma::load_async(token[lane_id], G.pre_tokens[src_dev], {src_tok, 0}, token_arrived[lane_id]);
            wait(token_arrived[lane_id], 0);
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{d, k}];
                if (slot >= 0)
                    tma::store_async(G.activations, token[lane_id], {slot, 0});
            }
            tma::store_async_wait();  // all TOP_K copies committed before signaling
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{d, k}];
                if (slot >= 0)
                    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                                 :: "l"(&G.barrier[G.dev_idx][{slot / gemm_config::ROW_BLOCK}]), "r"(1) : "memory");
            }
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
// reset-to-SLACK (not zero): counters must start each iteration pre-seeded with
// the row block's padding slack so real bumps bring them exactly to ROW_BLOCK.
// Python seeds once at setup; this keeps them seeded between iterations.
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = G.slack[{i}];
}
void entry(kittens::py::TKParallelTensor &pre_tokens, at::Tensor &post_tokens,
           at::Tensor &weights, at::Tensor &outputs, at::Tensor &padded_tokens_per_expert,
           at::Tensor &tp_slots, at::Tensor &slack, at::Tensor &pull_order,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens, const int num_tokens) {
    const int dev_idx = barrier.local_rank_;
    // TP geometry: padded_tokens_per_expert covers ALL experts, all of them local.
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "TP weights first dim must equal the FULL expert count");
    TORCH_CHECK(num_local_experts <= gemm_config::MAX_LOCAL_EXPERTS, "too many experts");
    TORCH_CHECK(tp_slots.size(1) == globals::TOP_K, "tp_slots second dim must be TOP_K");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    const int s_max = static_cast<int>(tp_slots.size(0));
    TORCH_CHECK(s_max == globals::NUM_DEVICES * num_tokens, "tp_slots rows must be world*T");
    TORCH_CHECK(pull_order.size(0) == s_max, "pull_order must cover world*T tokens");
    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .tp_slots = kittens::py::tensor_to_gl<globals::slots_gl>(tp_slots),
        .slack = kittens::py::tensor_to_gl<globals::slack_gl>(slack),
        .pull_order = kittens::py::tensor_to_gl<globals::order_gl>(pull_order),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = 0,
        .num_padded_local_tokens = num_padded_local_tokens, .num_tokens = num_tokens,
        .s_max = s_max, .num_comp_sms = num_comp_sms
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    const int dispatch_blocks = (s_max + globals::TOKENS_PER_BLOCK - 1) / globals::TOKENS_PER_BLOCK;
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<num_comp_sms + dispatch_blocks, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tpdisp

/* ===================================================================== *
 * 2c-v2. TP pull dispatch ⊕ dispenser GEMM ⊕ fused SwiGLU (docs/30).
 *
 * Two structural fixes over tpdisp (v1), both aimed at the comm-slows-compute
 * account (L0 exposure ~255us at comm24 == the pure SM-yield cost, data-wait
 * ~0; and a separate 109us torch silu pass):
 *
 *   1. COMM SMs JOIN THE GEMM once the AllGather drains. v1's dispatch blocks
 *      were short-lived (one block per 12 tokens, queued on the comm SMs) and
 *      the GEMM walk was statically partitioned, so after the pull front
 *      passed, 24 SMs idled for the rest of L0. v2 launches num_comm_sms
 *      PERSISTENT comm blocks that pull in waves over pull_order, then join
 *      the dispenser-fed GEMM (grouped_gemm_sm120_dispenser) alongside the
 *      comp blocks — the same all-hands drain trick layer1 already uses in
 *      the opposite direction (gemm_push_kernel_tp, docs/20).
 *   2. SwiGLU IS THE GEMM EPILOGUE (glu_store_policy). Weights are column-
 *      interleaved at setup ([gate64 | up64] per 128-col block), so each
 *      output tile holds both halves of the same intermediate columns; the
 *      consumer computes silu(gate)*up on the fp32 accumulators and stores
 *      the 64-wide act tile directly — the separate silu kernel (109us +
 *      75MB of HBM round-trip) disappears, and accuracy IMPROVES (silu on
 *      fp32 accs instead of rounded bf16). TK_L0_GLU=0 falls back to the
 *      plain store (gateup_out) + torch silu, independently of the dispenser.
 *
 * Join uses the dedicated named barrier (bar.sync 2) — NOT __syncthreads —
 * for the docs/24 reason (barrier 0 is cycled with a 256-thread count inside
 * the GEMM consumer group).
 * ===================================================================== */
namespace tpdisp2 {
using globals = tpdisp::globals;   // same tables; outputs = act (GLU) or gateup_out (plain)

// Persistent comm block: pull waves over pull_order (stride = all comm blocks),
// TMA-scatter each token row to its TOP_K slots, bump row-block counters.
// Same protocol as tpdisp::dispatch, but one resident block loops many waves;
// the per-lane mbarrier phase flips once per completed expect+arrive cycle.
__device__ inline void dispatch_persistent(const globals &G, const int cb_idx, const int num_cb) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK)
        init_semaphore(token_arrived[lane_id], 0, 1);
    __syncthreads();
    int phase = 0;
    for (int base = cb_idx * globals::TOKENS_PER_BLOCK; base < G.s_max;
         base += num_cb * globals::TOKENS_PER_BLOCK) {
        const int i = base + lane_id;
        if (lane_id < globals::TOKENS_PER_BLOCK && i < G.s_max) {
            const int d = G.pull_order[{i}];
            const int src_dev = d / G.num_tokens;
            const int src_tok = d % G.num_tokens;
            tma::expect_bytes(token_arrived[lane_id], sizeof(globals::token_vec));
            tma::load_async(token[lane_id], G.pre_tokens[src_dev], {src_tok, 0}, token_arrived[lane_id]);
            wait(token_arrived[lane_id], phase);
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{d, k}];
                if (slot >= 0)
                    tma::store_async(G.activations, token[lane_id], {slot, 0});
            }
            tma::store_async_wait();
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{d, k}];
                if (slot >= 0)
                    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                                 :: "l"(&G.barrier[G.dev_idx][{slot / gemm_config::ROW_BLOCK}]), "r"(1) : "memory");
            }
            phase ^= 1;   // this lane's mbarrier completed one cycle
        }
        __syncthreads(); // wave's smem fully consumed before reuse
    }
}

template <bool GLU>
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
            int *__restrict__ task_next, const int num_tasks) {
    if (blockIdx.x >= G.num_comp_sms) {
        dispatch_persistent(G, blockIdx.x - G.num_comp_sms, gridDim.x - G.num_comp_sms);
        // join the GEMM pool on the DEDICATED named barrier (docs/24: barrier 0
        // is cycled with a 256-count inside the GEMM consumer group).
        asm volatile("bar.sync 2, %0;" :: "n"(gemm_config::NUM_THREADS));
    }
    if constexpr (GLU) {
        grouped_gemm_sm120_dispenser(G, tpdisp::dispatch_gate{G}, noop_epilogue{},
                                     glu_store_policy<globals::outputs_gl>{G.outputs},
                                     blk_expert, task_next, num_tasks);
    } else {
        grouped_gemm_sm120_dispenser(G, tpdisp::dispatch_gate{G}, noop_epilogue{},
                                     plain_store_policy<globals::outputs_gl>{G.outputs},
                                     blk_expert, task_next, num_tasks);
    }
}

void entry(kittens::py::TKParallelTensor &pre_tokens, at::Tensor &post_tokens,
           at::Tensor &weights, at::Tensor &outputs, at::Tensor &padded_tokens_per_expert,
           at::Tensor &tp_slots, at::Tensor &slack, at::Tensor &pull_order,
           at::Tensor &blk_expert, at::Tensor &gemm_next,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens, const int num_tokens,
           const bool glu) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "TP weights first dim must equal the FULL expert count");
    TORCH_CHECK(num_local_experts <= gemm_config::MAX_LOCAL_EXPERTS, "too many experts");
    TORCH_CHECK(tp_slots.size(1) == globals::TOP_K, "tp_slots second dim must be TOP_K");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms must be >= 1 (deadlock otherwise)");
    TORCH_CHECK(gemm_next.numel() == 1, "gemm_next must be a single int counter");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    const int s_max = static_cast<int>(tp_slots.size(0));
    TORCH_CHECK(s_max == globals::NUM_DEVICES * num_tokens, "tp_slots rows must be world*T");
    TORCH_CHECK(pull_order.size(0) == s_max, "pull_order must cover world*T tokens");
    const int nblk = num_padded_local_tokens / gemm_config::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert must have one entry per row block");
    const int col_blocks = static_cast<int>(weights.size(2)) / gemm_config::COL_BLOCK;
    const int num_tasks = nblk * col_blocks;
    // GLU output is (P, inter) in 64-wide tiles; plain is (P, 2*inter).
    TORCH_CHECK(outputs.size(1) == (glu ? weights.size(2) / 2 : weights.size(2)),
                "outputs width mismatch for the chosen store policy");
    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .tp_slots = kittens::py::tensor_to_gl<globals::slots_gl>(tp_slots),
        .slack = kittens::py::tensor_to_gl<globals::slack_gl>(slack),
        .pull_order = kittens::py::tensor_to_gl<globals::order_gl>(pull_order),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = 0,
        .num_padded_local_tokens = num_padded_local_tokens, .num_tokens = num_tokens,
        .s_max = s_max, .num_comp_sms = num_comp_sms
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    if (glu) {
        CUDACHECK(cudaFuncSetAttribute(kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel<true><<<sm, gemm_config::NUM_THREADS, smem, stream>>>(
            G, blk_expert.data_ptr<int>(), gemm_next.data_ptr<int>(), num_tasks);
    } else {
        CUDACHECK(cudaFuncSetAttribute(kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        kernel<false><<<sm, gemm_config::NUM_THREADS, smem, stream>>>(
            G, blk_expert.data_ptr<int>(), gemm_next.data_ptr<int>(), num_tasks);
    }
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    tpdisp::reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tpdisp2

/* ===================================================================== *
 * 2c-fp8. FP8 TP layer0(docs/39 P2): fp8 AG-dispatch ⊕ fp8 dispenser
 *   GEMM ⊕ GLU epilogue。与 tpdisp2 同构, 三处不同:
 *   - pull 的 token 行是 fp8(4KB)+ 1×128 group scales(128B), 两个 TMA
 *     一个 mbarrier(expect = 4224B), 各自 scatter 到 gathered /
 *     gathered_scales 的 TOP_K 个 slot(AG 字节减半, docs/37 §1);
 *   - GEMM 走 grouped_gemm_sm120_fp8_dispenser(B^T (E,N,K) 交织后重量化,
 *     w_scales (E,N/128,K/128); a_scales = gathered_scales);
 *   - 输出经 glu_store_policy 直存 bf16 act(fp32 epilogue, L1 保持 bf16)。
 * ===================================================================== */
namespace tpdisp8 {
struct globals {
    using cfg = gemm_config_fp8;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    static constexpr int NSC = H / 128;                 // scale groups per token
    using token_vec = sv_fp8e4m3<H>;                    // 4KB
    using scale_vec = sv_fl<NSC>;                       // 128B
    static constexpr int TOKENS_PER_BLOCK = 20;         // 20×(4096+128+pad) ≤ 96KB
    using pre_tokens_pgl  = pgl<gl<fp8e4m3, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using pre_scales_pgl  = pgl<gl<float, 1, 1, -1, NSC, scale_vec>, NUM_DEVICES, false>;
    using gathered_gl     = gl<fp8e4m3, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using gscales_gl      = gl<float, 1, 1, -1, NSC, scale_vec>;
    using weights_gl      = gl<fp8e4m3, 1, -1, -1, -1, cfg::B_tile>;   // (E, N, K)
    using w_scales_gl     = gl<float, 1, -1, -1, -1>;                  // (E, N/128, K/128)
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;                    // act (P, inter)
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using slots_gl        = gl<int, 1, 1, -1, TOP_K>;
    using slack_gl        = gl<int, 1, 1, 1, -1>;
    using order_gl        = gl<int, 1, 1, 1, -1>;
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_pgl pre_tokens;
    pre_scales_pgl pre_scales;
    gathered_gl activations;      // gathered fp8 (LOCAL)
    gscales_gl a_scales;          // gathered scales (LOCAL, GEMM 直读)
    weights_gl weights;
    w_scales_gl w_scales;
    outputs_gl outputs;           // act (bf16)
    counts_gl padded_tokens_per_expert;
    slots_gl tp_slots;
    slack_gl slack;
    order_gl pull_order;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;
    const int expert_offset;
    const int num_padded_local_tokens;
    const int num_tokens;
    const int s_max;
    const int num_comp_sms;
};
struct dispatch_gate {
    const globals &G;
    __device__ inline void operator()(int row_idx) const {
        int v;
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        while (v != gemm_config_fp8::ROW_BLOCK) {
            __nanosleep(32);
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        }
    }
};
// 常驻 comm 块: 分波拉取 fp8 行 + scale 行(一个 mbarrier 两个 TMA),
// 各自 scatter 到 TOP_K 个 slot; 协议与 tpdisp2 相同。
__device__ inline void dispatch_persistent(const globals &G, const int cb_idx, const int num_cb) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    typename globals::scale_vec (&scales)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::scale_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id < globals::TOKENS_PER_BLOCK)
        init_semaphore(token_arrived[lane_id], 0, 1);
    __syncthreads();
    int phase = 0;
    for (int base = cb_idx * globals::TOKENS_PER_BLOCK; base < G.s_max;
         base += num_cb * globals::TOKENS_PER_BLOCK) {
        const int i = base + lane_id;
        if (lane_id < globals::TOKENS_PER_BLOCK && i < G.s_max) {
            const int d = G.pull_order[{i}];
            const int src_dev = d / G.num_tokens;
            const int src_tok = d % G.num_tokens;
            tma::expect_bytes(token_arrived[lane_id],
                              sizeof(globals::token_vec) + sizeof(globals::scale_vec));
            tma::load_async(token[lane_id], G.pre_tokens[src_dev], {src_tok, 0}, token_arrived[lane_id]);
            tma::load_async(scales[lane_id], G.pre_scales[src_dev], {src_tok, 0}, token_arrived[lane_id]);
            wait(token_arrived[lane_id], phase);
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{d, k}];
                if (slot >= 0) {
                    tma::store_async(G.activations, token[lane_id], {slot, 0});
                    tma::store_async(G.a_scales, scales[lane_id], {slot, 0});
                }
            }
            tma::store_async_wait();
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{d, k}];
                if (slot >= 0)
                    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                                 :: "l"(&G.barrier[G.dev_idx][{slot / gemm_config_fp8::ROW_BLOCK}]), "r"(1) : "memory");
            }
            phase ^= 1;
        }
        __syncthreads();
    }
}
__global__ __launch_bounds__(gemm_config_fp8::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
            int *__restrict__ task_next, const int num_tasks) {
    if (blockIdx.x >= G.num_comp_sms) {
        dispatch_persistent(G, blockIdx.x - G.num_comp_sms, gridDim.x - G.num_comp_sms);
        asm volatile("bar.sync 2, %0;" :: "n"(gemm_config_fp8::NUM_THREADS));  // docs/24
    }
    grouped_gemm_sm120_fp8_dispenser(G, dispatch_gate{G}, noop_epilogue{},
                                     glu_store_policy<globals::outputs_gl>{G.outputs},
                                     blk_expert, task_next, num_tasks);
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config_fp8::ROW_BLOCK - 1) / gemm_config_fp8::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = G.slack[{i}];
}
void entry(kittens::py::TKParallelTensor &pre_tokens, kittens::py::TKParallelTensor &pre_scales,
           at::Tensor &gathered, at::Tensor &gathered_scales,
           at::Tensor &weights, at::Tensor &w_scales, at::Tensor &act,
           at::Tensor &padded_tokens_per_expert, at::Tensor &tp_slots,
           at::Tensor &slack, at::Tensor &pull_order, at::Tensor &blk_expert,
           at::Tensor &gemm_next, kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens, const int num_tokens) {
    using cfg = gemm_config_fp8;
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts, "weights (E,N,K) E mismatch");
    TORCH_CHECK(weights.size(2) == globals::H, "weights (E,N,K) K must be H");
    TORCH_CHECK(act.size(1) == weights.size(1) / 2, "act width must be N/2 (GLU)");
    TORCH_CHECK(gathered_scales.size(1) == globals::NSC, "gathered_scales (P, H/128)");
    TORCH_CHECK(gemm_next.numel() == 1, "gemm_next must be a single int counter");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms >= 1");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    const int s_max = static_cast<int>(tp_slots.size(0));
    TORCH_CHECK(s_max == globals::NUM_DEVICES * num_tokens, "tp_slots rows must be world*T");
    const int nblk = num_padded_local_tokens / cfg::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert per row block");
    const int num_tasks = nblk * (static_cast<int>(weights.size(1)) / cfg::COL_BLOCK);
    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .pre_scales = kittens::py::parallel_tensor_to_pgl<globals::pre_scales_pgl>(pre_scales),
        .activations = kittens::py::tensor_to_gl<globals::gathered_gl>(gathered),
        .a_scales = kittens::py::tensor_to_gl<globals::gscales_gl>(gathered_scales),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .w_scales = kittens::py::tensor_to_gl<globals::w_scales_gl>(w_scales),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(act),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .tp_slots = kittens::py::tensor_to_gl<globals::slots_gl>(tp_slots),
        .slack = kittens::py::tensor_to_gl<globals::slack_gl>(slack),
        .pull_order = kittens::py::tensor_to_gl<globals::order_gl>(pull_order),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = 0,
        .num_padded_local_tokens = num_padded_local_tokens, .num_tokens = num_tokens,
        .s_max = s_max, .num_comp_sms = num_comp_sms
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // dispatch 波需要 20×(4KB+128B) ≈ 84.4KB > GEMM 的 48KB, 取大者
    constexpr int smem = globals::TOKENS_PER_BLOCK *
        (sizeof(globals::token_vec) + sizeof(globals::scale_vec)) + 2048 >
        cfg::DYNAMIC_SHARED_MEMORY + 1024
        ? globals::TOKENS_PER_BLOCK * (sizeof(globals::token_vec) + sizeof(globals::scale_vec)) + 2048
        : cfg::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<sm, cfg::NUM_THREADS, smem, stream>>>(
        G, blk_expert.data_ptr<int>(), gemm_next.data_ptr<int>(), num_tasks);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / cfg::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tpdisp8

/* ===================================================================== *
 * 2d. TP PUSH AllGather-dispatch ⊕ grouped GEMM (layer0, TP-T1; docs/23).
 *
 * Microbench verdict (docs/22 §0): SM pull is the WEAK path on this box
 * (cross-pair 29GB/s, 23.5GB/s/card under 4-way concurrency, needs 16 SMs);
 * SM push is the STRONG path (50.9GB/s, 4 SMs saturate, zero concurrent
 * degradation). TP's AllGather is dense and routing-independent, so the push
 * form is trivial: source card s TMA-pushes its own shard rows into every
 * peer's ag_staging plane [s] (single writer, no atomics), in the CANONICAL
 * min-slot order (tp_slots must be identical on all ranks — the schedule
 * builders drop the per-rank ring for this), with a per-(dst, CHUNK-of-rows)
 * watermark: local acq_rel election -> single-writer st.release.sys seq into
 * the dst's barrier row 2+s, col chunk (the proven preredpush chain).
 *
 * One persistent launch, three roles:
 *   [0, comp)                    grouped GEMM, gate = row-block counter == RB
 *   [comp, comp+push)            push_role: my shard -> peers (strong path)
 *   [comp+push, comp+push+scat)  scatter_role: consume (src, pos) in arrival
 *                                order — wait chunk watermark (remote srcs
 *                                only; own shard reads pre_tokens directly,
 *                                so no self-dependency), copy the row to its
 *                                TOP_K gathered slots, bump row-block counters.
 * All blocks resident (no churn); scatter spins only on REMOTE watermarks,
 * which peers' resident push blocks deliver -> deadlock-free.
 * ===================================================================== */
namespace tppdisp {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    static constexpr int CHUNK = 64;   // rows per watermark chunk (per source)
    using token_vec = sv_bf<H>;
    static constexpr int TOKENS_PER_BLOCK = cfg::DYNAMIC_SHARED_MEMORY / sizeof(token_vec);
    using pre_tokens_pgl  = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using staging_pgl     = pgl<gl<bf16, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;  // (world*T, H), plane s = rows [s*T, s*T+T)
    using post_tokens_gl  = gl<bf16, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using weights_gl      = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl      = gl<bf16, 1, 1, -1, -1>;
    using counts_gl       = gl<int, 1, 1, 1, -1>;
    using slots_gl        = gl<int, 1, 1, -1, TOP_K>;
    using slack_gl        = gl<int, 1, 1, 1, -1>;
    using order2d_gl      = gl<int, 1, 1, -1, -1>;     // push_order (world, T)
    using cnt_gl          = gl<int, 1, 1, 1, -1>;      // (world*nchunks,) local election counters
    using barrier_pgl     = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_pgl pre_tokens;
    staging_pgl ag_staging;
    post_tokens_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    slots_gl tp_slots;
    slack_gl slack;
    order2d_gl push_order;
    cnt_gl push_cnt;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_local_experts;    // == E
    const int expert_offset;        // == 0
    const int num_padded_local_tokens;
    const int num_tokens;           // per-card T
    const int s_max;                // world * T
    const int nchunks;              // ceil(T / CHUNK)
    const int num_comp_sms;
    const int num_push_sms;
    const int num_scatter_sms;
    const int seq;
};
// push role: persistent lanes stream MY shard rows to every peer's plane [me]
// in push_order (canonical min-slot order == every consumer's scatter order),
// electing the per-(dst, chunk) watermark after each chunk fully commits.
__device__ inline void push_role(const globals &G, const int pb_idx) {
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::token_vec (&token)[globals::TOKENS_PER_BLOCK] =
        al.allocate<typename globals::token_vec, globals::TOKENS_PER_BLOCK>();
    __shared__ semaphore token_arrived[globals::TOKENS_PER_BLOCK];
    const int lane_id = threadIdx.x;
    if (lane_id >= globals::TOKENS_PER_BLOCK) return;
    init_semaphore(token_arrived[lane_id], 0, 1);
    int phase = 0;
    const int stride = G.num_push_sms * globals::TOKENS_PER_BLOCK;
    for (int p = pb_idx * globals::TOKENS_PER_BLOCK + lane_id; p < G.num_tokens; p += stride) {
        const int tok = G.push_order[{G.dev_idx, p}];
        tma::expect_bytes(token_arrived[lane_id], sizeof(typename globals::token_vec));
        tma::load_async(token[lane_id], G.pre_tokens[G.dev_idx], {tok, 0}, token_arrived[lane_id]);
        wait(token_arrived[lane_id], phase);
        phase ^= 1;
        const int dst_row = G.dev_idx * G.num_tokens + tok;
        #pragma unroll
        for (int d = 0; d < globals::NUM_DEVICES; d++)
            if (d != G.dev_idx)
                tma::store_async(G.ag_staging[d], token[lane_id], {dst_row, 0});
        tma::store_async_wait();   // my row committed on every peer
        const int chunk = p / globals::CHUNK;
        const int expected = min(globals::CHUNK, G.num_tokens - chunk * globals::CHUNK);
        #pragma unroll
        for (int d = 0; d < globals::NUM_DEVICES; d++) {
            if (d == G.dev_idx) continue;
            int old;
            asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                         : "=r"(old) : "l"(&G.push_cnt[{d * G.nchunks + chunk}]) : "memory");
            if (old + 1 == expected) {   // last row of this chunk for dst d: elected
                __threadfence_system();
                pcie_sync::signal_slot(G.barrier, d, 2 + G.dev_idx, chunk, G.seq);
            }
        }
    }
}
// scatter role: persistent block-per-token rounds in ARRIVAL order (i encodes
// (pos, src) with pos major — earliest chunks of every source first).
__device__ inline void scatter_role(const globals &G, const int sb_idx) {
    constexpr int H = globals::H, VEC = 8, HVEC = H / VEC;
    for (int i = sb_idx; i < G.s_max; i += G.num_scatter_sms) {
        const int s = i % globals::NUM_DEVICES;
        const int p = i / globals::NUM_DEVICES;
        const int tok = G.push_order[{s, p}];
        if (threadIdx.x == 0 && s != G.dev_idx)
            pcie_sync::wait_slot(G.barrier, G.dev_idx, 2 + s, p / globals::CHUNK, G.seq);
        __syncthreads();   // watermark acquired -> whole block may read the row
        const int drow = s * G.num_tokens + tok;
        const bf16 *src = (s == G.dev_idx)
            ? &G.pre_tokens[G.dev_idx][{tok, 0}]      // own shard: never staged
            : &G.ag_staging[G.dev_idx][{drow, 0}];
        const float4 *src_v = reinterpret_cast<const float4 *>(src);
        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) {
            const int slot = G.tp_slots[{drow, k}];
            if (slot < 0) continue;
            float4 *dst_v = reinterpret_cast<float4 *>(&G.activations[{slot, 0}]);
            for (int c = threadIdx.x; c < HVEC; c += blockDim.x)
                dst_v[c] = src_v[c];
        }
        __syncthreads();   // all copies of this token done before signaling
        if (threadIdx.x == 0) {
            __threadfence();   // gpu-scope: block's writes visible before counters
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) {
                const int slot = G.tp_slots[{drow, k}];
                if (slot >= 0)
                    asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                                 :: "l"(&G.barrier[G.dev_idx][{slot / gemm_config::ROW_BLOCK}]), "r"(1) : "memory");
            }
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
    else if (blockIdx.x < G.num_comp_sms + G.num_push_sms)
        push_role(G, blockIdx.x - G.num_comp_sms);
    else
        scatter_role(G, blockIdx.x - G.num_comp_sms - G.num_push_sms);
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = G.slack[{i}];
}
void entry(kittens::py::TKParallelTensor &pre_tokens, kittens::py::TKParallelTensor &ag_staging,
           at::Tensor &post_tokens, at::Tensor &weights, at::Tensor &outputs,
           at::Tensor &padded_tokens_per_expert, at::Tensor &tp_slots, at::Tensor &slack,
           at::Tensor &push_order, at::Tensor &push_cnt,
           kittens::py::TKParallelTensor &barrier, const int num_push_sms,
           const int num_scatter_sms, const int num_padded_local_tokens,
           const int num_tokens, const int seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "TP weights first dim must equal the FULL expert count");
    TORCH_CHECK(num_local_experts <= gemm_config::MAX_LOCAL_EXPERTS, "too many experts");
    TORCH_CHECK(tp_slots.size(1) == globals::TOP_K, "tp_slots second dim must be TOP_K");
    TORCH_CHECK(push_order.size(0) == globals::NUM_DEVICES && push_order.size(1) == num_tokens,
                "push_order must be (world, T)");
    TORCH_CHECK(num_push_sms >= 1 && num_scatter_sms >= 1, "need >=1 push and scatter block");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_push_sms + num_scatter_sms < sm, "comm blocks must leave room for compute");
    const int num_comp_sms = sm - num_push_sms - num_scatter_sms;
    const int s_max = static_cast<int>(tp_slots.size(0));
    TORCH_CHECK(s_max == globals::NUM_DEVICES * num_tokens, "tp_slots rows must be world*T");
    const int nchunks = (num_tokens + globals::CHUNK - 1) / globals::CHUNK;
    TORCH_CHECK(push_cnt.numel() >= globals::NUM_DEVICES * nchunks, "push_cnt too small");
    globals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<globals::pre_tokens_pgl>(pre_tokens),
        .ag_staging = kittens::py::parallel_tensor_to_pgl<globals::staging_pgl>(ag_staging),
        .activations = kittens::py::tensor_to_gl<globals::post_tokens_gl>(post_tokens),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .tp_slots = kittens::py::tensor_to_gl<globals::slots_gl>(tp_slots),
        .slack = kittens::py::tensor_to_gl<globals::slack_gl>(slack),
        .push_order = kittens::py::tensor_to_gl<globals::order2d_gl>(push_order),
        .push_cnt = kittens::py::tensor_to_gl<globals::cnt_gl>(push_cnt),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_local_experts = num_local_experts,
        .expert_offset = 0,
        .num_padded_local_tokens = num_padded_local_tokens, .num_tokens = num_tokens,
        .s_max = s_max, .nchunks = nchunks, .num_comp_sms = num_comp_sms,
        .num_push_sms = num_push_sms, .num_scatter_sms = num_scatter_sms, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<sm, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tppdisp

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

/* ===================================================================== *
 * 4b. W2 GEMM ⊕ pre-reduction PUSH (layer1, T6-v1; docs/18). Same math as v0
 *     prered, but the data plane is reversed to the verified dpush3 push+
 *     election pattern: as soon as an expert card computes a partial row for
 *     (src_dev=s, src_tok=t), it TMA-PUSHES it to source card s's staging plane
 *     [my_rank] row t (strong path), edge-triggered so the transfer streams
 *     under the ongoing W2 GEMM (docs/17: v0 had ~0% overlap). Per-(expert card)
 *     watermark election (dpush3-style) replaces v0's full barrier; the source's
 *     final_reduce_push waits only on the <=world cards that actually send it.
 *
 * Signal layout: barrier_l1 rows — 0 = local GEMM col-block counter (reset each
 * iter), 1 = local W2 completion signal (job slot gate, same as v0), 2+d =
 * expert card d's cross-card watermark ("d finished all pushes to me").
 * ===================================================================== */
namespace preredpush {
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    using row_vec        = sv_bf<H>;                          // one partial row in smem for TMA push
    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl     = gl<bf16, 1, 1, -1, H>;             // local W2 output (expert_out)
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    using staging_pgl    = pgl<gl<bf16, 1, 1, -1, H, row_vec>, NUM_DEVICES, false>;  // peers push here
    using dst_gl         = gl<int, 1, 1, -1, 2>;              // (J,2) -> (src_dev, src_tok)
    using slots_gl       = gl<int, 1, 1, -1, TOP_K>;          // (J,TOP_K) local slots (-1 pad)
    using w_gl           = gl<float, 1, 1, -1, TOP_K>;        // (J,TOP_K) weights (0 pad)
    using cnt1d_gl       = gl<int, 1, 1, 1, -1>;              // local_cnt / push_expected (world,)
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    activations_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
    staging_pgl staging;           // this card's + peers' staging (push target)
    dst_gl prered_dst;
    slots_gl prered_slots;
    w_gl prered_w;
    cnt1d_gl local_cnt;            // (world,) per-source-card push counter, zeroed each iter
    cnt1d_gl push_expected_l1;     // (world,) rows this card pushes to each source card
    barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_source_tokens;
    const int num_jobs;
    const int num_comp_sms;
    const int seq;
};
struct final_globals {
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    using staging_pgl    = pgl<gl<bf16, 1, 1, -1, H>, NUM_DEVICES, false>;
    using contrib_gl     = gl<int, 1, 1, -1, -1>;             // (num_tokens, world)
    using cnt1d_gl       = gl<int, 1, 1, 1, -1>;              // recv_from (world,)
    using combine_out_gl = gl<bf16, 1, 1, -1, H>;
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    staging_pgl staging;
    contrib_gl final_contrib;
    cnt1d_gl recv_from;
    combine_out_gl combine_out;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_source_tokens;
    const int seq;
};
// LOCAL W2 completion signal — identical to v0 prered_signal_epilogue.
struct signal_epilogue {
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
// one push job: FP32 weighted-sum this card's local slots hitting (s,t) into an
// smem row, TMA-push it to source card s's staging[my_rank][t], then the elected
// last-completer for s fences+signals s's watermark.
__device__ inline void push_job(const globals &G, const int j) {
    if (j >= G.num_jobs) return;
    constexpr int H = globals::H, VEC = 8, HVEC = H / VEC;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::row_vec &row = al.allocate<typename globals::row_vec>();
    __shared__ int s_slot[globals::TOP_K];
    __shared__ float s_w[globals::TOP_K];
    __shared__ int s_s, s_t, s_has;
    if (threadIdx.x < globals::TOP_K) {
        const int k = threadIdx.x;
        const int slot = G.prered_slots[{j, k}];
        s_slot[k] = slot;
        s_w[k] = G.prered_w[{j, k}];
        if (slot >= 0)
            pcie_sync::wait_slot(G.barrier, G.dev_idx, 1, slot / gemm_config::ROW_BLOCK, G.seq);
    }
    if (threadIdx.x == 0) {
        s_s = G.prered_dst[{j, 0}];
        s_t = G.prered_dst[{j, 1}];
        int has = 0;
        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) has |= (G.prered_slots[{j, k}] >= 0);
        s_has = has;
    }
    __syncthreads();
    if (!s_has) return;  // empty job: no push, no count (matches host push_expected)

    // FP32 weighted-sum into the smem row (bf16)
    bf16 *row_ptr = reinterpret_cast<bf16 *>(&row);
    float4 *row_v = reinterpret_cast<float4 *>(row_ptr);
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
        row_v[c] = *reinterpret_cast<const float4 *>(res);
    }
    __syncthreads();
    // TMA-push the smem row to source card s's staging plane [my_rank] row t.
    if (threadIdx.x == 0) {
        const int dst_row = G.dev_idx * G.num_source_tokens + s_t;
        tma::store_async(G.staging[s_s], row, {dst_row, 0});
        tma::store_async_wait();  // my remote bulk write committed
        // LOCAL election (gpu scope): last completer for source s fences+signals.
        int old;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(old) : "l"(&G.local_cnt[{s_s}]) : "memory");
        if (old + 1 == G.push_expected_l1[{s_s}]) {
            __threadfence_system();
            pcie_sync::signal_slot(G.barrier, s_s, 2 + G.dev_idx, 0, G.seq);
        }
    }
}
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void gemm_push_kernel(const __grid_constant__ globals G) {
    const int col_blocks = static_cast<int>(G.weights.cols()) / gemm_config::COL_BLOCK;
    if (blockIdx.x < G.num_comp_sms)
        grouped_gemm_sm120(G, no_gate{}, signal_epilogue{G, col_blocks}, blockIdx.x, G.num_comp_sms);
    else
        push_job(G, blockIdx.x - G.num_comp_sms);
}
// TP variant (docs/20): PERSISTENT all-hands job drain instead of one short-
// lived block per job. First-round finding: 2048 job blocks queued on 16 comm
// SMs = 128 sequential waves of 97KB-smem block churn (~the whole slowdown at
// low num_comm_sms), and since a TP job depends on its token's MAX slot (the 8
// experts span the whole table) most jobs only unblock near the GEMM's end.
// Fix: grid = comp + comm blocks only, all resident; every block claims jobs
// from an atomic dispenser through `job_order` (max-slot sorted = readiness
// order, so comm blocks stream the earliest-ready pushes UNDER the GEMM), and
// comp blocks join the pool the moment their GEMM tasks finish — the tail
// drains on all ~110 SMs instead of num_comm_sms. Each job claimed exactly
// once (the single-writer staging/election invariants are untouched).
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void gemm_push_kernel_tp(const __grid_constant__ globals G,
                         const int *__restrict__ job_order, int *job_next) {
    const int col_blocks = static_cast<int>(G.weights.cols()) / gemm_config::COL_BLOCK;
    if (blockIdx.x < G.num_comp_sms) {
        grouped_gemm_sm120(G, no_gate{}, signal_epilogue{G, col_blocks}, blockIdx.x, G.num_comp_sms);
        // Join ALL warps on a DEDICATED named barrier before reusing smem.
        // NOT __syncthreads(): that is hw barrier 0 with a 288-thread count,
        // while inside the GEMM the consumer group still cycles barrier 0/1
        // with a 256-thread count (producer lanes 1..31 exit the GEMM
        // immediately) — concurrent mixed-count arrivals on one barrier are
        // UB (observed as cudaErrorIllegalInstruction, round 4 / docs/24).
        asm volatile("bar.sync 2, %0;" :: "n"(gemm_config::NUM_THREADS));
    }
    // Post-join (or comm block): barrier 0 is free again, __syncthreads is safe.
    __shared__ int s_j;
    while (true) {
        if (threadIdx.x == 0) s_j = atomicAdd(job_next, 1);
        __syncthreads();
        const int idx = s_j;
        if (idx >= G.num_jobs) break;
        push_job(G, job_order[idx]);
        __syncthreads();  // job's smem fully consumed before the next claim reuses it
    }
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config::ROW_BLOCK - 1) / gemm_config::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{0, i}] = 0;
}
// source-card final reduce: wait per-card watermark, then sum contributing rows.
__global__ void final_reduce_push_kernel(const __grid_constant__ final_globals G) {
    const int t = blockIdx.x;
    if (t >= G.num_source_tokens) return;
    constexpr int H = final_globals::H, VEC = 8, HVEC = H / VEC;
    __shared__ int s_contrib[final_globals::NUM_DEVICES];
    // per-card watermark wait: only the cards that send me anything (recv_from)
    if (threadIdx.x < final_globals::NUM_DEVICES) {
        const int d = threadIdx.x;
        s_contrib[d] = G.final_contrib[{t, d}];
        if (G.recv_from[{d}] != 0)
            pcie_sync::wait_slot(G.barrier, G.dev_idx, 2 + d, 0, G.seq);
    }
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
            const int row = d * G.num_source_tokens + t;
            const bf16 *p_row = &G.staging[G.dev_idx][{row, 0}];
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
static void _launch_gemm_push(globals &G, at::Tensor &padded_tokens_per_expert,
                              int num_padded_local_tokens, int num_comm_sms) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(gemm_push_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    gemm_push_kernel<<<G.num_comp_sms + G.num_jobs, gemm_config::NUM_THREADS, smem, stream>>>(G);
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
static globals _make_globals(at::Tensor &activations, at::Tensor &weights,
           kittens::py::TKParallelTensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           kittens::py::TKParallelTensor &staging, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w, at::Tensor &local_cnt,
           at::Tensor &push_expected_l1, kittens::py::TKParallelTensor &barrier,
           int num_comm_sms, int num_padded_local_tokens, int num_source_tokens,
           int num_jobs, int seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0)) / globals::NUM_DEVICES;
    TORCH_CHECK(weights.size(0) == num_local_experts, "weights first dim mismatch");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms >= 1");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    const int num_comp_sms = sm - num_comm_sms;
    return globals {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(expert_outputs.data_),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = dev_idx * num_local_experts,
        .staging = kittens::py::parallel_tensor_to_pgl<globals::staging_pgl>(staging),
        .prered_dst = kittens::py::tensor_to_gl<globals::dst_gl>(prered_dst),
        .prered_slots = kittens::py::tensor_to_gl<globals::slots_gl>(prered_slots),
        .prered_w = kittens::py::tensor_to_gl<globals::w_gl>(prered_w),
        .local_cnt = kittens::py::tensor_to_gl<globals::cnt1d_gl>(local_cnt),
        .push_expected_l1 = kittens::py::tensor_to_gl<globals::cnt1d_gl>(push_expected_l1),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens, .num_jobs = num_jobs,
        .num_comp_sms = num_comp_sms, .seq = seq
    };
}
void gemm_push_entry(at::Tensor &activations, at::Tensor &weights,
           kittens::py::TKParallelTensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           kittens::py::TKParallelTensor &staging, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w, at::Tensor &local_cnt,
           at::Tensor &push_expected_l1, kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens,
           const int num_source_tokens, const int num_jobs, const int seq) {
    globals G = _make_globals(activations, weights, expert_outputs, padded_tokens_per_expert,
                              staging, prered_dst, prered_slots, prered_w, local_cnt,
                              push_expected_l1, barrier, num_comm_sms,
                              num_padded_local_tokens, num_source_tokens, num_jobs, seq);
    _launch_gemm_push(G, padded_tokens_per_expert, num_padded_local_tokens, num_comm_sms);
}
// TP variant: identical kernel, two geometry changes — (a) padded covers ALL
// experts and they are all local (expert_offset = 0), (b) expert_out is a plain
// LOCAL tensor (peers never touch it; in TP the only cross-card layer1 traffic
// is the prered-row push into `staging`). The prered math is unchanged: in TP
// every job (src_dev, src_tok) has exactly TOP_K local hits (all experts are
// here), the pushed row is this card's I-shard partial of the full top-k
// weighted sum, and final_reduce_push's Σ over cards IS the TP all-reduce.
void gemm_push_entry_tp(at::Tensor &activations, at::Tensor &weights,
           at::Tensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           kittens::py::TKParallelTensor &staging, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w, at::Tensor &local_cnt,
           at::Tensor &push_expected_l1, at::Tensor &job_order, at::Tensor &job_next,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens,
           const int num_source_tokens, const int num_jobs, const int seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "TP weights first dim must equal the FULL expert count");
    TORCH_CHECK(num_local_experts <= gemm_config::MAX_LOCAL_EXPERTS, "too many experts");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms >= 1");
    TORCH_CHECK(job_order.size(0) == num_jobs, "job_order must cover num_jobs");
    TORCH_CHECK(job_next.numel() == 1, "job_next must be a single int counter");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(expert_outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = 0,
        .staging = kittens::py::parallel_tensor_to_pgl<globals::staging_pgl>(staging),
        .prered_dst = kittens::py::tensor_to_gl<globals::dst_gl>(prered_dst),
        .prered_slots = kittens::py::tensor_to_gl<globals::slots_gl>(prered_slots),
        .prered_w = kittens::py::tensor_to_gl<globals::w_gl>(prered_w),
        .local_cnt = kittens::py::tensor_to_gl<globals::cnt1d_gl>(local_cnt),
        .push_expected_l1 = kittens::py::tensor_to_gl<globals::cnt1d_gl>(push_expected_l1),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens, .num_jobs = num_jobs,
        .num_comp_sms = num_comp_sms, .seq = seq
    };
    // persistent grid: comp + comm blocks only (jobs come from the dispenser)
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(gemm_push_kernel_tp, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    gemm_push_kernel_tp<<<sm, gemm_config::NUM_THREADS, smem, stream>>>(
        G, job_order.data_ptr<int>(), job_next.data_ptr<int>());
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / gemm_config::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
void final_reduce_push_entry(kittens::py::TKParallelTensor &staging, at::Tensor &final_contrib,
           at::Tensor &recv_from, at::Tensor &combine_out, kittens::py::TKParallelTensor &barrier,
           const int num_source_tokens, const int seq) {
    const int dev_idx = barrier.local_rank_;
    final_globals G {
        .staging = kittens::py::parallel_tensor_to_pgl<final_globals::staging_pgl>(staging),
        .final_contrib = kittens::py::tensor_to_gl<final_globals::contrib_gl>(final_contrib),
        .recv_from = kittens::py::tensor_to_gl<final_globals::cnt1d_gl>(recv_from),
        .combine_out = kittens::py::tensor_to_gl<final_globals::combine_out_gl>(combine_out),
        .barrier = kittens::py::parallel_tensor_to_pgl<final_globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_source_tokens = num_source_tokens, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    final_reduce_push_kernel<<<num_source_tokens, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace preredpush

/* ===================================================================== *
 * 4c. TP layer1 v2: N 维分解的 combine(docs/32, Comet layer1-N 教训)。
 *
 * v1(gemm_push_kernel_tp)按 M 维分解 combine:job = 一个 token,就绪 =
 * 它 TOP_K 个 slot 的行块全部算完。topk=8 时 max slot 期望在表的 ~8/9 处
 * -> 约九成 job 拖到 GEMM 尾部才解锁,push 挤在尾巴,comm 块大部分时间
 * 在 spin(experience/02 §2 表里 "M 维不可" 的原因: combine 对同一 token
 * 的 topk 行归约,行与行强耦合)。
 *
 * v2 沿 N(输出列)分解,配套两件事:
 *   1. W2 GEMM 换 COL_MAJOR dispenser(列外层扫描): 第 c 个列扫在
 *      ~(c+1)/col_blocks 的 GEMM 进度处完成 —— 完整列切片提前成型;
 *   2. 信号按列扫聚合(docs/02 §3 计数聚合): 每个 (rb,cb) tile 完成给
 *      cb 计数器 +1,凑满 nblk 个行块 = 该列全表就绪,ONE 信号放行该列
 *      全部 token 的 combine —— per-job 的 max-slot wait 和 job_order
 *      排序整个删掉,协议反而更简单。
 *   push job = (token, chunk): 归约该 token TOP_K 行的 CHUNK_COLS 列段,
 *   TMA 推 1KB 段到源卡 staging 的行内偏移。job 序 chunk-major,
 *   comm 块从 chunk 0 起顺次消化 —— push 从 GEMM 的 ~CHUNK_CB/32 进度
 *   就开始流,而不是尾部倾泻。角色结构与 v1 相同(comp 块 GEMM 完
 *   bar.sync 2 转岗入池)。watermark 选举不变,expected × NCHUNKS。
 * ===================================================================== */
namespace tppr2 {
#ifndef TK_L1_CHUNK_CB
#define TK_L1_CHUNK_CB 4
#endif
struct globals {
    using cfg = gemm_config;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    static constexpr int CHUNK_CB = TK_L1_CHUNK_CB;              // col blocks per push chunk
    static constexpr int CHUNK_COLS = CHUNK_CB * cfg::COL_BLOCK; // 512 cols = 1KB bf16
    static constexpr int NCHUNKS = H / CHUNK_COLS;
    static_assert(H % CHUNK_COLS == 0, "H % (CHUNK_CB*COL_BLOCK) != 0");
    using chunk_vec       = sv_bf<CHUNK_COLS>;
    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;
    using outputs_gl     = gl<bf16, 1, 1, -1, H>;
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    using staging_pgl    = pgl<gl<bf16, 1, 1, -1, H, chunk_vec>, NUM_DEVICES, false>;
    using dst_gl         = gl<int, 1, 1, -1, 2>;
    using slots_gl       = gl<int, 1, 1, -1, TOP_K>;
    using w_gl           = gl<float, 1, 1, -1, TOP_K>;
    using cnt1d_gl       = gl<int, 1, 1, 1, -1>;
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    activations_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
    staging_pgl staging;
    dst_gl prered_dst;
    slots_gl prered_slots;
    w_gl prered_w;
    cnt1d_gl local_cnt;
    cnt1d_gl push_expected_l1;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_source_tokens;
    const int num_jobs;        // world*T tokens; push jobs = num_jobs * NCHUNKS
    const int num_comp_sms;
    const int seq;
};
struct no_gate { __device__ inline void operator()(int) const {} };
// 列扫聚合信号: 每个 (rb, cb) tile 完成 -> cb 计数器 +1; 满 nblk = 该列
// 全表就绪, 单写者把 seq 盖进 barrier 行 1 列 cb(本地 gpu 序即可,
// 跨卡的一步在 push 的 watermark 上)。barrier 行 0 = cb 计数器(每迭代
// 清零), 行 1 = cb 就绪信号(seq 单调, 无需清)。
struct colsweep_epilogue {
    const globals &G;
    const int nblk;
    __device__ inline void operator()(int, int col_idx) const {
        __threadfence();
        kittens::group<gemm_config::CONSUMER_WARPS>::sync(1);
        if (kittens::laneid() != 0 || kittens::warpid() != 0) return;
        int done;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(done) : "l"(&G.barrier[G.dev_idx][{0, col_idx}]) : "memory");
        if (done + 1 == nblk)
            asm volatile("st.release.gpu.global.s32 [%0], %1;"
                         :: "l"(&G.barrier[G.dev_idx][{1, col_idx}]), "r"(G.seq) : "memory");
    }
};
// push job = (chunk c, 16-token 组 g)(docs/33 修复: 第十二轮首测 job 粒度
// 1 token×1KB, 每 job 一次 TMA+wait 串行化 -> 每块仅 1 个 1KB 写在飞,
// PCIe 延迟被暴露 16384 次, L1_fused 691->1294)。关键: j = src_dev*T +
// src_tok 天然目的卡优先, 连续 GRP 个 j 同目的卡且目的行连续 -> 一个 job
// 归约 GRP 个 token 的列段(GRP KB smem), 背靠背发 GRP 个 TMA 再等一次,
// 延迟摊薄 GRP 倍, job 数回到 (world*T/GRP)*NCHUNKS = 1024。
// (docs/02 §3: 工作粒度 = 能触发一次高效通信的最小单位。)
static constexpr int GRP = 16;   // tokens per push job; 需 T % GRP == 0
__device__ inline void push_group(const globals &G, const int idx) {
    const int ngroups = G.num_jobs / GRP;
    if (idx >= ngroups * globals::NCHUNKS) return;
    const int c = idx / ngroups;
    const int g = idx - c * ngroups;
    const int j0 = g * GRP;
    constexpr int VEC = 8, CVEC = globals::CHUNK_COLS / VEC;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::chunk_vec (&rows)[GRP] =
        al.allocate<typename globals::chunk_vec, GRP>();
    __shared__ int s_slot[GRP][globals::TOP_K];
    __shared__ float s_w[GRP][globals::TOP_K];
    __shared__ int s_dst, s_t0;
    if (threadIdx.x < globals::CHUNK_CB)   // 列扫聚合信号, 每列一个槽
        pcie_sync::wait_slot(G.barrier, G.dev_idx, 1,
                             c * globals::CHUNK_CB + threadIdx.x, G.seq);
    if (threadIdx.x < GRP * globals::TOP_K) {
        const int r = threadIdx.x / globals::TOP_K;
        const int k = threadIdx.x - r * globals::TOP_K;
        s_slot[r][k] = G.prered_slots[{j0 + r, k}];
        s_w[r][k] = G.prered_w[{j0 + r, k}];
    }
    if (threadIdx.x == 0) {
        s_dst = G.prered_dst[{j0, 0}];   // 组内目的卡相同(T % GRP == 0)
        s_t0 = G.prered_dst[{j0, 1}];
    }
    __syncthreads();

    const int col0 = c * globals::CHUNK_COLS;
    for (int u = threadIdx.x; u < GRP * CVEC; u += blockDim.x) {
        const int r = u / CVEC, cc = u - (u / CVEC) * CVEC;
        float acc[VEC];
        #pragma unroll
        for (int i = 0; i < VEC; i++) acc[i] = 0.0f;
        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) {
            const int slot = s_slot[r][k];
            if (slot < 0) continue;
            const float w = s_w[r][k];
            const bf16 *e_row = &G.outputs[{slot, col0}];
            const float4 packed = reinterpret_cast<const float4 *>(e_row)[cc];
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
        reinterpret_cast<float4 *>(&rows[r])[cc] = *reinterpret_cast<const float4 *>(res);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int nreal = 0;
        #pragma unroll 1
        for (int r = 0; r < GRP; r++) {
            int has = 0;
            #pragma unroll
            for (int k = 0; k < globals::TOP_K; k++) has |= (s_slot[r][k] >= 0);
            if (!has) continue;   // 空 token 不推不计数(与 push_expected 口径一致)
            nreal++;
            const int dst_row = G.dev_idx * G.num_source_tokens + s_t0 + r;
            tma::store_async(G.staging[s_dst], rows[r], {dst_row, c});
        }
        tma::store_async_wait();   // GRP 个 1KB 写一次性排空(延迟摊薄)
        if (nreal) {
            int old;
            asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], %2;}"
                         : "=r"(old) : "l"(&G.local_cnt[{s_dst}]), "r"(nreal) : "memory");
            if (old + nreal == G.push_expected_l1[{s_dst}] * globals::NCHUNKS) {
                __threadfence_system();
                pcie_sync::signal_slot(G.barrier, s_dst, 2 + G.dev_idx, 0, G.seq);
            }
        }
    }
}
__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
            int *__restrict__ gemm_next, int *__restrict__ job_next) {
    if (blockIdx.x < G.num_comp_sms) {
        const int col_blocks = static_cast<int>(G.weights.cols()) / gemm_config::COL_BLOCK;
        const int nblk = G.num_padded_local_tokens / gemm_config::ROW_BLOCK;
        grouped_gemm_sm120_dispenser<true>(
            G, no_gate{}, colsweep_epilogue{G, nblk},
            plain_store_policy<globals::outputs_gl>{G.outputs},
            blk_expert, gemm_next, nblk * col_blocks);
        asm volatile("bar.sync 2, %0;" :: "n"(gemm_config::NUM_THREADS));  // docs/24
    }
    __shared__ int s_j;
    const int njobs = (G.num_jobs / GRP) * globals::NCHUNKS;
    while (true) {
        if (threadIdx.x == 0) s_j = atomicAdd(job_next, 1);
        __syncthreads();
        const int idx = s_j;
        if (idx >= njobs) break;
        push_group(G, idx);
        __syncthreads();
    }
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int cb = globals::H / gemm_config::COL_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{0, i}] = 0;
}
void entry(at::Tensor &activations, at::Tensor &weights,
           at::Tensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           kittens::py::TKParallelTensor &staging, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w, at::Tensor &local_cnt,
           at::Tensor &push_expected_l1, at::Tensor &blk_expert,
           at::Tensor &gemm_next, at::Tensor &job_next,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens,
           const int num_source_tokens, const int num_jobs, const int seq) {
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts,
                "TP weights first dim must equal the FULL expert count");
    TORCH_CHECK(num_local_experts <= gemm_config::MAX_LOCAL_EXPERTS, "too many experts");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms >= 1");
    TORCH_CHECK(gemm_next.numel() == 1 && job_next.numel() == 1, "counters must be single ints");
    const int nblk = num_padded_local_tokens / gemm_config::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert must have one entry per row block");
    TORCH_CHECK(static_cast<int>(weights.size(2)) == globals::H,
                "L1 weights cols must be H (down projection)");
    TORCH_CHECK(num_source_tokens % GRP == 0,
                "T must be divisible by the push group size (GRP)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(activations),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(expert_outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = 0,
        .staging = kittens::py::parallel_tensor_to_pgl<globals::staging_pgl>(staging),
        .prered_dst = kittens::py::tensor_to_gl<globals::dst_gl>(prered_dst),
        .prered_slots = kittens::py::tensor_to_gl<globals::slots_gl>(prered_slots),
        .prered_w = kittens::py::tensor_to_gl<globals::w_gl>(prered_w),
        .local_cnt = kittens::py::tensor_to_gl<globals::cnt1d_gl>(local_cnt),
        .push_expected_l1 = kittens::py::tensor_to_gl<globals::cnt1d_gl>(push_expected_l1),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens, .num_jobs = num_jobs,
        .num_comp_sms = num_comp_sms, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<sm, gemm_config::NUM_THREADS, smem, stream>>>(
        G, blk_expert.data_ptr<int>(), gemm_next.data_ptr<int>(), job_next.data_ptr<int>());
    CUDACHECK(cudaGetLastError());
    reset_kernel<<<1, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tppr2

/* ===================================================================== *
 * 4d. FP8 TP layer1(docs/42 P3): fp8 dispenser W2 GEMM + v1 的 M 维
 *   push/排空(N 维分解是已定负结果, docs/35)。集成零阻力:
 *   B^T = problem.w2 原布局 (E, H, inter) = (E, N, K), qc.w2_scale
 *   (E, N/128, K/128) 原样可用 —— 无转置、无重量化、无二次量化误差。
 *   A = act 经 rowgroup_quant_fp8(P, 768 -> 6 组/行)。GEMM 出 bf16
 *   expert_out, signal_epilogue / push_job / watermark / final_reduce
 *   与 v1 完全同构(push/combine 精度不变)。
 * ===================================================================== */
namespace tppr8 {
struct globals {
    using cfg = gemm_config_fp8;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    using row_vec        = sv_bf<H>;
    using activations_gl = gl<fp8e4m3, 1, 1, -1, -1, cfg::A_tile>;   // act fp8 (P, inter)
    using a_scales_gl    = gl<float, 1, 1, -1, -1>;                  // (P, inter/128)
    using weights_gl     = gl<fp8e4m3, 1, -1, -1, -1, cfg::B_tile>;  // w2 (E, H, inter) = B^T
    using w_scales_gl    = gl<float, 1, -1, -1, -1>;                 // (E, H/128, inter/128)
    using outputs_gl     = gl<bf16, 1, 1, -1, H>;                    // expert_out (bf16)
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    using staging_pgl    = pgl<gl<bf16, 1, 1, -1, H, row_vec>, NUM_DEVICES, false>;
    using dst_gl         = gl<int, 1, 1, -1, 2>;
    using slots_gl       = gl<int, 1, 1, -1, TOP_K>;
    using w_gl           = gl<float, 1, 1, -1, TOP_K>;
    using cnt1d_gl       = gl<int, 1, 1, 1, -1>;
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    activations_gl activations;
    a_scales_gl a_scales;
    weights_gl weights;
    w_scales_gl w_scales;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;
    const int num_local_experts;
    const int expert_offset;
    staging_pgl staging;
    dst_gl prered_dst;
    slots_gl prered_slots;
    w_gl prered_w;
    cnt1d_gl local_cnt;
    cnt1d_gl push_expected_l1;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_source_tokens;
    const int num_jobs;
    const int num_comp_sms;
    const int seq;
};
struct no_gate { __device__ inline void operator()(int) const {} };
// 与 preredpush::signal_epilogue 同构(fp8 config 的 CONSUMER_WARPS 同值)
struct signal_epilogue {
    const globals &G;
    const int col_blocks;
    __device__ inline void operator()(int row_idx, int) const {
        __threadfence();
        kittens::group<gemm_config_fp8::CONSUMER_WARPS>::sync(1);
        if (kittens::laneid() != 0 || kittens::warpid() != 0) return;
        int done;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(done) : "l"(&G.barrier[G.dev_idx][{0, row_idx}]) : "memory");
        if (done + 1 == col_blocks)
            asm volatile("st.release.gpu.global.s32 [%0], %1;"
                         :: "l"(&G.barrier[G.dev_idx][{1, row_idx}]), "r"(G.seq) : "memory");
    }
};
// push_job: 与 preredpush::push_job 逐行同构(读 bf16 expert_out, 推 8KB 行)
__device__ inline void push_job(const globals &G, const int j) {
    if (j >= G.num_jobs) return;
    constexpr int H = globals::H, VEC = 8, HVEC = H / VEC;
    extern __shared__ int __shm[];
    tma_swizzle_allocator al((int*)&__shm[0]);
    typename globals::row_vec &row = al.allocate<typename globals::row_vec>();
    __shared__ int s_slot[globals::TOP_K];
    __shared__ float s_w[globals::TOP_K];
    __shared__ int s_s, s_t, s_has;
    if (threadIdx.x < globals::TOP_K) {
        const int k = threadIdx.x;
        const int slot = G.prered_slots[{j, k}];
        s_slot[k] = slot;
        s_w[k] = G.prered_w[{j, k}];
        if (slot >= 0)
            pcie_sync::wait_slot(G.barrier, G.dev_idx, 1, slot / gemm_config_fp8::ROW_BLOCK, G.seq);
    }
    if (threadIdx.x == 0) {
        s_s = G.prered_dst[{j, 0}];
        s_t = G.prered_dst[{j, 1}];
        int has = 0;
        #pragma unroll
        for (int k = 0; k < globals::TOP_K; k++) has |= (G.prered_slots[{j, k}] >= 0);
        s_has = has;
    }
    __syncthreads();
    if (!s_has) return;

    bf16 *row_ptr = reinterpret_cast<bf16 *>(&row);
    float4 *row_v = reinterpret_cast<float4 *>(row_ptr);
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
        row_v[c] = *reinterpret_cast<const float4 *>(res);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const int dst_row = G.dev_idx * G.num_source_tokens + s_t;
        tma::store_async(G.staging[s_s], row, {dst_row, 0});
        tma::store_async_wait();
        int old;
        asm volatile("{atom.acq_rel.gpu.global.add.s32 %0, [%1], 1;}"
                     : "=r"(old) : "l"(&G.local_cnt[{s_s}]) : "memory");
        if (old + 1 == G.push_expected_l1[{s_s}]) {
            __threadfence_system();
            pcie_sync::signal_slot(G.barrier, s_s, 2 + G.dev_idx, 0, G.seq);
        }
    }
}
__global__ __launch_bounds__(gemm_config_fp8::NUM_THREADS, 1)
void kernel(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
            int *__restrict__ gemm_next,
            const int *__restrict__ job_order, int *__restrict__ job_next) {
    if (blockIdx.x < G.num_comp_sms) {
        const int col_blocks = static_cast<int>(G.weights.rows()) / gemm_config_fp8::COL_BLOCK;
        const int nblk = G.num_padded_local_tokens / gemm_config_fp8::ROW_BLOCK;
        grouped_gemm_sm120_fp8_dispenser(
            G, no_gate{}, signal_epilogue{G, col_blocks},
            plain_store_policy<globals::outputs_gl>{G.outputs},
            blk_expert, gemm_next, nblk * col_blocks);
        asm volatile("bar.sync 2, %0;" :: "n"(gemm_config_fp8::NUM_THREADS));  // docs/24
    }
    __shared__ int s_j;
    while (true) {
        if (threadIdx.x == 0) s_j = atomicAdd(job_next, 1);
        __syncthreads();
        const int idx = s_j;
        if (idx >= G.num_jobs) break;
        push_job(G, job_order[idx]);
        __syncthreads();
    }
}
__global__ __launch_bounds__(256)
void reset_kernel(const __grid_constant__ globals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config_fp8::ROW_BLOCK - 1) / gemm_config_fp8::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{0, i}] = 0;
}
void entry(at::Tensor &act_fp8, at::Tensor &act_scales,
           at::Tensor &weights, at::Tensor &w_scales,
           at::Tensor &expert_outputs, at::Tensor &padded_tokens_per_expert,
           kittens::py::TKParallelTensor &staging, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w, at::Tensor &local_cnt,
           at::Tensor &push_expected_l1, at::Tensor &blk_expert,
           at::Tensor &gemm_next, at::Tensor &job_order, at::Tensor &job_next,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens,
           const int num_source_tokens, const int num_jobs, const int seq) {
    using cfg = gemm_config_fp8;
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts, "w2 (E,H,inter) E mismatch");
    TORCH_CHECK(weights.size(1) == globals::H, "w2 rows must be H (B^T)");
    TORCH_CHECK(weights.size(2) == act_fp8.size(1), "w2 K must match act inter");
    TORCH_CHECK(act_fp8.size(1) % cfg::SCALE_K == 0, "inter % 128");
    TORCH_CHECK(act_scales.size(1) == act_fp8.size(1) / cfg::SCALE_K, "act_scales (P, inter/128)");
    TORCH_CHECK(w_scales.size(1) == globals::H / cfg::COL_BLOCK &&
                w_scales.size(2) == static_cast<int>(weights.size(2)) / cfg::SCALE_K,
                "w2_scales must be (E, H/128, inter/128)");
    TORCH_CHECK(gemm_next.numel() == 1 && job_next.numel() == 1, "counters");
    TORCH_CHECK(job_order.size(0) == num_jobs, "job_order");
    TORCH_CHECK(num_comm_sms >= 1, "num_comm_sms >= 1");
    const int nblk = num_padded_local_tokens / cfg::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert per row block");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int num_comp_sms = sm - num_comm_sms;
    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(act_fp8),
        .a_scales = kittens::py::tensor_to_gl<globals::a_scales_gl>(act_scales),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .w_scales = kittens::py::tensor_to_gl<globals::w_scales_gl>(w_scales),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(expert_outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = num_local_experts, .expert_offset = 0,
        .staging = kittens::py::parallel_tensor_to_pgl<globals::staging_pgl>(staging),
        .prered_dst = kittens::py::tensor_to_gl<globals::dst_gl>(prered_dst),
        .prered_slots = kittens::py::tensor_to_gl<globals::slots_gl>(prered_slots),
        .prered_w = kittens::py::tensor_to_gl<globals::w_gl>(prered_w),
        .local_cnt = kittens::py::tensor_to_gl<globals::cnt1d_gl>(local_cnt),
        .push_expected_l1 = kittens::py::tensor_to_gl<globals::cnt1d_gl>(push_expected_l1),
        .barrier = kittens::py::parallel_tensor_to_pgl<globals::barrier_pgl>(barrier),
        .dev_idx = dev_idx, .num_padded_local_tokens = num_padded_local_tokens,
        .num_source_tokens = num_source_tokens, .num_jobs = num_jobs,
        .num_comp_sms = num_comp_sms, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = cfg::DYNAMIC_SHARED_MEMORY + 1024;   // 48KB GEMM > 8KB push row
    CUDACHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel<<<sm, cfg::NUM_THREADS, smem, stream>>>(
        G, blk_expert.data_ptr<int>(), gemm_next.data_ptr<int>(),
        job_order.data_ptr<int>(), job_next.data_ptr<int>());
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / cfg::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tppr8

#include <torch/csrc/utils/pybind.h>
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_TK_PARALLEL_TENSOR(m);
    m.def("grouped_gemm", &gg::entry);
    m.def("grouped_gemm_nb", &gg::entry_nb);
    m.def("moe_dispatch_gemm", &disp::entry);
    m.def("moe_dispatch_dedup", &ddisp::entry);
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
    m.def("moe_gemm_prered_push_fused", &preredpush::gemm_push_entry);
    m.def("moe_final_reduce_push", &preredpush::final_reduce_push_entry);
    m.def("moe_tp_dispatch_gemm", &tpdisp::entry);
    m.def("moe_tp_dispatch_gemm_v2", &tpdisp2::entry);
    m.def("moe_tp_dispatch_gemm_fp8", &tpdisp8::entry);
    m.def("grouped_gemm_glu", &gg::entry_glu);
    m.def("grouped_gemm_cm", &gg::entry_cm);
    m.def("grouped_gemm_fp8", &gg8::entry);
    m.def("rowgroup_quant_fp8", &gg8::rowgroup_quant_entry);
    m.def("moe_tp_dispatch_push_gemm", &tppdisp::entry);
    m.def("moe_tp_gemm_prered_push", &preredpush::gemm_push_entry_tp);
    m.def("moe_tp_gemm_prered_push_v2", &tppr2::entry);
    m.def("moe_tp_gemm_prered_push_fp8", &tppr8::entry);
}
