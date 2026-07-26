/**
 * @file tk_moe.cu
 * @brief TK 通算融合 MoE 扩展(FP8, TP)—— tk_tp_scheme.TKFusedTP 的全部 kernel。
 *
 * 一个 PYBIND11_MODULE 里导出:
 *   - TKParallelTensor bindings (IPC symmetric buffers + broker)
 *   - pcie_device_barrier(barrier, seq)                 [跨卡 barrier]
 *   - rowgroup_quant_fp8(in, out, scales)               [1×128 group 量化]
 *   - grouped_gemm_fp8(...)                             [单卡 GEMM 裁决探针]
 *   - moe_tp_dispatch_gemm_fp8_push(...)                [layer0: AG ⊕ GEMM ⊕ SwiGLU]
 *   - moe_tp_gemm_prered_push_fp8(...)                  [layer1: W2 GEMM ⊕ 预归约 ⊕ push]
 *   - moe_final_reduce_push(...)                        [源卡最终归约]
 *
 * 数据流与协议见 docs/02, 优化账见 docs/03, 已判负的分支见 docs/04。
 * 按 world size / hidden 编译(-DTK_NUM_DEVICES=N -DTK_HIDDEN=H)。
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
 * 1. FP8 grouped GEMM 单卡入口: A 1×128 group 量化 + W 128×128 block
 *    量化, dispenser 结构, 输出 bf16。融合 kernel 用的是同一个引擎,
 *    这里只是把它单独暴露出来做正确性/效率裁决
 *    (tools/verify_fp8_gemm.py 对拍 fp32 反量化参考并计时)。
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
// 裸 mma 吞吐探针(docs/03): 跳过重标定, 结果错, 只测 fp8+f32acc 硬上限
__global__ __launch_bounds__(gemm_config_fp8::NUM_THREADS, 1)
void kernel_raw(const __grid_constant__ globals G, const int *__restrict__ blk_expert,
                int *__restrict__ task_next, const int num_tasks) {
    grouped_gemm_sm120_fp8_dispenser<false>(
        G, no_gate{}, noop_epilogue{},
        plain_store_policy<globals::outputs_gl>{G.outputs},
        blk_expert, task_next, num_tasks);
}
void entry(const at::Tensor &inputs, const at::Tensor &a_scales,
           const at::Tensor &weights, const at::Tensor &w_scales, at::Tensor &outputs,
           const at::Tensor &padded_tokens_per_expert, const at::Tensor &blk_expert,
           at::Tensor &task_next, const int expert_offset, const bool raw) {
    using cfg = gemm_config_fp8;
    // 布局(docs/02): weights = B^T (E, N, K)(w1 原始布局, 免转置),
    // w_scales (E, N/128, K/128); mma_ABt + row-layout 加载。
    TORCH_CHECK(inputs.size(0) % cfg::ROW_BLOCK == 0, "tokens % ROW_BLOCK");
    TORCH_CHECK(inputs.size(1) % cfg::SCALE_K == 0, "K % 128 (scale blocks)");
    TORCH_CHECK(weights.size(2) == inputs.size(1), "weights must be (E, N, K), K match");
    TORCH_CHECK(weights.size(1) % cfg::COL_BLOCK == 0, "N % 128");
    TORCH_CHECK(a_scales.size(0) == inputs.size(0) &&
                a_scales.size(1) == inputs.size(1) / cfg::SCALE_K,
                "a_scales must be (rows, K/128)");
    TORCH_CHECK(w_scales.size(1) == weights.size(1) / cfg::SCALE_K &&
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
// 1×128 row-group 量化(docs/03): bf16 (rows, groups*128) -> fp8 + scales
// (rows, groups)。torch 的五连发小 kernel 链要 ~80µs, 单 kernel 版 ~10-15µs。
// L0 的 token 量化(groups=H/128)与 L1 的 act 量化(groups=inter/128)共用。
// block = 一行, 8 warps 按 group 跨步, warp 内 shfl 归约 amax。
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
 * 2. 跨卡 barrier(每迭代包住 pre_tokens 的覆写窗口, 见 tk_tp_scheme.run)
 * ===================================================================== */
namespace disp {
using barrier_pgl = pgl<gl<int, 1, 1, -1, -1>, TK_NUM_DEVICES, false>;
__global__ __launch_bounds__(32)
void barrier_kernel(const __grid_constant__ barrier_pgl bar, const int dev_idx, const int seq) {
    pcie_sync::pcie_barrier_all(bar, dev_idx, seq);
}
void barrier_entry(kittens::py::TKParallelTensor &barrier, const int seq) {
    auto bar = kittens::py::parallel_tensor_to_pgl<barrier_pgl>(barrier);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    barrier_kernel<<<1, 32, 0, stream>>>(bar, barrier.local_rank_, seq);
    CUDACHECK(cudaGetLastError());
}
} // namespace disp

/* ===================================================================== *
 * 3. FP8 TP layer0: AllGather ⊕ grouped GEMM ⊕ SwiGLU, 单 kernel(docs/02)
 *   - 源侧 push(posted write): 每张卡按**消费序** push_order 把自己的
 *     fp8 token 行(4KB)+ 1×128 group scales(128B)写进 3 个 peer 的
 *     staging plane, 写完发 per-token 的 st.release.sys 到达 flag;
 *   - 收侧 scatter: 本地 acquire 等 flag, 从本地 staging 读一次, 散到该
 *     token 的 TOP_K 个 gathered slot, 每散完一个 slot 给行块计数 +1;
 *   - GEMM: grouped_gemm_sm120_fp8_dispenser + dispatch_gate_p(行块计数
 *     满 ROW_BLOCK 才放行), 输出经 glu_store_policy 在 fp32 累加器上算
 *     silu(gate)*up 直存 bf16 act;
 *   - comm 块推完自己的活就在命名 barrier 2 汇合后转岗领 GEMM task。
 *   pull 版 / per-lane 版 / warp 版 / copy-engine 版都已判负, 见 docs/04。
 * ===================================================================== */
namespace tpdisp8 {
struct pglobals {
    using cfg = gemm_config_fp8;
    static constexpr int NUM_DEVICES = TK_NUM_DEVICES;
    static constexpr int H = TK_HIDDEN;
    static constexpr int TOP_K = 8;
    static constexpr int NSC = H / 128;
    using token_vec = sv_fp8e4m3<H>;
    using scale_vec = sv_fl<NSC>;
    static constexpr int SLOTS = 20;             // 每 comm 块 in-flight 槽数
    using pre_tokens_pgl = pgl<gl<fp8e4m3, 1, 1, -1, H, token_vec>, NUM_DEVICES, false>;
    using pre_scales_pgl = pgl<gl<float, 1, 1, -1, NSC, scale_vec>, NUM_DEVICES, false>;
    using flags_pgl      = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    using gathered_gl    = gl<fp8e4m3, 1, 1, -1, H, token_vec, cfg::A_tile>;
    using gscales_gl     = gl<float, 1, 1, -1, NSC, scale_vec>;
    using weights_gl     = gl<fp8e4m3, 1, -1, -1, -1, cfg::B_tile>;
    using w_scales_gl    = gl<float, 1, -1, -1, -1>;
    using outputs_gl     = gl<bf16, 1, 1, -1, -1>;
    using counts_gl      = gl<int, 1, 1, 1, -1>;
    using slots_gl       = gl<int, 1, 1, -1, TOP_K>;
    using slack_gl       = gl<int, 1, 1, 1, -1>;
    using order_gl       = gl<int, 1, 1, 1, -1>;
    using porder_gl      = gl<int, 1, 1, -1, -1>;   // (world, T)
    using barrier_pgl    = pgl<gl<int, 1, 1, -1, -1>, NUM_DEVICES, false>;
    pre_tokens_pgl pre_tokens;    // 本 rank 量化行(源侧读 + 自己分片直读)
    pre_scales_pgl pre_scales;
    pre_tokens_pgl ag_staging;    // peer 可写 (S, H) fp8, plane s = 源 s 单写者
    pre_scales_pgl ag_sscales;    // peer 可写 (S, NSC)
    flags_pgl ag_flags;           // peer 可写 (1, S), 值 = 到达 seq
    gathered_gl activations;      // gathered(本地)
    gscales_gl a_scales;
    weights_gl weights;
    w_scales_gl w_scales;
    outputs_gl outputs;
    slots_gl tp_slots;
    slack_gl slack;
    order_gl pull_order;
    porder_gl push_order;
    barrier_pgl barrier;
    const int dev_idx;
    const int num_padded_local_tokens;
    const int num_tokens;
    const int s_max;
    const int num_comp_sms;
    const int num_push_sms;
    const int seq;
};
struct dispatch_gate_p {
    const pglobals &G;
    __device__ inline void operator()(int row_idx) const {
        int v;
        asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        PCIE_SPIN_GUARD_DECL;
        while (v != gemm_config_fp8::ROW_BLOCK) {
            __nanosleep(32); PCIE_SPIN_GUARD_TICK;
            asm volatile("{ld.relaxed.gpu.global.s32 %0, [%1];}" : "=r"(v) : "l"(&G.barrier[G.dev_idx][{row_idx}]) : "memory");
        }
    }
};
__device__ inline void push_lane(const pglobals &G, int *__restrict__ push_next,
                                 typename pglobals::token_vec &tok,
                                 typename pglobals::scale_vec &sc, semaphore &sem) {
    int phase = 0;
    while (true) {
        const int j = atomicAdd(push_next, 1);
        if (j >= G.num_tokens) return;
        const int t = G.push_order[{G.dev_idx, j}];   // 本源在各 dest 的消费序
        tma::expect_bytes(sem, sizeof(typename pglobals::token_vec) +
                               sizeof(typename pglobals::scale_vec));
        tma_cta::load_async(tok, G.pre_tokens[G.dev_idx], {t, 0}, sem);
        tma_cta::load_async(sc, G.pre_scales[G.dev_idx], {t, 0}, sem);
        pcie_sync::guarded_wait(sem, phase);
        phase ^= 1;
        const int d = G.dev_idx * G.num_tokens + t;
        #pragma unroll
        for (int dst = 0; dst < pglobals::NUM_DEVICES; dst++) {
            if (dst == G.dev_idx) continue;
            tma::store_async(G.ag_staging[dst], tok, {d, 0});
            tma::store_async(G.ag_sscales[dst], sc, {d, 0});
        }
        tma::store_async_wait();      // 本 token 的远端 bulk 写已提交
        __threadfence_system();
        #pragma unroll
        for (int dst = 0; dst < pglobals::NUM_DEVICES; dst++) {
            if (dst == G.dev_idx) continue;
            asm volatile("st.release.sys.global.s32 [%0], %1;"
                         :: "l"(&G.ag_flags[dst][{0, d}]), "r"(G.seq) : "memory");
        }
    }
}
__device__ inline void scatter_lane(const pglobals &G, int *__restrict__ pull_next,
                                    typename pglobals::token_vec &tok,
                                    typename pglobals::scale_vec &sc, semaphore &sem) {
    int phase = 0;
    while (true) {
        const int i = atomicAdd(pull_next, 1);
        if (i >= G.s_max) return;
        const int d = G.pull_order[{i}];
        const int src = d / G.num_tokens;
        const int t = d % G.num_tokens;
        if (src != G.dev_idx) {       // 远端行: 本地 acquire 自旋等到达
            // 有界自旋(踩坑加固): 协议 bug 导致 flag 永不到达时, ~30s 后
            // trap 杀死整个 kernel -> CUDA error -> 进程干净退出, 避免
            // 持久 kernel 自旋 wedge 整机(不可抢占 + IPC 级联, docs/06)。
            long long spins = 0;
            int v;
            do {
                asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                             : "=r"(v) : "l"(&G.ag_flags[G.dev_idx][{0, d}]) : "memory");
                if (v < G.seq) {
                    __nanosleep(64);
                    if (++spins > 500000000LL) asm volatile("trap;");
                }
            } while (v < G.seq);
        }
        tma::expect_bytes(sem, sizeof(typename pglobals::token_vec) +
                               sizeof(typename pglobals::scale_vec));
        if (src == G.dev_idx) {       // 自己分片直读(免 staging 一跳)
            tma_cta::load_async(tok, G.pre_tokens[src], {t, 0}, sem);
            tma_cta::load_async(sc, G.pre_scales[src], {t, 0}, sem);
        } else {                      // 本地 staging 读(~0.5µs, 无 PCIe RTT)
            tma_cta::load_async(tok, G.ag_staging[G.dev_idx], {d, 0}, sem);
            tma_cta::load_async(sc, G.ag_sscales[G.dev_idx], {d, 0}, sem);
        }
        pcie_sync::guarded_wait(sem, phase);
        phase ^= 1;
        #pragma unroll
        for (int k = 0; k < pglobals::TOP_K; k++) {
            const int s = G.tp_slots[{d, k}];
            if (s >= 0) {
                tma::store_async(G.activations, tok, {s, 0});
                tma::store_async(G.a_scales, sc, {s, 0});
            }
        }
        tma::store_async_wait();
        #pragma unroll
        for (int k = 0; k < pglobals::TOP_K; k++) {
            const int s = G.tp_slots[{d, k}];
            if (s >= 0)
                asm volatile("{red.release.gpu.global.add.s32 [%0], %1;}"
                             :: "l"(&G.barrier[G.dev_idx][{s / gemm_config_fp8::ROW_BLOCK}]), "r"(1) : "memory");
        }
    }
}
__global__ __launch_bounds__(gemm_config_fp8::NUM_THREADS, 1)
void kernel_push(const __grid_constant__ pglobals G, const int *__restrict__ blk_expert,
                 int *__restrict__ task_next, const int num_tasks,
                 int *__restrict__ push_next, int *__restrict__ pull_next) {
    using cfg = gemm_config_fp8;
    if (blockIdx.x >= G.num_comp_sms) {
        const int cb = blockIdx.x - G.num_comp_sms;
        extern __shared__ int __shm[];
        tma_swizzle_allocator al((int*)&__shm[0]);
        typename pglobals::token_vec (&tok)[pglobals::SLOTS] =
            al.allocate<typename pglobals::token_vec, pglobals::SLOTS>();
        typename pglobals::scale_vec (&sc)[pglobals::SLOTS] =
            al.allocate<typename pglobals::scale_vec, pglobals::SLOTS>();
        __shared__ semaphore arrived[pglobals::SLOTS];
        const bool is_slot = (threadIdx.x % 8 == 0) &&
                             (threadIdx.x / 8 < pglobals::SLOTS);
        const int slot = threadIdx.x / 8;
        if (is_slot)
            init_semaphore(arrived[slot], 0, 1);
        __syncthreads();
        if (is_slot) {
            if (cb < G.num_push_sms)
                push_lane(G, push_next, tok[slot], sc[slot], arrived[slot]);
            else
                scatter_lane(G, pull_next, tok[slot], sc[slot], arrived[slot]);
        }
        kittens::group<cfg::NUM_WARPS>::sync(2);  // 全块汇合后转岗(docs/04 barrier 纪律)
    }
    grouped_gemm_sm120_fp8_dispenser(G, dispatch_gate_p{G}, noop_epilogue{},
                                     glu_store_policy<pglobals::outputs_gl>{G.outputs},
                                     blk_expert, task_next, num_tasks);
}
__global__ __launch_bounds__(256)
void reset_kernel_p(const __grid_constant__ pglobals G) {
    const int nb = (G.num_padded_local_tokens + gemm_config_fp8::ROW_BLOCK - 1) / gemm_config_fp8::ROW_BLOCK;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < nb; i += gridDim.x * blockDim.x)
        G.barrier[G.dev_idx][{i}] = G.slack[{i}];
}
void entry_push(kittens::py::TKParallelTensor &pre_tokens, kittens::py::TKParallelTensor &pre_scales,
           kittens::py::TKParallelTensor &ag_staging, kittens::py::TKParallelTensor &ag_sscales,
           kittens::py::TKParallelTensor &ag_flags,
           at::Tensor &gathered, at::Tensor &gathered_scales,
           at::Tensor &weights, at::Tensor &w_scales, at::Tensor &act,
           at::Tensor &padded_tokens_per_expert, at::Tensor &tp_slots,
           at::Tensor &slack, at::Tensor &pull_order, at::Tensor &push_order,
           at::Tensor &blk_expert, at::Tensor &gemm_next,
           at::Tensor &push_next, at::Tensor &pull_next,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_push_sms,
           const int num_padded_local_tokens, const int num_tokens, const int seq) {
    using cfg = gemm_config_fp8;
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts, "weights (E,N,K) E mismatch");
    TORCH_CHECK(weights.size(2) == pglobals::H, "weights (E,N,K) K must be H");
    TORCH_CHECK(act.size(1) == weights.size(1) / 2, "act width must be N/2 (GLU)");
    TORCH_CHECK(gathered_scales.size(1) == pglobals::NSC, "gathered_scales (P, H/128)");
    TORCH_CHECK(gemm_next.numel() == 1 && push_next.numel() == 1 && pull_next.numel() == 1, "counters");
    TORCH_CHECK(num_comm_sms >= 2 && num_push_sms >= 1 && num_push_sms < num_comm_sms,
                "need >=1 push SM and >=1 scatter SM");
    const int s_max = static_cast<int>(tp_slots.size(0));
    TORCH_CHECK(s_max == pglobals::NUM_DEVICES * num_tokens, "tp_slots rows must be world*T");
    TORCH_CHECK(ag_staging.data_.size(0) == s_max, "ag_staging rows must be world*T");
    TORCH_CHECK(ag_flags.data_.numel() >= s_max, "ag_flags must cover world*T");
    TORCH_CHECK(push_order.size(0) == pglobals::NUM_DEVICES && push_order.size(1) == num_tokens,
                "push_order must be (world, T)");
    int sm; CUDACHECK(cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev_idx));
    TORCH_CHECK(num_comm_sms < sm, "num_comm_sms must leave room for compute");
    const int nblk = num_padded_local_tokens / cfg::ROW_BLOCK;
    TORCH_CHECK(blk_expert.size(0) == nblk, "blk_expert per row block");
    const int num_tasks = nblk * (static_cast<int>(weights.size(1)) / cfg::COL_BLOCK);
    pglobals G {
        .pre_tokens = kittens::py::parallel_tensor_to_pgl<pglobals::pre_tokens_pgl>(pre_tokens),
        .pre_scales = kittens::py::parallel_tensor_to_pgl<pglobals::pre_scales_pgl>(pre_scales),
        .ag_staging = kittens::py::parallel_tensor_to_pgl<pglobals::pre_tokens_pgl>(ag_staging),
        .ag_sscales = kittens::py::parallel_tensor_to_pgl<pglobals::pre_scales_pgl>(ag_sscales),
        .ag_flags = kittens::py::parallel_tensor_to_pgl<pglobals::flags_pgl>(ag_flags),
        .activations = kittens::py::tensor_to_gl<pglobals::gathered_gl>(gathered),
        .a_scales = kittens::py::tensor_to_gl<pglobals::gscales_gl>(gathered_scales),
        .weights = kittens::py::tensor_to_gl<pglobals::weights_gl>(weights),
        .w_scales = kittens::py::tensor_to_gl<pglobals::w_scales_gl>(w_scales),
        .outputs = kittens::py::tensor_to_gl<pglobals::outputs_gl>(act),
        .tp_slots = kittens::py::tensor_to_gl<pglobals::slots_gl>(tp_slots),
        .slack = kittens::py::tensor_to_gl<pglobals::slack_gl>(slack),
        .pull_order = kittens::py::tensor_to_gl<pglobals::order_gl>(pull_order),
        .push_order = kittens::py::tensor_to_gl<pglobals::porder_gl>(push_order),
        .barrier = kittens::py::parallel_tensor_to_pgl<pglobals::barrier_pgl>(barrier),
        .dev_idx = dev_idx,
        .num_padded_local_tokens = num_padded_local_tokens, .num_tokens = num_tokens,
        .s_max = s_max, .num_comp_sms = sm - num_comm_sms,
        .num_push_sms = num_push_sms, .seq = seq
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = pglobals::SLOTS *
        (sizeof(pglobals::token_vec) + sizeof(pglobals::scale_vec)) + 2048 >
        cfg::DYNAMIC_SHARED_MEMORY + 1024
        ? pglobals::SLOTS * (sizeof(pglobals::token_vec) + sizeof(pglobals::scale_vec)) + 2048
        : cfg::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(kernel_push, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kernel_push<<<sm, cfg::NUM_THREADS, smem, stream>>>(
        G, blk_expert.data_ptr<int>(), gemm_next.data_ptr<int>(), num_tasks,
        push_next.data_ptr<int>(), pull_next.data_ptr<int>());
    CUDACHECK(cudaGetLastError());
    const int rb = (num_padded_local_tokens / cfg::ROW_BLOCK + 255) / 256 + 1;
    reset_kernel_p<<<rb, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace tpdisp8
/* ===================================================================== *
 * 4. 源卡最终归约(TP layer1 push 的收端): 等每张卡的 watermark, 把
 *    world 个 partial plane 加起来得到本卡 token 的最终输出。
 * ===================================================================== */
namespace preredpush {
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
// grid-stride + 入口限栅格(docs/04): 满铺 block-per-token 会占满 SM 并
// 自旋等 watermark, 饿死同设备的其它辅助 kernel -> 死锁。
__global__ void final_reduce_push_kernel(const __grid_constant__ final_globals G) {
    for (int t = blockIdx.x; t < G.num_source_tokens; t += gridDim.x) {
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
    __syncthreads();  // 下一条带复用 s_contrib 前, 本条带读完
    }
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
    // grid 限到 sm-2(docs/06): 满铺自旋会饿死同设备的其它辅助 kernel
    int sm_frp; CUDACHECK(cudaDeviceGetAttribute(&sm_frp, cudaDevAttrMultiProcessorCount, dev_idx));
    const int frp_blocks = std::min(num_source_tokens, sm_frp - 2);
    final_reduce_push_kernel<<<frp_blocks, 256, 0, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}
} // namespace preredpush

/* ===================================================================== *
 * 5. FP8 TP layer1: W2 GEMM ⊕ 本地 top-k 预归约 ⊕ 稠密 ReduceScatter
 *    push, 单 kernel(docs/02)。
 *    - A = act 经 rowgroup_quant_fp8(P, inter -> inter/128 组/行);
 *      B^T = problem.w2 原布局 (E, H, inter) = (E, N, K), qc.w2_scale
 *      (E, N/128, K/128) 原样可用 —— 无转置、无重量化、无二次量化误差;
 *    - signal_epilogue: 行块的所有列块写完后, 由本地选举出的唯一写者
 *      st.release.gpu 写"行块就绪"信号(不依赖远端原子, docs/04);
 *    - push_job: comm 块按 job_order(= 就绪序)领 job, 等齐 8 个 slot 的
 *      行块信号 -> fp32 加权求和 -> TMA 推源卡 staging -> 本地计数选举
 *      -> 最后一条发 watermark。GEMM 出完的块也全员加入排空。
 *    N 维分解(Comet layer1-N)在本平台是负结果, 见 docs/04。
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
    using partial_gl     = gl<float, 1, 1, -1, H>;                   // combine_partial (num_jobs, H) fp32
    using sjob_gl        = gl<int, 1, 1, 1, -1>;                     // slot_job (P,)
    using sw_gl          = gl<float, 1, 1, 1, -1>;                   // slot_w (P,)
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
    partial_gl partial;
    sjob_gl slot_job;
    sw_gl slot_w;
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
    const int use_epired;
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
// EPIRED store policy(TK_L1_EPIRED): C tile 不再写 expert_out, 在寄存器里
// 乘 w 后直接 red.add 进 combine_partial 的对应 job 行 —— 省掉 expert_out
// 写 134MB + push_job 重读 8 行 134MB(H=4096, P=16K 时)。red 是
// fire-and-forget(无返回、无等待), 可见性由 signal_epilogue 既有的
// threadfence + 行块放行信号保证; padding 行 slot_job=-1 跳过。
// rt 布局(docs/10, global_to_register 实测): data[k] 偶 → 行 r0, 奇 → r0+8;
// k>>1 → 列 +8; float2 = 相邻 2 列。
struct wred_store_policy {
    const globals &G;
    template <int COL = gemm_config_fp8::COL_BLOCK>
    __device__ inline void operator()(rt_fl<16, COL> &acc,
                                      int row_idx, int col_idx) const {
        if (!G.use_epired) {
            kittens::group<gemm_config_fp8::CONSUMER_WARPS>::store(
                G.outputs, acc, {row_idx, col_idx});
            return;
        }
        constexpr int WG = gemm_config_fp8::CONSUMER_WARPS;
        const int warp_id = kittens::warpid();
        const int lane = kittens::laneid();
        const int strip = (WG % 4 == 0) ? (warp_id / 4 + (warp_id % 4) * (WG / 4)) : warp_id;
        const int r0 = row_idx * gemm_config_fp8::ROW_BLOCK + strip * 16 + (lane >> 2);
        const int job0 = G.slot_job[{r0}];
        const int job1 = G.slot_job[{r0 + 8}];
        const float w0 = G.slot_w[{r0}];
        const float w1 = G.slot_w[{r0 + 8}];
        const int c0 = col_idx * COL + (lane & 3) * 2;
        #pragma unroll
        for (int j = 0; j < acc.width; j++) {
            #pragma unroll
            for (int k = 0; k < acc.tiles[0][j].packed_per_thread; k++) {
                const int job = (k & 1) ? job1 : job0;
                if (job < 0) continue;
                const float w = (k & 1) ? w1 : w0;
                float *dst = &G.partial[{job, c0 + j * 16 + ((k >> 1) << 3)}];
                const float2 v = acc.tiles[0][j].data[k];
                asm volatile("red.global.add.f32 [%0], %1;" :: "l"(dst), "f"(v.x * w) : "memory");
                asm volatile("red.global.add.f32 [%0], %1;" :: "l"(dst + 1), "f"(v.y * w) : "memory");
            }
        }
    }
};
// push_job: 行块就绪后做本地 top-k 加权归约(EPIRED 时归约已在 GEMM
// epilogue 完成, 这里只读 fp32 部分和转 bf16 并顺手清零, 供下一迭代直接用),
// 把整行推到源卡 staging, 最后一条到达时由本地选举出的唯一写者发
// watermark(源卡 final reduce 等它)。
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

    // 归约结果先落 smem 行, 再 TMA 推远端
    float4 *row_v = reinterpret_cast<float4 *>(reinterpret_cast<bf16 *>(&row));
    if (G.use_epired) {
        // epilogue 已把 w 乘进 partial; 读 fp32 行(16KB)转 bf16, 读位清零。
        // 本卡私有缓冲, 唯一读者就是 push_job, 无竞争; 清零保证下一迭代从 0 累加。
        float4 *p_v = reinterpret_cast<float4 *>(&G.partial[{j, 0}]);
        constexpr int HF8 = H / 8;   // 每迭代 8 个 fp32 -> 16B bf16
        for (int c = threadIdx.x; c < HF8; c += blockDim.x) {
            float4 lo = p_v[2 * c], hi = p_v[2 * c + 1];
            bf16_2 res[4];
            res[0] = __floats2bfloat162_rn(lo.x, lo.y);
            res[1] = __floats2bfloat162_rn(lo.z, lo.w);
            res[2] = __floats2bfloat162_rn(hi.x, hi.y);
            res[3] = __floats2bfloat162_rn(hi.z, hi.w);
            row_v[c] = *reinterpret_cast<const float4 *>(res);
            p_v[2 * c]     = make_float4(0.f, 0.f, 0.f, 0.f);
            p_v[2 * c + 1] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    } else {
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
            wred_store_policy{G},
            blk_expert, gemm_next, nblk * col_blocks);
        kittens::group<gemm_config_fp8::NUM_WARPS>::sync(2);  // 专用命名 barrier(docs/04)
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
           at::Tensor &expert_outputs,
           at::Tensor &padded_tokens_per_expert,
           at::Tensor &partial, at::Tensor &slot_job, at::Tensor &slot_w,
           kittens::py::TKParallelTensor &staging, at::Tensor &prered_dst,
           at::Tensor &prered_slots, at::Tensor &prered_w, at::Tensor &local_cnt,
           at::Tensor &push_expected_l1, at::Tensor &blk_expert,
           at::Tensor &gemm_next, at::Tensor &job_order, at::Tensor &job_next,
           kittens::py::TKParallelTensor &barrier,
           const int num_comm_sms, const int num_padded_local_tokens,
           const int num_source_tokens, const int num_jobs, const int seq,
           const int use_epired) {
    using cfg = gemm_config_fp8;
    const int dev_idx = barrier.local_rank_;
    const int num_local_experts = static_cast<int>(padded_tokens_per_expert.size(0));
    TORCH_CHECK(weights.size(0) == num_local_experts, "w2 (E,H,inter) E mismatch");
    TORCH_CHECK(weights.size(1) == globals::H, "w2 rows must be H (B^T)");
    TORCH_CHECK(weights.size(2) == act_fp8.size(1), "w2 K must match act inter");
    TORCH_CHECK(act_fp8.size(1) % cfg::SCALE_K == 0, "inter % 128");
    TORCH_CHECK(act_scales.size(1) == act_fp8.size(1) / cfg::SCALE_K, "act_scales (P, inter/128)");
    TORCH_CHECK(w_scales.size(1) == globals::H / cfg::SCALE_K &&
                w_scales.size(2) == static_cast<int>(weights.size(2)) / cfg::SCALE_K,
                "w2_scales must be (E, H/128, inter/128)");
    TORCH_CHECK(gemm_next.numel() == 1 && job_next.numel() == 1, "counters");
    TORCH_CHECK(job_order.size(0) == num_jobs, "job_order");
    TORCH_CHECK(partial.size(0) == num_jobs && partial.size(1) == globals::H &&
                partial.scalar_type() == at::ScalarType::Float,
                "partial must be fp32 (num_jobs, H)");
    TORCH_CHECK(slot_job.numel() >= num_padded_local_tokens &&
                slot_w.numel() >= num_padded_local_tokens,
                "slot_job/slot_w must cover all padded rows");
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
        .partial = kittens::py::tensor_to_gl<globals::partial_gl>(partial),
        .slot_job = kittens::py::tensor_to_gl<globals::sjob_gl>(slot_job),
        .slot_w = kittens::py::tensor_to_gl<globals::sw_gl>(slot_w),
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
        .num_comp_sms = num_comp_sms, .seq = seq, .use_epired = use_epired
    };
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    constexpr int smem = cfg::DYNAMIC_SHARED_MEMORY + 1024;   // 96KB GEMM > 8KB push row
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
    m.def("pcie_device_barrier", &disp::barrier_entry);
    m.def("grouped_gemm_fp8", &gg8::entry);
    m.def("rowgroup_quant_fp8", &gg8::rowgroup_quant_entry);
    m.def("moe_tp_dispatch_gemm_fp8_push", &tpdisp8::entry_push);
    m.def("moe_tp_gemm_prered_push_fp8", &tppr8::entry);
    m.def("moe_final_reduce_push", &preredpush::final_reduce_push_entry);
}
