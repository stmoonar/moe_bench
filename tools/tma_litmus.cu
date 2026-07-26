/**
 * @file tma_litmus.cu — 编译期试金石（只编译，**不要运行**，不占 GPU）。
 *
 * 目的：裁决本平台（sm_120a 工作站卡）ptxas 对各形态 cp.async.bulk{.tensor}
 * 发射的是原生 SASS 指令还是驱动 syscall（CALL.ABS → __cuda_syscall_*）。
 * docs/10 §6 实测：TK 现用的 4d/5d tile 形态是 syscall，且 ABI call 使 ptxas
 * 预留 ~56 regs/thread（有效上限 224→168），这是 COL_BLOCK=64 的根因。
 * 本文件逐形态各写一个空 kernel，编完看 SASS 即可判定哪条路是原生的。
 *
 * 用法（GPU 机器上编译，与生产 build 同一 nvcc；不需要空闲卡）：
 *   nvcc -arch=sm_120a -cubin --ptxas-options=-v \
 *        -o tma_litmus.cubin moe_bench/tools/tma_litmus.cu 2> tma_litmus.ptxas
 *   cuobjdump -sass      tma_litmus.cubin > tma_litmus.sass
 *   cuobjdump -res-usage tma_litmus.cubin > tma_litmus.res
 *   # 若 .shared::cta 目标的 load 变体编译报错（PTX 版本不支持），加 -DNO_CTA_DST
 *   # 重编——报错本身就是"cta 形态不可用"的结论，记下来即可。
 *
 * 判读（对每个 Function 的 SASS 正文）：
 *   - 出现 CALL.ABS.NOINC / __cuda_syscall_cp_async_bulk_* → syscall 路径；
 *   - 出现单条 UTMALDG / UTMASTG / UBLKCP 类指令、无 CALL → 原生路径。
 *   press_* 三个看 tma_litmus.res 的 REG/STACK：
 *   - press_none（无 TMA 对照）应 REG≈188、STACK=0（180 个活跃 fp32 塞得下 224 上限）；
 *   - press_2d 若 REG>168 且 STACK=0 → 2d 形态不吃 ABI 预留，COL=128 路线开绿灯；
 *   - press_4d 应复现 REG=168 + STACK>0（spill），证明本试金石对预留敏感。
 *
 * 注：这是编译期探针，不跑任何 benchmark，不涉及模型形状，故不读主配置 YAML。
 * kernel 都不做任何有意义的事（tensormap 是假的），仅为让 ptxas 生成指令。
 */

struct alignas(64) FakeTmap { unsigned long long q[16]; };  // 只要一个 64 位地址

__device__ __forceinline__ unsigned s32(const void *p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

#define SMEM_DECLS \
    __shared__ alignas(1024) char buf[4096]; \
    __shared__ alignas(8) unsigned long long mbar

/* ---------------- loads: global -> shared, mbarrier 完成通知 ---------------- */

__global__ void ld1d_cluster(const void *src) {
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
            " [%0], [%1], %2, [%3];"
            :: "r"(s32(buf)), "l"(src), "r"(4096), "r"(s32(&mbar)) : "memory");
}

__global__ void ld2d_cluster(const FakeTmap *tm) {
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)), "r"(0), "r"(0) : "memory");
}

__global__ void ld3d_cluster(const FakeTmap *tm) {
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)), "r"(0), "r"(0), "r"(0) : "memory");
}

__global__ void ld4d_cluster(const FakeTmap *tm) {   // TK 现用形态（对照, 应为 syscall）
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)),
               "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}

__global__ void ld5d_cluster(const FakeTmap *tm) {   // TK swizzle 形态（对照, 应为 syscall）
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)),
               "n"(0), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}

/* .shared::cta 目标变体（PTX 8.6+）：若原生而 cluster 形态是 syscall，
 * 修法退化成改 TK 的一行 asm 字符串，全 kernel 受益。 */
#ifndef NO_CTA_DST
__global__ void ld2d_cta(const FakeTmap *tm) {
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)), "r"(0), "r"(0) : "memory");
}

__global__ void ld4d_cta(const FakeTmap *tm) {
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)),
               "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}

__global__ void ld5d_cta(const FakeTmap *tm) {   // tma_cta 的 swizzled tile 路径(A/B 加载)
    SMEM_DECLS;
    if (threadIdx.x == 0)
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)),
               "n"(0), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
}
#endif

/* ---------------- stores: shared -> global, bulk_group 完成语义 ---------------- */

__global__ void st1d_cta(void *dst) {
    SMEM_DECLS;
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile(
            "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
            :: "l"(dst), "r"(s32(buf)), "r"(4096) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
}

__global__ void st2d_cta(const FakeTmap *tm) {
    SMEM_DECLS;
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile(
            "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3}], [%1];"
            :: "l"(tm), "r"(s32(buf)), "r"(0), "r"(0) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
}

__global__ void st4d_cta(const FakeTmap *tm) {   // TK 现用形态（对照）
    SMEM_DECLS;
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile(
            "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5}], [%1];"
            :: "l"(tm), "r"(s32(buf)), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
}

__global__ void st5d_cta(const FakeTmap *tm) {   // TK swizzle 形态（对照）
    SMEM_DECLS;
    if (threadIdx.x == 0) {
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile(
            "cp.async.bulk.tensor.5d.global.shared::cta.tile.bulk_group"
            " [%0, {%2, %3, %4, %5, %6}], [%1];"
            :: "l"(tm), "r"(s32(buf)), "n"(0), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
    }
}

/* ------------- 寄存器预留探针: 180 个活跃 fp32 + 循环内一次 TMA -------------
 * __launch_bounds__(288,1) 与生产 kernel 相同 → 静态上限 224。
 * 需求 ≈ 180 + 寻址/计数 ≈ 190: 168 上限必 spill(STACK>0), 224 上限则 0 spill。 */

#define NREG 180

__global__ __launch_bounds__(288, 1) void press_none(float *out, int n) {  // 无 TMA 对照
    __shared__ alignas(1024) char buf[4096];
    float acc[NREG];
    #pragma unroll
    for (int i = 0; i < NREG; i++) acc[i] = out[i];
    for (int it = 0; it < n; it++) {
        __syncthreads();
        const float *s = reinterpret_cast<const float *>(buf);
        #pragma unroll
        for (int i = 0; i < NREG; i++) acc[i] = fmaf(acc[i], 1.000001f, s[i & 255]);
    }
    #pragma unroll
    for (int i = 0; i < NREG; i++) out[i] = acc[i];
}

__global__ __launch_bounds__(288, 1) void press_2d(const FakeTmap *tm, float *out, int n) {
    SMEM_DECLS;
    float acc[NREG];
    #pragma unroll
    for (int i = 0; i < NREG; i++) acc[i] = out[i];
    for (int it = 0; it < n; it++) {
        if (threadIdx.x == 0)
            asm volatile(
                "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4}], [%2];"
                :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)), "r"(it), "r"(0) : "memory");
        __syncthreads();
        const float *s = reinterpret_cast<const float *>(buf);
        #pragma unroll
        for (int i = 0; i < NREG; i++) acc[i] = fmaf(acc[i], 1.000001f, s[i & 255]);
    }
    #pragma unroll
    for (int i = 0; i < NREG; i++) out[i] = acc[i];
}

__global__ __launch_bounds__(288, 1) void press_4d(const FakeTmap *tm, float *out, int n) {
    SMEM_DECLS;                                      // 对照: 应复现 REG=168 + spill
    float acc[NREG];
    #pragma unroll
    for (int i = 0; i < NREG; i++) acc[i] = out[i];
    for (int it = 0; it < n; it++) {
        if (threadIdx.x == 0)
            asm volatile(
                "cp.async.bulk.tensor.4d.shared::cluster.global.tile.mbarrier::complete_tx::bytes"
                " [%0], [%1, {%3, %4, %5, %6}], [%2];"
                :: "r"(s32(buf)), "l"(tm), "r"(s32(&mbar)),
                   "r"(it), "r"(0), "r"(0), "r"(0) : "memory");
        __syncthreads();
        const float *s = reinterpret_cast<const float *>(buf);
        #pragma unroll
        for (int i = 0; i < NREG; i++) acc[i] = fmaf(acc[i], 1.000001f, s[i & 255]);
    }
    #pragma unroll
    for (int i = 0; i < NREG; i++) out[i] = acc[i];
}
