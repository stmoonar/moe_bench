/**
 * @file sm120_common.cuh
 * @brief Shared building blocks for the TileOverlap SM120 + PCIe MoE kernels.
 *
 * Two things live here:
 *   1. pcie_sync — cross-device synchronization that is legal on PCIe:
 *      plain st.release.sys writes into per-writer slots + local polling.
 *      NO remote atomics, NO multimem (neither exists on this platform).
 *   2. grouped_gemm_sm120_fp8_dispenser — the persistent FP8 grouped-GEMM
 *      device function every fused kernel is built on: warp-level mma.sync
 *      (SM120 has no wgmma/tcgen05), TMA loads, a 4-stage smem pipeline
 *      inside the 99KB budget, and an atomic task dispenser. The producer
 *      warp calls a caller-supplied Gate before touching each row block,
 *      which is where communication readiness gets fused in.
 *
 * 平台约束(无远端原子/无 multimem/TMA 是驱动 syscall)见 docs/04;
 * GEMM 主循环与寄存器预算的推导见 docs/03。
 */

#pragma once

#include "kittens.cuh"
#include "prototype.cuh"

namespace tileoverlap {

using namespace kittens;
using namespace kittens::prototype;

/* ==========================================================================
 * 1. PCIe-safe cross-device synchronization
 * ======================================================================== */

namespace pcie_sync {

/* 死锁红线(AGENTS.md, 2026-07-16 整机 wedge 事故): 所有跨卡/跨块自旋必须
 * 有界。~32s(5e8 次 nanosleep(64))等不到就 trap 杀死整个 kernel -> CUDA
 * error -> 进程干净退出。持久 kernel 的无界自旋一旦挂死不可抢占, 上下文
 * 销毁抱着 RM GPU 锁永不返回, nvidia-smi/新 CUDA 进程全部排队 -> 整机只能
 * 重启宿主机。合法等待最长 ~几 ms, 32s 裕量 1000 倍, miss 路径加一次计数
 * 零成本。 */
#define PCIE_SPIN_GUARD_DECL  long long _spin_guard = 0
#define PCIE_SPIN_GUARD_TICK  do { if (++_spin_guard > 500000000LL) asm volatile("trap;"); } while (0)

/* 有界 mbarrier 等待(红线第 2 条的 mbarrier 版): 跨卡 TMA pull 喂的
 * semaphore 在对端 rank 崩溃/被 kill 时永不 arrive, 普通 wait() 无界挂死
 * 且不经过任何自旋 guard(2026-07-17 审计缝隙: 所有 gate 已过、只剩尾部
 * dispatch 块挂在对端 TMA 上时, 传染性 trap 覆盖失效)。try_wait 是硬件
 * 挂起等待, 唤醒延迟与 wait() 相同, 就绪路径仅多一次 clock64 读; 超时按
 * GPU 时钟 1e11 周期(2.5GHz 下 ~40s, 与自旋 guard 同数量级) trap 杀
 * kernel -> CUDA error -> 进程干净退出。 */
constexpr long long PCIE_GUARD_WAIT_CYCLES = 100000000000LL;
__device__ static inline void guarded_wait(kittens::semaphore &sem, int phase) {
    const long long start = clock64();
    while (!kittens::try_wait(sem, phase))
        if (clock64() - start > PCIE_GUARD_WAIT_CYCLES) asm volatile("trap;");
}

/**
 * Slot-based all-device barrier. Legal on PCIe because every slot has exactly
 * one writer (plain release store, no atomics) and every wait polls local
 * memory only.
 *
 * Barrier tensor layout convention: row 1, column d = arrival slot written by
 * device d. Slots hold a monotonically increasing sequence number, so the
 * barrier never needs resetting; the caller passes a fresh `seq` (>=1,
 * strictly increasing) per use.
 *
 * Must be called by at least BAR::num_devices threads of one block.
 */
template <kittens::ducks::pgl::all BAR>
__device__ static inline void pcie_barrier_all(const BAR &bar, const int dev_idx, const int seq) {
    static_assert(!BAR::multicast, "pcie_barrier_all is for unicast (PCIe) barriers");
    constexpr int N = BAR::num_devices;
    if (threadIdx.x < N) {
        // announce my arrival on every device (including myself)
        asm volatile("st.release.sys.global.s32 [%0], %1;"
                     :: "l"(&bar[threadIdx.x][{1, dev_idx}]), "r"(seq) : "memory");
        // poll my local copy until every device has arrived
        int val = 0;
        PCIE_SPIN_GUARD_DECL;
        do {
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                         : "=r"(val) : "l"(&bar[dev_idx][{1, static_cast<int>(threadIdx.x)}]) : "memory");
            if (val < seq) { __nanosleep(64); PCIE_SPIN_GUARD_TICK; }
        } while (val < seq);
    }
    __syncthreads();
}

/**
 * Single-writer cross-device signal: write `seq` into `dst_dev`'s barrier
 * copy at (row, col). The (row, col) slot must be owned by exactly one
 * writer across the whole system.
 */
template <kittens::ducks::pgl::all BAR>
__device__ static inline void signal_slot(const BAR &bar, const int dst_dev, const int row, const int col, const int seq) {
    asm volatile("st.release.sys.global.s32 [%0], %1;"
                 :: "l"(&bar[dst_dev][{row, col}]), "r"(seq) : "memory");
}

/** Poll a local slot until it reaches `seq`. */
template <kittens::ducks::pgl::all BAR>
__device__ static inline void wait_slot(const BAR &bar, const int my_dev, const int row, const int col, const int seq) {
    int val = 0;
    PCIE_SPIN_GUARD_DECL;
    do {
        asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                     : "=r"(val) : "l"(&bar[my_dev][{row, col}]) : "memory");
        if (val < seq) { __nanosleep(64); PCIE_SPIN_GUARD_TICK; }
    } while (val < seq);
}

} // namespace pcie_sync

/* ==========================================================================
 * 1.5 tma_cta — .shared::cta 形态的 TMA load(docs/11)
 *
 * 2026-07-26 编译期试金石(tools/tma_litmus.cu)实锤: 本平台(sm_120a)上
 * cp.async.bulk.tensor 的 **.shared::cluster 目标形态走驱动 syscall**
 * (CALL.ABS.NOINC → __cuda_syscall_*, 每 kernel 预留 ~56 regs 调用帧,
 * docs/10 §6 的 168 寄存器帽根因), 而 **.shared::cta 目标形态是原生单条
 * UTMALDG**(REG:4, 零栈帧, 2d/4d 已验证; 5d 见 litmus ld5d_cta)。TK 上游
 * (子模块, 不可改)的 load_async 固定发射 cluster 形态, 这里复刻其地址/
 * 坐标计算, 仅把状态空间修饰符换成 cta。store 路径 TK 本来就是
 * .global.shared::cta(原生 UTMASTG/UBLKCP), 无需处理。
 *
 * 语义等价性: sm120 无 thread block cluster, CTA == cluster, 两种形态的
 * dst/mbar 操作数(CTA 本地 shared 地址)与完成语义(mbarrier complete_tx)
 * 完全一致, 仅指令编码不同。mbarrier 的 expect_tx(tma::expect_bytes)
 * 是原生 mbarrier 指令, 与拷贝指令的状态空间修饰符无耦合, 不用动。
 * ======================================================================== */

namespace tma_cta {

/** tile 版(swizzled → 5d, 否则 4d): 对应 kittens::tma::load_async 的
 *  NORMAL/dim::ROW 特化, 坐标计算原样复用 TK 的 detail::tma_coords。 */
template<ducks::st::all ST, ducks::gl::all GL, ducks::coord::tile COORD = coord<ST>>
__device__ static inline void load_async(ST &dst, const GL &src, const COORD &idx, semaphore &bar) {
    constexpr int AXIS = dim::ROW;
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<ST, AXIS>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    auto unit_coord = idx.template unit_coord<AXIS, 3>();
    if constexpr (ST::swizzle) {
        int4 tc = tma::detail::tma_coords<ST, AXIS>(unit_coord);
        asm volatile(
            "cp.async.bulk.tensor.5d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6, %7}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
              "n"(0), "r"(tc.x), "r"(tc.y), "r"(tc.z), "r"(tc.w)
            : "memory");
    } else {
        static_assert(AXIS == 2, "For non-swizzled tiles, only axis 2 is supported.");
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :
            : "r"(dst_ptr), "l"(tma_ptr), "r"(mbar_ptr),
              "r"(unit_coord.c), "r"(unit_coord.r), "r"(unit_coord.d), "r"(unit_coord.b)
            : "memory");
    }
}

/** vec 版(sv → 4d, 按 sv_tma_dim2 分片): 对应 kittens::tma::load_async
 *  的 sv 重载, 分片/偏移逻辑原样复用 TK 的 sv_tma_dim1/2。 */
template<ducks::sv::all SV, ducks::gl::all GL, ducks::coord::vec COORD = coord<SV>>
__device__ static inline void load_async(SV &dst, const GL &src, const COORD &idx, semaphore &bar) {
    coord<> unit_coord = idx.template unit_coord<-1, 3>();
    uint64_t tma_ptr  = reinterpret_cast<uint64_t>(src.template get_tma<SV, -1>());
    uint32_t mbar_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(&bar));
    uint32_t dst_ptr  = static_cast<uint32_t>(__cvta_generic_to_shared(&dst));
    for (int i = 0; i < ::kittens::detail::tma::sv_tma_dim2<SV>; i++) {
        coord<> tma_coord = unit_coord;
        tma_coord.c += i * ::kittens::detail::tma::sv_tma_dim1<SV>;
        uint32_t dst_i_ptr = dst_ptr + i * ::kittens::detail::tma::sv_tma_dim1<SV> * sizeof(typename SV::dtype);
        asm volatile(
            "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
            " [%0], [%1, {%3, %4, %5, %6}], [%2];"
            :
            : "r"(dst_i_ptr), "l"(tma_ptr), "r"(mbar_ptr),
              "r"(tma_coord.c), "r"(tma_coord.r), "r"(tma_coord.d), "r"(tma_coord.b)
            : "memory");
    }
}

} // namespace tma_cta

struct noop_epilogue { __device__ inline void operator()(int, int) const {} };

/* ROW_BLOCK = 每个 tile 的 token 行数, 同时是 expert 的 padding 单位。
 * 编译期开关 -DTK_ROW_BLOCK: 128(默认)或 64。64 时每 expert 的 padding
 * 减半, 但 B tile 重载翻倍 —— 实测净负(docs/09), 保留开关只为复现。
 * CONSUMER_WARPS = ROW_BLOCK/16 让每个 warp 固定负责一条 16 行带。 */
#ifndef TK_ROW_BLOCK
#define TK_ROW_BLOCK 128
#endif

/* ==========================================================================
 * 2. SM120 FP8 grouped GEMM(docs/10;本仓库唯一在用的 GEMM 引擎)
 *
 * 量化方案(DeepSeek 式):A 按 1×128 group(每行每 128 个 K 一个 fp32
 * scale),W 按 128×128 block。mma.sync m16n8k32 e4m3(SM120 无 wgmma)。
 *
 * 结构 = 原子 dispenser 发任务的持久 kernel:
 *  - **P1: K-tile = 128 == 量化块**(CUTLASS 87c blockwise 主循环同构,
 *    docs/08 §5): 每个 stage 一次 fp32 重标定 acc += sub × (a_scale[row]
 *    × w_scale[kblk,cblk]); 下一 stage 的 wait 提到末尾 QMMA 之前,
 *    重标定点与旧 per-2-step 版一致 → 数值逐比特等价;
 *  - **P2: COL_BLOCK = 64**(docs/10 §7): 本平台 TMA 是驱动 syscall(ABI
 *    call) → ptxas 预留 ~56 regs/thread, 有效上限 168 且 setmaxnreg 被
 *    忽略(C7506); 128 宽 tile 需求 ~230 必 spill acc 进 local。64 宽后
 *    acc/sub 各 32, A/B 全双缓冲总需求 ~135 ≤ 168。A tile 16KB + B tile
 *    8KB, 4 stage 96KB ≤ 99KB。GLU 配对改 [gate32|up32] 交织(同一 128
 *    列 scale 块内置换, 量化零改动); w_scales 按 col_idx>>1 取 128 列块;
 *  - **P3: 重标定切片交织**(docs/11 §3/§9, docs/12): 前一量化块的 FFMA
 *    重标定按 16 列 base-tile 切片, 移进下一 stage kk0 与其 QMMA 交错
 *    发射, 拆掉"末 QMMA→FFMA→清零→次 stage QMMA"的固定延迟 RAW 串行链
 *    (NCU stall_wait 第一大空转, 发射槽 60% 空闲可吸收); 每个元素的运算
 *    次序不变 → 逐比特等价;
 *  - scale 直接从 global 读(L2 广播,每 K 块每线程 3 个 float,不进
 *    smem,不动 TMA expect 字节数);
 *  - 行内 scale 映射:rt 行布局 data[偶] → 行 lane/4,data[奇] → +8
 *    (global_to_register.cuh 实测确认);
 *  - **B 用转置布局 (N, K) + mma_ABt**:TK 的 col-layout fp8 寄存器加载
 *    路径没写完(shared_to_register.cuh 对 fp8x4 用 .x/.y,编译不过),
 *    而 row-layout 走 ldmatrix 是通的;mma_ABt 的 fp8 特化在 SM120 齐全。
 *    副作用是好事:权重保持 w1 的原始 (E, N, K) 布局,免转置;
 *  - 输出 fp32 累加器 → 现有 store policy(plain / glu)直接复用。
 * ======================================================================== */

struct gemm_config_fp8 {
    static constexpr int ROW_BLOCK = TK_ROW_BLOCK;
    // P2: COL_BLOCK 128→64(docs/10 §7)。本平台 TMA 是驱动 syscall(ABI call)
    // → ptxas 预留 ~56 regs/thread → 有效上限 168(setmaxnreg 被 C7506 忽略)。
    // 128 宽 tile 需求 ~230(acc/sub 128 + frags + 寻址) 必 spill acc 进 local;
    // 64 宽后 acc/sub 各 32, 全双缓冲下总需求 ~135, 稳进 168。GLU 配对改
    // [gate32|up32] 交织(同一 128 列 scale 块内置换, 量化零改动)。
    static constexpr int COL_BLOCK = 64;
    static constexpr int RED_BLOCK = 128;            // K-tile == 量化块(P1)
    static constexpr int SCALE_K = 128;              // 量化块 K 宽
    static constexpr int PIPELINE_STAGES = 4;        // 4 × 24KB = 96KB ≤ 99KB
    static constexpr int MMA_K = 32;                 // m16n8k32

    static constexpr int CONSUMER_WARPS = ROW_BLOCK / 16;
    static constexpr int NUM_WARPS = CONSUMER_WARPS + 1;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    using A_tile = st_fp8e4m3<ROW_BLOCK, RED_BLOCK>; // 16KB (RB=128)
    using B_tile = st_fp8e4m3<COL_BLOCK, RED_BLOCK>; // 8KB, B^T (N-major)

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    static constexpr int DYNAMIC_SHARED_MEMORY = PIPELINE_STAGES * sizeof(pipeline_inputs);
    static_assert(SCALE_K == RED_BLOCK, "P1: K-tile == 量化块, 每 stage 重标定一次");
    static_assert(PIPELINE_STAGES * (ROW_BLOCK + COL_BLOCK) * RED_BLOCK <= 101376 - 2048,
                  "fp8 pipeline 4x24KB=96KB, 须给静态 smem 留余量(sm120 上限 101376)");
};

/* ---- output store policies -----------------------------------------------
 * The consumer group's register->global store is a policy so layer0 can fuse
 * the SwiGLU activation into the GEMM epilogue instead of a separate torch
 * silu pass over 75MB of HBM traffic. Called by EVERY consumer warp with its
 * own accumulator (group<CONSUMER_WARPS>::store composes the full tile). */

/** Default: store the full COL_BLOCK-wide fp32 accumulator as bf16.
 *  COL 由实参 rt 宽度推导(fp8 tile = 64, docs/10), 默认值只作占位。 */
template <typename OutGL>
struct plain_store_policy {
    const OutGL &out;
    template <int COL = gemm_config_fp8::COL_BLOCK>
    __device__ inline void operator()(rt_fl<16, COL> &acc,
                                      int row_idx, int col_idx) const {
        kittens::group<gemm_config_fp8::CONSUMER_WARPS>::store(out, acc, {row_idx, col_idx});
    }
};

/** SwiGLU store: weights are column-INTERLEAVED so each output tile holds
 * [gate | up] halves of the SAME intermediate columns ([gate32|up32] per
 * 64-col fp8 block, docs/10 §7). Compute act = silu(gate) * up in fp32
 * registers and store the half-width act tile at the same tile coordinate
 * (tile units follow rt width). More accurate than the old path (silu on
 * fp32 accs, not on rounded bf16).
 * NOTE: the A-load strip permutation (store_strip, docs/05) depends only on
 * CONSUMER_WARPS, not tile width, so acc rows line up for the half-width
 * store exactly as for the full-width one. */
template <typename OutGL>
struct glu_store_policy {
    const OutGL &out;
    template <int COL = gemm_config_fp8::COL_BLOCK>
    __device__ inline void operator()(rt_fl<16, COL> &acc,
                                      int row_idx, int col_idx) const {
        constexpr int HW = COL / 32;  // half-width in 16-col base tiles
        rt_fl<16, COL / 2> act;
        #pragma unroll
        for (int j = 0; j < HW; j++) {
            #pragma unroll
            for (int k = 0; k < acc.tiles[0][j].packed_per_thread; k++) {
                const float2 g = acc.tiles[0][j].data[k];
                const float2 u = acc.tiles[0][j + HW].data[k];
                float2 r;
                r.x = (g.x / (1.0f + __expf(-g.x))) * u.x;
                r.y = (g.y / (1.0f + __expf(-g.y))) * u.y;
                act.tiles[0][j].data[k] = r;
            }
        }
        kittens::group<gemm_config_fp8::CONSUMER_WARPS>::store(out, act, {row_idx, col_idx});
    }
};

/**
 * FP8 dispenser grouped GEMM。任务空间是扁平的: task t -> 行块 t/col_blocks、
 * 列块 t%col_blocks, 行块所属 expert 从 blk_expert 表查(与调度表一起重建,
 * 有 host golden 对拍)。任务由**全局原子 dispenser** 发放, 所以块可以迟到:
 * layer0 的 comm 块推完 AllGather 后转岗领 GEMM task, 而不是空转(docs/02)。
 * producer 通过小的 smem 描述符环(TASK_Q=2, mbarrier 握手)把领到的 task 交给
 * consumer warp, 输入流水线跨 task 边界连续推进(无 per-task 块内 barrier、
 * 无流水排空); row=-1 的哨兵终止 consumer。
 *
 * Globals 额外要求(duck-typed):
 *   G.activations : gl<fp8e4m3, 1, 1, -1(rows), -1(K), cfg8::A_tile>
 *   G.weights     : gl<fp8e4m3, 1, -1(E), -1(N), -1(K), cfg8::B_tile>  (B^T)
 *   G.a_scales    : gl<float, 1, 1, -1(rows), -1(K/128)>
 *   G.w_scales    : gl<float, 1, -1(E), -1(N/128), -1(K/128)>
 * Gate: `__device__ void operator()(int row_idx) const`, producer 在为该行块
 * 发起任何加载前调用 —— 通信就绪性就是在这里融进 GEMM 的(纯计算传 no_gate)。
 * Epilogue: tile 存完后由 consumer 组调用, layer1 在这里发"行块已写"信号。
 * Store: plain_store_policy / glu_store_policy。
 */
/* RESCALE=false = 裸 mma 吞吐探针(docs/03):跳过重标定 FFMA 与 scale 读,
 * 结果不正确,只用于测 fp8+fp32acc 的硬上限(tools/verify_fp8_gemm.py --raw)。 */
template <bool RESCALE = true, typename Globals, typename Gate, typename Epilogue, typename Store>
// 必须 __forceinline__: 若 dispenser 以 ABI 调用形式存在, ptxas 会忽略
// 函数体内的 setmaxnreg(C7506 'extern call', 2026-07-25 实测) → 寄存器
// 分配退回 168 + acc spill。
__device__ __forceinline__ void grouped_gemm_sm120_fp8_dispenser(
        const Globals &G, const Gate &gate, const Epilogue &epilogue, const Store &store,
        const int *__restrict__ blk_expert, int *__restrict__ task_next, const int num_tasks) {
    using cfg = gemm_config_fp8;
    using consumers = kittens::group<cfg::CONSUMER_WARPS>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    typename cfg::pipeline_inputs (&inputs)[cfg::PIPELINE_STAGES] =
        allocator.allocate<typename cfg::pipeline_inputs, cfg::PIPELINE_STAGES>();

    static constexpr int TASK_Q = 2;
    __shared__ semaphore inputs_arrived[cfg::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[cfg::PIPELINE_STAGES];
    __shared__ semaphore task_ready[TASK_Q];
    __shared__ semaphore task_done[TASK_Q];
    __shared__ int2 task_desc[TASK_Q];
    if (threadIdx.x == 0) {
        for (int i = 0; i < cfg::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);
            init_semaphore(inputs_finished[i], 0, cfg::CONSUMER_WARPS);
        }
        for (int q = 0; q < TASK_Q; ++q) {
            init_semaphore(task_ready[q], 0, 1);
            init_semaphore(task_done[q], 0, cfg::CONSUMER_WARPS);
        }
    }
    __syncthreads();

    const int warp_id = kittens::warpid();
    const int lane_id = kittens::laneid();
    const int num_iters = static_cast<int>(G.activations.cols()) / cfg::RED_BLOCK;
    // B^T 布局: weights (E, N, K) -> N 在 rows 维
    const int col_blocks = static_cast<int>(G.weights.rows()) / cfg::COL_BLOCK;
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;
    uint32_t qphase    = 0xFFFF0000;

    if (warp_id == cfg::CONSUMER_WARPS) {
        // ------------------------------------------------------ producer warp
        if (lane_id == 0) {
            int q = 0;
            while (true) {
                const int t = atomicAdd(task_next, 1);
                int row_idx = -1, col_idx = -1;
                if (t < num_tasks) {
                    row_idx = t / col_blocks;
                    col_idx = t - row_idx * col_blocks;
                }
                wait(task_done[q], get_phasebit<1>(qphase, q));
                update_phasebit<1>(qphase, q);
                task_desc[q] = make_int2(row_idx, col_idx);
                warp::arrive(task_ready[q]);
                if (row_idx < 0) break;
                gate(row_idx);
                const int e = blk_expert[row_idx];
                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(typename cfg::pipeline_inputs));
                    tma_cta::load_async(inputs[stage].A, G.activations, {row_idx, red_idx}, inputs_arrived[stage]);
                    // B^T: (E, N, K) 布局, tile 坐标 {N 块, K 块}
                    tma_cta::load_async(inputs[stage].B, G.weights, {e, col_idx, red_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % cfg::PIPELINE_STAGES;
                }
                q = (q + 1) % TASK_Q;
            }
        }
    } else {
        // ---------------------------------------------------- consumer warps
        // P1 主循环(CUTLASS sm120 blockwise 同构, docs/08 §5): 每 stage 恰好一
        // 个量化块(K=128 = RED_BLOCK), KK=4 个 MMA step, A/B 全双缓冲(P2
        // COL=64 后寄存器预算宽裕, docs/10 §7); 下一 stage 的 arrived wait
        // 提前到本 stage 最后一条 QMMA 之前(mbarrier 延迟全遮蔽); finished
        // arrive 保持在最后一条 QMMA 之后(此时该 stage 的 LDSM 已全部被
        // QMMA 消费完毕, smem 可读覆)。重标定点与旧 per-2-step 版完全相同
        // (每 128 K), 块内 MMA 顺序一致(K 升序) → 数值逐比特等价。
        // P3(交织): 重标定 FFMA 不再在 stage 边界串行执行, 而是移进下一
        // stage 的 kk0, 按 16 列 base-tile 切片与 QMMA 交错; 首块以
        // s0=s1=0 走同一路径, 尾块在循环外收尾。等待点/信号序完全不变。
        // 注: 不要在这里加 setmaxnreg —— 本平台 TMA 是驱动 syscall(ABI
        // call), ptxas C7506 全忽略, docs/10 §6。
        constexpr int KK = cfg::RED_BLOCK / cfg::MMA_K;   // 4
        int q = 0;
        while (true) {
            wait(task_ready[q], get_phasebit<0>(qphase, q));
            update_phasebit<0>(qphase, q);
            const int row_idx = task_desc[q].x;
            const int col_idx = task_desc[q].y;
            if (row_idx < 0) break;
            const int e = blk_expert[row_idx];

            rt_fl<16, cfg::COL_BLOCK> acc;    // 重标定后的主累加器
            rt_fl<16, cfg::COL_BLOCK> sub;    // 单个量化块(K=128)的子累加器
            warp::zero(acc);
            warp::zero(sub);
            constexpr int WG = cfg::CONSUMER_WARPS;
            const int store_strip = (WG % 4 == 0) ? (warp_id / 4 + (warp_id % 4) * (WG / 4)) : warp_id;
            // rt 行布局: data[偶] → 行 r0, data[奇] → r0+8(global_to_register)
            const int r0 = row_idx * cfg::ROW_BLOCK + store_strip * 16 + (lane_id >> 2);

            // scale 预取(docs/03: 首测耗时降低 21.3%, 边界处的 3 个 global scale 读
            // 在关键路径上, 32 个边界 × L2 延迟 ≈ 40% 气泡)。边界只消费
            // 已在寄存器的值, 同时发起下一块的加载(1 个 stage 的着陆窗)。
            // w_scales 块 = 128 列: 每个 128 列 scale 块含两个 64 列 GEMM tile
            float bsc_n, s0_n, s1_n;
            // P3: 待并入的"前一量化块"scale 积。首块用 s0=s1=0 走同一条交织
            // 路径(acc += 0×0, fma(+0,+0,+0)=+0 逐比特无操作), 免 per-iter 分支。
            float s0 = 0.0f, s1 = 0.0f;
            if constexpr (RESCALE) {
                bsc_n = G.w_scales[{e, col_idx >> 1, 0}];
                s0_n = G.a_scales[{r0, 0}];
                s1_n = G.a_scales[{r0 + 8, 0}];
            }
            (void)e;

            // COL=64 后寄存器预算宽裕(docs/10 §7: 总需求 ~135 ≤ 168),
            // 恢复 A/B 全双缓冲(P1 原设计): kk+1 的 LDSM 在 kk 的 QMMA 之前
            // 发射; 下一 stage 的 arrived wait + 首个预取提到末尾 QMMA 之前
            // (CUTLASS 序); finished arrive 保持在末尾 QMMA 之后(该 stage 的
            // LDSM 已全部被 QMMA 消费, smem 可读覆)。
            rt_fp8e4m3<16, cfg::MMA_K> a_reg[2];
            rt_fp8e4m3<cfg::COL_BLOCK, cfg::MMA_K> b_reg[2];
            auto load_kk = [&](int buf, int st, int kk) {
                auto a_sub = inputs[st].A.template subtile<16, cfg::MMA_K>({store_strip, kk});
                warp::load(a_reg[buf], a_sub);
                // B^T 行布局加载(ldmatrix 路径; col-layout fp8 加载在 TK
                // 里没写完), mma_ABt 的 fp8 特化做 (M,K)x(N,K)^T
                auto b_sub = inputs[st].B.template subtile<cfg::COL_BLOCK, cfg::MMA_K>({0, kk});
                warp::load(b_reg[buf], b_sub);
            };

            wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
            update_phasebit<0>(phasebits, stage);
            load_kk(0, stage, 0);

            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                const int nxt = (stage + 1) % cfg::PIPELINE_STAGES;
                #pragma unroll
                for (int kk = 0; kk < KK; kk++) {
                    if (RESCALE && kk == 0) {
                        // P3 交织(docs/11 §3/§9): 前一量化块的重标定按 16 列
                        // base-tile 切片, 与本块 kk0 的 QMMA 交错 ——
                        // FFMA(片j)→清零(片j)→QMMA(片j)。片 j 的 FFMA 只等
                        // 上一 stage 末 QMMA 中最早发射的两条原子(j 升序),
                        // 片 j 的 QMMA 与片 j+1 的 FFMA 无依赖、在发射流里
                        // 互相掩护。每元素运算次序与串行版一致(旧块 QMMA
                        // K 升序→FFMA→清零→新块 QMMA K 升序) → 逐比特等价。
                        // mma_ABt_base = mma_ABt 对 (n=0,m=j,k=0) 的同一原子
                        // 调用(TK warp.cuh: d.tiles[0][m] 配 b.tiles[m][0])。
                        // v2: load_kk(1) 移到交织块之后 —— v1 里它先发射,
                        // a/b_reg[1] (~20 regs) 在整个交织期被迫存活, 把
                        // gg8::kernel 顶到 168+spill(156→168, 8B, 实测判负
                        // -5/-10.6µs); 后置让交织期只有单套 frag 存活,
                        // kk1 LDSM 延迟由 kk0 的 8 条 QMMA 在 tensor 管线
                        // 的积压掩护。
                        #pragma unroll
                        for (int j = 0; j < acc.width; j++) {
                            #pragma unroll
                            for (int k = 0; k < acc.tiles[0][j].packed_per_thread; k++) {
                                const float s = (k & 1) ? s1 : s0;
                                acc.tiles[0][j].data[k].x += sub.tiles[0][j].data[k].x * s;
                                acc.tiles[0][j].data[k].y += sub.tiles[0][j].data[k].y * s;
                                sub.tiles[0][j].data[k].x = 0.0f;
                                sub.tiles[0][j].data[k].y = 0.0f;
                            }
                            warp::mma_ABt_base(sub.tiles[0][j], a_reg[0].tiles[0][0],
                                               b_reg[0].tiles[j][0], sub.tiles[0][j]);
                        }
                        load_kk(1, stage, 1);
                        continue;
                    }
                    if (kk + 1 < KK) {
                        load_kk((kk + 1) & 1, stage, kk + 1);
                    } else if (red_idx + 1 < num_iters) {
                        // CUTLASS 序: 下一 stage 的 wait + 预取提到末尾 QMMA 前
                        wait(inputs_arrived[nxt], get_phasebit<0>(phasebits, nxt));
                        update_phasebit<0>(phasebits, nxt);
                        load_kk(0, nxt, 0);
                    }
                    warp::mma_ABt(sub, a_reg[kk & 1], b_reg[kk & 1], sub);
                }
                warp::arrive(inputs_finished[stage]);
                stage = nxt;

                // P3: 边界只算下一次交织用的 scale 积 + 预取下一块的 3 个
                // global scale; FFMA 本体已移进下一块 kk0(尾块在循环外收尾)
                if constexpr (RESCALE) {
                    s0 = s0_n * bsc_n;
                    s1 = s1_n * bsc_n;
                    const int kblk1 = red_idx + 1;
                    if (kblk1 < num_iters) {  // 预取下一块
                        bsc_n = G.w_scales[{e, col_idx >> 1, kblk1}];
                        s0_n = G.a_scales[{r0, kblk1}];
                        s1_n = G.a_scales[{r0 + 8, kblk1}];
                    }
                }
            }

            if constexpr (RESCALE) {
                // P3 尾块重标定: 交织只并入"前一块", 最后一个量化块在此收尾
                // (sub 此后不再使用, 免清零)
                #pragma unroll
                for (int j = 0; j < acc.width; j++) {
                    #pragma unroll
                    for (int k = 0; k < acc.tiles[0][j].packed_per_thread; k++) {
                        const float s = (k & 1) ? s1 : s0;
                        acc.tiles[0][j].data[k].x += sub.tiles[0][j].data[k].x * s;
                        acc.tiles[0][j].data[k].y += sub.tiles[0][j].data[k].y * s;
                    }
                }
                store(acc, row_idx, col_idx);
            } else store(sub, row_idx, col_idx);   // raw 探针: 存未标定值
            consumers::sync(0);
            epilogue(row_idx, col_idx);
            warp::arrive(task_done[q]);
            q = (q + 1) % TASK_Q;
        }
    }
}

} // namespace tileoverlap
