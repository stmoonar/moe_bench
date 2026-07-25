/**
 * @file sm120_common.cuh
 * @brief Shared building blocks for the TileOverlap SM120 + PCIe MoE kernels.
 *
 * Two things live here:
 *   1. pcie_sync — cross-device synchronization that is legal on PCIe:
 *      plain st.release.sys writes into per-writer slots + local polling.
 *      NO remote atomics, NO multimem (neither exists on this platform).
 *   2. grouped_gemm_sm120 — a persistent grouped-GEMM device function built
 *      on warp-level mma.sync (SM120 has no wgmma/tcgen05), TMA loads and a
 *      3-stage smem pipeline sized for the 99KB smem budget. The producer
 *      warp calls a caller-supplied Gate before touching each row block,
 *      which is where communication readiness gets fused in.
 *
 * Design references: kernels/parallel/moe_dispatch_gemm/moe_dispatch_gemm_h100.cu
 * (structure) and experience/12_SM120与PCIe拓扑适配.md (platform constraints).
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
 * 2. SM120 grouped GEMM (persistent, TMA + warp-level mma.sync)
 * ======================================================================== */

struct gemm_config {
    // ROW_BLOCK is the tokens-per-tile AND the expert padding unit. Compile-time
    // switchable via -DTK_ROW_BLOCK (T5, docs/16): 128 (default) or 64. At 64 the
    // per-expert padding halves (NE=256: 64 real tokens no longer pad to 128), so
    // both GEMM layers compute ~half the rows. CONSUMER_WARPS = ROW_BLOCK/16 keeps
    // each warp on one 16-row strip (rt_fl<16,COL_BLOCK> accumulator). Note WG=4
    // (ROW_BLOCK=64) makes group::store's warpgroup-interleave map to identity
    // (w/4 + (w%4)*(WG/4) == w), unlike WG=8; the consumer load uses the same
    // store_strip formula so it stays correct either way (docs/05).
#ifndef TK_ROW_BLOCK
#define TK_ROW_BLOCK 128
#endif
    static constexpr int ROW_BLOCK = TK_ROW_BLOCK;
    static constexpr int COL_BLOCK = 128;  // output columns per tile
    static constexpr int RED_BLOCK = 64;   // K-dim chunk per pipeline stage
    static constexpr int PIPELINE_STAGES = 3; // 3 x 32KB = 96KB <= 99KB smem

    static constexpr int CONSUMER_WARPS = ROW_BLOCK / 16; // each owns a 16-row strip
    static constexpr int NUM_WARPS = CONSUMER_WARPS + 1; // +1 producer warp
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS; // 288 (WG=8) / 160 (WG=4)

    static constexpr int MAX_LOCAL_EXPERTS = 256;

    using A_tile = st_bf<ROW_BLOCK, RED_BLOCK>; // 16 KB (WG=8) / 8 KB (WG=4)
    using B_tile = st_bf<RED_BLOCK, COL_BLOCK>; // 16 KB

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    static constexpr int DYNAMIC_SHARED_MEMORY = PIPELINE_STAGES * sizeof(pipeline_inputs);
};

/**
 * Grouped GEMM over the local experts:
 *   outputs[row_range(e), :] = activations[row_range(e), :] @ weights[e, :, :]
 * where row ranges come from the (128-aligned) padded_tokens_per_expert prefix sums.
 *
 * Globals requirements (duck-typed):
 *   G.activations : gl<bf16, 1, 1, -1(tokens), -1(H), ..., gemm_config::A_tile>
 *   G.weights     : gl<bf16, 1, -1(E), -1(H), -1(I), gemm_config::B_tile>
 *   G.outputs     : gl<bf16, 1, 1, -1(tokens), -1(I)>   (register-path store, no TMA type needed)
 *   G.padded_tokens_per_expert : gl<int, 1, 1, 1, -1>   (global expert ids)
 *   G.num_local_experts, G.expert_offset : int
 *
 * Gate: `__device__ void operator()(int row_idx) const` — called by the
 * producer warp once per (row block, col block) task BEFORE issuing any load
 * for that task. Spin here until the 128-token row block `row_idx` is ready.
 * Pass a no-op for the compute-only baseline.
 *
 * Assumptions: H % (PIPELINE_STAGES * RED_BLOCK) == 0 is NOT required, but
 * H % RED_BLOCK == 0, I % COL_BLOCK == 0, and per-expert padding to
 * ROW_BLOCK are.
 */
template <typename Globals, typename Gate>
__device__ inline void grouped_gemm_sm120(const Globals &G, const Gate &gate, const int sm_idx, const int num_sms) {
    struct no_epilogue { __device__ inline void operator()(int, int) const {} };
    grouped_gemm_sm120(G, gate, no_epilogue{}, sm_idx, num_sms);
}

/* ---- output store policies (docs/30) -------------------------------------
 * The consumer group's register->global store is a policy so layer0 can fuse
 * the SwiGLU activation into the GEMM epilogue instead of a separate torch
 * silu pass over 75MB of HBM traffic. Called by EVERY consumer warp with its
 * own accumulator (group<CONSUMER_WARPS>::store composes the full tile). */

/** Default: store the full COL_BLOCK-wide fp32 accumulator as bf16. */
template <typename OutGL>
struct plain_store_policy {
    const OutGL &out;
    __device__ inline void operator()(rt_fl<16, gemm_config::COL_BLOCK> &acc,
                                      int row_idx, int col_idx) const {
        kittens::group<gemm_config::CONSUMER_WARPS>::store(out, acc, {row_idx, col_idx});
    }
};

/** SwiGLU store: weights are column-INTERLEAVED per COL_BLOCK so each output
 * tile is [gate(64) | up(64)] for the SAME intermediate columns. Compute
 * act = silu(gate) * up in fp32 registers and store the 64-wide act tile at
 * the same tile coordinate (act tensor is (rows, inter), 64-col tile units).
 * More accurate than the old path (silu on fp32 accs, not on rounded bf16).
 * NOTE: the A-load strip permutation (store_strip, docs/05) depends only on
 * CONSUMER_WARPS, not tile width, so acc rows line up for the 64-wide store
 * exactly as for the 128-wide one. */
template <typename OutGL>
struct glu_store_policy {
    const OutGL &out;
    __device__ inline void operator()(rt_fl<16, gemm_config::COL_BLOCK> &acc,
                                      int row_idx, int col_idx) const {
        constexpr int HW = gemm_config::COL_BLOCK / 32;  // half-width in 16-col base tiles
        rt_fl<16, gemm_config::COL_BLOCK / 2> act;
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
        kittens::group<gemm_config::CONSUMER_WARPS>::store(out, act, {row_idx, col_idx});
    }
};

/**
 * Same as above, plus an output Epilogue functor called by the consumer group
 * right after each (row_idx, col_idx) tile is stored to G.outputs. Signature:
 *   `__device__ void operator()(int row_idx, int col_idx) const`
 * All consumer threads reach it; the functor guards to a single thread as
 * needed. This is where layer1 fuses in the "tile written -> signal source
 * cards" step. Pass no_epilogue (default overload above) for plain GEMM.
 */
template <typename Globals, typename Gate, typename Epilogue>
__device__ inline void grouped_gemm_sm120(const Globals &G, const Gate &gate, const Epilogue &epilogue, const int sm_idx, const int num_sms) {
    using cfg = gemm_config;
    using consumers = kittens::group<cfg::CONSUMER_WARPS>;

    // Shared memory
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    typename cfg::pipeline_inputs (&inputs)[cfg::PIPELINE_STAGES] =
        allocator.allocate<typename cfg::pipeline_inputs, cfg::PIPELINE_STAGES>();

    // Per-expert padded token counts (small, cached in smem)
    __shared__ int padded_tokens_smem[cfg::MAX_LOCAL_EXPERTS];
    for (int i = threadIdx.x; i < G.num_local_experts; i += blockDim.x)
        padded_tokens_smem[i] = G.padded_tokens_per_expert[{G.expert_offset + i}];

    // Pipeline mbarriers
    __shared__ semaphore inputs_arrived[cfg::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[cfg::PIPELINE_STAGES];
    if (threadIdx.x == 0) {
        for (int i = 0; i < cfg::PIPELINE_STAGES; ++i) {
            init_semaphore(inputs_arrived[i], 0, 1);                   // TMA transaction count
            init_semaphore(inputs_finished[i], 0, cfg::CONSUMER_WARPS); // one arrive per consumer warp
        }
    }
    __syncthreads();

    // Common variables
    const int warp_id = kittens::warpid();
    const int lane_id = kittens::laneid();
    const int num_iters = static_cast<int>(G.activations.cols()) / cfg::RED_BLOCK;
    const int col_blocks = static_cast<int>(G.weights.cols()) / cfg::COL_BLOCK;
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;

    if (warp_id == cfg::CONSUMER_WARPS) {
        // ------------------------------------------------------ producer warp
        if (lane_id == 0) {
            for (int task_id = sm_idx, cum = 0, e = 0; e < G.num_local_experts; e++) {
                const int row_block_start = cum / cfg::ROW_BLOCK;
                cum += padded_tokens_smem[e];
                const int row_block_end = (cum + cfg::ROW_BLOCK - 1) / cfg::ROW_BLOCK;
                const int num_blocks = (row_block_end - row_block_start) * col_blocks;

                for (; task_id < num_blocks; task_id += num_sms) {
                    const int row_idx = task_id / col_blocks + row_block_start;
                    const int col_idx = task_id % col_blocks;

                    gate(row_idx); // <-- communication readiness fuses in here

                    for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                        wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                        update_phasebit<1>(phasebits, stage);
                        tma::expect_bytes(inputs_arrived[stage], sizeof(typename cfg::pipeline_inputs));
                        tma::load_async(inputs[stage].A, G.activations, {row_idx, red_idx}, inputs_arrived[stage]);
                        tma::load_async(inputs[stage].B, G.weights, {e, red_idx, col_idx}, inputs_arrived[stage]);
                        stage = (stage + 1) % cfg::PIPELINE_STAGES;
                    }
                }
                task_id -= num_blocks;
            }
        }
    } else {
        // ---------------------------------------------------- consumer warps
        for (int task_id = sm_idx, cum = 0, e = 0; e < G.num_local_experts; e++) {
            const int row_block_start = cum / cfg::ROW_BLOCK;
            cum += padded_tokens_smem[e];
            const int row_block_end = (cum + cfg::ROW_BLOCK - 1) / cfg::ROW_BLOCK;
            const int num_blocks = (row_block_end - row_block_start) * col_blocks;

            for (; task_id < num_blocks; task_id += num_sms) {
                const int row_idx = task_id / col_blocks + row_block_start;
                const int col_idx = task_id % col_blocks;

                rt_fl<16, cfg::COL_BLOCK> acc;
                warp::zero(acc);

                // group<8>::store permutes the row strip a warp writes to:
                // local_warpid = w/4 + (w%4)*(WARPS/4) (warpgroup interleave, since
                // CONSUMER_WARPS%4==0). We must therefore COMPUTE the rows we will
                // STORE, i.e. load A from the same permuted 16-row strip — otherwise
                // every warp but 0 lands in the wrong rows.
                constexpr int WG = cfg::CONSUMER_WARPS;
                const int store_strip = (WG % 4 == 0) ? (warp_id / 4 + (warp_id % 4) * (WG / 4)) : warp_id;

                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                    update_phasebit<0>(phasebits, stage);
                    #pragma unroll
                    for (int kk = 0; kk < cfg::RED_BLOCK / 16; kk++) {
                        rt_bf<16, 16> a_reg;
                        auto a_sub = inputs[stage].A.template subtile<16, 16>({store_strip, kk});
                        warp::load(a_reg, a_sub);
                        rt_bf<16, cfg::COL_BLOCK, ducks::rt_layout::col> b_reg;
                        auto b_sub = inputs[stage].B.template subtile<16, cfg::COL_BLOCK>({kk, 0});
                        warp::load(b_reg, b_sub);
                        warp::mma_AB(acc, a_reg, b_reg, acc);
                    }
                    warp::arrive(inputs_finished[stage]);
                    stage = (stage + 1) % cfg::PIPELINE_STAGES;
                }

                // Register -> global store (float->bf16 conversion happens inside).
                // Each consumer warp writes its own 16-row strip; group<8> composes 128x128.
                consumers::store(G.outputs, acc, {row_idx, col_idx});
                consumers::sync(0); // all strips of this tile are in global before we signal
                epilogue(row_idx, col_idx);
            }
            task_id -= num_blocks;
        }
    }
}

/**
 * Dispenser-fed grouped GEMM (docs/30). Same math/pipeline as the static-walk
 * grouped_gemm_sm120, but tasks come from a GLOBAL atomic counter so blocks
 * can join LATE: layer0's comm blocks finish the AllGather pulls and then
 * take GEMM tasks instead of idling (the static walk pre-assigns tasks by
 * sm_idx, which is why the 24 comm SMs used to sit dead for the L0 tail).
 *
 * Task space is flat: task t -> row block t/col_blocks, col block t%col_blocks;
 * the row block's expert comes from the blk_expert table (rebuilt with the
 * schedule, host-golden adjudicated). Claim order == the static walk's
 * expert-major order, so the readiness pipelining vs pull_order is unchanged.
 *
 * The producer streams claimed tasks to the consumer warps through a small
 * smem descriptor ring (TASK_Q=2, mbarrier handshake, same phasebit pattern
 * as the stage pipeline) — the 3-stage input pipeline runs CONTINUOUSLY
 * across task boundaries, exactly like the static walk (no per-task block
 * barrier, no pipeline drain). Sentinel row=-1 terminates the consumers.
 *
 * Store is a policy (plain_store_policy / glu_store_policy above); the
 * Globals only need .activations and .weights here.
 */
struct noop_epilogue { __device__ inline void operator()(int, int) const {} };

/* COL_MAJOR (docs/32, the Comet layer1-N lesson): task t sweeps COLUMN-outer
 * (all row blocks at col 0, then col 1, ...) so complete OUTPUT COLUMN SLICES
 * materialize early — the N-decomposed combine can start after ~1/col_blocks
 * of the GEMM instead of waiting for row blocks that finish last. Claim order
 * == completion order either way; only the t -> (row, col) map changes. */
template <bool COL_MAJOR = false, typename Globals, typename Gate, typename Epilogue, typename Store>
__device__ inline void grouped_gemm_sm120_dispenser(
        const Globals &G, const Gate &gate, const Epilogue &epilogue, const Store &store,
        const int *__restrict__ blk_expert, int *__restrict__ task_next, const int num_tasks) {
    using cfg = gemm_config;
    using consumers = kittens::group<cfg::CONSUMER_WARPS>;

    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);
    typename cfg::pipeline_inputs (&inputs)[cfg::PIPELINE_STAGES] =
        allocator.allocate<typename cfg::pipeline_inputs, cfg::PIPELINE_STAGES>();

    static constexpr int TASK_Q = 2;
    __shared__ semaphore inputs_arrived[cfg::PIPELINE_STAGES];
    __shared__ semaphore inputs_finished[cfg::PIPELINE_STAGES];
    __shared__ semaphore task_ready[TASK_Q];   // producer -> consumers
    __shared__ semaphore task_done[TASK_Q];    // consumers -> producer (slot free)
    __shared__ int2 task_desc[TASK_Q];         // (row_idx, col_idx); row -1 = exit
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
    const int col_blocks = static_cast<int>(G.weights.cols()) / cfg::COL_BLOCK;
    int stage = 0;
    uint32_t phasebits = 0xFFFF0000;   // stage pipeline (low: arrived, high: finished)
    uint32_t qphase    = 0xFFFF0000;   // task ring (low: ready, high: done-free)

    if (warp_id == cfg::CONSUMER_WARPS) {
        // ------------------------------------------------------ producer warp
        if (lane_id == 0) {
            const int nblk = num_tasks / col_blocks;
            int q = 0;
            while (true) {
                const int t = atomicAdd(task_next, 1);
                int row_idx = -1, col_idx = -1;
                if (t < num_tasks) {
                    if constexpr (COL_MAJOR) { row_idx = t % nblk; col_idx = t / nblk; }
                    else { row_idx = t / col_blocks; col_idx = t - (t / col_blocks) * col_blocks; }
                }
                // slot q free? (consumers finished the task TASK_Q rounds ago)
                wait(task_done[q], get_phasebit<1>(qphase, q));
                update_phasebit<1>(qphase, q);
                task_desc[q] = make_int2(row_idx, col_idx);
                warp::arrive(task_ready[q]);   // publish (mbarrier orders the smem write)
                if (row_idx < 0) break;
                gate(row_idx); // <-- communication readiness fuses in here
                const int e = blk_expert[row_idx];
                for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                    wait(inputs_finished[stage], get_phasebit<1>(phasebits, stage));
                    update_phasebit<1>(phasebits, stage);
                    tma::expect_bytes(inputs_arrived[stage], sizeof(typename cfg::pipeline_inputs));
                    tma::load_async(inputs[stage].A, G.activations, {row_idx, red_idx}, inputs_arrived[stage]);
                    tma::load_async(inputs[stage].B, G.weights, {e, red_idx, col_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % cfg::PIPELINE_STAGES;
                }
                q = (q + 1) % TASK_Q;
            }
        }
    } else {
        // ---------------------------------------------------- consumer warps
        int q = 0;
        while (true) {
            wait(task_ready[q], get_phasebit<0>(qphase, q));
            update_phasebit<0>(qphase, q);
            const int row_idx = task_desc[q].x;
            const int col_idx = task_desc[q].y;
            if (row_idx < 0) break;

            rt_fl<16, cfg::COL_BLOCK> acc;
            warp::zero(acc);
            constexpr int WG = cfg::CONSUMER_WARPS;
            const int store_strip = (WG % 4 == 0) ? (warp_id / 4 + (warp_id % 4) * (WG / 4)) : warp_id;

            for (int red_idx = 0; red_idx < num_iters; red_idx++) {
                wait(inputs_arrived[stage], get_phasebit<0>(phasebits, stage));
                update_phasebit<0>(phasebits, stage);
                #pragma unroll
                for (int kk = 0; kk < cfg::RED_BLOCK / 16; kk++) {
                    rt_bf<16, 16> a_reg;
                    auto a_sub = inputs[stage].A.template subtile<16, 16>({store_strip, kk});
                    warp::load(a_reg, a_sub);
                    rt_bf<16, cfg::COL_BLOCK, ducks::rt_layout::col> b_reg;
                    auto b_sub = inputs[stage].B.template subtile<16, cfg::COL_BLOCK>({kk, 0});
                    warp::load(b_reg, b_sub);
                    warp::mma_AB(acc, a_reg, b_reg, acc);
                }
                warp::arrive(inputs_finished[stage]);
                stage = (stage + 1) % cfg::PIPELINE_STAGES;
            }

            store(acc, row_idx, col_idx);
            consumers::sync(0); // full tile in global before the epilogue signal
            epilogue(row_idx, col_idx);
            warp::arrive(task_done[q]); // slot free for the producer
            q = (q + 1) % TASK_Q;
        }
    }
}

/* ==========================================================================
 * 3. SM120 FP8 grouped GEMM(docs/37 P1,分支 fp8_tp)
 *
 * 量化方案(DeepSeek 式):A 按 1×128 group(每行每 128 个 K 一个 fp32
 * scale),W 按 128×128 block。mma.sync m16n8k32 e4m3(experience/12)。
 *
 * 结构 = grouped_gemm_sm120_dispenser 的 fp8 变体:
 *  - **P1: K-tile = 128 == 量化块**(CUTLASS 87c blockwise 主循环同构,
 *    docs/08 §5): A/B tile 各 16KB, 3 stage 96KB ≤ 99KB; 每个 stage 一次
 *    fp32 重标定 acc += sub × (a_scale[row] × w_scale[kblk,cblk])。
 *    consumer 为 KK=4 步寄存器双缓冲主循环(kk+1 的 LDSM 先于 kk 的 QMMA
 *    发射, 下一 stage 的 wait/预取提到末尾 QMMA 之前), 重标定点与旧
 *    per-2-step 版一致 → 数值逐比特等价;
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
    static constexpr int COL_BLOCK = 128;
    static constexpr int RED_BLOCK = 128;            // K-tile == 量化块(P1)
    static constexpr int SCALE_K = 128;              // 量化块 K 宽
    static constexpr int PIPELINE_STAGES = 3;        // 3 × 32KB = 96KB ≤ 99KB
    static constexpr int MMA_K = 32;                 // m16n8k32

    static constexpr int CONSUMER_WARPS = ROW_BLOCK / 16;
    static constexpr int NUM_WARPS = CONSUMER_WARPS + 1;
    static constexpr int NUM_THREADS = NUM_WARPS * WARP_THREADS;

    using A_tile = st_fp8e4m3<ROW_BLOCK, RED_BLOCK>; // 16KB (RB=128)
    using B_tile = st_fp8e4m3<COL_BLOCK, RED_BLOCK>; // 16KB, B^T (N-major)

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    static constexpr int DYNAMIC_SHARED_MEMORY = PIPELINE_STAGES * sizeof(pipeline_inputs);
    static_assert(SCALE_K == RED_BLOCK, "P1: K-tile == 量化块, 每 stage 重标定一次");
    static_assert(PIPELINE_STAGES * ROW_BLOCK * RED_BLOCK * 2 <= 101376 - 2048,
                  "fp8 pipeline 3x32KB=96KB, 须给静态 smem 留余量(sm120 上限 101376)");
};

/**
 * FP8 dispenser grouped GEMM。Globals 额外要求(duck-typed):
 *   G.activations : gl<fp8e4m3, 1, 1, -1(rows), -1(K), cfg8::A_tile>
 *   G.weights     : gl<fp8e4m3, 1, -1(E), -1(N), -1(K), cfg8::B_tile>  (B^T)
 *   G.a_scales    : gl<float, 1, 1, -1(rows), -1(K/128)>
 *   G.w_scales    : gl<float, 1, -1(E), -1(N/128), -1(K/128)>
 * 其余(gate/epilogue/store/blk_expert/task_next)与 bf16 dispenser 相同。
 */
/* RESCALE=false = 裸 mma 吞吐探针(docs/38):跳过重标定 FFMA 与 scale 读,
 * 结果不正确,只用于测 fp8+fp32acc 的硬上限,裁决 "GeForce fp32 累加税"
 * 假说(mma.f32.e4m3 指令率 = f16 版的一半 → fp8 峰值 ≈ bf16 峰值)。 */
template <bool COL_MAJOR = false, bool RESCALE = true, typename Globals, typename Gate, typename Epilogue, typename Store>
__device__ inline void grouped_gemm_sm120_fp8_dispenser(
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
            const int nblk = num_tasks / col_blocks;
            int q = 0;
            while (true) {
                const int t = atomicAdd(task_next, 1);
                int row_idx = -1, col_idx = -1;
                if (t < num_tasks) {
                    if constexpr (COL_MAJOR) { row_idx = t % nblk; col_idx = t / nblk; }
                    else { row_idx = t / col_blocks; col_idx = t - (t / col_blocks) * col_blocks; }
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
                    tma::load_async(inputs[stage].A, G.activations, {row_idx, red_idx}, inputs_arrived[stage]);
                    // B^T: (E, N, K) 布局, tile 坐标 {N 块, K 块}
                    tma::load_async(inputs[stage].B, G.weights, {e, col_idx, red_idx}, inputs_arrived[stage]);
                    stage = (stage + 1) % cfg::PIPELINE_STAGES;
                }
                q = (q + 1) % TASK_Q;
            }
        }
    } else {
        // ---------------------------------------------------- consumer warps
        // P1 主循环(CUTLASS sm120 blockwise 同构, docs/08 §5): 每 stage 恰好一
        // 个量化块(K=128 = RED_BLOCK), KK=4 个 MMA step 寄存器双缓冲 —— kk+1
        // 的 LDSM 在 kk 的 QMMA 之前发射; 下一 stage 的 arrived wait + 首个
        // 预取放在本 stage 最后一条 QMMA 之前(与末尾 QMMA 重叠); finished
        // arrive 保持在最后一条 QMMA 之后(此时该 stage 的 LDSM 已全部被
        // QMMA 消费完毕, smem 可读覆)。重标定点与旧 per-2-step 版完全相同
        // (每 128 K), 块内 MMA 顺序一致(K 升序) → 数值逐比特等价。
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

            // scale 预取(docs/38: 首测耗时降低 21.3%, 边界处的 3 个 global scale 读
            // 在关键路径上, 32 个边界 × L2 延迟 ≈ 40% 气泡)。边界只消费
            // 已在寄存器的值, 同时发起下一块的加载(1 个 stage 的着陆窗)。
            float bsc_n = G.w_scales[{e, col_idx, 0}];
            float s0_n = G.a_scales[{r0, 0}];
            float s1_n = G.a_scales[{r0 + 8, 0}];

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
                    if (kk + 1 < KK) {
                        load_kk((kk + 1) & 1, stage, kk + 1);
                    } else if (red_idx + 1 < num_iters) {
                        // CUTLASS 序: 下一 stage 的 wait + 预取放在末尾 QMMA 前
                        wait(inputs_arrived[nxt], get_phasebit<0>(phasebits, nxt));
                        update_phasebit<0>(phasebits, nxt);
                        load_kk(0, nxt, 0);
                    }
                    warp::mma_ABt(sub, a_reg[kk & 1], b_reg[kk & 1], sub);
                }
                warp::arrive(inputs_finished[stage]);
                stage = nxt;

                // 每 stage = 一个量化块(K=128): fp32 重标定并入主累加器
                if constexpr (RESCALE) {
                    const float s0 = s0_n * bsc_n;
                    const float s1 = s1_n * bsc_n;
                    const int kblk1 = red_idx + 1;
                    if (kblk1 < num_iters) {  // 预取下一块
                        bsc_n = G.w_scales[{e, col_idx, kblk1}];
                        s0_n = G.a_scales[{r0, kblk1}];
                        s1_n = G.a_scales[{r0 + 8, kblk1}];
                    }
                    #pragma unroll
                    for (int j = 0; j < acc.width; j++) {
                        #pragma unroll
                        for (int k = 0; k < acc.tiles[0][j].packed_per_thread; k++) {
                            const float s = (k & 1) ? s1 : s0;
                            acc.tiles[0][j].data[k].x += sub.tiles[0][j].data[k].x * s;
                            acc.tiles[0][j].data[k].y += sub.tiles[0][j].data[k].y * s;
                        }
                    }
                    warp::zero(sub);
                }
            }

            if constexpr (RESCALE) store(acc, row_idx, col_idx);
            else store(sub, row_idx, col_idx);   // raw 探针: 存未标定值
            consumers::sync(0);
            epilogue(row_idx, col_idx);
            warp::arrive(task_done[q]);
            q = (q + 1) % TASK_Q;
        }
    }
}

} // namespace tileoverlap
