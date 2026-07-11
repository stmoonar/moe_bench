/**
 * @file probe.cu
 * @brief Single-process multi-GPU platform probe for SM120 + PCIe machines.
 *
 * Answers, with numbers, the questions everything downstream depends on:
 *   [A] Is P2P available between each device pair? (attributes + real access)
 *   [B] Do remote atomics work over this fabric? (expected: NO on PCIe —
 *       this probe *verifies* the failure instead of assuming it)
 *   [C] Peer bandwidth: copy engine (cudaMemcpyPeerAsync) vs SM-driven
 *       pull/push, and how many blocks it takes to saturate.
 *   [D] Cross-device signal round-trip latency (st.release.sys / ld.acquire.sys
 *       ping-pong) — the cost model for every tile-granularity signal.
 *
 * Pure CUDA runtime, no ThunderKittens, no torch. Run: ./probe
 * NOTE: uses cudaDeviceEnablePeerAccess (runtime P2P). TK itself maps peer
 * memory via VMM+IPC across processes; both paths require the same fabric
 * capability, so this is a valid (if not byte-identical) proxy.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while (0)

/* ------------------------------------------------------------------ [C] -- */

__global__ void copy_kernel(const float4 *__restrict__ src, float4 *__restrict__ dst, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride)
        dst[i] = src[i];
}

/* ------------------------------------------------------------------ [B] -- */

__global__ void atomic_dev_probe(int *peer_counter, int iters) {
    for (int i = 0; i < iters; i++)
        atomicAdd(peer_counter, 1); // device-scope atomic on peer memory
}

__global__ void atomic_sys_probe(int *peer_counter, int iters) {
    for (int i = 0; i < iters; i++)
        atomicAdd_system(peer_counter, 1); // system-scope atomic on peer memory
}

__global__ void red_sys_probe(int *peer_counter, int iters) {
    for (int i = 0; i < iters; i++)
        asm volatile("red.release.sys.global.add.s32 [%0], %1;" :: "l"(peer_counter), "r"(1) : "memory");
}

/* ------------------------------------------------------------------ [D] -- */

// Two single-thread kernels, one per device. Each writes a sequence number
// into the *other* device's flag and polls its *own* flag (local memory).
__global__ void pingpong_kernel(int *my_flag, int *peer_flag, int iters, int am_first, int *error_out) {
    const unsigned long long timeout = 20ULL * 1000ULL * 1000ULL * 1000ULL; // ~seconds at GHz clocks
    for (int i = 1; i <= iters; i++) {
        if (am_first) {
            asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(peer_flag), "r"(i) : "memory");
        }
        int v = 0;
        unsigned long long t0 = clock64();
        do {
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];" : "=r"(v) : "l"(my_flag) : "memory");
            if (clock64() - t0 > timeout) { *error_out = i; return; }
        } while (v < i);
        if (!am_first) {
            asm volatile("st.release.sys.global.s32 [%0], %1;" :: "l"(peer_flag), "r"(i) : "memory");
        }
    }
    *error_out = 0;
}

/* ------------------------------------------------------------------------- */

int main() {
    int num_devices = 0;
    CHECK(cudaGetDeviceCount(&num_devices));
    printf("==== TileOverlap platform probe ====\n");
    printf("Devices: %d\n\n", num_devices);

    // ---- Device properties -------------------------------------------------
    for (int i = 0; i < num_devices; i++) {
        cudaDeviceProp p;
        CHECK(cudaGetDeviceProperties(&p, i));
        printf("[dev %d] %s | SM %d.%d | %d SMs | %.1f GB | smem/block(optin) %zu KB | regs/SM %d\n",
               i, p.name, p.major, p.minor, p.multiProcessorCount,
               p.totalGlobalMem / 1024.0 / 1024.0 / 1024.0,
               p.sharedMemPerBlockOptin / 1024, p.regsPerMultiprocessor);
    }
    printf("\n");

    if (num_devices < 2) {
        printf("Only one device visible - nothing to probe. Set CUDA_VISIBLE_DEVICES?\n");
        return 0;
    }

    // ---- [A] P2P matrix -----------------------------------------------------
    printf("---- [A] P2P access / attributes (row=src dev, col=dst dev) ----\n");
    printf("format: access/nativeAtomics/perfRank\n");
    for (int i = 0; i < num_devices; i++) {
        printf("dev %d: ", i);
        for (int j = 0; j < num_devices; j++) {
            if (i == j) { printf("   -/-/-  "); continue; }
            int access = 0, atomics = 0, rank = -1;
            CHECK(cudaDeviceCanAccessPeer(&access, i, j));
            if (access) {
                CHECK(cudaDeviceGetP2PAttribute(&atomics, cudaDevP2PAttrNativeAtomicSupported, i, j));
                CHECK(cudaDeviceGetP2PAttribute(&rank, cudaDevP2PAttrPerformanceRank, i, j));
            }
            printf("   %d/%d/%-2d ", access, atomics, rank);
        }
        printf("\n");
    }
    printf("=> P2P access must be 1 for the TK pgl path to work at all.\n");
    printf("=> nativeAtomics is expected to be 0 on PCIe; kernels must not use remote red/atom.\n\n");

    // Enable peer access everywhere possible
    std::vector<std::vector<bool>> p2p(num_devices, std::vector<bool>(num_devices, false));
    for (int i = 0; i < num_devices; i++) {
        CHECK(cudaSetDevice(i));
        for (int j = 0; j < num_devices; j++) {
            if (i == j) continue;
            int access = 0;
            CHECK(cudaDeviceCanAccessPeer(&access, i, j));
            if (access) {
                cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
                if (e == cudaErrorPeerAccessAlreadyEnabled) { cudaGetLastError(); e = cudaSuccess; }
                if (e == cudaSuccess) p2p[i][j] = true;
                else { printf("WARNING: enable peer %d->%d failed: %s\n", i, j, cudaGetErrorString(e)); cudaGetLastError(); }
            }
        }
    }

    // ---- Buffers ------------------------------------------------------------
    const size_t BYTES = 256ULL * 1024 * 1024;
    const size_t N4 = BYTES / sizeof(float4);
    std::vector<float4 *> buf_a(num_devices), buf_b(num_devices);
    std::vector<int *> flags(num_devices);
    for (int i = 0; i < num_devices; i++) {
        CHECK(cudaSetDevice(i));
        CHECK(cudaMalloc(&buf_a[i], BYTES));
        CHECK(cudaMalloc(&buf_b[i], BYTES));
        CHECK(cudaMalloc(&flags[i], 4096));
        CHECK(cudaMemset(flags[i], 0, 4096));
        CHECK(cudaMemset(buf_a[i], 1, BYTES));
    }

    // ---- [C1] Copy-engine bandwidth ------------------------------------------
    printf("---- [C1] cudaMemcpyPeerAsync bandwidth (copy engine, %zu MB) ----\n", BYTES >> 20);
    for (int i = 0; i < num_devices; i++) {
        for (int j = 0; j < num_devices; j++) {
            if (i == j) continue;
            CHECK(cudaSetDevice(i));
            cudaEvent_t s, e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
            CHECK(cudaMemcpyPeerAsync(buf_b[j], j, buf_a[i], i, BYTES, 0)); // warmup
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(s, 0));
            for (int r = 0; r < 4; r++)
                CHECK(cudaMemcpyPeerAsync(buf_b[j], j, buf_a[i], i, BYTES, 0));
            CHECK(cudaEventRecord(e, 0));
            CHECK(cudaEventSynchronize(e));
            float ms; CHECK(cudaEventElapsedTime(&ms, s, e));
            printf("  %d -> %d : %7.2f GB/s %s\n", i, j, 4.0 * BYTES / (ms * 1e6),
                   p2p[i][j] ? "" : "(no P2P: staged through host!)");
            CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
        }
    }
    printf("\n");

    // ---- [C2] SM-driven pull/push bandwidth (dev pairs with P2P) --------------
    printf("---- [C2] SM-driven peer bandwidth (float4 loop, %zu MB) ----\n", BYTES >> 20);
    printf("This is the path the fused kernels use. Watch blocks-to-saturation.\n");
    const int block_counts[] = {1, 2, 4, 8, 16, 32};
    for (int i = 0; i < num_devices && i < 2; i++) { // representative pairs only: 0->1, 1->0
        int j = (i + 1) % num_devices;
        if (!p2p[i][j]) { printf("  %d <-> %d : P2P unavailable, skipped\n", i, j); continue; }
        CHECK(cudaSetDevice(i));
        cudaEvent_t s, e; CHECK(cudaEventCreate(&s)); CHECK(cudaEventCreate(&e));
        for (int bc : block_counts) {
            // pull: kernel on dev i reads dev j's memory, writes locally
            copy_kernel<<<bc, 256>>>(buf_a[j], buf_b[i], N4); // warmup
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(s));
            copy_kernel<<<bc, 256>>>(buf_a[j], buf_b[i], N4);
            CHECK(cudaEventRecord(e));
            CHECK(cudaEventSynchronize(e));
            float pull_ms; CHECK(cudaEventElapsedTime(&pull_ms, s, e));
            // push: kernel on dev i reads locally, writes dev j's memory
            copy_kernel<<<bc, 256>>>(buf_a[i], buf_b[j], N4);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaEventRecord(s));
            copy_kernel<<<bc, 256>>>(buf_a[i], buf_b[j], N4);
            CHECK(cudaEventRecord(e));
            CHECK(cudaEventSynchronize(e));
            float push_ms; CHECK(cudaEventElapsedTime(&push_ms, s, e));
            printf("  dev %d, %2d blocks: pull %7.2f GB/s | push %7.2f GB/s\n",
                   i, bc, BYTES / (pull_ms * 1e6), BYTES / (push_ms * 1e6));
        }
        CHECK(cudaEventDestroy(s)); CHECK(cudaEventDestroy(e));
    }
    printf("\n");

    // ---- [B] Remote atomic probe ---------------------------------------------
    printf("---- [B] Remote atomic probe (dev 0 -> dev 1 memory) ----\n");
    if (num_devices >= 2 && p2p[0][1]) {
        const int ITERS = 1000, BLOCKS = 8, THREADS = 128;
        const long long expected = (long long)ITERS * BLOCKS * THREADS;
        struct { const char *name; void (*k)(int *, int); } probes[] = {
            {"atomicAdd (device scope) ", atomic_dev_probe},
            {"atomicAdd_system         ", atomic_sys_probe},
            {"red.release.sys.add      ", red_sys_probe},
        };
        for (auto &pr : probes) {
            CHECK(cudaSetDevice(1));
            CHECK(cudaMemset(flags[1], 0, 4));
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaSetDevice(0));
            pr.k<<<BLOCKS, THREADS>>>(flags[1], ITERS);
            cudaError_t launch_err = cudaDeviceSynchronize();
            if (launch_err != cudaSuccess) {
                printf("  %s : KERNEL ERROR (%s) -> remote atomics NOT usable\n", pr.name, cudaGetErrorString(launch_err));
                cudaGetLastError();
                continue;
            }
            int result = 0;
            CHECK(cudaSetDevice(1));
            CHECK(cudaMemcpy(&result, flags[1], 4, cudaMemcpyDeviceToHost));
            printf("  %s : got %d, expected %lld -> %s\n", pr.name, result, expected,
                   (result == expected) ? "OK (unexpected on PCIe! recheck topo)" : "WRONG/UNSUPPORTED (as expected on PCIe)");
        }
    } else {
        printf("  skipped (need P2P between dev 0 and dev 1)\n");
    }
    printf("\n");

    // ---- [D] Signal ping-pong latency -----------------------------------------
    printf("---- [D] Cross-device signal RTT (st.release.sys + local poll) ----\n");
    if (num_devices >= 2 && p2p[0][1] && p2p[1][0]) {
        const int ITERS = 20000;
        int *err0, *err1;
        CHECK(cudaSetDevice(0)); CHECK(cudaMalloc(&err0, 4)); CHECK(cudaMemset(flags[0], 0, 4096));
        CHECK(cudaSetDevice(1)); CHECK(cudaMalloc(&err1, 4)); CHECK(cudaMemset(flags[1], 0, 4096));
        CHECK(cudaSetDevice(0)); CHECK(cudaDeviceSynchronize());
        CHECK(cudaSetDevice(1)); CHECK(cudaDeviceSynchronize());

        cudaStream_t s0, s1;
        CHECK(cudaSetDevice(0)); CHECK(cudaStreamCreate(&s0));
        CHECK(cudaSetDevice(1)); CHECK(cudaStreamCreate(&s1));

        cudaEvent_t t_start, t_stop;
        CHECK(cudaSetDevice(0));
        CHECK(cudaEventCreate(&t_start)); CHECK(cudaEventCreate(&t_stop));

        // launch responder first (dev 1), then pinger (dev 0)
        CHECK(cudaSetDevice(1));
        pingpong_kernel<<<1, 1, 0, s1>>>(flags[1], flags[0], ITERS, 0, err1);
        CHECK(cudaSetDevice(0));
        CHECK(cudaEventRecord(t_start, s0));
        pingpong_kernel<<<1, 1, 0, s0>>>(flags[0], flags[1], ITERS, 1, err0);
        CHECK(cudaEventRecord(t_stop, s0));
        CHECK(cudaSetDevice(0)); CHECK(cudaStreamSynchronize(s0));
        CHECK(cudaSetDevice(1)); CHECK(cudaStreamSynchronize(s1));

        int h_err0 = -1, h_err1 = -1;
        CHECK(cudaMemcpy(&h_err0, err0, 4, cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(&h_err1, err1, 4, cudaMemcpyDeviceToHost));
        if (h_err0 != 0 || h_err1 != 0) {
            printf("  TIMEOUT at iteration %d/%d -> cross-device release-store visibility is broken; STOP HERE.\n", h_err0, h_err1);
        } else {
            float ms;
            CHECK(cudaSetDevice(0));
            CHECK(cudaEventElapsedTime(&ms, t_start, t_stop));
            printf("  %d roundtrips in %.2f ms -> RTT %.2f us, one-way signal ~%.2f us\n",
                   ITERS, ms, ms * 1000.0 / ITERS, ms * 500.0 / ITERS);
        }
    } else {
        printf("  skipped (need bidirectional P2P between dev 0 and dev 1)\n");
    }

    printf("\n==== probe done ====\n");
    return 0;
}
