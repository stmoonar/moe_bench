// SPDX-License-Identifier: Apache-2.0
// microbench 通信/干扰探针:纯 CUDA runtime + torch 扩展,不依赖 ThunderKittens。
//
// 单进程多卡,cudaDeviceEnablePeerAccess 打开 P2P(与 TK 的 VMM+IPC 走同一 fabric
// 能力,见 ThunderKittens/tileoverlap/00_probe 的说明)。提供:
//   enable_p2p  : 全对全打开 peer access
//   sm_copy     : SM 驱动的连续 float4 copy(在哪张卡上执行、指针在哪张卡,
//                 决定了是 pull(读远端)还是 push(写远端))
//   row_copy    : 按行散布 copy(行 = 8KB bf16 token 行,可带 src/dst 行索引,
//                 模拟 dispatch/combine 的散行访问模式)
//   memcpy_peer : copy engine(cudaMemcpyPeerAsync)
//   pingpong    : st.release.sys / ld.acquire.sys 跨卡信号往返延迟
//   overlap     : 通信+计算干扰探针,inter-SM(整 block 分工)与
//                 intra-SM(block 内按 warp 分工)两种编排
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <chrono>

#define MB_CHECK(call)                                                        \
    do {                                                                      \
        cudaError_t _e = (call);                                              \
        TORCH_CHECK(_e == cudaSuccess, "CUDA error: ", cudaGetErrorString(_e)); \
    } while (0)

// ---------------------------------------------------------------- P2P ----

void enable_p2p() {
    int n = 0;
    MB_CHECK(cudaGetDeviceCount(&n));
    for (int i = 0; i < n; i++) {
        MB_CHECK(cudaSetDevice(i));
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            int can = 0;
            MB_CHECK(cudaDeviceCanAccessPeer(&can, i, j));
            TORCH_CHECK(can, "P2P not available between dev ", i, " and ", j);
            cudaError_t e = cudaDeviceEnablePeerAccess(j, 0);
            if (e == cudaErrorPeerAccessAlreadyEnabled) {
                cudaGetLastError();  // clear
            } else {
                MB_CHECK(e);
            }
        }
    }
}

int64_t sm_count(int64_t dev) {
    int c = 0;
    MB_CHECK(cudaDeviceGetAttribute(&c, cudaDevAttrMultiProcessorCount, (int)dev));
    return c;
}

// ------------------------------------------------------------- sm_copy ----

__global__ void copy_kernel(float4 *__restrict__ dst,
                            const float4 *__restrict__ src, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) dst[i] = src[i];
}

void sm_copy(at::Tensor dst, at::Tensor src, int64_t exec_dev, int64_t blocks,
             int64_t threads) {
    const int64_t bytes = dst.numel() * dst.element_size();
    TORCH_CHECK(src.numel() * src.element_size() == bytes, "size mismatch");
    TORCH_CHECK(bytes % 16 == 0, "bytes must be multiple of 16");
    c10::cuda::CUDAGuard guard((int)exec_dev);
    auto stream = at::cuda::getCurrentCUDAStream();
    copy_kernel<<<(int)blocks, (int)threads, 0, stream.stream()>>>(
        (float4 *)dst.data_ptr(), (const float4 *)src.data_ptr(),
        (size_t)(bytes / 16));
    MB_CHECK(cudaGetLastError());
}

// ------------------------------------------------------------ row_copy ----

__global__ void row_copy_kernel(char *__restrict__ dst,
                                const char *__restrict__ src,
                                const int *__restrict__ dst_idx,
                                const int *__restrict__ src_idx, int num_rows,
                                long long row_bytes) {
    const long long n16 = row_bytes / 16;
    for (int r = blockIdx.x; r < num_rows; r += gridDim.x) {
        const int sr = src_idx ? src_idx[r] : r;
        const int dr = dst_idx ? dst_idx[r] : r;
        const int4 *s = (const int4 *)(src + (long long)sr * row_bytes);
        int4 *d = (int4 *)(dst + (long long)dr * row_bytes);
        for (long long i = threadIdx.x; i < n16; i += blockDim.x) d[i] = s[i];
    }
}

void row_copy(at::Tensor dst, at::Tensor src, c10::optional<at::Tensor> dst_idx,
              c10::optional<at::Tensor> src_idx, int64_t num_rows,
              int64_t row_bytes, int64_t exec_dev, int64_t blocks,
              int64_t threads) {
    TORCH_CHECK(row_bytes % 16 == 0, "row_bytes must be multiple of 16");
    TORCH_CHECK(dst.numel() * dst.element_size() >= num_rows * row_bytes ||
                    dst_idx.has_value(),
                "dst too small");
    const int *dip = dst_idx.has_value() ? (const int *)dst_idx->data_ptr() : nullptr;
    const int *sip = src_idx.has_value() ? (const int *)src_idx->data_ptr() : nullptr;
    c10::cuda::CUDAGuard guard((int)exec_dev);
    auto stream = at::cuda::getCurrentCUDAStream();
    row_copy_kernel<<<(int)blocks, (int)threads, 0, stream.stream()>>>(
        (char *)dst.data_ptr(), (const char *)src.data_ptr(), dip, sip,
        (int)num_rows, (long long)row_bytes);
    MB_CHECK(cudaGetLastError());
}

// --------------------------------------------------------- memcpy_peer ----

void memcpy_peer(at::Tensor dst, at::Tensor src, int64_t exec_dev) {
    const int64_t bytes = dst.numel() * dst.element_size();
    TORCH_CHECK(src.numel() * src.element_size() == bytes, "size mismatch");
    c10::cuda::CUDAGuard guard((int)exec_dev);
    auto stream = at::cuda::getCurrentCUDAStream();
    MB_CHECK(cudaMemcpyPeerAsync(dst.data_ptr(), dst.get_device(),
                                 src.data_ptr(), src.get_device(),
                                 (size_t)bytes, stream.stream()));
}

// ------------------------------------------------------------ pingpong ----

// 与 tileoverlap/00_probe 同款:两张卡各跑一个单线程 kernel,向对方 flag
// st.release.sys 写序号,自旋 ld.acquire.sys 等自己的 flag。
__global__ void pingpong_kernel(int *my_flag, int *peer_flag, int iters,
                                int am_first, int *error_out) {
    const unsigned long long timeout = 20ULL * 1000ULL * 1000ULL * 1000ULL;
    for (int i = 1; i <= iters; i++) {
        if (am_first) {
            asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(peer_flag),
                         "r"(i)
                         : "memory");
        }
        int v = 0;
        unsigned long long t0 = clock64();
        do {
            asm volatile("ld.acquire.sys.global.s32 %0, [%1];"
                         : "=r"(v)
                         : "l"(my_flag)
                         : "memory");
            if (clock64() - t0 > timeout) {
                *error_out = i;
                return;
            }
        } while (v < i);
        if (!am_first) {
            asm volatile("st.release.sys.global.s32 [%0], %1;" ::"l"(peer_flag),
                         "r"(i)
                         : "memory");
        }
    }
    *error_out = 0;
}

// 返回总耗时(秒)。调用方负责事先把 flag/err 清零。RTT = 秒 / iters。
double pingpong(at::Tensor flag_a, at::Tensor flag_b, at::Tensor err_a,
                at::Tensor err_b, int64_t iters) {
    const int dev_a = flag_a.get_device();
    const int dev_b = flag_b.get_device();
    TORCH_CHECK(dev_a != dev_b, "flags must live on two different devices");
    auto t0 = std::chrono::steady_clock::now();
    {
        c10::cuda::CUDAGuard g(dev_a);
        pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
            (int *)flag_a.data_ptr(), (int *)flag_b.data_ptr(), (int)iters, 1,
            (int *)err_a.data_ptr());
        MB_CHECK(cudaGetLastError());
    }
    {
        c10::cuda::CUDAGuard g(dev_b);
        pingpong_kernel<<<1, 1, 0, at::cuda::getCurrentCUDAStream()>>>(
            (int *)flag_b.data_ptr(), (int *)flag_a.data_ptr(), (int)iters, 0,
            (int *)err_b.data_ptr());
        MB_CHECK(cudaGetLastError());
    }
    {
        c10::cuda::CUDAGuard g(dev_a);
        MB_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    }
    {
        c10::cuda::CUDAGuard g(dev_b);
        MB_CHECK(cudaStreamSynchronize(at::cuda::getCurrentCUDAStream()));
    }
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

// ------------------------------------------------------------- overlap ----

// 通信(float4 copy)+ 计算(FMA 循环)干扰探针。
// mode 0 = inter-SM:前 comm_blocks 个 block 只做通信,其余 block 只做计算。
// mode 1 = intra-SM:每个 block 的前 comm_warps 个 warp 做通信,其余 warp 计算。
// total_fma_iters 是全体计算线程合计的循环次数(每次循环 = 8 条 fmaf),
// kernel 内部按计算线程数均摊,保证不同编排下总计算量一致。
// 动态 smem 仅用于占位,把每 SM 限到 1 个 block(编排语义才干净)。
__global__ void overlap_kernel(float4 *__restrict__ dst,
                               const float4 *__restrict__ src, size_t n4,
                               float *__restrict__ sink,
                               long long total_fma_iters, int comm_blocks,
                               int comm_warps, int mode) {
    extern __shared__ char _mb_smem_pad[];
    (void)_mb_smem_pad;
    bool is_comm;
    long long comm_tid = -1;
    long long num_comm_threads;
    if (mode == 0) {
        is_comm = (int)blockIdx.x < comm_blocks;
        num_comm_threads = (long long)comm_blocks * blockDim.x;
        if (is_comm) comm_tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    } else {
        const int warp = threadIdx.x / 32;
        is_comm = warp < comm_warps;
        num_comm_threads = (long long)gridDim.x * comm_warps * 32;
        if (is_comm)
            comm_tid = (long long)blockIdx.x * (comm_warps * 32) + threadIdx.x;
    }
    if (is_comm) {
        if (n4 == 0 || num_comm_threads <= 0) return;
        for (size_t i = (size_t)comm_tid; i < n4; i += (size_t)num_comm_threads)
            dst[i] = src[i];
    } else {
        const long long nct =
            (mode == 0)
                ? (long long)(gridDim.x - comm_blocks) * blockDim.x
                : (long long)gridDim.x * (blockDim.x - comm_warps * 32);
        if (nct <= 0 || total_fma_iters <= 0) return;
        const long long iters = total_fma_iters / nct;
        float a0 = 1.1f, a1 = 1.2f, a2 = 1.3f, a3 = 1.4f;
        float a4 = 1.5f, a5 = 1.6f, a6 = 1.7f, a7 = 1.8f;
        const float m = 1.0000001f, c = 1e-7f;
        for (long long i = 0; i < iters; i++) {
            a0 = fmaf(a0, m, c);
            a1 = fmaf(a1, m, c);
            a2 = fmaf(a2, m, c);
            a3 = fmaf(a3, m, c);
            a4 = fmaf(a4, m, c);
            a5 = fmaf(a5, m, c);
            a6 = fmaf(a6, m, c);
            a7 = fmaf(a7, m, c);
        }
        const float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
        if (s == 12345.678f) sink[0] = s;  // 永假,防死代码消除
    }
}

void overlap(at::Tensor dst, at::Tensor src, at::Tensor sink, int64_t exec_dev,
             int64_t total_fma_iters, int64_t comm_blocks, int64_t comm_warps,
             int64_t mode, int64_t grid, int64_t threads, int64_t smem_bytes) {
    const int64_t bytes = dst.numel() * dst.element_size();
    TORCH_CHECK(src.numel() * src.element_size() == bytes, "size mismatch");
    TORCH_CHECK(bytes % 16 == 0, "bytes must be multiple of 16");
    TORCH_CHECK(mode == 0 || mode == 1, "mode must be 0 (inter) or 1 (intra)");
    TORCH_CHECK(threads % 32 == 0, "threads must be multiple of 32");
    TORCH_CHECK(comm_warps * 32 <= threads, "comm_warps too large");
    TORCH_CHECK(comm_blocks <= grid, "comm_blocks too large");
    c10::cuda::CUDAGuard guard((int)exec_dev);
    auto stream = at::cuda::getCurrentCUDAStream();
    if (smem_bytes > 48 * 1024) {
        MB_CHECK(cudaFuncSetAttribute(overlap_kernel,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      (int)smem_bytes));
    }
    overlap_kernel<<<(int)grid, (int)threads, (int)smem_bytes, stream.stream()>>>(
        (float4 *)dst.data_ptr(), (const float4 *)src.data_ptr(),
        (size_t)(bytes / 16), (float *)sink.data_ptr(),
        (long long)total_fma_iters, (int)comm_blocks, (int)comm_warps,
        (int)mode);
    MB_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("enable_p2p", &enable_p2p, "enable all-pair peer access");
    m.def("sm_count", &sm_count, "SM count of a device");
    m.def("sm_copy", &sm_copy, "SM-driven float4 copy (pull/push by placement)");
    m.def("row_copy", &row_copy, "scattered row copy (8KB token rows)");
    m.def("memcpy_peer", &memcpy_peer, "copy-engine peer copy");
    m.def("pingpong", &pingpong, "cross-device signal round-trip (seconds)");
    m.def("overlap", &overlap, "comm+compute interference probe (inter/intra SM)");
}
