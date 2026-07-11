/**
 * @file grouped_gemm_sm120.cu
 * @brief Single-GPU grouped GEMM for SM120 — the compute-only correctness and
 *        performance anchor for the fused MoE kernels (no communication).
 *
 *   outputs[rows(e), :] = inputs[rows(e), :] @ weights[e, :, :]
 *
 * The kernel body is exactly tileoverlap::grouped_gemm_sm120 with a no-op
 * gate; the fused version (02) only swaps the gate and adds dispatch blocks.
 */

#include "kittens.cuh"
#include "prototype.cuh"
#include "pyutils/torchutils.cuh"

#include "../common/sm120_common.cuh"

using namespace kittens;
using namespace tileoverlap;

struct globals {
    using cfg = gemm_config;

    using activations_gl = gl<bf16, 1, 1, -1, -1, cfg::A_tile>;    // (num_padded_tokens, H)
    using weights_gl     = gl<bf16, 1, -1, -1, -1, cfg::B_tile>;   // (E, H, I)
    using outputs_gl     = gl<bf16, 1, 1, -1, -1>;                 // (num_padded_tokens, I)
    using counts_gl      = gl<int, 1, 1, 1, -1>;                   // (E,)

    activations_gl activations;
    weights_gl weights;
    outputs_gl outputs;
    counts_gl padded_tokens_per_expert;

    const int num_local_experts;
    const int expert_offset;
};

struct no_gate {
    __device__ inline void operator()(int) const {}
};

__global__ __launch_bounds__(gemm_config::NUM_THREADS, 1)
void grouped_gemm_kernel(const __grid_constant__ globals G) {
    grouped_gemm_sm120(G, no_gate{}, blockIdx.x, gridDim.x);
}

void entrypoint(
    const at::Tensor &inputs,
    const at::Tensor &weights,
    at::Tensor &outputs,
    const at::Tensor &padded_tokens_per_expert
) {
    TORCH_CHECK(inputs.size(0) % gemm_config::ROW_BLOCK == 0, "num_padded_tokens must be a multiple of 128");
    TORCH_CHECK(inputs.size(1) % gemm_config::RED_BLOCK == 0, "H must be a multiple of 64");
    TORCH_CHECK(weights.size(2) % gemm_config::COL_BLOCK == 0, "I must be a multiple of 128");
    TORCH_CHECK(weights.size(1) == inputs.size(1), "H mismatch between inputs and weights");
    TORCH_CHECK(outputs.size(0) == inputs.size(0) && outputs.size(1) == weights.size(2), "bad outputs shape");
    TORCH_CHECK(padded_tokens_per_expert.size(0) == weights.size(0), "expert count mismatch");
    TORCH_CHECK(weights.size(0) <= gemm_config::MAX_LOCAL_EXPERTS, "too many experts");

    globals G {
        .activations = kittens::py::tensor_to_gl<globals::activations_gl>(inputs),
        .weights = kittens::py::tensor_to_gl<globals::weights_gl>(weights),
        .outputs = kittens::py::tensor_to_gl<globals::outputs_gl>(outputs),
        .padded_tokens_per_expert = kittens::py::tensor_to_gl<globals::counts_gl>(padded_tokens_per_expert),
        .num_local_experts = static_cast<int>(weights.size(0)),
        .expert_offset = 0
    };

    int sm_count;
    CUDACHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, inputs.device().index()));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    // +1024 slack: tma_swizzle_allocator enforces 1024B alignment and can bump
    // the dynamic-smem base forward, pushing the last stage's tile past a tight
    // (exactly-96KB) reservation. Reserve the bump so the store stays in range.
    constexpr int smem_bytes = gemm_config::DYNAMIC_SHARED_MEMORY + 1024;
    CUDACHECK(cudaFuncSetAttribute(grouped_gemm_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
    grouped_gemm_kernel<<<sm_count, gemm_config::NUM_THREADS, smem_bytes, stream>>>(G);
    CUDACHECK(cudaGetLastError());
}

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(_C, m) {
    m.def("grouped_gemm", &entrypoint);
}
