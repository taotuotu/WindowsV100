// SM70 packed D72 attention through the vendored llama.cpp Volta flash-MMA kernel.
// This translation unit alone includes the vendored headers because their common.cuh
// intentionally shares names with ninfer's private CUDA headers.

#include "fattn-mma-f16.cuh"

#include "core/device.h"
#include "core/layout.h"
#include "ops/softmax_attention/dense/packed/launch.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr int kLogicalHeadDim = 72;
constexpr int kPaddedHeadDim  = 80;
constexpr int kHeads          = 16;
constexpr int kQueryTileRows  = 32;
constexpr int kQueryBlockRows = 256;
constexpr int kNcols1         = kQueryTileRows;
constexpr int kNcols2         = 1;
constexpr int kNcols          = kNcols1 * kNcols2;
constexpr int kMinimumTokens  = 1024;
constexpr int kMaximumTokens  = 16'384;
constexpr float kAttentionScale = 0.11785113019775792073F;

struct VoltaFlashConfig {
    int threads          = 0;
    int warps            = 0;
    int key_rows_per_mma = 0;
    int max_blocks       = 0;
    int shared_bytes     = 0;
};

const VoltaFlashConfig& flash_config() {
    static const VoltaFlashConfig config = [] {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp properties{};
        CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
        if (properties.major != 7 || properties.minor != 0) {
            throw std::runtime_error("packed Volta flash attention requires SM70");
        }
        const int cc = properties.major * 100 + properties.minor * 10;

        const int nbatch_k2 =
            ggml_cuda_fattn_mma_get_nbatch_K2(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int nbatch_v2 =
            ggml_cuda_fattn_mma_get_nbatch_V2(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int nbatch_combine =
            ggml_cuda_fattn_mma_get_nbatch_combine(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const bool q_in_registers =
            ggml_cuda_fattn_mma_get_Q_in_reg(kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        const int nstages = ggml_cuda_fattn_mma_get_nstages(
            kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2, cc);

        VoltaFlashConfig out;
        out.threads = ggml_cuda_fattn_mma_get_nthreads(
            kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        out.warps            = out.threads / WARP_SIZE;
        out.key_rows_per_mma = ggml_cuda_fattn_mma_get_nbatch_fa(
            kPaddedHeadDim, kPaddedHeadDim, kNcols, cc);
        if (out.threads <= 0 || out.warps <= 0 || out.key_rows_per_mma <= 0 ||
            properties.multiProcessorCount <= 0) {
            throw std::runtime_error("vendored Volta flash configuration is invalid for D80");
        }

        const int cols_per_warp = std::min(kNcols, get_cols_per_warp(cc));
        const std::size_t shared_kv_1stage =
            static_cast<std::size_t>(out.key_rows_per_mma) *
            static_cast<std::size_t>(std::max(nbatch_k2 + 4, nbatch_v2 + 4)) * sizeof(half2);
        const std::size_t shared_kv_2stage =
            static_cast<std::size_t>(out.key_rows_per_mma) *
            static_cast<std::size_t>(nbatch_k2 + 4 + nbatch_v2 + 4) * sizeof(half2);
        const std::size_t shared_q =
            static_cast<std::size_t>(kNcols) * (kPaddedHeadDim / 2 + 4) * sizeof(half2);
        const std::size_t shared_mask =
            static_cast<std::size_t>(kNcols1) * (out.key_rows_per_mma / 2 + 4) * sizeof(half2);
        const std::size_t shared_combine =
            static_cast<std::size_t>(out.warps) * static_cast<std::size_t>(cols_per_warp) *
            static_cast<std::size_t>(nbatch_combine + 4) * sizeof(half2);
        const std::size_t shared_kv = nstages <= 1 ? shared_kv_1stage : shared_kv_2stage;
        const std::size_t shared_bytes =
            std::max(shared_combine,
                     q_in_registers ? std::max(shared_q, shared_kv + shared_mask)
                                    : shared_q + shared_kv + shared_mask);
        if (shared_bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::overflow_error("packed Volta flash shared-memory extent exceeds int32");
        }
        out.shared_bytes = static_cast<int>(shared_bytes);

        const auto kernel = flash_attn_ext_f16<kPaddedHeadDim, kPaddedHeadDim, kNcols1, kNcols2,
                                               false, false>;
        CUDA_CHECK(cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel),
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        out.shared_bytes));
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &out.max_blocks, reinterpret_cast<const void*>(kernel), out.threads,
            static_cast<std::size_t>(out.shared_bytes)));
        out.max_blocks *= properties.multiProcessorCount;
        if (out.max_blocks <= 0) {
            throw std::runtime_error("packed Volta flash kernel has no resident blocks");
        }
        return out;
    }();
    return config;
}

int padded_key_rows(int tokens) {
    const int tile = flash_config().key_rows_per_mma;
    if (tokens <= 0 || tokens > std::numeric_limits<int>::max() - (tile - 1)) {
        throw std::invalid_argument("packed Volta flash key extent is invalid");
    }
    return ((tokens + tile - 1) / tile) * tile;
}

int block_count(int query_tokens, int key_tokens) {
    const VoltaFlashConfig& config = flash_config();
    const int query_tiles          = (query_tokens + kNcols1 - 1) / kNcols1;
    const int output_tiles         = query_tiles * kHeads;
    const int key_tiles            = (key_tokens + config.key_rows_per_mma - 1) /
                          config.key_rows_per_mma;
    // Bound each MMA value accumulation to about 1024 keys. Stream-K's FP32 fixup combines
    // the partial maxima, sums and outputs; this avoids long FP16 accumulation drift on Volta.
    const int tiles_per_split = std::max(1, 1024 / config.key_rows_per_mma);
    const int splits = (key_tiles + tiles_per_split - 1) / tiles_per_split;
    return output_tiles * splits;
}

std::int32_t metadata_elements(int tokens) {
    const int query_tokens = std::min(tokens, kQueryBlockRows);
    const int output_tiles = ((query_tokens + kNcols1 - 1) / kNcols1) * kHeads;
    const int keys         = padded_key_rows(tokens);
    const int blocks       = block_count(query_tokens, keys);
    const std::int64_t elements =
        static_cast<std::int64_t>(std::max(blocks, output_tiles)) * kNcols *
        (2 + kPaddedHeadDim / 2);
    if (elements <= 0 || elements > std::numeric_limits<std::int32_t>::max()) {
        throw std::overflow_error("packed Volta flash metadata extent exceeds int32");
    }
    return static_cast<std::int32_t>(elements);
}

std::size_t workspace_capacity(int tokens) {
    if (tokens < kMinimumTokens || tokens > kMaximumTokens) { return 0; }
    const int keys = padded_key_rows(tokens);
    WorkspaceLayoutBuilder layout;
    (void)layout.alloc(DType::FP16, {kPaddedHeadDim, kHeads, keys});
    (void)layout.alloc(DType::FP16, {kPaddedHeadDim, kHeads, keys});
    if (keys != tokens) { (void)layout.alloc(DType::FP16, {keys}); }
    const int query_rows = std::min(tokens, kQueryBlockRows);
    (void)layout.alloc(DType::FP32, {kPaddedHeadDim, kHeads, query_rows});
    (void)layout.alloc(DType::FP32, {kPaddedHeadDim, kHeads, query_rows});
    (void)layout.alloc(DType::FP32, {2, metadata_elements(tokens)});
    return layout.peak_bytes(256);
}

__global__ void stage_kv_bf16_to_f16_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    std::int64_t k_stride_head, std::int64_t k_stride_token,
    std::int64_t v_stride_head, std::int64_t v_stride_token,
    half* __restrict__ k_out, half* __restrict__ v_out, int tokens, int padded_tokens,
    std::int64_t elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    const int d     = static_cast<int>(index % kPaddedHeadDim);
    const int head  = static_cast<int>((index / kPaddedHeadDim) % kHeads);
    const int token = static_cast<int>(index / (kPaddedHeadDim * kHeads));
    half k_value = __float2half_rn(0.0F);
    half v_value = __float2half_rn(0.0F);
    if (token < tokens && d < kLogicalHeadDim) {
        const std::int64_t k_index = static_cast<std::int64_t>(token) * k_stride_token +
                                     static_cast<std::int64_t>(head) * k_stride_head + d;
        const std::int64_t v_index = static_cast<std::int64_t>(token) * v_stride_token +
                                     static_cast<std::int64_t>(head) * v_stride_head + d;
        k_value = __float2half_rn(__bfloat162float(k[k_index]));
        v_value = __float2half_rn(__bfloat162float(v[v_index]));
    }
    const std::int64_t destination = static_cast<std::int64_t>(token) * kHeads * kPaddedHeadDim +
                                    static_cast<std::int64_t>(head) * kPaddedHeadDim + d;
    k_out[destination] = k_value;
    v_out[destination] = v_value;
    (void)padded_tokens;
}

__global__ void stage_q_bf16_to_f32_kernel(const __nv_bfloat16* __restrict__ q,
                                            std::int64_t q_stride_head,
                                            std::int64_t q_stride_token,
                                            float* __restrict__ q_out, int token_begin,
                                            int query_tokens, std::int64_t elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    const int d     = static_cast<int>(index % kPaddedHeadDim);
    const int head  = static_cast<int>((index / kPaddedHeadDim) % kHeads);
    const int token = static_cast<int>(index / (kPaddedHeadDim * kHeads));
    float value = 0.0F;
    if (token < query_tokens && d < kLogicalHeadDim) {
        const std::int64_t source = static_cast<std::int64_t>(token_begin + token) * q_stride_token +
                                   static_cast<std::int64_t>(head) * q_stride_head + d;
        value = __bfloat162float(q[source]);
    }
    q_out[index] = value;
}

__global__ void build_padding_mask_kernel(half* __restrict__ mask, int real_keys,
                                           int padded_keys) {
    const int key = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (key >= padded_keys) { return; }
    mask[key] = __float2half_rn(key < real_keys ? 0.0F : -CUDART_INF_F);
}

__global__ void copy_flash_output_to_bf16_kernel(const float* __restrict__ source,
                                                  __nv_bfloat16* __restrict__ destination,
                                                  int token_begin, int query_tokens,
                                                  std::int64_t elements) {
    const std::int64_t index = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) { return; }
    const int d     = static_cast<int>(index % kLogicalHeadDim);
    const int head  = static_cast<int>((index / kLogicalHeadDim) % kHeads);
    const int token = static_cast<int>(index / (kLogicalHeadDim * kHeads));
    const std::int64_t source_index =
        static_cast<std::int64_t>(token) * kHeads * kPaddedHeadDim +
        static_cast<std::int64_t>(head) * kPaddedHeadDim + d;
    const std::int64_t destination_index =
        static_cast<std::int64_t>(token_begin + token) * kHeads * kLogicalHeadDim +
        static_cast<std::int64_t>(head) * kLogicalHeadDim + d;
    destination[destination_index] = __float2bfloat16_rn(source[source_index]);
    (void)query_tokens;
}

void launch_flash_block(const float* q, const half* k, const half* v, const half* mask,
                        float* output, float2* metadata, int query_tokens, int key_tokens,
                        cudaStream_t stream) {
    const VoltaFlashConfig& config = flash_config();
    const int query_tiles          = (query_tokens + kNcols1 - 1) / kNcols1;
    const int output_tiles         = query_tiles * kHeads;
    const int key_tiles            = (key_tokens + config.key_rows_per_mma - 1) /
                          config.key_rows_per_mma;
    const std::int64_t work_tiles = static_cast<std::int64_t>(output_tiles) * key_tiles;
    const int blocks = block_count(query_tokens, key_tokens);
    if (blocks <= 0) { throw std::logic_error("packed Volta flash launch has no work blocks"); }

    constexpr int kD = kPaddedHeadDim;
    const int bytes_q_token = kD * kHeads * static_cast<int>(sizeof(float));
    const int bytes_q_head  = kD * static_cast<int>(sizeof(float));
    const int bytes_k_token = kD * kHeads * static_cast<int>(sizeof(half));
    const int bytes_k_head  = kD * static_cast<int>(sizeof(half));
    const uint3 query_fastdiv = init_fastdiv_values(query_tokens);
    const dim3 block_dim(WARP_SIZE, config.warps, 1);
    const dim3 grid(static_cast<unsigned>(blocks), 1u, 1u);

    const auto kernel = flash_attn_ext_f16<kD, kD, kNcols1, kNcols2, false, false>;
    kernel<<<grid, block_dim, static_cast<std::size_t>(config.shared_bytes), stream>>>(
        reinterpret_cast<const char*>(q), reinterpret_cast<const char*>(k),
        reinterpret_cast<const char*>(v), reinterpret_cast<const char*>(mask), nullptr, nullptr,
        output, metadata, kAttentionScale, 0.0F, 1.0F, 1.0F, 16u, 0.0F,
        kD, query_fastdiv, kHeads, 1, bytes_q_token, bytes_q_head, 0,
        kD, key_tokens, kHeads, 1, bytes_k_token, bytes_k_head, 0,
        bytes_k_token, bytes_k_head, 0, query_tokens, 1, 1, 0, 0, 0);
    CUDA_CHECK(cudaGetLastError());

    if (blocks % output_tiles == 0 && blocks > output_tiles) {
        const uint3 fd0 = init_fastdiv_values(output_tiles);
        const uint3 fd1 = init_fastdiv_values(query_tiles);
        const uint3 fd2 = init_fastdiv_values(query_tiles);
        flash_attn_stream_k_fixup_uniform<kD, kNcols1, kNcols2>
            <<<dim3(static_cast<unsigned>(output_tiles), kNcols1, kNcols2), dim3(kD, 1, 1), 0,
               stream>>>(output, metadata, query_tokens, kHeads, kHeads, blocks, 1,
                         blocks / output_tiles, fd0, fd1, fd2);
        CUDA_CHECK(cudaGetLastError());
    } else if (output_tiles % blocks != 0) {
        const std::int64_t total_work = work_tiles;
        const uint3 fd_k_j_z_ne12 = init_fastdiv_values(
            static_cast<std::int32_t>(key_tiles * query_tiles * kHeads));
        const uint3 fd_k_j_z = init_fastdiv_values(
            static_cast<std::int32_t>(key_tiles * query_tiles));
        const uint3 fd_k_j = init_fastdiv_values(static_cast<std::int32_t>(key_tiles * query_tiles));
        const uint3 fd_k   = init_fastdiv_values(key_tiles);
        flash_attn_stream_k_fixup_general<kD, kNcols1, kNcols2>
            <<<dim3(static_cast<unsigned>(blocks), kNcols1, kNcols2), dim3(kD, 1, 1), 0,
               stream>>>(output, metadata, query_tokens, kHeads, 1, total_work,
                         fd_k_j_z_ne12, fd_k_j_z, fd_k_j, fd_k);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <class Allocator>
struct FlashWorkspace {
    Tensor k;
    Tensor v;
    Tensor mask;
    Tensor q;
    Tensor output;
    Tensor metadata;
};

template <class Allocator>
FlashWorkspace<Allocator> allocate_flash_workspace(Allocator& workspace, int tokens) {
    const int keys       = padded_key_rows(tokens);
    const int query_rows = std::min(tokens, kQueryBlockRows);
    return {
        workspace.alloc(DType::FP16, {kPaddedHeadDim, kHeads, keys}),
        workspace.alloc(DType::FP16, {kPaddedHeadDim, kHeads, keys}),
        keys == tokens ? Tensor{} : workspace.alloc(DType::FP16, {keys}),
        workspace.alloc(DType::FP32, {kPaddedHeadDim, kHeads, query_rows}),
        workspace.alloc(DType::FP32, {kPaddedHeadDim, kHeads, query_rows}),
        workspace.alloc(DType::FP32, {2, metadata_elements(tokens)}),
    };
}

} // namespace

bool packed_attention_volta_flash_eligible(std::int32_t tokens) noexcept {
    return tokens >= kMinimumTokens && tokens <= kMaximumTokens;
}

std::size_t packed_attention_volta_flash_workspace_capacity_bytes(std::int32_t min_tokens,
                                                                   std::int32_t max_tokens) {
    if (min_tokens <= 0 || max_tokens < min_tokens) { return 0; }
    const int first = std::max(min_tokens, kMinimumTokens);
    const int last  = std::min(max_tokens, kMaximumTokens);
    if (first > last) { return 0; }
    return workspace_capacity(last);
}

void packed_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         Tensor& out, WorkspaceArena& workspace,
                                         cudaStream_t stream) {
    const int tokens = q.ne[2];
    if (!packed_attention_volta_flash_eligible(tokens)) {
        throw std::invalid_argument("packed Volta flash attention extent is unsupported");
    }
    const std::size_t required = workspace_capacity(tokens);
    if (workspace.capacity() - workspace.used() < required) {
        throw std::invalid_argument("packed Volta flash attention workspace is too small");
    }
    auto scope = workspace.scope();
    FlashWorkspace<WorkspaceArena> buffers = allocate_flash_workspace(workspace, tokens);
    const int keys = padded_key_rows(tokens);
    const std::int64_t stage_elements =
        static_cast<std::int64_t>(keys) * kHeads * kPaddedHeadDim;
    constexpr int kThreads = 256;
    const int stage_blocks = static_cast<int>((stage_elements + kThreads - 1) / kThreads);
    stage_kv_bf16_to_f16_kernel<<<stage_blocks, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(k.data), static_cast<const __nv_bfloat16*>(v.data),
        k.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        k.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        v.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        v.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
        static_cast<half*>(buffers.k.data), static_cast<half*>(buffers.v.data), tokens, keys,
        stage_elements);
    CUDA_CHECK(cudaGetLastError());

    if (keys != tokens) {
        const int mask_blocks = (keys + kThreads - 1) / kThreads;
        build_padding_mask_kernel<<<mask_blocks, kThreads, 0, stream>>>(
            static_cast<half*>(buffers.mask.data), tokens, keys);
        CUDA_CHECK(cudaGetLastError());
    }

    const int query_stride = std::min(tokens, kQueryBlockRows);
    for (int begin = 0; begin < tokens; begin += query_stride) {
        const int query_tokens = std::min(query_stride, tokens - begin);
        const std::int64_t query_elements =
            static_cast<std::int64_t>(query_tokens) * kHeads * kPaddedHeadDim;
        const int query_blocks = static_cast<int>((query_elements + kThreads - 1) / kThreads);
        stage_q_bf16_to_f32_kernel<<<query_blocks, kThreads, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(q.data),
            q.nb[1] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
            q.nb[2] / static_cast<std::int64_t>(sizeof(__nv_bfloat16)),
            static_cast<float*>(buffers.q.data), begin, query_tokens, query_elements);
        CUDA_CHECK(cudaGetLastError());

        const std::int64_t padded_output_elements =
            static_cast<std::int64_t>(query_tokens) * kHeads * kPaddedHeadDim;
        CUDA_CHECK(cudaMemsetAsync(buffers.output.data, 0,
                                   static_cast<std::size_t>(padded_output_elements) * sizeof(float),
                                   stream));
        launch_flash_block(static_cast<const float*>(buffers.q.data),
                           static_cast<const half*>(buffers.k.data),
                           static_cast<const half*>(buffers.v.data),
                           buffers.mask.data == nullptr ? nullptr
                                                        : static_cast<const half*>(buffers.mask.data),
                           static_cast<float*>(buffers.output.data),
                           static_cast<float2*>(buffers.metadata.data), query_tokens, keys, stream);

        const std::int64_t output_elements =
            static_cast<std::int64_t>(query_tokens) * kHeads * kLogicalHeadDim;
        const int output_blocks = static_cast<int>((output_elements + kThreads - 1) / kThreads);
        copy_flash_output_to_bf16_kernel<<<output_blocks, kThreads, 0, stream>>>(
            static_cast<const float*>(buffers.output.data),
            static_cast<__nv_bfloat16*>(out.data), begin, query_tokens, output_elements);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace ninfer::ops::detail
