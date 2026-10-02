#pragma once

#include "core/arena.h"
#include "core/tensor.h"

#include <cstddef>

#include <cuda_runtime.h>

namespace ninfer::ops::detail {

void packed_attention_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                             const Tensor& cu_seqlens, Tensor* tiles, Tensor& out,
                             cudaStream_t stream);

std::int32_t packed_attention_uniform_tile(std::int32_t segment_length);

void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, Tensor& out, cudaStream_t stream);

void packed_attention_uniform_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                     std::int32_t segment_length, WorkspaceArena& workspace,
                                     Tensor& out, cudaStream_t stream);

void packed_attention_uniform_launch_with_tile(const Tensor& q, const Tensor& k, const Tensor& v,
                                               std::int32_t segment_length, std::int32_t tile_size,
                                               Tensor& out, cudaStream_t stream);

#ifdef NINFER_VOLTA_BUILD
[[nodiscard]] bool packed_attention_volta_flash_eligible(std::int32_t tokens) noexcept;
[[nodiscard]] std::size_t packed_attention_volta_flash_workspace_capacity_bytes(
    std::int32_t min_tokens, std::int32_t max_tokens);
void packed_attention_volta_flash_launch(const Tensor& q, const Tensor& k, const Tensor& v,
                                         Tensor& out, WorkspaceArena& workspace,
                                         cudaStream_t stream);
#endif

} // namespace ninfer::ops::detail
