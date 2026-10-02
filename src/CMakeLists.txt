# Modified for the Windows/V100 port by taotuotu, 2026; see NOTICE.
function(ninfer_internal_includes target)
  target_include_directories(${target} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/third_party
    ${PROJECT_SOURCE_DIR}/third_party/utf8proc)
endfunction()

function(ninfer_cuda_archive target)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endfunction()

# Device/tensor/layout/graph primitives and the model-independent request runtime.
add_library(ninfer_core STATIC
  core/arena.cu
  core/device.cu
  core/dtype.cpp
  core/host_worker_pool.cpp
  core/gdn_replay_records.cpp
  core/cyclic_kv_cache.cpp
  core/host_kv_arena.cpp
  core/paged_kv_cache.cpp
  core/linear_attention_state.cpp
  core/layout.cpp
  core/tensor.cpp
  core/decode_graph.cpp
  runtime/engine/admission_policy.cpp
  runtime/engine/context_cost.cpp
  runtime/engine/context_cost_defaults.cpp
  runtime/engine/kv_capacity.cpp
  runtime/engine/public_types.cpp)
ninfer_internal_includes(ninfer_core)
ninfer_cuda_archive(ninfer_core)
target_link_libraries(ninfer_core PUBLIC CUDA::cudart CUDA::nvtx3 Threads::Threads)

# Generic .ninfer framing, binding, and direct final materialization.
add_library(ninfer_artifact STATIC
  artifact/reader.cpp
  artifact/storage_layouts.cpp
  artifact/binder.cpp
  artifact/materializer.cpp
  artifact/typed_binding.cpp)
ninfer_internal_includes(ninfer_artifact)
target_link_libraries(ninfer_artifact PUBLIC ninfer_core)

# Warp-specialized SM120 NVFP4 kernels use setmaxnreg to transfer registers from producer
# warpgroups to consumer warpgroups. Relocatable device code makes ptxas discard that contract, so
# these self-contained kernels are intentionally compiled in a non-RDC archive and called through
# host launchers from ninfer_ops.
add_library(ninfer_nvfp4_non_rdc STATIC
  ops/linear/nvfp4/nvfp4_w4a4_tma.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4_tma.cu)
if(NOT CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  target_sources(ninfer_nvfp4_non_rdc PRIVATE
    ops/softmax_attention/dense/causal_cache/prompt_nvfp4_non_rdc.cu)
endif()
ninfer_internal_includes(ninfer_nvfp4_non_rdc)
set_target_properties(ninfer_nvfp4_non_rdc PROPERTIES
  CUDA_SEPARABLE_COMPILATION OFF
  CUDA_RESOLVE_DEVICE_SYMBOLS OFF)
target_compile_options(ninfer_nvfp4_non_rdc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
target_link_libraries(ninfer_nvfp4_non_rdc PRIVATE ninfer_core CUDA::cudart CUDA::cuda_driver)

# Shared mathematical Ops. The list is intentionally explicit: adding a
# source is a build-boundary decision, not an accidental recursive-glob side effect.
add_library(ninfer_ops STATIC
  ops/launcher/add_bias.cu
  ops/launcher/argmax.cu
  ops/launcher/cast.cu
  ops/launcher/causal_conv1d.cu
  ops/launcher/embed_gather.cu
  ops/launcher/gdn_gating.cu
  ops/launcher/gelu.cu
  ops/launcher/l2norm.cu
  ops/launcher/layer_norm.cu
  ops/launcher/mtp_pack.cu
  ops/launcher/mtp_round.cu
  ops/launcher/speculative_round.cu
  ops/launcher/position.cu
  ops/launcher/prepare_ragged_prefix.cu
  ops/launcher/prepare_masked_block.cu
  ops/launcher/residual_add.cu
  ops/launcher/rmsnorm.cu
  ops/launcher/rmsnorm_pack_tail.cu
  ops/launcher/rope.cu
  ops/launcher/sampling.cu
  ops/launcher/scalar.cu
  ops/launcher/scatter.cu
  ops/launcher/sigmoid_gate_mul.cu
  ops/launcher/silu_and_mul.cu
  ops/launcher/target_logprobs.cu
  ops/launcher/vision_pos_embed.cu
  ops/rmsnorm_rope/rmsnorm_rope.cpp
  ops/rmsnorm_rope/launch.cu
  ops/context_kv_materialize/context_kv_materialize.cpp
  ops/context_kv_materialize/materialize.cu
  ops/candidate_selector/bf16/candidate_selector_path.cu
  ops/candidate_selector/bf16/candidate_selector_path_plan.cpp
  # Softmax Attention and KV-cache state transitions.
  ops/softmax_attention/dense/causal_cache/causal_softmax_attention.cpp
  ops/softmax_attention/dense/causal_cache/small_t.cu
  ops/softmax_attention/dense/causal_cache/small_t_fp8.cu
  ops/softmax_attention/dense/causal_cache/small_t_nvfp4.cu
  ops/softmax_attention/dense/causal_cache/small_t_k8v4.cu
  ops/softmax_attention/dense/causal_cache/prompt.cu
  ops/softmax_attention/dense/causal_cache/prompt_fp8.cu
  ops/softmax_attention/dense/causal_cache/prompt_nvfp4.cu
  ops/softmax_attention/dense/causal_cache/prompt_k8v4.cu
  ops/softmax_attention/dense/packed/packed_softmax_attention.cpp
  ops/softmax_attention/dense/packed/launch.cu
  ops/softmax_attention/dense/context/context_softmax_attention.cpp
  ops/softmax_attention/dense/context/launch.cu
  ops/softmax_attention/sliding_window/sliding_window_attention.cpp
  ops/softmax_attention/sliding_window/launch.cu
  ops/kv_cache/append/kv_cache_append.cpp
  ops/kv_cache/append/launch.cu
  ops/kv_cache/append/nvfp4_launch.cu
  ops/kv_cache/append/k8v4_launch.cu
  # Gated DeltaNet Linear Attention core.
  ops/linear_attention/gated_delta_net/gated_delta_net.cpp
  ops/linear_attention/gated_delta_net/replay.cpp
  ops/linear_attention/gated_delta_net/recurrent.cu
  ops/linear_attention/gated_delta_net/chunked/launch.cu
  ops/linear_attention/gated_delta_net/chunked/prepare_wy_wu.cu
  ops/linear_attention/gated_delta_net/chunked/state_passing.cu
  ops/linear_attention/gated_delta_net/chunked/output.cu
  ops/attn_input_proj/bf16/bf16_attn_input_decode.cu
  ops/attn_input_proj/bf16/bf16_attn_input_gemm_mma.cu
  ops/attn_input_proj/bf16/bf16_attn_input_small_t.cu
  ops/attn_input_proj/bf16/bf16_attn_input_plan.cpp
  ops/attn_input_proj/fp8/fp8_attn_input_decode.cu
  ops/attn_input_proj/fp8/fp8_attn_input_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a16_small_t.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a16_gemm.cu
  ops/attn_input_proj/fp8/fp8_attn_input_a8.cu
  ops/attn_input_proj/fp8/fp8_attn_input_plan.cpp
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_decode.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_small_t.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_w4a4.cu
  ops/attn_input_proj/nvfp4/nvfp4_attn_input_plan.cpp
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_gemm_mma.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_small_t.cu
  ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp
  ops/attn_input_proj/w8/w8_attn_input_decode.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_mma.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_simt.cu
  ops/attn_input_proj/w8/w8_attn_input_gemm_splitk.cu
  ops/attn_input_proj/w8/w8_dflash2_attn_input.cu
  ops/attn_input_proj/w8/w8_attn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_input_decode.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_small_t.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_a8.cu
  ops/gdn_input_proj/fp8/fp8_gdn_input_plan.cpp
  ops/gdn_input_proj/fp8/fp8_gdn_conv_fused.cu
  ops/gdn_input_proj/fp8/fp8_gdn_conv_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_w4a4.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_input_plan.cpp
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_decode.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_small_t.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_post.cu
  ops/gdn_input_proj/nvfp4/nvfp4_gdn_snapshot_plan.cpp
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_independent.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_conv_snapshot.cu
  ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp
  ops/gdn_input_proj/gdn_projected_conv.cu
  ops/gdn_input_proj/w8/w8_gdn_input_decode.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_mma.cu
  ops/gdn_input_proj/w8/w8_gdn_input_gemm_splitk.cu
  ops/gdn_input_proj/w8/w8_gdn_input_plan.cpp
  ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_kernels.cu
  ops/gdn_gating_proj/bf16/bf16_gdn_gating_proj_plan.cpp
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu
  ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp
  ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_materialized.cu
  ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_plan.cpp
  ops/linear/bf16/bf16_gemv.cu
  ops/linear/bf16/bf16_gemm_mma.cu
  ops/linear/bf16/bf16_n256_k5120.cu
  ops/linear/bf16/bf16_small_t.cu
  ops/linear/bf16/bf16_dispatch.cpp
  ops/linear/fp8/fp8_format.cpp
  ops/linear/fp8/fp8_gemv.cu
  ops/linear/fp8/fp8_small_t.cu
  ops/linear/fp8/fp8_a16_small_t_mma.cu
  ops/linear/fp8/fp8_a16_gemm_mma.cu
  ops/linear/fp8/fp8_a8.cu
  ops/linear/fp8/fp8_dispatch.cpp
  ops/linear/nvfp4/nvfp4_format.cpp
  ops/linear/nvfp4/nvfp4_gemv.cu
  ops/linear/nvfp4/nvfp4_small_t.cu
  ops/linear/nvfp4/nvfp4_w4a4.cu
  ops/linear/nvfp4/nvfp4_dispatch.cpp
  ops/linear/q4/q4_small_t_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_mma.cu
  ops/linear/q4/q4_rowsplit_gemm_simt.cu
  ops/linear/q4/q4_rowsplit_gemv.cu
  ops/linear/q4/q4_dispatch.cpp
  ops/linear/q5/q5_dispatch.cpp
  ops/linear/q5/q5_rowsplit_gemm_mma.cu
  ops/linear/q5/q5_rowsplit_gemm_simt.cu
  ops/linear/q5/q5_rowsplit_gemv.cu
  ops/linear_add/bf16/bf16_linear_add_decode.cu
  ops/linear_add/bf16/bf16_linear_add_gemm_mma.cu
  ops/linear_add/bf16/bf16_linear_add_small_t.cu
  ops/linear_add/bf16/bf16_linear_add_plan.cpp
  ops/linear_add/nvfp4/nvfp4_linear_add_decode.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_small_t.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_w4a4.cu
  ops/linear_add/nvfp4/nvfp4_linear_add_plan.cpp
  ops/linear_add/fp8/fp8_linear_add_decode.cu
  ops/linear_add/fp8/fp8_linear_add_small_t.cu
  ops/linear_add/fp8/fp8_linear_add_a8.cu
  ops/linear_add/fp8/fp8_linear_add_plan.cpp
  ops/linear_add/q5/q5_linear_add_gemm_mma.cu
  ops/linear_add/q5/q5_linear_add_gemm_simt.cu
  ops/linear_add/q5/q5_linear_add_gemv.cu
  ops/linear_add/q5/q5_linear_add_plan.cpp
  ops/linear_add/w8/w8_linear_add_gemm_mma.cu
  ops/linear_add/w8/w8_linear_add_gemm_simt.cu
  ops/linear_add/w8/w8_linear_add_gemm_splitk.cu
  ops/linear_add/w8/w8_linear_add_plan.cpp
  ops/linear_swiglu/q4/q4_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_gemv.cu
  ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_decode.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_small_t.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_w4a4.cu
  ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_plan.cpp
  ops/linear_swiglu/fp8/fp8_linear_swiglu_decode.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_small_t.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_a8.cu
  ops/linear_swiglu/fp8/fp8_linear_swiglu_plan.cpp
  ops/linear_swiglu/w8/w8_linear_swiglu_decode.cu
  ops/linear_swiglu/w8/w8_dflash2_linear_swiglu.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_mma.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_gemm_splitk.cu
  ops/linear_swiglu/w8/w8_linear_swiglu_plan.cpp
  ops/linear/q6/q6_dispatch.cpp
  ops/linear/q6/q6_rowsplit_gemm_mma.cu
  ops/linear/q6/q6_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_feature.cu
  ops/linear/w8/w8_rowsplit_gemm_mma.cu
  ops/linear/w8/w8_rowsplit_gemm_simt.cu
  ops/linear/w8/w8_rowsplit_gemm_decode.cu
  ops/linear/w8/w8_small_t.cu
  ops/linear/w8/w8_rowsplit_gemm_splitk.cu
  ops/linear/w8/w8_dispatch.cpp
  ops/linear_pair/w8/w8_pair_decode.cu
  ops/linear_pair/w8/w8_pair_gemm_concat.cu
  ops/linear_pair/w8/w8_pair_gemm_mma.cu
  ops/linear_pair/w8/w8_pair_gemm_splitk.cu
  ops/linear_pair/w8/w8_pair_plan.cpp
  ops/linear_topk/linear_topk.cpp
  ops/linear_topk/w8.cu
  ops/linear_topk/w8_m64.cu
  ops/linear_topk/fp8.cu
  ops/linear_topk/fp8_m64.cu
  ops/linear_topk/q4.cu
  ops/linear_topk/q4_m64.cu
  ops/linear_topk/merge.cu
  ops/sparse_moe/decode/sparse_moe_decode_kernels.cu
  ops/sparse_moe/decode/sparse_moe_decode_plan.cpp
  ops/sparse_moe/prefill/sparse_moe_prefill_kernels.cu
  ops/sparse_moe/prefill/sparse_moe_prefill_plan.cpp
  ops/sparse_moe/small_t/sparse_moe_small_t_kernels.cu
  ops/sparse_moe/small_t/sparse_moe_small_t_plan.cpp
  ops/linear/linear.cpp
  ops/wrapper/add_bias.cpp
  ops/wrapper/argmax.cpp
  ops/wrapper/cast.cpp
  ops/wrapper/causal_conv1d_silu.cpp
  ops/dflash2_sm70_stub.cu
  ops/wrapper/candidate_selector.cpp
  ops/wrapper/dynamic_grouped_conv.cpp
  ops/wrapper/embedding.cpp
  ops/wrapper/gdn_gating.cpp
  ops/wrapper/gdn_gating_proj.cpp
  ops/wrapper/gdn_input_proj.cpp
  ops/wrapper/gelu.cpp
  ops/wrapper/attn_input_proj.cpp
  ops/wrapper/l2norm.cpp
  ops/wrapper/layer_norm.cpp
  ops/wrapper/linear_add.cpp
  ops/wrapper/linear_pair.cpp
  ops/wrapper/linear_swiglu.cpp
  ops/wrapper/mtp_pack.cpp
  ops/wrapper/mtp_round.cpp
  ops/wrapper/speculative_round.cpp
  ops/wrapper/position.cpp
  ops/wrapper/prepare_ragged_prefix.cpp
  ops/wrapper/prepare_masked_block.cpp
  ops/wrapper/residual_add.cpp
  ops/wrapper/rmsnorm.cpp
  ops/wrapper/rmsnorm_pack_tail.cpp
  ops/wrapper/rope.cpp
  ops/wrapper/sampling.cpp
  ops/wrapper/scalar.cpp
  ops/wrapper/scatter.cpp
  ops/wrapper/sigmoid_mul.cpp
  ops/wrapper/silu_mul.cpp
  ops/wrapper/sparse_moe.cpp
  ops/wrapper/target_logprobs.cpp
  ops/wrapper/vision_pos_embed.cpp)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "70")
  # Volta-only execution leaves. Shared Op contracts and planners remain in the
  # ordinary source list above; these files provide the sm_70 implementations.
  target_sources(ninfer_ops PRIVATE
    ops/attn_input_proj/fp8/fp8_attn_input_volta_qpn.cu
    ops/attn_input_proj/fp8/fp8_attn_input_cutlass_sm70.cu
    ops/attn_input_proj/bf16/bf16_attn_input_cutlass_sm70.cu
    ops/attn_input_proj/q4_q5/q4_q5_attn_input_cutlass_sm70.cu
    ops/gdn_input_proj/fp8/fp8_gdn_input_volta_qpn.cu
    ops/gdn_input_proj/fp8/fp8_gdn_input_cutlass_sm70.cu
    ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_cutlass_sm70.cu
    ops/gdn_input_proj/w8/w8_gdn_input_cutlass_sm70.cu
    ops/linear/fp8/fp8_volta_qpn_gemm.cu
    ops/linear/fp8/fp8_prepack_sm70.cu
    ops/linear/fp8/fp8_cutlass_sm70.cu
    ops/linear/bf16/bf16_cutlass_sm70.cu
    ops/linear/nvfp4/nvfp4_volta_qpn_gemm.cu
    ops/linear/nvfp4/nvfp4_volta_mma_gemm.cu
    ops/linear/nvfp4/nvfp4_cutlass_sm70.cu
    ops/linear/nvfp4/nvfp4_prepack_sm70.cu
    ops/linear/q4/q4_volta_mma_gemm.cu
    ops/linear/q4/q4_volta_qpn_gemm.cu
    ops/linear/q5/q5_volta_mma_gemm.cu
    ops/linear/w8/w8_volta_mma_gemm.cu
    ops/linear/w8/w8_volta_qpn_gemm.cu
    ops/linear_add/q5/q5_linear_add_cutlass_sm70.cu
    ops/linear_add/bf16/bf16_linear_add_volta.cu
    ops/linear_swiglu/fp8/fp8_linear_swiglu_volta_qpn.cu
    ops/linear_swiglu/fp8/fp8_linear_swiglu_qpn_split.cu
    ops/linear_swiglu/fp8/fp8_linear_swiglu_cutlass_sm70.cu
    ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_volta_qpn.cu
    ops/linear_swiglu/nvfp4/nvfp4_linear_swiglu_qpn_split.cu
    ops/linear_swiglu/q4/q4_linear_swiglu_cutlass_sm70.cu
    ops/launcher/gqa_attention_volta_flash.cu
    ops/softmax_attention/dense/packed/volta_flash.cu)
  set_source_files_properties(
    ops/launcher/gqa_attention_volta_flash.cu
    ops/softmax_attention/dense/packed/volta_flash.cu
    TARGET_DIRECTORY ninfer_ops
    PROPERTIES
      INCLUDE_DIRECTORIES "${PROJECT_SOURCE_DIR}/third_party/llama_cpp_fattn"
      COMPILE_OPTIONS "$<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;-diag-suppress=940>")
  # --- DFlash2 port: these sm_80-only translation units (cp.async / bf16 mma /
  #     __reduce_*_sync) have no Volta kernel yet. Exclude them from the sm_70 build;
  #     the op wrappers throw std::logic_error under NINFER_VOLTA_BUILD until P2+.
  set_source_files_properties(ops/launcher/rmsnorm_pack_tail.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/rmsnorm_rope/launch.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/context_kv_materialize/materialize.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/candidate_selector/bf16/candidate_selector_path.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/attn_input_proj/fp8/fp8_attn_input_a16_small_t.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/attn_input_proj/fp8/fp8_attn_input_a16_gemm.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/attn_input_proj/w8/w8_dflash2_attn_input.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/gdn_gating_proj/bf16/bf16_gdn_norm_gating_proj_27.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_partial.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_reduce.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/dynamic_grouped_conv/w8/w8_dynamic_grouped_conv_add_materialized.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear/bf16/bf16_n256_k5120.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_swiglu/w8/w8_dflash2_linear_swiglu.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear/w8/w8_feature.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_topk/w8.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_topk/w8_m64.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_topk/fp8.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_topk/fp8_m64.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_topk/q4.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  set_source_files_properties(ops/linear_topk/q4_m64.cu TARGET_DIRECTORY ninfer_ops PROPERTIES HEADER_FILE_ONLY ON)
  target_sources(ninfer_ops PRIVATE ops/dynamic_grouped_conv/volta/dflash2_dynamic_conv_volta.cu)
  target_sources(ninfer_ops PRIVATE ops/attn_input_proj/w8/w8_dflash2_attn_input_volta.cu)
  target_sources(ninfer_ops PRIVATE ops/launcher/rmsnorm_pack_tail_volta.cu)
  target_sources(ninfer_ops PRIVATE ops/linear_topk/dflash2_linear_topk_volta.cu)
  target_sources(ninfer_ops PRIVATE ops/linear/bf16/bf16_n256_k5120_volta.cu)
  target_sources(ninfer_ops PRIVATE ops/candidate_selector/bf16/dflash2_selector_volta.cu)
  target_sources(ninfer_ops PRIVATE ops/common/dflash2_rmsnorm_rope_volta.cu)
  target_link_libraries(ninfer_ops PRIVATE nvidia::cutlass::cutlass)
  target_compile_options(ninfer_ops PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()
ninfer_internal_includes(ninfer_ops)
ninfer_cuda_archive(ninfer_ops)
target_link_libraries(ninfer_ops PUBLIC ninfer_core PRIVATE ninfer_nvfp4_non_rdc)

# Product-independent text and media mechanisms.
add_library(ninfer_text STATIC
  text/unicode.cpp
  ${PROJECT_SOURCE_DIR}/third_party/utf8proc/utf8proc.c)
ninfer_internal_includes(ninfer_text)
target_compile_definitions(ninfer_text PUBLIC UTF8PROC_STATIC)

if(NINFER_BUILD_MEDIA_DECODE)
  add_library(ninfer_media_decode STATIC
    media/decode/decode.cpp)
  ninfer_internal_includes(ninfer_media_decode)
  target_link_libraries(ninfer_media_decode PRIVATE PkgConfig::FFMPEG)
endif()

if(NINFER_BUILD_MEDIA_ACQUIRE)
  # Product-only path/data/HTTP acquisition. No target package links this library.
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/acquire.cpp)
  ninfer_internal_includes(ninfer_media_acquire)
  target_link_libraries(ninfer_media_acquire PRIVATE PkgConfig::LIBCURL)
elseif(NINFER_BUILD_WINDOWS_SERVE)
  # Native Windows byte/data/HTTP acquisition; no target package links transport code.
  add_library(ninfer_media_acquire STATIC
    product/media_acquire/windows.cpp)
  ninfer_internal_includes(ninfer_media_acquire)
  target_link_libraries(ninfer_media_acquire PRIVATE winhttp ws2_32)
  target_compile_definitions(ninfer_media_acquire PRIVATE
    NOMINMAX WIN32_LEAN_AND_MEAN _WIN32_WINNT=0x0A00)
endif()

if(NINFER_BUILD_PROMPT_INPUT)
  # Owning command-line message input shared by product entry points and target diagnostics.
  add_library(ninfer_product_prompt_input STATIC
    product/prompt_input/prompt_input.cpp)
  ninfer_internal_includes(ninfer_product_prompt_input)
  target_link_libraries(ninfer_product_prompt_input PRIVATE ninfer_media_acquire)
endif()

if(NINFER_BUILD_APPS OR BUILD_TESTING OR NINFER_BUILD_WINDOWS_SERVE)
  add_library(ninfer_product_logging STATIC
    product/logging/logging.cpp
    product/logging/pretty_format.cpp
    product/logging/startup_log.cpp)
  ninfer_internal_includes(ninfer_product_logging)
  target_link_libraries(ninfer_product_logging PUBLIC spdlog::spdlog)
endif()

# One cold closed registry plus the opaque public product Engine. Family and exact-target
# directories contribute their sources directly to this single compile owner below.
add_library(ninfer_engine STATIC
  targets/registry.cpp
  runtime/contract/sampling.cpp
  runtime/engine/engine.cpp)
ninfer_internal_includes(ninfer_engine)
target_include_directories(ninfer_engine PUBLIC ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(targets/qwen3_6)
add_subdirectory(targets/qwen3_6_27b)
add_subdirectory(targets/qwen3_6_35b_a3b)

target_link_libraries(ninfer_engine
  PRIVATE
    ninfer_artifact
    ninfer_core
    ninfer_ops
    ninfer_text)
if(NINFER_BUILD_MEDIA_DECODE)
  target_link_libraries(ninfer_engine PRIVATE ninfer_media_decode)
endif()
add_library(ninfer::engine ALIAS ninfer_engine)

if(NINFER_BUILD_SERVE)
  if(WIN32 AND NINFER_BUILD_WINDOWS_SERVE)
    set(NINFER_WINDOWS_CHAT_PAGE_HTML_FILE
      "${CMAKE_CURRENT_SOURCE_DIR}/serve/web/index.html")
    if(NOT EXISTS "${NINFER_WINDOWS_CHAT_PAGE_HTML_FILE}")
      message(FATAL_ERROR
        "Windows serving requires src/serve/web/index.html")
    endif()
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
      "${NINFER_WINDOWS_CHAT_PAGE_HTML_FILE}")
    file(READ "${NINFER_WINDOWS_CHAT_PAGE_HTML_FILE}" NINFER_WINDOWS_CHAT_PAGE_HTML)
    string(FIND "${NINFER_WINDOWS_CHAT_PAGE_HTML}" ")NINFERCHAT\""
      NINFER_WINDOWS_CHAT_PAGE_DELIMITER_POSITION)
    if(NOT NINFER_WINDOWS_CHAT_PAGE_DELIMITER_POSITION EQUAL -1)
      message(FATAL_ERROR
        "src/serve/web/index.html contains the reserved raw-string delimiter )NINFERCHAT\"")
    endif()
    set(NINFER_WINDOWS_CHAT_PAGE_REMAINING "${NINFER_WINDOWS_CHAT_PAGE_HTML}")
    set(NINFER_WINDOWS_CHAT_PAGE_LINE 1)
    while(TRUE)
      string(FIND "${NINFER_WINDOWS_CHAT_PAGE_REMAINING}" "\n"
        NINFER_WINDOWS_CHAT_PAGE_NEWLINE_POSITION)
      if(NINFER_WINDOWS_CHAT_PAGE_NEWLINE_POSITION EQUAL -1)
        set(NINFER_WINDOWS_CHAT_PAGE_CURRENT_LINE "${NINFER_WINDOWS_CHAT_PAGE_REMAINING}")
        set(NINFER_WINDOWS_CHAT_PAGE_AT_END TRUE)
      else()
        string(SUBSTRING "${NINFER_WINDOWS_CHAT_PAGE_REMAINING}" 0
          ${NINFER_WINDOWS_CHAT_PAGE_NEWLINE_POSITION} NINFER_WINDOWS_CHAT_PAGE_CURRENT_LINE)
        math(EXPR NINFER_WINDOWS_CHAT_PAGE_NEXT_LINE_START
          "${NINFER_WINDOWS_CHAT_PAGE_NEWLINE_POSITION} + 1")
        string(SUBSTRING "${NINFER_WINDOWS_CHAT_PAGE_REMAINING}"
          ${NINFER_WINDOWS_CHAT_PAGE_NEXT_LINE_START} -1 NINFER_WINDOWS_CHAT_PAGE_REMAINING)
        set(NINFER_WINDOWS_CHAT_PAGE_AT_END FALSE)
      endif()
      string(HEX "${NINFER_WINDOWS_CHAT_PAGE_CURRENT_LINE}" NINFER_WINDOWS_CHAT_PAGE_LINE_HEX)
      string(LENGTH "${NINFER_WINDOWS_CHAT_PAGE_LINE_HEX}" NINFER_WINDOWS_CHAT_PAGE_LINE_HEX_LENGTH)
      math(EXPR NINFER_WINDOWS_CHAT_PAGE_LINE_BYTES
        "${NINFER_WINDOWS_CHAT_PAGE_LINE_HEX_LENGTH} / 2")
      if(NINFER_WINDOWS_CHAT_PAGE_LINE_BYTES GREATER 12000)
        message(FATAL_ERROR
          "src/serve/web/index.html line ${NINFER_WINDOWS_CHAT_PAGE_LINE} exceeds 12000 bytes; "
          "split this line so MSVC can compile the embedded page")
      endif()
      if(NINFER_WINDOWS_CHAT_PAGE_AT_END)
        break()
      endif()
      math(EXPR NINFER_WINDOWS_CHAT_PAGE_LINE "${NINFER_WINDOWS_CHAT_PAGE_LINE} + 1")
    endwhile()
    string(REPLACE "\n" "\n)NINFERCHAT\"\nR\"NINFERCHAT("
      NINFER_WINDOWS_CHAT_PAGE_HTML_CHUNKS "${NINFER_WINDOWS_CHAT_PAGE_HTML}")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated/serve/web")
    configure_file(
      "${CMAKE_CURRENT_SOURCE_DIR}/serve/web/chat_page.h.in"
      "${CMAKE_CURRENT_BINARY_DIR}/generated/serve/web/chat_page.h"
      @ONLY)
  endif()

  # HTTP protocol/product adapter. No serving header is part of the public API.
  add_library(ninfer_serve STATIC
    serve/http_transport.cpp
    serve/request_validation.cpp
    serve/openai_chat_request.cpp
    serve/openai_chat_response.cpp
    serve/openai_chat_http.cpp
    serve/openai_common.cpp
    serve/openai_responses_request.cpp
    serve/openai_responses_state.cpp
    serve/openai_responses_response.cpp
    serve/anthropic_messages_request.cpp
    serve/anthropic_messages_response.cpp
    serve/anthropic_messages_http.cpp
    serve/openai_responses_store.cpp
    serve/translate.cpp
    serve/generation_service.cpp
    serve/request_events.cpp
    serve/operational_log.cpp
    serve/request_log.cpp
    serve/http_server.cpp
    serve/openai_responses_http.cpp
    serve/serve_options.cpp)
  ninfer_internal_includes(ninfer_serve)
  target_include_directories(ninfer_serve PRIVATE
    ${PROJECT_SOURCE_DIR}/third_party/cpp-httplib)
  if(WIN32 AND NINFER_BUILD_WINDOWS_SERVE)
    target_sources(ninfer_serve PRIVATE serve/session_cache.cpp)
    target_link_libraries(ninfer_serve PRIVATE bcrypt)
    target_include_directories(ninfer_serve PRIVATE
      "${CMAKE_CURRENT_BINARY_DIR}/generated")
  endif()
  target_link_libraries(ninfer_serve
    PUBLIC ninfer_engine Threads::Threads
    PRIVATE ninfer_media_acquire ninfer_product_logging CUDA::cudart)
  if(WIN32 AND NINFER_BUILD_WINDOWS_SERVE)
    target_compile_definitions(ninfer_serve
      PUBLIC NINFER_WINDOWS_SERVE=1
      _WIN32_WINNT=0x0A00 NOMINMAX WIN32_LEAN_AND_MEAN)
  endif()
  if(WIN32)
    target_link_libraries(ninfer_serve PRIVATE ws2_32)
  endif()
endif()
