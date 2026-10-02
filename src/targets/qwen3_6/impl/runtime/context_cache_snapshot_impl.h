#pragma once

#include "targets/qwen3_6/impl/runtime/context_cache_snapshot.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {
namespace {

using namespace context_cache_snapshot;

enum class SnapshotObjectKind : std::uint8_t {
    StateImage = 1,
    MainKVPage = 2,
    BackendKVPage = 3,
};

struct SnapshotObjectKey {
    SnapshotObjectKind kind = SnapshotObjectKind::StateImage;
    std::uint64_t process_key = 0;

    [[nodiscard]] friend bool operator==(const SnapshotObjectKey&,
                                         const SnapshotObjectKey&) noexcept = default;
};

struct SnapshotObjectKeyHash {
    [[nodiscard]] std::size_t operator()(const SnapshotObjectKey& value) const noexcept {
        const std::uint64_t mixed = value.process_key ^
                                    (static_cast<std::uint64_t>(value.kind) << 57U) ^
                                    (value.process_key >> 29U);
        return static_cast<std::size_t>(mixed);
    }
};

struct OwnerCapabilityKey {
    std::uint32_t index = 0;
    std::uint64_t generation = 0;

    [[nodiscard]] friend bool operator==(const OwnerCapabilityKey&,
                                         const OwnerCapabilityKey&) noexcept = default;
};

struct OwnerCapabilityKeyHash {
    [[nodiscard]] std::size_t operator()(const OwnerCapabilityKey& value) const noexcept {
        std::uint64_t mixed = value.generation ^
                              (static_cast<std::uint64_t>(value.index) * 0x9e3779b97f4a7c15ULL);
        mixed ^= mixed >> 30U;
        mixed *= 0xbf58476d1ce4e5b9ULL;
        mixed ^= mixed >> 27U;
        mixed *= 0x94d049bb133111ebULL;
        mixed ^= mixed >> 31U;
        return static_cast<std::size_t>(mixed);
    }
};

struct ContextCacheExportState {
    const void* owner = nullptr;
    std::uint64_t next_link_id = 1;
    std::unordered_map<SnapshotObjectKey, std::uint64_t, SnapshotObjectKeyHash> object_ids;
    std::unordered_set<OwnerCapabilityKey, OwnerCapabilityKeyHash> private_owners;
    std::unordered_set<OwnerCapabilityKey, OwnerCapabilityKeyHash> shared_owners;
    bool closed = false;
};

struct ContextCacheImportObject {
    std::uint64_t link_id = 0;
    SnapshotObjectKind kind = SnapshotObjectKind::StateImage;
    StateImageHandle state;
    LogicalKVPageHandle page;
    std::optional<HostKVAllocation> host_allocation;
    std::optional<HostKVExtentCapability> published_host_extent;
    std::uint32_t committed_columns = 0;
    std::uint32_t state_references = 0;
    std::uint32_t primary_owners = 0;
    std::uint32_t address_references = 0;
    bool device_replica = false;
    bool host_replica = false;
    bool published = false;
};

struct ContextCacheImportAddress {
    KVAddressSpaceHandle address;
    std::uint32_t frontier = 0;
    std::uint32_t checkpoint_frontier = 0;
    std::vector<std::uint64_t> page_links;
    bool backend = false;
    bool installed = false;
};

struct ContextCacheStagedAddress {
    KVAddressSpaceHandle address;
    bool backend = false;
};

struct ContextCacheImportedPrivateOwner {
    SequenceState sequence;
    ContextCacheImportAddress main_address;
    std::optional<ContextCacheImportAddress> backend_address;
    qwen3_6::ContinuationSummary summary;
    std::uint32_t slot_index = 0;
};

struct ContextCacheImportedSharedOwner {
    SharedPrefixState shared;
    ContextCacheImportAddress main_address;
    std::optional<ContextCacheImportAddress> backend_address;
    qwen3_6::SharedPrefixSummary summary;
    std::uint32_t slot_index = 0;
};

using ContextCacheImportedOwner =
    std::variant<ContextCacheImportedPrivateOwner, ContextCacheImportedSharedOwner>;

struct ContextCacheImportState {
    ProgramImplCore* owner = nullptr;
    std::size_t expected_owner_count = 0;
    std::uint64_t next_link_id = 1;
    std::unordered_map<std::uint64_t, std::size_t> object_indices;
    std::vector<ContextCacheImportObject> objects;
    std::vector<ContextCacheImportedOwner> owners;
    std::vector<ContextCacheStagedAddress> allocated_addresses;
    std::vector<HostKVExtentCapability> published_host_extents;
    std::vector<std::uint32_t> private_slot_indices;
    std::vector<std::uint32_t> shared_slot_indices;
    std::vector<qwen3_6::ContextCacheOwnerImport<Variant>> returned_owners;
    DeviceKVPageReservation main_reservation;
    DeviceKVPageReservation backend_reservation;
    bool committed = false;
    bool rollback_failed = false;
};

[[nodiscard]] std::uint64_t snapshot_state_layout_signature(
    const qwen3_6::StateImageHostLayout& layout) noexcept {
    std::uint64_t hash = 1469598103934665603ULL;
    const auto mix = [&](std::uint64_t value) {
        for (std::uint32_t byte = 0; byte < 8; ++byte) {
            hash ^= static_cast<std::uint8_t>(value >> (8U * byte));
            hash *= 1099511628211ULL;
        }
    };
    const auto region = [&](const LayoutRegion& value) {
        mix(value.offset);
        mix(value.bytes);
        mix(value.alignment);
    };
    mix(layout.spec.linear.layers);
    mix(static_cast<std::uint32_t>(layout.spec.linear.conv_channels));
    mix(static_cast<std::uint32_t>(layout.spec.linear.conv_width));
    mix(static_cast<std::uint32_t>(layout.spec.linear.value_heads));
    mix(static_cast<std::uint32_t>(layout.spec.linear.value_head_dim));
    mix(static_cast<std::uint32_t>(layout.spec.linear.key_head_dim));
    mix(static_cast<std::uint8_t>(layout.spec.linear.conv_dtype));
    mix(static_cast<std::uint32_t>(layout.spec.hidden));
    mix(layout.spec.dflash_local.has_value());
    if (layout.spec.dflash_local) {
        mix(layout.spec.dflash_local->layers);
        mix(layout.spec.dflash_local->capacity);
        mix(static_cast<std::uint32_t>(layout.spec.dflash_local->kv_heads));
        mix(static_cast<std::uint32_t>(layout.spec.dflash_local->head_dim));
    }
    region(layout.linear_conv);
    mix(layout.linear_conv_layer_bytes);
    region(layout.linear_recurrent);
    mix(layout.linear_recurrent_layer_bytes);
    region(layout.continuation_hidden);
    mix(layout.dflash_local_k.has_value());
    if (layout.dflash_local_k) { region(*layout.dflash_local_k); }
    mix(layout.dflash_local_v.has_value());
    if (layout.dflash_local_v) { region(*layout.dflash_local_v); }
    mix(layout.dflash_local_layer_bytes);
    mix(layout.image_bytes);
    return hash == 0 ? 1 : hash;
}

[[nodiscard]] bool supported_snapshot_profile(const ProgramImplCore& program) noexcept {
    constexpr bool supported_target = TextConfig::hidden == 5120 && TextConfig::layers == 64 &&
                                      Variant::maximum_context == 262144;
    const bool supported_kv = program.kv_storage == KvCacheStorage::BFloat16 ||
                              program.kv_storage == KvCacheStorage::Int8Group64 ||
                              program.kv_storage == KvCacheStorage::Fp8E4M3Row256;
    const bool supported_backend = program.speculative_backend == SpeculativeBackend::None ||
                                   program.speculative_backend == SpeculativeBackend::Mtp;
    return supported_target && supported_kv && supported_backend && !program.causal_scoring &&
           program.state_images && !program.state_images->host_layout().spec.dflash_local &&
           program.text_kv_pages && program.text_kv_addresses &&
           ((program.speculative_backend == SpeculativeBackend::Mtp &&
             program.backend_kv_pages && program.backend_kv_addresses) ||
            (program.speculative_backend == SpeculativeBackend::None &&
             !program.backend_kv_pages && !program.backend_kv_addresses));
}

void require_snapshot_profile(const ProgramImplCore& program) {
    if (!supported_snapshot_profile(program)) {
        throw ninfer::ContextCacheUnsupported(
            "this Qwen backend or KV profile has no persistent context-cache codec");
    }
}

void write_target_header(ninfer::ContextCacheWriter& writer, const ProgramImplCore& program,
                         ninfer::ContextCacheOwnerKind kind) {
    write_u8(writer, static_cast<std::uint8_t>(kind));
    write_u8(writer, static_cast<std::uint8_t>(program.kv_storage));
    write_u8(writer, static_cast<std::uint8_t>(program.speculative_backend));
    write_u8(writer, static_cast<std::uint8_t>(program.proposal_head));
    write_u32(writer, program.draft_window);
    write_u32(writer, program.capacity);
    write_u32(writer, Variant::maximum_context);
    write_u64(writer, snapshot_state_layout_signature(program.state_images->host_layout()));
    write_kv_geometry(writer, program.text_kv_pages->physical_pool().geometry());
    write_bool(writer, program.backend_kv_pages != nullptr);
    if (program.backend_kv_pages) {
        write_kv_geometry(writer, program.backend_kv_pages->physical_pool().geometry());
    }
}

void validate_target_header(ninfer::ContextCacheReader& reader, const ProgramImplCore& program,
                            ninfer::ContextCacheOwnerKind expected_kind) {
    const std::uint8_t kind = read_u8(reader);
    const std::uint8_t kv_storage = read_u8(reader);
    const std::uint8_t backend = read_u8(reader);
    const std::uint8_t proposal = read_u8(reader);
    const std::uint32_t draft_window = read_u32(reader);
    const std::uint32_t capacity = read_u32(reader);
    const std::uint32_t maximum_context = read_u32(reader);
    const std::uint64_t state_signature = read_u64(reader);
    if (kind != static_cast<std::uint8_t>(expected_kind) ||
        kv_storage != static_cast<std::uint8_t>(program.kv_storage) ||
        backend != static_cast<std::uint8_t>(program.speculative_backend) ||
        proposal != static_cast<std::uint8_t>(program.proposal_head) ||
        draft_window != program.draft_window || capacity != program.capacity ||
        maximum_context != Variant::maximum_context ||
        state_signature != snapshot_state_layout_signature(program.state_images->host_layout())) {
        throw std::invalid_argument("context-cache Program compatibility metadata does not match");
    }
    validate_kv_geometry(reader, program.text_kv_pages->physical_pool().geometry());
    const bool has_backend = read_bool(reader);
    if (has_backend != (program.backend_kv_pages != nullptr)) {
        throw std::invalid_argument("context-cache backend pool presence does not match");
    }
    if (has_backend) {
        validate_kv_geometry(reader, program.backend_kv_pages->physical_pool().geometry());
    }
}

[[nodiscard]] OwnerCapabilityKey owner_link_key(std::uint32_t index,
                                                 std::uint64_t generation) noexcept {
    return OwnerCapabilityKey{.index = index, .generation = generation};
}

void write_prefix_identity(ninfer::ContextCacheWriter& writer,
                            const qwen3_6::detail::ResidentPrefixIdentity& identity) {
    write_little_endian_span(writer, std::span<const std::uint8_t>(identity.token_types()));
    for (const auto& axis : identity.positions()) {
        write_little_endian_span(writer, std::span<const std::int32_t>(axis));
    }
    const auto& vision = identity.vision_items();
    if (vision.size() > kMaximumVisionItems) {
        throw std::invalid_argument("context-cache Vision identity has too many items");
    }
    write_u32(writer, static_cast<std::uint32_t>(vision.size()));
    for (const qwen3_6::VisionItem& item : vision) {
        write_u8(writer, static_cast<std::uint8_t>(item.modality));
        write_i32(writer, item.grid.temporal);
        write_i32(writer, item.grid.height);
        write_i32(writer, item.grid.width);
        write_u64(writer, item.patch_begin);
        write_u64(writer, item.patch_count);
        write_raw(writer, std::as_bytes(std::span(item.content_digest)));
        if (item.timestamps.size() > kMaximumVisionTimestamps ||
            item.token_spans.size() > kMaximumVisionSpans) {
            throw std::invalid_argument("context-cache Vision metadata is too large");
        }
        write_u32(writer, static_cast<std::uint32_t>(item.timestamps.size()));
        for (const double timestamp : item.timestamps) {
            write_u64(writer, std::bit_cast<std::uint64_t>(timestamp));
        }
        write_u32(writer, static_cast<std::uint32_t>(item.token_spans.size()));
        for (const qwen3_6::TokenSpan& span : item.token_spans) {
            write_u64(writer, span.begin);
            write_u64(writer, span.count);
        }
    }
    const auto& rewrites = identity.rewrite_execution_frontiers();
    if (rewrites.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("context-cache rewrite identity is too large");
    }
    write_u32(writer, static_cast<std::uint32_t>(rewrites.size()));
    write_little_endian_span(writer, std::span<const std::uint32_t>(rewrites));
}

[[nodiscard]] qwen3_6::detail::ResidentPrefixIdentity
read_prefix_identity(ninfer::ContextCacheReader& reader, std::uint32_t token_count) {
    std::vector<std::uint8_t> token_types(token_count);
    read_little_endian_span(reader, std::span<std::uint8_t>(token_types));
    std::array<std::vector<std::int32_t>, 3> positions;
    for (auto& axis : positions) {
        axis.resize(token_count);
        read_little_endian_span(reader, std::span<std::int32_t>(axis));
    }
    const std::uint32_t vision_count =
        read_count(reader, kMaximumVisionItems, "context-cache Vision item count is invalid");
    std::vector<qwen3_6::VisionItem> vision;
    vision.reserve(vision_count);
    for (std::uint32_t index = 0; index < vision_count; ++index) {
        qwen3_6::VisionItem item;
        item.modality = static_cast<qwen3_6::PromptModality>(read_u8(reader));
        item.grid.temporal = read_i32(reader);
        item.grid.height = read_i32(reader);
        item.grid.width = read_i32(reader);
        const std::uint64_t patch_begin = read_u64(reader);
        const std::uint64_t patch_count = read_u64(reader);
        if (patch_begin > std::numeric_limits<std::size_t>::max() ||
            patch_count > std::numeric_limits<std::size_t>::max()) {
            throw std::invalid_argument("context-cache Vision patch range is not representable");
        }
        item.patch_begin = static_cast<std::size_t>(patch_begin);
        item.patch_count = static_cast<std::size_t>(patch_count);
        reader.read_exact(std::as_writable_bytes(std::span(item.content_digest)));
        const std::uint32_t timestamp_count = read_count(
            reader, kMaximumVisionTimestamps, "context-cache Vision timestamp count is invalid");
        item.timestamps.reserve(timestamp_count);
        for (std::uint32_t timestamp_index = 0; timestamp_index < timestamp_count;
             ++timestamp_index) {
            item.timestamps.push_back(std::bit_cast<double>(read_u64(reader)));
        }
        const std::uint32_t span_count =
            read_count(reader, kMaximumVisionSpans, "context-cache Vision span count is invalid");
        item.token_spans.reserve(span_count);
        for (std::uint32_t span_index = 0; span_index < span_count; ++span_index) {
            const std::uint64_t begin = read_u64(reader);
            const std::uint64_t count = read_u64(reader);
            if (begin > std::numeric_limits<std::size_t>::max() ||
                count > std::numeric_limits<std::size_t>::max()) {
                throw std::invalid_argument("context-cache Vision token span is not representable");
            }
            item.token_spans.push_back(qwen3_6::TokenSpan{
                .begin = static_cast<std::size_t>(begin),
                .count = static_cast<std::size_t>(count),
            });
        }
        vision.push_back(std::move(item));
    }
    const std::uint32_t rewrite_count =
        read_count(reader, token_count, "context-cache rewrite frontier count is invalid");
    std::vector<std::uint32_t> rewrites(rewrite_count);
    read_little_endian_span(reader, std::span<std::uint32_t>(rewrites));

    qwen3_6::detail::ResidentPrefixIdentity identity;
    identity.assign_snapshot(std::move(token_types), std::move(positions), std::move(vision),
                             std::move(rewrites));
    return identity;
}

void write_token_ledger(ninfer::ContextCacheWriter& writer, std::span<const TokenId> tokens) {
    write_little_endian_span(writer, tokens);
}

[[nodiscard]] std::vector<TokenId> read_token_ledger(ninfer::ContextCacheReader& reader,
                                                     std::uint32_t count,
                                                     std::uint32_t maximum_context) {
    if (count > maximum_context) {
        throw std::invalid_argument("context-cache token ledger exceeds target context");
    }
    std::vector<TokenId> tokens(count);
    read_little_endian_span(reader, std::span<TokenId>(tokens));
    return tokens;
}

} // namespace
} // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
