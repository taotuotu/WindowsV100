#pragma once

#include "core/host_kv_arena.h"
#include "ninfer/context_cache.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::context_cache_snapshot {

inline constexpr std::uint32_t kOwnerRecordMagic = 0x31534351U; // QSC1, little endian
inline constexpr std::uint32_t kOwnerRecordVersion = 3;
inline constexpr std::size_t kMetadataChunkBytes = 64U << 10;
inline constexpr std::uint64_t kMaximumVisionItems = 1U << 16;
inline constexpr std::uint64_t kMaximumVisionSpans = 1U << 20;
inline constexpr std::uint64_t kMaximumVisionTimestamps = 1U << 20;
inline constexpr std::uint64_t kMaximumKVPlanes = 4096;

class CountingContextCacheWriter final : public ninfer::ContextCacheWriter {
public:
    void write(std::span<const std::byte> bytes) override { add(bytes.size()); }

    void add(std::uint64_t count) {
        if (count > std::numeric_limits<std::uint64_t>::max() - bytes_) {
            throw std::overflow_error("context-cache owner record size overflow");
        }
        bytes_ += count;
    }

    [[nodiscard]] std::uint64_t bytes() const noexcept { return bytes_; }

private:
    std::uint64_t bytes_ = 0;
};

class BoundedContextCacheReader final : public ninfer::ContextCacheReader {
public:
    BoundedContextCacheReader(ninfer::ContextCacheReader& source, std::uint64_t limit) noexcept
        : source_(&source), remaining_(limit) {}

    void read_exact(std::span<std::byte> bytes) override {
        if (bytes.size() > remaining_) {
            throw std::invalid_argument("context-cache owner record is truncated");
        }
        source_->read_exact(bytes);
        remaining_ -= bytes.size();
    }

    [[nodiscard]] std::uint64_t remaining_bytes() const override { return remaining_; }

private:
    ninfer::ContextCacheReader* source_ = nullptr;
    std::uint64_t remaining_ = 0;
};

inline void write_raw(ninfer::ContextCacheWriter& writer, std::span<const std::byte> bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t count = std::min(kMetadataChunkBytes, bytes.size() - offset);
        writer.write(bytes.subspan(offset, count));
        offset += count;
    }
}

inline void write_payload(ninfer::ContextCacheWriter& writer,
                          std::span<const std::byte> payload) {
    if (auto* const counter = dynamic_cast<CountingContextCacheWriter*>(&writer)) {
        counter->add(payload.size());
        return;
    }
    write_raw(writer, payload);
}

inline void read_raw(ninfer::ContextCacheReader& reader, std::span<std::byte> bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const std::size_t count = std::min(kMetadataChunkBytes, bytes.size() - offset);
        reader.read_exact(bytes.subspan(offset, count));
        offset += count;
    }
}

template <class T>
inline void write_little_endian_span(ninfer::ContextCacheWriter& writer,
                                     std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::endian::native == std::endian::little,
                  "context-cache snapshot currently requires a little-endian host");
    write_raw(writer, std::as_bytes(values));
}

template <class T>
inline void read_little_endian_span(ninfer::ContextCacheReader& reader, std::span<T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    static_assert(std::endian::native == std::endian::little,
                  "context-cache snapshot currently requires a little-endian host");
    read_raw(reader, std::as_writable_bytes(values));
}

inline void write_u8(ninfer::ContextCacheWriter& writer, std::uint8_t value) {
    const std::byte byte{value};
    writer.write({&byte, 1});
}

inline void write_u32(ninfer::ContextCacheWriter& writer, std::uint32_t value) {
    const std::array<std::byte, 4> bytes{
        std::byte(value & 0xffU), std::byte((value >> 8U) & 0xffU),
        std::byte((value >> 16U) & 0xffU), std::byte((value >> 24U) & 0xffU)};
    writer.write(bytes);
}

inline void write_u64(ninfer::ContextCacheWriter& writer, std::uint64_t value) {
    const std::array<std::byte, 8> bytes{
        std::byte(value & 0xffU),        std::byte((value >> 8U) & 0xffU),
        std::byte((value >> 16U) & 0xffU), std::byte((value >> 24U) & 0xffU),
        std::byte((value >> 32U) & 0xffU), std::byte((value >> 40U) & 0xffU),
        std::byte((value >> 48U) & 0xffU), std::byte((value >> 56U) & 0xffU)};
    writer.write(bytes);
}

inline void write_i32(ninfer::ContextCacheWriter& writer, std::int32_t value) {
    write_u32(writer, static_cast<std::uint32_t>(value));
}

inline void write_bool(ninfer::ContextCacheWriter& writer, bool value) {
    write_u8(writer, value ? 1U : 0U);
}

inline std::uint8_t read_u8(ninfer::ContextCacheReader& reader) {
    std::byte byte{};
    reader.read_exact({&byte, 1});
    return std::to_integer<std::uint8_t>(byte);
}

inline std::uint32_t read_u32(ninfer::ContextCacheReader& reader) {
    std::array<std::byte, 4> bytes{};
    reader.read_exact(bytes);
    return std::to_integer<std::uint32_t>(bytes[0]) |
           (std::to_integer<std::uint32_t>(bytes[1]) << 8U) |
           (std::to_integer<std::uint32_t>(bytes[2]) << 16U) |
           (std::to_integer<std::uint32_t>(bytes[3]) << 24U);
}

inline std::uint64_t read_u64(ninfer::ContextCacheReader& reader) {
    std::array<std::byte, 8> bytes{};
    reader.read_exact(bytes);
    std::uint64_t value = 0;
    for (std::uint32_t index = 0; index < bytes.size(); ++index) {
        value |= std::to_integer<std::uint64_t>(bytes[index]) << (8U * index);
    }
    return value;
}

inline std::int32_t read_i32(ninfer::ContextCacheReader& reader) {
    return static_cast<std::int32_t>(read_u32(reader));
}

inline bool read_bool(ninfer::ContextCacheReader& reader) {
    const std::uint8_t value = read_u8(reader);
    if (value > 1) { throw std::invalid_argument("context-cache boolean field is invalid"); }
    return value != 0;
}

inline std::uint32_t read_count(ninfer::ContextCacheReader& reader, std::uint64_t maximum,
                                const char* label) {
    const std::uint32_t count = read_u32(reader);
    if (count > maximum) { throw std::invalid_argument(label); }
    return count;
}

inline void write_prefill_work(ninfer::ContextCacheWriter& writer,
                               const runtime::PrefillWork& work) {
    write_u64(writer, work.chunks);
    write_u64(writer, work.tokens);
    write_u64(writer, work.attention_pairs);
    write_u64(writer, work.vision_items);
    write_u64(writer, work.vision_patches);
}

inline runtime::PrefillWork read_prefill_work(ninfer::ContextCacheReader& reader) {
    return runtime::PrefillWork{.chunks = read_u64(reader),
                                .tokens = read_u64(reader),
                                .attention_pairs = read_u64(reader),
                                .vision_items = read_u64(reader),
                                .vision_patches = read_u64(reader)};
}

inline void write_kv_geometry(ninfer::ContextCacheWriter& writer,
                              const ninfer::KVPageGeometry& geometry) {
    write_u32(writer, geometry.page_tokens);
    write_u8(writer, static_cast<std::uint8_t>(geometry.device_plane_order));
    if (geometry.planes.size() > kMaximumKVPlanes) {
        throw std::invalid_argument("context-cache KV geometry has too many planes");
    }
    write_u32(writer, static_cast<std::uint32_t>(geometry.planes.size()));
    for (const ninfer::KVPlaneGeometry& plane : geometry.planes) {
        write_u8(writer, static_cast<std::uint8_t>(plane.dtype));
        write_i32(writer, plane.leading_extent);
        write_i32(writer, plane.head_extent);
        write_u64(writer, plane.alignment);
    }
}

inline void validate_kv_geometry(ninfer::ContextCacheReader& reader,
                                 const ninfer::KVPageGeometry& expected) {
    const std::uint32_t page_tokens = read_u32(reader);
    const std::uint8_t order = read_u8(reader);
    const std::uint32_t plane_count = read_count(reader, kMaximumKVPlanes,
                                                 "context-cache KV plane count is invalid");
    if (page_tokens != expected.page_tokens ||
        order != static_cast<std::uint8_t>(expected.device_plane_order) ||
        plane_count != expected.planes.size()) {
        throw std::invalid_argument("context-cache KV geometry does not match the Program");
    }
    for (std::size_t index = 0; index < expected.planes.size(); ++index) {
        const std::uint8_t dtype = read_u8(reader);
        const std::int32_t leading = read_i32(reader);
        const std::int32_t heads = read_i32(reader);
        const std::uint64_t alignment = read_u64(reader);
        const ninfer::KVPlaneGeometry& plane = expected.planes[index];
        if (dtype != static_cast<std::uint8_t>(plane.dtype) || leading != plane.leading_extent ||
            heads != plane.head_extent || alignment != plane.alignment) {
            throw std::invalid_argument("context-cache KV plane geometry does not match");
        }
    }
}

} // namespace ninfer::targets::qwen3_6::detail::context_cache_snapshot
