#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <stdexcept>

namespace ninfer {

enum class ContextCacheOwnerKind : std::uint8_t {
    PrivateContinuation,
    SharedPrefix,
};

class ContextCacheUnsupported final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

class ContextCacheOwnershipError final : public std::logic_error {
public:
    using std::logic_error::logic_error;
};

class ContextCacheBudgetExceeded final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Streaming sink/source used by the Engine's opaque context-cache archive. Calls receive bounded
// chunks; implementations must consume or produce the full span and throw on an I/O failure.
// Framing and on-disk transactions belong to the caller.
class ContextCacheWriter {
public:
    virtual ~ContextCacheWriter() = default;

    virtual void write(std::span<const std::byte> bytes) = 0;
};

class ContextCacheReader {
public:
    virtual ~ContextCacheReader() = default;

    virtual void read_exact(std::span<std::byte> bytes) = 0;
    [[nodiscard]] virtual std::uint64_t remaining_bytes() const = 0;
};

struct ContextCacheSnapshotStats {
    std::uint64_t bytes                  = 0;
    std::uint64_t private_continuations = 0;
    std::uint64_t shared_prefixes       = 0;
    std::uint64_t checkpoints           = 0;
    // Private owners that carry a complete SessionIndex binding and can resume a session.
    std::uint64_t session_continuations = 0;
};

inline constexpr std::uint64_t kDefaultMaximumContextCacheSnapshotBytes = 16ULL << 30;

// Immutable host-owned archive captured by an Engine. The snapshot can outlive the Engine and
// can be written by a background thread without touching Program, CUDA, or Engine state.
class ContextCacheSnapshot {
public:
    ContextCacheSnapshot() noexcept;
    ~ContextCacheSnapshot();

    ContextCacheSnapshot(ContextCacheSnapshot&&) noexcept;
    ContextCacheSnapshot& operator=(ContextCacheSnapshot&&) noexcept;

    ContextCacheSnapshot(const ContextCacheSnapshot&)            = delete;
    ContextCacheSnapshot& operator=(const ContextCacheSnapshot&) = delete;

    [[nodiscard]] explicit operator bool() const noexcept;
    [[nodiscard]] const ContextCacheSnapshotStats& stats() const noexcept;
    // Actual immutable chunk storage capacity; it can exceed stats().bytes by one partial chunk.
    [[nodiscard]] std::uint64_t allocated_bytes() const noexcept;
    void write_to(ContextCacheWriter& writer) const;

private:
    class Impl;
    explicit ContextCacheSnapshot(std::unique_ptr<Impl> impl) noexcept;
    std::unique_ptr<Impl> impl_;

    friend class Engine;
};

} // namespace ninfer
