#pragma once

#if defined(NINFER_WINDOWS_SERVE)

#include "ninfer/engine.h"
#include "ninfer/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace ninfer::serve {

struct SessionCacheStoreOptions {
    std::filesystem::path directory;
    std::uint64_t max_bytes = 32ULL << 30;
    std::size_t max_sessions = 8;
};

// Fields that determine whether an on-disk continuation belongs to this loaded serving target.
// The store hashes the complete artifact once at construction; it does not trust file size or
// timestamps as an artifact identity.
struct SessionCacheIdentity {
    std::filesystem::path artifact_path;
    std::string model_id;
    std::string weights_id;
    bool vision_enabled = false;
    ninfer::KvCacheStorage kv_dtype = ninfer::KvCacheStorage::BFloat16;
    ninfer::SpeculativeBackend speculative_backend = ninfer::SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ninfer::ProposalHead proposal_head = ninfer::ProposalHead::Full;
};

struct SessionCacheStoreStats {
    bool enabled = false;
    std::uint64_t disk_bytes = 0;
    std::uint32_t sessions = 0;
    std::uint64_t saves = 0;
    std::uint64_t restores = 0;
    std::uint64_t misses = 0;
    std::uint64_t failures = 0;
    // Bytes in the last committed/read container, including outer framing.
    std::uint64_t last_saved_bytes = 0;
    std::uint64_t last_restored_bytes = 0;

    // The queue includes the active disk writer and at most one replaceable latest snapshot.
    // queued_snapshot_bytes reports actual host chunk allocation for queued/in-flight snapshots;
    // capture_reserved_bytes reports the maximum host allocation allowed for an Engine capture.
    std::uint32_t queued_snapshots = 0;
    std::uint64_t queued_snapshot_bytes = 0;
    std::uint64_t capture_reserved_bytes = 0;
    std::uint64_t coalesced_snapshots = 0;
    std::uint64_t cancelled_snapshots = 0;
    double last_capture_seconds = 0.0;
    double last_write_seconds = 0.0;
    bool capture_in_progress = false;

    // Current operation is one of: idle, capturing, saving, restoring, or clearing.
    std::string operation = "idle";
    // A closed, non-sensitive status code; never contains a path, session key, prompt, or output.
    // Current values include none, disabled, no_checkpoint, miss, identity_mismatch,
    // checksum_mismatch, budget_exceeded, queue_full, io_error, engine_error, fallback_restored,
    // and cancelled.
    std::string last_error_code = "none";
};

enum class SessionCacheCaptureResult { Queued, NoCheckpoint, Skipped };

class SessionCacheStore {
public:
    SessionCacheStore(SessionCacheStoreOptions options, SessionCacheIdentity identity);
    ~SessionCacheStore();

    SessionCacheStore(SessionCacheStore&&) noexcept;
    SessionCacheStore& operator=(SessionCacheStore&&) noexcept;

    SessionCacheStore(const SessionCacheStore&) = delete;
    SessionCacheStore& operator=(const SessionCacheStore&) = delete;

    // Captures an immutable bounded Engine archive, then queues it for disk commit. The capture is
    // the only Engine operation; the background writer owns host bytes and never touches Engine,
    // Program, or CUDA state. The result belongs to this call, independently of background status.
    // Queue admission coalesces an unstarted older snapshot and never waits for disk I/O.
    [[nodiscard]] SessionCacheCaptureResult capture_and_enqueue(
        ninfer::Engine& engine, std::string_view session,
        std::function<void(std::string_view)> failure_observer = {});

    // Restores the newest valid generation for this session. A missing, incompatible, or corrupt
    // cache is a miss and returns false. The checkpoint runs during bounded disk reads and before
    // each Engine reader chunk; its exception is propagated without treating the cache as corrupt.
    // If replace_in_memory is true, the current Engine catalog is cleared once, only after a
    // candidate has passed full identity and checksum validation; a miss leaves it untouched.
    [[nodiscard]] bool restore(ninfer::Engine& engine, std::string_view session,
                               std::function<void()> checkpoint = {},
                               bool replace_in_memory = false);

    // Removes every cache generation owned by this store's directory.
    void clear();

    // Copies bounded, non-sensitive counters and status. This uses an independent short lock and
    // never waits for snapshot I/O.
    [[nodiscard]] SessionCacheStoreStats snapshot_stats() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ninfer::serve

#endif // defined(NINFER_WINDOWS_SERVE)
