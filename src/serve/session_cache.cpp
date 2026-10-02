#include "serve/session_cache.h"

#if defined(NINFER_WINDOWS_SERVE)

#ifndef NOMINMAX
#    define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <bcrypt.h>
#pragma comment(lib, "bcrypt.lib")
#include <process.h>

#include "ninfer/context_cache.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <initializer_list>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::serve {
namespace {

using Json = nlohmann::json;
using Digest = std::array<std::byte, 32>;

constexpr std::array<std::byte, 8> kHeadMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'S'},
    std::byte{'C'}, std::byte{'H'}, std::byte{0}, std::byte{2},
};
constexpr std::array<std::byte, 8> kSnapshotMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'S'},
    std::byte{'C'}, std::byte{'A'}, std::byte{0}, std::byte{2},
};
constexpr std::array<std::byte, 8> kSnapshotEndMagic = {
    std::byte{'N'}, std::byte{'I'}, std::byte{'N'}, std::byte{'S'},
    std::byte{'C'}, std::byte{'E'}, std::byte{0}, std::byte{2},
};
constexpr std::string_view kOwnedPrefix = "ninfer-session-cache-";
constexpr std::uint32_t kStoreFormatVersion = 2;
// Wide Volta NVFP4 SwiGLU now preserves FP32 gate/up before its final BF16 cast.
// Keep the framing unchanged while isolating continuations computed by older mathematics.
constexpr std::uint32_t kEngineMathRevision = 1;
constexpr std::size_t kSnapshotHeaderBytes = 16U * 1024U;
constexpr std::size_t kSnapshotPrefixBytes = 8U + sizeof(std::uint32_t) * 2U + Digest{}.size();
constexpr std::size_t kSnapshotFooterBytes = 8U + sizeof(std::uint64_t) + Digest{}.size();
constexpr std::size_t kHeadPrefixBytes = 8U + sizeof(std::uint32_t) + Digest{}.size() * 2U +
                                         sizeof(std::uint64_t) * 3U;
constexpr std::size_t kHeadFileBytes = kHeadPrefixBytes + Digest{}.size();
constexpr std::size_t kIoBufferBytes = 1U << 20;
constexpr std::uint64_t kCommitReserveBytes = kSnapshotFooterBytes + kHeadFileBytes;
constexpr std::uint64_t kMaximumSnapshotArchiveBytes =
    ninfer::kDefaultMaximumContextCacheSnapshotBytes;
constexpr std::uint64_t kMaximumQueuedSnapshotBytes = 32ULL << 30;
constexpr std::uint64_t kMaxIdentityStringBytes = 1024;

class StoreError final : public std::runtime_error {
public:
    enum class Kind : std::uint8_t { Io, Budget };

    StoreError(Kind kind, const char* message) : std::runtime_error(message), kind_(kind) {}
    [[nodiscard]] Kind kind() const noexcept { return kind_; }

private:
    Kind kind_;
};

class CandidateError final : public std::runtime_error {
public:
    CandidateError(std::string code, const char* message)
        : std::runtime_error(message), code_(std::move(code)) {}
    [[nodiscard]] const std::string& code() const noexcept { return code_; }

private:
    std::string code_;
};

class FileHandle {
public:
    explicit FileHandle(HANDLE handle = INVALID_HANDLE_VALUE) noexcept : handle_(handle) {}
    ~FileHandle() { reset(); }

    FileHandle(FileHandle&& other) noexcept : handle_(std::exchange(other.handle_, INVALID_HANDLE_VALUE)) {}
    FileHandle& operator=(FileHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, INVALID_HANDLE_VALUE);
        }
        return *this;
    }

    FileHandle(const FileHandle&) = delete;
    FileHandle& operator=(const FileHandle&) = delete;

    [[nodiscard]] HANDLE get() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != INVALID_HANDLE_VALUE; }

    void reset() noexcept {
        if (handle_ != INVALID_HANDLE_VALUE) {
            (void)::CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
    }

private:
    HANDLE handle_ = INVALID_HANDLE_VALUE;
};

class Sha256 {
public:
    Sha256() {
        NTSTATUS status = ::BCryptOpenAlgorithmProvider(&algorithm_, BCRYPT_SHA256_ALGORITHM,
                                                        nullptr, 0);
        if (!BCRYPT_SUCCESS(status)) { throw std::runtime_error("session cache SHA-256 unavailable"); }

        ULONG object_bytes = 0;
        ULONG result_bytes = 0;
        status = ::BCryptGetProperty(algorithm_, BCRYPT_OBJECT_LENGTH,
                                     reinterpret_cast<PUCHAR>(&object_bytes), sizeof(object_bytes),
                                     &result_bytes, 0);
        if (!BCRYPT_SUCCESS(status) || result_bytes != sizeof(object_bytes) || object_bytes == 0) {
            close_algorithm();
            throw std::runtime_error("session cache SHA-256 initialization failed");
        }
        object_.resize(object_bytes);
        status = ::BCryptCreateHash(algorithm_, &hash_, reinterpret_cast<PUCHAR>(object_.data()),
                                    object_bytes, nullptr, 0, 0);
        if (!BCRYPT_SUCCESS(status)) {
            close_algorithm();
            throw std::runtime_error("session cache SHA-256 initialization failed");
        }
    }

    ~Sha256() {
        if (hash_ != nullptr) { (void)::BCryptDestroyHash(hash_); }
        close_algorithm();
    }

    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;

    void update(std::span<const std::byte> bytes) {
        if (finished_) { throw std::logic_error("session cache digest is already finalized"); }
        std::size_t offset = 0;
        while (offset < bytes.size()) {
            const std::size_t count =
                std::min<std::size_t>(bytes.size() - offset, std::numeric_limits<ULONG>::max());
            const NTSTATUS status = ::BCryptHashData(
                hash_, reinterpret_cast<PUCHAR>(const_cast<std::byte*>(bytes.data() + offset)),
                static_cast<ULONG>(count), 0);
            if (!BCRYPT_SUCCESS(status)) { throw std::runtime_error("session cache SHA-256 failed"); }
            offset += count;
        }
    }

    [[nodiscard]] Digest finish() {
        if (finished_) { throw std::logic_error("session cache digest is already finalized"); }
        Digest digest{};
        const NTSTATUS status = ::BCryptFinishHash(
            hash_, reinterpret_cast<PUCHAR>(digest.data()), static_cast<ULONG>(digest.size()), 0);
        if (!BCRYPT_SUCCESS(status)) { throw std::runtime_error("session cache SHA-256 failed"); }
        finished_ = true;
        return digest;
    }

private:
    void close_algorithm() noexcept {
        if (algorithm_ != nullptr) {
            (void)::BCryptCloseAlgorithmProvider(algorithm_, 0);
            algorithm_ = nullptr;
        }
    }

    BCRYPT_ALG_HANDLE algorithm_ = nullptr;
    BCRYPT_HASH_HANDLE hash_ = nullptr;
    std::vector<std::byte> object_;
    bool finished_ = false;
};

[[nodiscard]] Digest digest_bytes(std::span<const std::byte> bytes) {
    Sha256 digest;
    digest.update(bytes);
    return digest.finish();
}

[[nodiscard]] Digest digest_text(std::string_view text) {
    return digest_bytes(std::as_bytes(std::span(text.data(), text.size())));
}

[[nodiscard]] std::string digest_hex(const Digest& digest) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const unsigned int byte = std::to_integer<unsigned int>(digest[i]);
        result[i * 2] = kHex[byte >> 4U];
        result[i * 2 + 1] = kHex[byte & 0xfU];
    }
    return result;
}

[[nodiscard]] bool parse_digest_hex(std::string_view value, Digest& digest) noexcept {
    if (value.size() != digest.size() * 2U) { return false; }
    const auto digit = [](char c) noexcept -> int {
        if (c >= '0' && c <= '9') { return c - '0'; }
        if (c >= 'a' && c <= 'f') { return c - 'a' + 10; }
        if (c >= 'A' && c <= 'F') { return c - 'A' + 10; }
        return -1;
    };
    for (std::size_t i = 0; i < digest.size(); ++i) {
        const int high = digit(value[i * 2]);
        const int low = digit(value[i * 2 + 1]);
        if (high < 0 || low < 0) { return false; }
        digest[i] = static_cast<std::byte>((high << 4) | low);
    }
    return true;
}

void append_u32(std::vector<std::byte>& output, std::uint32_t value) {
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

void append_u64(std::vector<std::byte>& output, std::uint64_t value) {
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        output.push_back(static_cast<std::byte>((value >> shift) & 0xffU));
    }
}

[[nodiscard]] std::uint32_t read_u32(std::span<const std::byte> data, std::size_t offset) {
    if (offset > data.size() || data.size() - offset < sizeof(std::uint32_t)) {
        throw std::runtime_error("invalid session cache metadata");
    }
    std::uint32_t result = 0;
    for (unsigned int shift = 0; shift < 32; shift += 8) {
        result |= std::to_integer<std::uint32_t>(data[offset++]) << shift;
    }
    return result;
}

[[nodiscard]] std::uint64_t read_u64(std::span<const std::byte> data, std::size_t offset) {
    if (offset > data.size() || data.size() - offset < sizeof(std::uint64_t)) {
        throw std::runtime_error("invalid session cache metadata");
    }
    std::uint64_t result = 0;
    for (unsigned int shift = 0; shift < 64; shift += 8) {
        result |= std::to_integer<std::uint64_t>(data[offset++]) << shift;
    }
    return result;
}

void append_string(std::vector<std::byte>& output, std::string_view value) {
    if (value.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::invalid_argument("session cache identity string is too large");
    }
    append_u32(output, static_cast<std::uint32_t>(value.size()));
    const auto bytes = std::as_bytes(std::span(value.data(), value.size()));
    output.insert(output.end(), bytes.begin(), bytes.end());
}

[[nodiscard]] std::uint64_t json_u64(const Json& value, const char* name) {
    const auto found = value.find(name);
    if (found == value.end() || (!found->is_number_unsigned() && !found->is_number_integer())) {
        throw std::runtime_error("invalid session cache metadata");
    }
    if (found->is_number_unsigned()) { return found->get<std::uint64_t>(); }
    const std::int64_t number = found->get<std::int64_t>();
    if (number < 0) { throw std::runtime_error("invalid session cache metadata"); }
    return static_cast<std::uint64_t>(number);
}

[[nodiscard]] std::string json_string(const Json& value, const char* name) {
    const auto found = value.find(name);
    if (found == value.end() || !found->is_string()) {
        throw std::runtime_error("invalid session cache metadata");
    }
    return found->get<std::string>();
}

void require_json_members(const Json& value, std::initializer_list<std::string_view> names) {
    if (!value.is_object() || value.size() != names.size()) {
        throw std::runtime_error("invalid session cache metadata");
    }
    for (const std::string_view name : names) {
        if (!value.contains(std::string(name))) {
            throw std::runtime_error("invalid session cache metadata");
        }
    }
}

[[nodiscard]] std::uint64_t utc_milliseconds() noexcept {
    const auto value = std::chrono::system_clock::now().time_since_epoch();
    const auto count = std::chrono::duration_cast<std::chrono::milliseconds>(value).count();
    return count > 0 ? static_cast<std::uint64_t>(count) : 0;
}

[[nodiscard]] std::uint64_t unique_nonce() noexcept {
    static std::atomic<std::uint64_t> sequence{0};
    const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
    return static_cast<std::uint64_t>(now) ^
           (static_cast<std::uint64_t>(::_getpid()) << 32U) ^
           sequence.fetch_add(1, std::memory_order_relaxed);
}

[[nodiscard]] std::string fixed_hex_u64(std::uint64_t value) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string result(16, '0');
    for (std::size_t i = 0; i < result.size(); ++i) {
        const unsigned int shift = static_cast<unsigned int>((result.size() - 1U - i) * 4U);
        result[i] = kHex[(value >> shift) & 0xfU];
    }
    return result;
}

[[nodiscard]] std::wstring extended_windows_path(const std::filesystem::path& path) {
    const std::wstring native =
        std::filesystem::absolute(path).lexically_normal().make_preferred().native();
    if (native.starts_with(L"\\\\?\\")) { return native; }
    if (native.starts_with(L"\\\\")) { return L"\\\\?\\UNC\\" + native.substr(2); }
    return L"\\\\?\\" + native;
}

[[nodiscard]] FileHandle open_file(const std::filesystem::path& path, DWORD access, DWORD share,
                                  DWORD disposition, DWORD flags = FILE_ATTRIBUTE_NORMAL) {
    const std::wstring native = extended_windows_path(path);
    HANDLE handle = ::CreateFileW(native.c_str(), access, share, nullptr, disposition, flags, nullptr);
    if (handle == INVALID_HANDLE_VALUE) { throw StoreError(StoreError::Kind::Io, "session cache file open failed"); }
    return FileHandle(handle);
}

void write_exact(HANDLE handle, std::span<const std::byte> bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
            bytes.size() - offset, std::numeric_limits<DWORD>::max()));
        DWORD written = 0;
        if (!::WriteFile(handle, bytes.data() + offset, request, &written, nullptr) || written == 0) {
            throw StoreError(StoreError::Kind::Io, "session cache file write failed");
        }
        offset += written;
    }
}

void read_exact(HANDLE handle, std::span<std::byte> bytes) {
    std::size_t offset = 0;
    while (offset < bytes.size()) {
        const DWORD request = static_cast<DWORD>(std::min<std::size_t>(
            bytes.size() - offset, std::numeric_limits<DWORD>::max()));
        DWORD read = 0;
        if (!::ReadFile(handle, bytes.data() + offset, request, &read, nullptr) || read == 0) {
            throw StoreError(StoreError::Kind::Io, "session cache file read failed");
        }
        offset += read;
    }
}

void seek_file(HANDLE handle, std::uint64_t position) {
    LARGE_INTEGER target{};
    target.QuadPart = static_cast<LONGLONG>(position);
    if (position > static_cast<std::uint64_t>(std::numeric_limits<LONGLONG>::max()) ||
        !::SetFilePointerEx(handle, target, nullptr, FILE_BEGIN)) {
        throw StoreError(StoreError::Kind::Io, "session cache file seek failed");
    }
}

[[nodiscard]] std::uint64_t file_size(HANDLE handle) {
    LARGE_INTEGER result{};
    if (!::GetFileSizeEx(handle, &result) || result.QuadPart < 0) {
        throw StoreError(StoreError::Kind::Io, "session cache file size query failed");
    }
    return static_cast<std::uint64_t>(result.QuadPart);
}

void flush_file(HANDLE handle) {
    if (!::FlushFileBuffers(handle)) { throw StoreError(StoreError::Kind::Io, "session cache flush failed"); }
}

void atomic_replace(const std::filesystem::path& source, const std::filesystem::path& destination) {
    if (!::MoveFileExW(extended_windows_path(source).c_str(),
                       extended_windows_path(destination).c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw StoreError(StoreError::Kind::Io, "session cache atomic replace failed");
    }
}

[[nodiscard]] std::string path_filename_utf8(const std::filesystem::path& path) {
    const std::u8string value = path.filename().u8string();
    return std::string(reinterpret_cast<const char*>(value.data()), value.size());
}

[[nodiscard]] Digest hash_artifact_file(const std::filesystem::path& path) {
    FileHandle file = open_file(path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING,
                                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN);
    const std::uint64_t expected_size = file_size(file.get());
    Sha256 digest;
    std::vector<std::byte> buffer(kIoBufferBytes);
    std::uint64_t remaining = expected_size;
    while (remaining != 0) {
        const std::size_t count = static_cast<std::size_t>(
            std::min<std::uint64_t>(remaining, buffer.size()));
        read_exact(file.get(), std::span(buffer).first(count));
        digest.update(std::span<const std::byte>(buffer).first(count));
        remaining -= count;
    }
    if (file_size(file.get()) != expected_size) {
        throw StoreError(StoreError::Kind::Io, "session cache artifact changed while hashing");
    }
    return digest.finish();
}

struct HeadRecord {
    Digest namespace_digest{};
    Digest session_digest{};
    std::uint64_t current_generation = 0;
    std::uint64_t previous_generation = 0;
    std::uint64_t last_access_ms = 0;
};

[[nodiscard]] std::vector<std::byte> encode_head(const HeadRecord& record) {
    std::vector<std::byte> bytes;
    bytes.reserve(kHeadFileBytes);
    bytes.insert(bytes.end(), kHeadMagic.begin(), kHeadMagic.end());
    append_u32(bytes, kStoreFormatVersion);
    bytes.insert(bytes.end(), record.namespace_digest.begin(), record.namespace_digest.end());
    bytes.insert(bytes.end(), record.session_digest.begin(), record.session_digest.end());
    append_u64(bytes, record.current_generation);
    append_u64(bytes, record.previous_generation);
    append_u64(bytes, record.last_access_ms);
    if (bytes.size() != kHeadPrefixBytes) { throw std::logic_error("session cache head size mismatch"); }
    const Digest checksum = digest_bytes(bytes);
    bytes.insert(bytes.end(), checksum.begin(), checksum.end());
    return bytes;
}

[[nodiscard]] HeadRecord decode_head(std::span<const std::byte> bytes) {
    if (bytes.size() != kHeadFileBytes ||
        !std::equal(kHeadMagic.begin(), kHeadMagic.end(), bytes.begin()) ||
        read_u32(bytes, kHeadMagic.size()) != kStoreFormatVersion) {
        throw std::runtime_error("invalid session cache head");
    }
    const Digest checksum = digest_bytes(bytes.first(kHeadPrefixBytes));
    if (!std::equal(checksum.begin(), checksum.end(), bytes.begin() + kHeadPrefixBytes)) {
        throw std::runtime_error("invalid session cache head checksum");
    }
    HeadRecord result;
    std::size_t offset = kHeadMagic.size() + sizeof(std::uint32_t);
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), result.namespace_digest.size(),
                result.namespace_digest.begin());
    offset += result.namespace_digest.size();
    std::copy_n(bytes.begin() + static_cast<std::ptrdiff_t>(offset), result.session_digest.size(),
                result.session_digest.begin());
    offset += result.session_digest.size();
    result.current_generation = read_u64(bytes, offset);
    offset += sizeof(std::uint64_t);
    result.previous_generation = read_u64(bytes, offset);
    offset += sizeof(std::uint64_t);
    result.last_access_ms = read_u64(bytes, offset);
    if (result.current_generation == 0 ||
        (result.previous_generation >= result.current_generation && result.previous_generation != 0)) {
        throw std::runtime_error("invalid session cache generations");
    }
    return result;
}

struct SnapshotHeader {
    std::uint64_t generation = 0;
    Digest namespace_digest{};
    Digest session_digest{};
    Digest artifact_digest{};
    std::string model_id;
    std::string weights_id;
    bool vision_enabled = false;
    ninfer::KvCacheStorage kv_dtype = ninfer::KvCacheStorage::BFloat16;
    ninfer::SpeculativeBackend backend = ninfer::SpeculativeBackend::None;
    std::uint32_t draft_tokens = 0;
    ninfer::ProposalHead proposal_head = ninfer::ProposalHead::Full;
    std::uint64_t archive_bytes = 0;
    Digest archive_digest{};
    std::uint64_t checkpoints = 0;
};

[[nodiscard]] bool is_lower_hex(std::string_view value) noexcept {
    return std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

constexpr std::size_t kOwnedBaseBytes = kOwnedPrefix.size() + 64U + 1U + 64U;

[[nodiscard]] bool is_valid_base(std::string_view base) noexcept {
    return base.size() == kOwnedBaseBytes && base.starts_with(kOwnedPrefix) &&
           base[kOwnedPrefix.size() + 64U] == '_' &&
           is_lower_hex(base.substr(kOwnedPrefix.size(), 64U)) &&
           is_lower_hex(base.substr(kOwnedPrefix.size() + 65U, 64U));
}

[[nodiscard]] bool parse_head_name(std::string_view name, std::string& base) {
    constexpr std::string_view suffix = ".head";
    if (name.size() != kOwnedBaseBytes + suffix.size() || !name.ends_with(suffix)) { return false; }
    base.assign(name.substr(0, kOwnedBaseBytes));
    return is_valid_base(base);
}

[[nodiscard]] bool parse_snapshot_name(std::string_view name, std::string& base,
                                       std::uint64_t& generation) {
    constexpr std::size_t kSuffixBytes = 2U + 16U + 5U; // .g<16 hex>.snap
    if (name.size() != kOwnedBaseBytes + kSuffixBytes || !name.starts_with(kOwnedPrefix)) {
        return false;
    }
    const std::string_view candidate_base = name.substr(0, kOwnedBaseBytes);
    if (!is_valid_base(candidate_base) || name[kOwnedBaseBytes] != '.' ||
        name[kOwnedBaseBytes + 1U] != 'g' || name.substr(kOwnedBaseBytes + 18U) != ".snap") {
        return false;
    }
    const std::string_view hex = name.substr(kOwnedBaseBytes + 2U, 16U);
    if (!is_lower_hex(hex)) { return false; }
    std::uint64_t parsed = 0;
    for (const char c : hex) {
        const std::uint64_t nibble = c <= '9' ? static_cast<std::uint64_t>(c - '0')
                                             : static_cast<std::uint64_t>(c - 'a' + 10);
        parsed = (parsed << 4U) | nibble;
    }
    if (parsed == 0) { return false; }
    base.assign(candidate_base);
    generation = parsed;
    return true;
}

[[nodiscard]] bool parse_temporary_name(std::string_view name, std::string& base) {
    const std::size_t marker = name.find(".tmp.");
    if (marker == std::string_view::npos || marker == 0) { return false; }
    const std::string_view original = name.substr(0, marker);
    std::uint64_t ignored_generation = 0;
    if (!parse_snapshot_name(original, base, ignored_generation)) {
        if (!parse_head_name(original, base)) { return false; }
    }
    const std::string_view nonce = name.substr(marker + 5U);
    return nonce.size() == 16U && is_lower_hex(nonce);
}

[[nodiscard]] bool is_owned_filename(std::string_view name) {
    std::string ignored_base;
    std::uint64_t ignored_generation = 0;
    return parse_head_name(name, ignored_base) ||
           parse_snapshot_name(name, ignored_base, ignored_generation) ||
           parse_temporary_name(name, ignored_base);
}

[[nodiscard]] std::string base_name(const Digest& namespace_digest, const Digest& session_digest) {
    return std::string(kOwnedPrefix) + digest_hex(namespace_digest) + "_" +
           digest_hex(session_digest);
}

[[nodiscard]] std::string snapshot_name(std::string_view base, std::uint64_t generation) {
    return std::string(base) + ".g" + fixed_hex_u64(generation) + ".snap";
}

[[nodiscard]] std::wstring widen_ascii(std::string_view value) {
    std::wstring result;
    result.reserve(value.size());
    for (const unsigned char character : value) { result.push_back(static_cast<wchar_t>(character)); }
    return result;
}

[[nodiscard]] Json make_snapshot_json(const SnapshotHeader& header) {
    return Json{
        {"artifact_sha256", digest_hex(header.artifact_digest)},
        {"archive_bytes", header.archive_bytes},
        {"archive_sha256", digest_hex(header.archive_digest)},
        {"checkpoints", header.checkpoints},
        {"draft_tokens", header.draft_tokens},
        {"format_version", kStoreFormatVersion},
        {"generation", header.generation},
        {"identity_sha256", digest_hex(header.namespace_digest)},
        {"kv_dtype", static_cast<std::uint32_t>(header.kv_dtype)},
        {"model_id", header.model_id},
        {"proposal_head", static_cast<std::uint32_t>(header.proposal_head)},
        {"session_sha256", digest_hex(header.session_digest)},
        {"speculative_backend", static_cast<std::uint32_t>(header.backend)},
        {"vision_enabled", header.vision_enabled},
        {"weights_id", header.weights_id},
    };
}

[[nodiscard]] std::vector<std::byte> encode_snapshot_region(const SnapshotHeader& header) {
    const std::string serialized = make_snapshot_json(header).dump();
    if (serialized.empty() || serialized.size() > kSnapshotHeaderBytes - kSnapshotPrefixBytes) {
        throw std::runtime_error("session cache metadata is too large");
    }
    std::vector<std::byte> region(kSnapshotHeaderBytes, std::byte{0});
    std::copy(kSnapshotMagic.begin(), kSnapshotMagic.end(), region.begin());
    const std::uint32_t version = kStoreFormatVersion;
    const std::uint32_t length = static_cast<std::uint32_t>(serialized.size());
    std::vector<std::byte> prefix;
    prefix.reserve(kSnapshotPrefixBytes);
    prefix.insert(prefix.end(), kSnapshotMagic.begin(), kSnapshotMagic.end());
    append_u32(prefix, version);
    append_u32(prefix, length);
    const auto json_bytes = std::as_bytes(std::span(serialized.data(), serialized.size()));
    const Digest checksum = digest_bytes(json_bytes);
    prefix.insert(prefix.end(), checksum.begin(), checksum.end());
    if (prefix.size() != kSnapshotPrefixBytes) {
        throw std::logic_error("session cache snapshot prefix size mismatch");
    }
    std::copy(prefix.begin(), prefix.end(), region.begin());
    std::copy(json_bytes.begin(), json_bytes.end(), region.begin() + kSnapshotPrefixBytes);
    return region;
}

[[nodiscard]] SnapshotHeader parse_snapshot_region(std::span<const std::byte> region) {
    if (region.size() != kSnapshotHeaderBytes ||
        !std::equal(kSnapshotMagic.begin(), kSnapshotMagic.end(), region.begin()) ||
        read_u32(region, kSnapshotMagic.size()) != kStoreFormatVersion) {
        throw std::runtime_error("invalid session cache snapshot header");
    }
    const std::uint32_t length = read_u32(region, kSnapshotMagic.size() + sizeof(std::uint32_t));
    if (length == 0 || length > region.size() - kSnapshotPrefixBytes) {
        throw std::runtime_error("invalid session cache metadata length");
    }
    Digest recorded_checksum{};
    std::copy_n(region.begin() + static_cast<std::ptrdiff_t>(kSnapshotMagic.size() +
                                                              sizeof(std::uint32_t) * 2U),
                recorded_checksum.size(), recorded_checksum.begin());
    const auto json_bytes = region.subspan(kSnapshotPrefixBytes, length);
    const Digest actual_checksum = digest_bytes(json_bytes);
    if (recorded_checksum != actual_checksum ||
        !std::all_of(region.begin() + static_cast<std::ptrdiff_t>(kSnapshotPrefixBytes + length),
                     region.end(), [](std::byte byte) { return byte == std::byte{0}; })) {
        throw std::runtime_error("invalid session cache metadata checksum");
    }

    const Json json = Json::parse(reinterpret_cast<const char*>(json_bytes.data()),
                                  reinterpret_cast<const char*>(json_bytes.data() + json_bytes.size()));
    require_json_members(json, {"artifact_sha256", "archive_bytes", "archive_sha256", "checkpoints",
                                "draft_tokens", "format_version", "generation", "identity_sha256",
                                "kv_dtype", "model_id", "proposal_head", "session_sha256",
                                "speculative_backend", "vision_enabled", "weights_id"});
    SnapshotHeader result;
    result.generation = json_u64(json, "generation");
    result.archive_bytes = json_u64(json, "archive_bytes");
    result.checkpoints = json_u64(json, "checkpoints");
    const std::uint64_t version = json_u64(json, "format_version");
    const std::uint64_t kv = json_u64(json, "kv_dtype");
    const std::uint64_t backend = json_u64(json, "speculative_backend");
    const std::uint64_t head = json_u64(json, "proposal_head");
    const std::uint64_t draft = json_u64(json, "draft_tokens");
    if (version != kStoreFormatVersion || result.generation == 0 || result.checkpoints == 0 ||
        kv > static_cast<std::uint64_t>(ninfer::KvCacheStorage::Fp8KeyNvfp4Value) ||
        backend > static_cast<std::uint64_t>(ninfer::SpeculativeBackend::DFlash2) ||
        head > static_cast<std::uint64_t>(ninfer::ProposalHead::Optimized) ||
        draft > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("invalid session cache identity metadata");
    }
    result.kv_dtype = static_cast<ninfer::KvCacheStorage>(kv);
    result.backend = static_cast<ninfer::SpeculativeBackend>(backend);
    result.proposal_head = static_cast<ninfer::ProposalHead>(head);
    result.draft_tokens = static_cast<std::uint32_t>(draft);
    result.model_id = json_string(json, "model_id");
    result.weights_id = json_string(json, "weights_id");
    const auto vision = json.find("vision_enabled");
    if (vision == json.end() || !vision->is_boolean()) {
        throw std::runtime_error("invalid session cache identity metadata");
    }
    result.vision_enabled = vision->get<bool>();
    if (result.model_id.empty() || result.model_id.size() > kMaxIdentityStringBytes ||
        result.weights_id.empty() || result.weights_id.size() > kMaxIdentityStringBytes) {
        throw std::runtime_error("invalid session cache identity metadata");
    }
    if (!parse_digest_hex(json_string(json, "identity_sha256"), result.namespace_digest) ||
        !parse_digest_hex(json_string(json, "session_sha256"), result.session_digest) ||
        !parse_digest_hex(json_string(json, "artifact_sha256"), result.artifact_digest) ||
        !parse_digest_hex(json_string(json, "archive_sha256"), result.archive_digest)) {
        throw std::runtime_error("invalid session cache digest metadata");
    }
    return result;
}

class RestoreInterrupted final : public std::exception {
public:
    explicit RestoreInterrupted(std::exception_ptr original) noexcept : original_(std::move(original)) {}
    [[nodiscard]] const char* what() const noexcept override { return "session cache restore interrupted"; }
    [[noreturn]] void rethrow_original() const { std::rethrow_exception(original_); }

private:
    std::exception_ptr original_;
};

void invoke_checkpoint(const std::function<void()>& checkpoint) {
    if (!checkpoint) { return; }
    try {
        checkpoint();
    } catch (...) {
        throw RestoreInterrupted(std::current_exception());
    }
}

} // namespace

class SessionCacheStore::Impl {
public:
    Impl(SessionCacheStoreOptions options, SessionCacheIdentity identity)
        : options_(std::move(options)), identity_(std::move(identity)) {
        stats_.enabled = options_.max_bytes != 0 && options_.max_sessions != 0;
        if (!stats_.enabled) {
            stats_.last_error_code = "disabled";
            return;
        }
        if (options_.directory.empty() || identity_.artifact_path.empty() ||
            identity_.model_id.empty() || identity_.weights_id.empty() ||
            identity_.model_id.size() > kMaxIdentityStringBytes ||
            identity_.weights_id.size() > kMaxIdentityStringBytes) {
            throw std::invalid_argument("session cache configuration or identity is invalid");
        }
        if (options_.max_bytes < kSnapshotHeaderBytes + kCommitReserveBytes) {
            throw std::invalid_argument("session cache budget is too small");
        }

        std::error_code error;
        std::filesystem::create_directories(options_.directory, error);
        if (error) { throw StoreError(StoreError::Kind::Io, "session cache directory creation failed"); }
        root_ = std::filesystem::weakly_canonical(options_.directory, error);
        if (error || root_.empty()) {
            throw StoreError(StoreError::Kind::Io, "session cache directory resolution failed");
        }
        const auto status = std::filesystem::status(root_, error);
        if (error || !std::filesystem::is_directory(status)) {
            throw StoreError(StoreError::Kind::Io, "session cache path is not a directory");
        }

        artifact_digest_ = hash_artifact_file(identity_.artifact_path);
        namespace_digest_ = make_namespace_digest();
        try {
            std::lock_guard operation_lock(operation_mutex_);
            cleanup_orphans_locked();
            refresh_disk_stats_locked();
        } catch (...) {
            update_stats([&](SessionCacheStoreStats& stats) {
                stats.enabled = false;
                stats.last_error_code = "io_error";
            });
            throw;
        }
        worker_ = std::thread([this] { save_worker_loop(); });
    }

    ~Impl() {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    [[nodiscard]] SessionCacheStoreStats snapshot_stats() const {
        std::lock_guard lock(stats_mutex_);
        return stats_;
    }

    [[nodiscard]] SessionCacheCaptureResult capture_and_enqueue(
        ninfer::Engine& engine, std::string_view session,
        std::function<void(std::string_view)> failure_observer) {
        if (!stats_.enabled) { set_error("disabled"); return SessionCacheCaptureResult::Skipped; }
        validate_session(session);

        std::uint64_t capture_limit = 0;
        std::uint64_t capture_epoch = 0;
        std::string admission_failure;
        {
            std::lock_guard lock(queue_mutex_);
            if (stopping_) {
                admission_failure = "cancelled";
            } else if (capture_in_progress_) {
                throw std::logic_error("session cache snapshot capture is already active");
            } else if (pending_job_ && pending_job_->session != session) {
                admission_failure = "queue_full";
            } else {
                const std::uint64_t occupied_bytes = active_bytes_ + pending_bytes_;
                const std::uint64_t available_bytes =
                    occupied_bytes < kMaximumQueuedSnapshotBytes
                        ? kMaximumQueuedSnapshotBytes - occupied_bytes
                        : 0;
                capture_limit = std::min(kMaximumSnapshotArchiveBytes, available_bytes);
                if (capture_limit == 0) { admission_failure = "budget_exceeded"; }
            }
            if (!admission_failure.empty()) {
                update_stats([&](SessionCacheStoreStats& stats) {
                    stats.last_error_code = admission_failure;
                    ++stats.failures;
                });
                publish_queue_stats_locked();
            } else {
                capture_in_progress_ = true;
                capture_session_ = session;
                capture_reserved_bytes_ = capture_limit;
                capture_epoch = epoch_;
                publish_queue_stats_locked();
            }
        }
        if (!admission_failure.empty()) {
            if (failure_observer) {
                const std::string detail = admission_failure == "queue_full"
                                               ? "session cache save queue is occupied by another session"
                                               : "session cache snapshot memory budget is exhausted";
                try { failure_observer(detail); } catch (...) {}
            }
            return SessionCacheCaptureResult::Skipped;
        }

        const auto capture_started = std::chrono::steady_clock::now();
        ninfer::ContextCacheSnapshot snapshot;
        try {
            snapshot = engine.capture_context_cache(capture_limit);
        } catch (...) {
            const double capture_seconds = elapsed_seconds(capture_started);
            finish_capture_failure(capture_seconds, exception_status(std::current_exception()));
            throw;
        }
        const double capture_seconds = elapsed_seconds(capture_started);
        const ninfer::ContextCacheSnapshotStats snapshot_stats = snapshot.stats();
        const std::uint64_t allocated_bytes = snapshot.allocated_bytes();
        if (!snapshot || snapshot_stats.bytes > capture_limit || allocated_bytes > capture_limit) {
            finish_capture_failure(capture_seconds, "budget_exceeded");
            if (failure_observer) {
                try { failure_observer("captured session snapshot exceeded its reserved memory budget"); }
                catch (...) {}
            }
            return SessionCacheCaptureResult::Skipped;
        }
        if (snapshot_stats.checkpoints == 0 || snapshot_stats.session_continuations == 0) {
            finish_empty_capture(capture_seconds);
            return SessionCacheCaptureResult::NoCheckpoint;
        }

        std::optional<SaveJob> replacement;
        try {
            replacement.emplace(SaveJob{.session = std::string(session),
                                        .snapshot = std::move(snapshot),
                                        .failure_observer = std::move(failure_observer),
                                        .capture_seconds = capture_seconds,
                                        .bytes = allocated_bytes,
                                        .epoch = capture_epoch});
        } catch (...) {
            finish_capture_failure(capture_seconds, "engine_error");
            throw;
        }

        std::string enqueue_failure;
        {
            std::lock_guard lock(queue_mutex_);
            capture_in_progress_ = false;
            capture_session_ = {};
            capture_reserved_bytes_ = 0;
            if (epoch_ != capture_epoch) {
                enqueue_failure = "cancelled";
            } else if (pending_job_ && pending_job_->session != session) {
                enqueue_failure = "queue_full";
            } else if (allocated_bytes > kMaximumQueuedSnapshotBytes - active_bytes_ - pending_bytes_) {
                enqueue_failure = "budget_exceeded";
            } else {
                if (pending_job_) {
                    ++coalesced_snapshots_;
                }
                pending_job_ = std::move(replacement);
                pending_bytes_ = allocated_bytes;
                update_capture_stats_locked(capture_seconds, "none", false);
            }
            if (!enqueue_failure.empty()) {
                ++cancelled_snapshots_;
                update_capture_stats_locked(capture_seconds, enqueue_failure, true);
            }
            publish_queue_stats_locked();
        }
        if (!enqueue_failure.empty()) {
            if (replacement && replacement->failure_observer) {
                const std::string detail = enqueue_failure == "queue_full"
                                               ? "session cache save queue changed during capture"
                                           : enqueue_failure == "cancelled"
                                               ? "session cache capture was invalidated by clear"
                                               : "session cache snapshot memory budget is exhausted";
                try { replacement->failure_observer(detail); } catch (...) {}
            }
            return SessionCacheCaptureResult::Skipped;
        }
        queue_cv_.notify_one();
        queue_cv_.notify_all();
        return SessionCacheCaptureResult::Queued;
    }

    [[nodiscard]] bool restore(ninfer::Engine& engine, std::string_view session,
                               const std::function<void()>& checkpoint,
                               bool replace_in_memory) {
        if (!stats_.enabled) { set_error("disabled"); return false; }
        validate_session(session);
        try {
            wait_for_session_saves(session, checkpoint);
        } catch (const RestoreInterrupted& interrupted) {
            finish_operation("cancelled", false, false, 0, 0);
            interrupted.rethrow_original();
        }
        std::unique_lock operation_lock(operation_mutex_);
        set_operation("restoring");
        std::string last_error = "miss";
        std::uint64_t restored_bytes = 0;
        try {
            invoke_checkpoint(checkpoint);
            const std::string base = make_base(session);
            std::optional<HeadRecord> head;
            try {
                head = read_head(base);
            } catch (const StoreError&) {
                refresh_disk_stats_best_effort();
                finish_operation("io_error", false, true, 0, 0);
                return false;
            }
            if (!head) {
                finish_operation("miss", false, false, 0, 0);
                return false;
            }
            const std::array<std::uint64_t, 2> generations = {
                head->current_generation, head->previous_generation};
            bool replacement_clear_attempted = false;
            for (std::size_t index = 0; index < generations.size(); ++index) {
                const std::uint64_t generation = generations[index];
                if (generation == 0) { continue; }
                invoke_checkpoint(checkpoint);
                try {
                    const CandidateInfo candidate = validate_candidate(
                        base, generation, head->session_digest, checkpoint);
                    if (replace_in_memory && !replacement_clear_attempted) {
                        replacement_clear_attempted = true;
                        engine.clear_context_cache();
                    }
                    seek_file(candidate.file.get(), kSnapshotHeaderBytes);
                    SnapshotFileReader reader(candidate.file.get(), candidate.header.archive_bytes,
                                              checkpoint);
                    const ninfer::ContextCacheSnapshotStats loaded = engine.load_context_cache(reader);
                    if (reader.remaining_bytes() != 0 || loaded.checkpoints == 0 ||
                        loaded.session_continuations == 0 ||
                        loaded.checkpoints != candidate.header.checkpoints ||
                        loaded.bytes != candidate.header.archive_bytes) {
                        engine.clear_context_cache();
                        throw std::runtime_error("Engine session cache restore summary mismatch");
                    }
                    restored_bytes = kSnapshotHeaderBytes + loaded.bytes + kSnapshotFooterBytes;
                    HeadRecord touched = *head;
                    touched.last_access_ms = utc_milliseconds();
                    try {
                        write_head_atomic(base, touched, false);
                    } catch (const StoreError& error) {
                        cleanup_orphans_best_effort();
                        refresh_disk_stats_best_effort();
                        last_error = error.kind() == StoreError::Kind::Budget
                                         ? "budget_exceeded"
                                         : "io_error";
                    } catch (...) {
                        // A failed LRU timestamp update must not invalidate an already-restored
                        // Engine catalog.
                        cleanup_orphans_best_effort();
                        refresh_disk_stats_best_effort();
                        last_error = "io_error";
                    }
                    if (index != 0) { last_error = "fallback_restored"; }
                    finish_operation(last_error == "miss" ? "none" : last_error,
                                     false, false, 0, restored_bytes);
                    return true;
                } catch (const RestoreInterrupted&) {
                    throw;
                } catch (const ninfer::ContextCacheOwnershipError&) {
                    // Engine rollback failed. Do not hide an unavailable/ownership-broken Engine
                    // by trying another generation or reporting a cache miss.
                    throw;
                } catch (const StoreError& error) {
                    last_error = error.kind() == StoreError::Kind::Budget ? "budget_exceeded" : "io_error";
                } catch (const CandidateError& error) {
                    last_error = error.code();
                } catch (...) {
                    if (!engine.is_available()) { throw; }
                    last_error = "engine_error";
                }
            }
            refresh_disk_stats_best_effort();
            finish_operation(last_error == "miss" ? "miss" : last_error, false, true, 0, 0);
            return false;
        } catch (const RestoreInterrupted& interrupted) {
            finish_operation("cancelled", false, false, 0, 0);
            interrupted.rethrow_original();
        } catch (...) {
            refresh_disk_stats_best_effort();
            finish_operation("engine_error", false, true, 0, 0);
            throw;
        }
    }

    void clear() {
        if (!stats_.enabled) { set_error("disabled"); return; }
        {
            std::lock_guard lock(queue_mutex_);
            ++epoch_;
            if (pending_job_) {
                pending_job_.reset();
                pending_bytes_ = 0;
                ++cancelled_snapshots_;
            }
            publish_queue_stats_locked();
        }
        queue_cv_.notify_all();
        std::unique_lock operation_lock(operation_mutex_);
        set_operation("clearing");
        try {
            const std::vector<std::string> files = list_owned_files();
            bool head_failure = false;
            for (const std::string& name : files) {
                std::string base;
                if (parse_head_name(name, base) || name.find(".head.tmp.") != std::string::npos) {
                    if (!delete_file_accounted(name)) { head_failure = true; }
                }
            }
            if (head_failure) {
                refresh_disk_stats_best_effort();
                finish_operation("io_error", false, true, 0, 0);
                throw StoreError(StoreError::Kind::Io, "session cache clear could not revoke all heads");
            }
            bool file_failure = false;
            for (const std::string& name : list_owned_files()) {
                if (!delete_file_accounted(name)) { file_failure = true; }
            }
            refresh_disk_stats_best_effort();
            if (file_failure) {
                finish_operation("io_error", false, true, 0, 0);
                throw StoreError(StoreError::Kind::Io, "session cache clear could not remove all files");
            }
            finish_operation("none", false, false, 0, 0);
        } catch (...) {
            if (current_operation() != "idle") { finish_operation("io_error", false, true, 0, 0); }
            throw;
        }
    }

private:
    struct SaveJob {
        std::string session;
        ninfer::ContextCacheSnapshot snapshot;
        std::function<void(std::string_view)> failure_observer;
        double capture_seconds = 0.0;
        std::uint64_t bytes = 0;
        std::uint64_t epoch = 0;
    };

    [[nodiscard]] static double elapsed_seconds(std::chrono::steady_clock::time_point start) {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }

    [[nodiscard]] static std::string exception_status(std::exception_ptr error) {
        try {
            if (error) { std::rethrow_exception(error); }
        } catch (const ninfer::ContextCacheBudgetExceeded&) {
            return "budget_exceeded";
        } catch (const StoreError& store_error) {
            return store_error.kind() == StoreError::Kind::Budget ? "budget_exceeded" : "io_error";
        } catch (const std::exception&) {
            return "engine_error";
        } catch (...) {
            return "engine_error";
        }
        return "engine_error";
    }

    [[nodiscard]] static std::string exception_detail(std::exception_ptr error) {
        try {
            if (error) { std::rethrow_exception(error); }
        } catch (const std::exception& exception) {
            return exception.what();
        } catch (...) {
            return "unknown context cache persistence error";
        }
        return "unknown context cache persistence error";
    }

    void publish_queue_stats_locked() {
        update_stats([&](SessionCacheStoreStats& stats) {
            stats.queued_snapshots = static_cast<std::uint32_t>(
                (active_bytes_ != 0 ? 1U : 0U) + (pending_job_ ? 1U : 0U));
            stats.queued_snapshot_bytes = active_bytes_ + pending_bytes_;
            stats.capture_reserved_bytes = capture_reserved_bytes_;
            stats.capture_in_progress = capture_in_progress_;
            stats.coalesced_snapshots = coalesced_snapshots_;
            stats.cancelled_snapshots = cancelled_snapshots_;
            if (!exclusive_operation_.empty()) {
                stats.operation = exclusive_operation_;
            } else if (capture_in_progress_) {
                stats.operation = "capturing";
            } else if (active_bytes_ != 0 || pending_job_) {
                stats.operation = "saving";
            } else {
                stats.operation = "idle";
            }
        });
    }

    void update_capture_stats_locked(double capture_seconds, std::string error, bool failed) {
        update_stats([&](SessionCacheStoreStats& stats) {
            stats.last_capture_seconds = capture_seconds;
            stats.last_error_code = std::move(error);
            if (failed) { ++stats.failures; }
        });
    }

    void finish_capture_failure(double capture_seconds, std::string error) {
        std::lock_guard lock(queue_mutex_);
        capture_in_progress_ = false;
        capture_session_ = {};
        capture_reserved_bytes_ = 0;
        update_capture_stats_locked(capture_seconds, std::move(error), true);
        publish_queue_stats_locked();
        queue_cv_.notify_all();
    }

    void finish_empty_capture(double capture_seconds) {
        std::lock_guard lock(queue_mutex_);
        capture_in_progress_ = false;
        capture_session_ = {};
        capture_reserved_bytes_ = 0;
        update_capture_stats_locked(capture_seconds, "no_checkpoint", false);
        publish_queue_stats_locked();
        queue_cv_.notify_all();
    }

    [[nodiscard]] bool job_is_current(std::uint64_t epoch) {
        std::lock_guard lock(queue_mutex_);
        return epoch == epoch_;
    }

    [[nodiscard]] bool session_save_pending_locked(std::string_view session) const {
        return (active_bytes_ != 0 && active_session_ == session) ||
               (pending_job_ && pending_job_->session == session) ||
               (capture_in_progress_ && capture_session_ == session);
    }

    void wait_for_session_saves(std::string_view session,
                                const std::function<void()>& checkpoint) {
        std::unique_lock lock(queue_mutex_);
        while (session_save_pending_locked(session)) {
            lock.unlock();
            invoke_checkpoint(checkpoint);
            lock.lock();
            if (!session_save_pending_locked(session)) { break; }
            queue_cv_.wait_for(lock, std::chrono::milliseconds(25));
        }
    }

    void save_worker_loop() noexcept {
        for (;;) {
            std::optional<SaveJob> job;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || pending_job_.has_value(); });
                if (!pending_job_) {
                    if (stopping_) { return; }
                    continue;
                }
                job.emplace(std::move(*pending_job_));
                pending_job_.reset();
                pending_bytes_ = 0;
                active_bytes_ = job->bytes;
                active_session_ = job->session;
                publish_queue_stats_locked();
            }

            double write_seconds = 0.0;
            std::exception_ptr failure;
            std::unique_lock operation_lock(operation_mutex_);
            const auto write_started = std::chrono::steady_clock::now();
            bool attempted_write = false;
            try {
                if (job_is_current(job->epoch)) {
                    attempted_write = true;
                    (void)save_snapshot(*job);
                } else {
                    {
                        std::lock_guard lock(queue_mutex_);
                        ++cancelled_snapshots_;
                    }
                    finish_operation("cancelled", false, false, 0, 0);
                }
            } catch (...) {
                failure = std::current_exception();
                finish_operation(exception_status(failure), false, true, 0, 0);
                cleanup_orphans_best_effort();
                refresh_disk_stats_best_effort();
            }
            if (attempted_write) { write_seconds = elapsed_seconds(write_started); }
            operation_lock.unlock();
            job->snapshot = ninfer::ContextCacheSnapshot{};
            {
                std::lock_guard lock(queue_mutex_);
                active_bytes_ = 0;
                active_session_ = {};
                update_stats([&](SessionCacheStoreStats& stats) {
                    stats.last_write_seconds = write_seconds;
                    if (failure != nullptr) { stats.last_error_code = exception_status(failure); }
                });
                publish_queue_stats_locked();
            }
            queue_cv_.notify_all();
            if (failure != nullptr && job->failure_observer) {
                try { job->failure_observer(exception_detail(failure)); } catch (...) {}
            }
        }
    }

    [[nodiscard]] bool save_snapshot(const SaveJob& job) {
        const std::string base = make_base(job.session);
        cleanup_orphans_locked();
        refresh_disk_stats_locked();
        auto head = read_head(base);
        const std::uint64_t generation = next_generation(head);
        const std::string final_name = snapshot_name(base, generation);
        const std::filesystem::path final_path = child_path(final_name);
        const std::filesystem::path temporary_path = child_path(temporary_name(final_name));

        ensure_room(kSnapshotHeaderBytes + kCommitReserveBytes, base);
        SnapshotFileWriter writer(*this, base, temporary_path, generation,
                                  session_digest(job.session));
        const ninfer::ContextCacheSnapshotStats& saved = job.snapshot.stats();
        job.snapshot.write_to(writer);
        if (saved.bytes != writer.archive_bytes()) {
            throw std::runtime_error("Engine session cache archive length mismatch");
        }
        writer.complete(saved);
        writer.close();
        atomic_replace(temporary_path, final_path);
        if (!job_is_current(job.epoch)) {
            cleanup_orphans_best_effort();
            refresh_disk_stats_best_effort();
            {
                std::lock_guard lock(queue_mutex_);
                ++cancelled_snapshots_;
            }
            finish_operation("cancelled", false, false, 0, 0);
            return false;
        }

        HeadRecord next;
        next.namespace_digest = namespace_digest_;
        next.session_digest = session_digest(job.session);
        next.current_generation = generation;
        next.previous_generation = head ? head->current_generation : 0;
        next.last_access_ms = utc_milliseconds();

        // Keep the previous committed generation available until the replacement is complete.
        // A cache clear advances the epoch before taking operation_mutex_; if it races this commit,
        // it will subsequently remove the just-committed head as part of the same barrier.
        ensure_session_slot(base);
        write_head_atomic(base, next);

        if (head && head->previous_generation != 0) {
            (void)delete_file_accounted(snapshot_name(base, head->previous_generation));
        }
        cleanup_orphans_best_effort();
        refresh_disk_stats_best_effort();
        finish_operation("none", true, false,
                         kSnapshotHeaderBytes + saved.bytes + kSnapshotFooterBytes, 0);
        return true;
    }

    class SnapshotFileWriter final : public ninfer::ContextCacheWriter {
    public:
        SnapshotFileWriter(Impl& owner, std::string base, std::filesystem::path temporary_path,
                           std::uint64_t generation,
                           Digest session_digest)
            : owner_(owner), base_(std::move(base)), temporary_path_(std::move(temporary_path)),
              generation_(generation), session_digest_(session_digest) {
            owner_.ensure_room(kSnapshotHeaderBytes + kCommitReserveBytes, base_);
            file_ = open_file(temporary_path_, GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ, CREATE_NEW,
                              FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN);
            std::vector<std::byte> blank(kSnapshotHeaderBytes, std::byte{0});
            write_exact(file_.get(), blank);
            owner_.used_bytes_ += kSnapshotHeaderBytes;
            owner_.publish_disk_bytes();
        }

        void write(std::span<const std::byte> bytes) override {
            if (completed_ || closed_) { throw std::logic_error("session cache writer is closed"); }
            if (bytes.empty()) { return; }
            if (archive_bytes_ > std::numeric_limits<std::uint64_t>::max() - bytes.size()) {
                throw StoreError(StoreError::Kind::Budget, "session cache archive length overflow");
            }
            if (bytes.size() > std::numeric_limits<std::uint64_t>::max() - kCommitReserveBytes) {
                throw StoreError(StoreError::Kind::Budget, "session cache archive length overflow");
            }
            owner_.ensure_room(bytes.size() + kCommitReserveBytes, base_);
            write_exact(file_.get(), bytes);
            archive_hash_.update(bytes);
            archive_bytes_ += bytes.size();
            owner_.used_bytes_ += bytes.size();
            owner_.publish_disk_bytes();
        }

        [[nodiscard]] std::uint64_t archive_bytes() const noexcept { return archive_bytes_; }

        void complete(const ninfer::ContextCacheSnapshotStats& stats) {
            if (completed_ || closed_) { throw std::logic_error("session cache writer cannot complete"); }
            if (archive_bytes_ == 0 || stats.checkpoints == 0) {
                throw std::logic_error("empty session cache archive cannot be committed");
            }
            if (stats.bytes != archive_bytes_) {
                throw std::runtime_error("Engine session cache archive length mismatch");
            }
            SnapshotHeader header;
            header.generation = generation_;
            header.namespace_digest = owner_.namespace_digest_;
            header.session_digest = session_digest_;
            header.artifact_digest = owner_.artifact_digest_;
            header.model_id = owner_.identity_.model_id;
            header.weights_id = owner_.identity_.weights_id;
            header.vision_enabled = owner_.identity_.vision_enabled;
            header.kv_dtype = owner_.identity_.kv_dtype;
            header.backend = owner_.identity_.speculative_backend;
            header.draft_tokens = owner_.identity_.draft_tokens;
            header.proposal_head = owner_.identity_.proposal_head;
            header.archive_bytes = archive_bytes_;
            header.archive_digest = archive_hash_.finish();
            header.checkpoints = stats.checkpoints;

            const std::vector<std::byte> region = encode_snapshot_region(header);
            seek_file(file_.get(), 0);
            write_exact(file_.get(), region);
            seek_file(file_.get(), kSnapshotHeaderBytes + archive_bytes_);
            std::vector<std::byte> footer;
            footer.reserve(kSnapshotFooterBytes);
            footer.insert(footer.end(), kSnapshotEndMagic.begin(), kSnapshotEndMagic.end());
            append_u64(footer, archive_bytes_);
            footer.insert(footer.end(), header.archive_digest.begin(), header.archive_digest.end());
            if (footer.size() != kSnapshotFooterBytes) {
                throw std::logic_error("session cache footer size mismatch");
            }
            owner_.ensure_room(kCommitReserveBytes, base_);
            write_exact(file_.get(), footer);
            owner_.used_bytes_ += footer.size();
            owner_.publish_disk_bytes();
            flush_file(file_.get());
            completed_ = true;
        }

        void close() {
            if (!completed_) { throw std::logic_error("session cache writer is incomplete"); }
            file_.reset();
            closed_ = true;
        }

        void discard() noexcept {
            file_.reset();
            closed_ = true;
            try { (void)owner_.delete_file_accounted(path_filename_utf8(temporary_path_)); } catch (...) {}
        }

        ~SnapshotFileWriter() override {
            if (!closed_) { discard(); }
        }

    private:
        Impl& owner_;
        std::string base_;
        std::filesystem::path temporary_path_;
        std::uint64_t generation_ = 0;
        Digest session_digest_{};
        FileHandle file_;
        Sha256 archive_hash_;
        std::uint64_t archive_bytes_ = 0;
        bool completed_ = false;
        bool closed_ = false;
    };

    class SnapshotFileReader final : public ninfer::ContextCacheReader {
    public:
        SnapshotFileReader(HANDLE handle, std::uint64_t remaining,
                           const std::function<void()>& checkpoint)
            : handle_(handle), remaining_(remaining), checkpoint_(checkpoint) {}

        void read_exact(std::span<std::byte> bytes) override {
            if (bytes.size() > remaining_) {
                throw StoreError(StoreError::Kind::Io, "Engine read beyond session cache archive");
            }
            std::size_t offset = 0;
            while (offset < bytes.size()) {
                invoke_checkpoint(checkpoint_);
                const std::size_t count = std::min<std::size_t>(bytes.size() - offset, kIoBufferBytes);
                ::ninfer::serve::read_exact(handle_, bytes.subspan(offset, count));
                offset += count;
            }
            remaining_ -= bytes.size();
        }

        [[nodiscard]] std::uint64_t remaining_bytes() const override { return remaining_; }

    private:
        HANDLE handle_ = INVALID_HANDLE_VALUE;
        std::uint64_t remaining_ = 0;
        const std::function<void()>& checkpoint_;
    };

    struct CandidateInfo {
        SnapshotHeader header;
        std::string filename;
        FileHandle file;
    };

    struct SessionEntry {
        std::string base;
        HeadRecord head;
    };

    [[nodiscard]] Digest make_namespace_digest() const {
        std::vector<std::byte> material;
        material.reserve(identity_.model_id.size() + identity_.weights_id.size() + 64U);
        append_u32(material, kStoreFormatVersion);
        append_u32(material, kEngineMathRevision);
        append_string(material, identity_.model_id);
        append_string(material, identity_.weights_id);
        material.insert(material.end(), artifact_digest_.begin(), artifact_digest_.end());
        append_u32(material, identity_.vision_enabled ? 1U : 0U);
        append_u32(material, static_cast<std::uint32_t>(identity_.kv_dtype));
        append_u32(material, static_cast<std::uint32_t>(identity_.speculative_backend));
        append_u32(material, identity_.draft_tokens);
        append_u32(material, static_cast<std::uint32_t>(identity_.proposal_head));
        return digest_bytes(material);
    }

    [[nodiscard]] Digest session_digest(std::string_view session) const { return digest_text(session); }

    [[nodiscard]] std::string make_base(std::string_view session) const {
        return base_name(namespace_digest_, session_digest(session));
    }

    static void validate_session(std::string_view session) {
        if (session.empty() || session.size() > ninfer::kMaximumContextCacheSessionKeyBytes) {
            throw std::invalid_argument("session cache key length is invalid");
        }
    }

    [[nodiscard]] std::filesystem::path child_path(std::string_view name) const {
        if (!is_owned_filename(name)) { throw std::invalid_argument("invalid session cache filename"); }
        const std::filesystem::path result = root_ / widen_ascii(name);
        if (result.parent_path().lexically_normal() != root_.lexically_normal()) {
            throw std::invalid_argument("session cache path escaped its root");
        }
        return result;
    }

    [[nodiscard]] std::filesystem::path head_path(std::string_view base) const {
        if (!is_valid_base(base)) { throw std::invalid_argument("invalid session cache namespace"); }
        return child_path(std::string(base) + ".head");
    }

    [[nodiscard]] std::string temporary_name(std::string_view destination_name) const {
        return std::string(destination_name) + ".tmp." + fixed_hex_u64(unique_nonce());
    }

    [[nodiscard]] std::optional<HeadRecord> read_head(std::string_view base) const {
        const std::filesystem::path path = head_path(base);
        const DWORD attributes = ::GetFileAttributesW(extended_windows_path(path).c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
            return std::nullopt;
        }
        FileHandle file = open_file(path, GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN);
        if (file_size(file.get()) != kHeadFileBytes) { return std::nullopt; }
        std::array<std::byte, kHeadFileBytes> bytes{};
        read_exact(file.get(), bytes);
        HeadRecord decoded;
        try {
            decoded = decode_head(bytes);
        } catch (const StoreError&) {
            throw;
        } catch (...) {
            return std::nullopt;
        }
        const std::string expected_base = base_name(decoded.namespace_digest, decoded.session_digest);
        if (expected_base != base) { return std::nullopt; }
        return decoded;
    }

    [[nodiscard]] std::vector<std::string> list_owned_files() const {
        std::vector<std::string> result;
        std::error_code error;
        for (std::filesystem::directory_iterator it(root_, error), end; it != end && !error;
             it.increment(error)) {
            const std::string name = path_filename_utf8(it->path());
            if (!is_owned_filename(name)) { continue; }
            const DWORD attributes = ::GetFileAttributesW(extended_windows_path(it->path()).c_str());
            if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
                (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
                continue;
            }
            result.push_back(name);
        }
        if (error) { throw StoreError(StoreError::Kind::Io, "session cache directory scan failed"); }
        return result;
    }

    [[nodiscard]] std::uint64_t named_file_size(std::string_view name) const {
        const std::filesystem::path path = child_path(name);
        const DWORD attributes = ::GetFileAttributesW(extended_windows_path(path).c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES || (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
            return 0;
        }
        FileHandle file = open_file(path, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE,
                                    OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN);
        return file_size(file.get());
    }

    [[nodiscard]] std::uint64_t calculate_disk_bytes() const {
        std::uint64_t total = 0;
        for (const std::string& name : list_owned_files()) {
            const std::uint64_t size = named_file_size(name);
            if (size > std::numeric_limits<std::uint64_t>::max() - total) {
                throw StoreError(StoreError::Kind::Io, "session cache disk size overflow");
            }
            total += size;
        }
        return total;
    }

    void refresh_disk_stats_locked() {
        used_bytes_ = calculate_disk_bytes();
        std::uint32_t sessions = 0;
        for (const std::string& name : list_owned_files()) {
            std::string base;
            if (parse_head_name(name, base) && read_head(base)) { ++sessions; }
        }
        update_stats([&](SessionCacheStoreStats& stats) {
            stats.disk_bytes = used_bytes_;
            stats.sessions = sessions;
        });
    }

    void refresh_disk_stats_best_effort() noexcept {
        try { refresh_disk_stats_locked(); } catch (...) {}
    }

    void update_stats(const std::function<void(SessionCacheStoreStats&)>& update) const {
        std::lock_guard lock(stats_mutex_);
        update(stats_);
    }

    void publish_disk_bytes() {
        update_stats([&](SessionCacheStoreStats& stats) { stats.disk_bytes = used_bytes_; });
    }

    void set_operation(std::string operation) {
        std::lock_guard lock(queue_mutex_);
        exclusive_operation_ = std::move(operation);
        publish_queue_stats_locked();
    }

    void set_error(std::string error) {
        update_stats([&](SessionCacheStoreStats& stats) {
            stats.last_error_code = std::move(error);
        });
    }

    [[nodiscard]] std::string current_operation() const {
        std::lock_guard lock(stats_mutex_);
        return stats_.operation;
    }

    void finish_operation(std::string error, bool saved, bool failed,
                          std::uint64_t saved_bytes, std::uint64_t restored_bytes) {
        std::lock_guard queue_lock(queue_mutex_);
        exclusive_operation_.clear();
        update_stats([&](SessionCacheStoreStats& stats) {
            stats.last_error_code = std::move(error);
            if (saved) { ++stats.saves; }
            if (restored_bytes != 0) { ++stats.restores; }
            if (failed) { ++stats.failures; }
            if (!saved && !restored_bytes && stats.last_error_code == "miss") { ++stats.misses; }
            if (saved) { stats.last_saved_bytes = saved_bytes; }
            if (restored_bytes != 0) { stats.last_restored_bytes = restored_bytes; }
        });
        publish_queue_stats_locked();
    }

    [[nodiscard]] std::uint64_t next_generation(const std::optional<HeadRecord>& head) const {
        if (!head) { return 1; }
        if (head->current_generation == std::numeric_limits<std::uint64_t>::max()) {
            throw StoreError(StoreError::Kind::Budget, "session cache generation overflow");
        }
        return head->current_generation + 1U;
    }

    void ensure_room(std::uint64_t additional_bytes, std::string_view preserve_base,
                     bool evict_sessions = true) {
        for (;;) {
            if (additional_bytes <= options_.max_bytes &&
                used_bytes_ <= options_.max_bytes - additional_bytes) {
                return;
            }
            if (!evict_sessions || !evict_one_lru(preserve_base)) {
                throw StoreError(StoreError::Kind::Budget, "session cache disk budget exceeded");
            }
        }
    }

    void ensure_session_slot(std::string_view preserve_base) {
        for (;;) {
            std::vector<SessionEntry> entries = session_entries();
            const bool already_exists = std::any_of(entries.begin(), entries.end(), [&](const auto& entry) {
                return entry.base == preserve_base;
            });
            if (already_exists || entries.size() < options_.max_sessions) { return; }
            if (!evict_one_lru(preserve_base)) {
                throw StoreError(StoreError::Kind::Budget, "session cache session limit reached");
            }
        }
    }

    [[nodiscard]] std::vector<SessionEntry> session_entries() const {
        std::vector<SessionEntry> result;
        for (const std::string& name : list_owned_files()) {
            std::string base;
            if (!parse_head_name(name, base)) { continue; }
            const auto head = read_head(base);
            if (head) { result.push_back(SessionEntry{base, *head}); }
        }
        return result;
    }

    [[nodiscard]] bool evict_one_lru(std::string_view preserve_base) {
        std::vector<SessionEntry> entries = session_entries();
        std::sort(entries.begin(), entries.end(), [](const SessionEntry& left, const SessionEntry& right) {
            if (left.head.last_access_ms != right.head.last_access_ms) {
                return left.head.last_access_ms < right.head.last_access_ms;
            }
            return left.base < right.base;
        });
        for (const SessionEntry& entry : entries) {
            if (entry.base == preserve_base) { continue; }
            if (erase_session(entry.base, entry.head)) {
                used_bytes_ = calculate_disk_bytes();
                update_stats([&](SessionCacheStoreStats& stats) {
                    stats.disk_bytes = used_bytes_;
                    if (stats.sessions != 0) { --stats.sessions; }
                });
                return true;
            }
        }
        return false;
    }

    [[nodiscard]] bool erase_session(std::string_view base, const HeadRecord& head) {
        if (!delete_file_accounted(std::string(base) + ".head")) { return false; }
        if (head.current_generation != 0) {
            (void)delete_file_accounted(snapshot_name(base, head.current_generation));
        }
        if (head.previous_generation != 0) {
            (void)delete_file_accounted(snapshot_name(base, head.previous_generation));
        }
        for (const std::string& name : list_owned_files()) {
            std::string candidate_base;
            std::uint64_t generation = 0;
            if (parse_snapshot_name(name, candidate_base, generation) && candidate_base == base) {
                (void)delete_file_accounted(name);
            }
        }
        return true;
    }

    [[nodiscard]] bool delete_file_accounted(std::string_view name) {
        if (!is_owned_filename(name)) { return false; }
        const std::filesystem::path path = child_path(name);
        std::error_code canonical_error;
        const std::filesystem::path resolved_parent =
            std::filesystem::weakly_canonical(path.parent_path(), canonical_error);
        if (canonical_error || resolved_parent != root_) { return false; }
        const DWORD attributes = ::GetFileAttributesW(extended_windows_path(path).c_str());
        if (attributes == INVALID_FILE_ATTRIBUTES) {
            const DWORD error = ::GetLastError();
            return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND;
        }
        if ((attributes & FILE_ATTRIBUTE_DIRECTORY) != 0 ||
            (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0) {
            return false;
        }
        std::uint64_t size = 0;
        try { size = named_file_size(name); } catch (...) { return false; }
        if (!::DeleteFileW(extended_windows_path(path).c_str())) {
            const DWORD error = ::GetLastError();
            return error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND;
        }
        used_bytes_ = used_bytes_ >= size ? used_bytes_ - size : 0;
        return true;
    }

    void write_head_atomic(std::string_view base, const HeadRecord& head,
                           bool evict_for_space = true) {
        const std::string final_name = std::string(base) + ".head";
        const std::string temp_name = temporary_name(final_name);
        const std::filesystem::path final_path = child_path(final_name);
        const std::filesystem::path temp_path = child_path(temp_name);
        const std::uint64_t replaced_size = named_file_size(final_name);
        ensure_room(kHeadFileBytes, base, evict_for_space);
        const std::vector<std::byte> bytes = encode_head(head);
        FileHandle file = open_file(temp_path, GENERIC_WRITE, 0, CREATE_NEW, FILE_ATTRIBUTE_NORMAL);
        write_exact(file.get(), bytes);
        flush_file(file.get());
        file.reset();
        used_bytes_ += bytes.size();
        atomic_replace(temp_path, final_path);
        used_bytes_ = used_bytes_ >= replaced_size ? used_bytes_ - replaced_size : 0;
        update_stats([&](SessionCacheStoreStats& stats) { stats.disk_bytes = used_bytes_; });
    }

    [[nodiscard]] CandidateInfo validate_candidate(std::string_view base, std::uint64_t generation,
                                                  const Digest& expected_session,
                                                  const std::function<void()>& checkpoint) const {
        const std::string filename = snapshot_name(base, generation);
        FileHandle file;
        try {
            file = open_file(child_path(filename), GENERIC_READ, FILE_SHARE_READ, OPEN_EXISTING,
                             FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN);
        } catch (const StoreError&) {
            throw;
        }
        const std::uint64_t size = file_size(file.get());
        if (size < kSnapshotHeaderBytes + kSnapshotFooterBytes || size > options_.max_bytes) {
            throw CandidateError("checksum_mismatch", "invalid session cache snapshot size");
        }
        std::vector<std::byte> region(kSnapshotHeaderBytes);
        read_exact(file.get(), region);
        SnapshotHeader header;
        try {
            header = parse_snapshot_region(region);
        } catch (...) {
            throw CandidateError("checksum_mismatch", "invalid session cache snapshot metadata");
        }
        if (header.generation != generation || header.namespace_digest != namespace_digest_ ||
            header.session_digest != expected_session || header.artifact_digest != artifact_digest_ ||
            header.model_id != identity_.model_id || header.weights_id != identity_.weights_id ||
            header.vision_enabled != identity_.vision_enabled ||
            header.kv_dtype != identity_.kv_dtype || header.backend != identity_.speculative_backend ||
            header.draft_tokens != identity_.draft_tokens ||
            header.proposal_head != identity_.proposal_head) {
            throw CandidateError("identity_mismatch", "session cache identity mismatch");
        }
        if (header.archive_bytes > std::numeric_limits<std::uint64_t>::max() -
                                       kSnapshotHeaderBytes - kSnapshotFooterBytes ||
            size != kSnapshotHeaderBytes + header.archive_bytes + kSnapshotFooterBytes) {
            throw CandidateError("checksum_mismatch", "invalid session cache archive length");
        }
        seek_file(file.get(), kSnapshotHeaderBytes);
        Sha256 digest;
        std::vector<std::byte> buffer(kIoBufferBytes);
        std::uint64_t remaining = header.archive_bytes;
        while (remaining != 0) {
            invoke_checkpoint(checkpoint);
            const std::size_t count = static_cast<std::size_t>(
                std::min<std::uint64_t>(remaining, buffer.size()));
            read_exact(file.get(), std::span(buffer).first(count));
            digest.update(std::span<const std::byte>(buffer).first(count));
            remaining -= count;
        }
        if (digest.finish() != header.archive_digest) {
            throw CandidateError("checksum_mismatch", "session cache archive checksum mismatch");
        }
        std::array<std::byte, kSnapshotFooterBytes> footer{};
        read_exact(file.get(), footer);
        if (!std::equal(kSnapshotEndMagic.begin(), kSnapshotEndMagic.end(), footer.begin()) ||
            read_u64(footer, kSnapshotEndMagic.size()) != header.archive_bytes ||
            !std::equal(header.archive_digest.begin(), header.archive_digest.end(),
                        footer.begin() + static_cast<std::ptrdiff_t>(kSnapshotEndMagic.size() +
                                                                      sizeof(std::uint64_t)))) {
            throw CandidateError("checksum_mismatch", "session cache complete marker is invalid");
        }
        return CandidateInfo{std::move(header), filename, std::move(file)};
    }

    void cleanup_orphans_locked() {
        const std::vector<std::string> names = list_owned_files();
        std::map<std::string, HeadRecord> valid_heads;
        std::set<std::string> protected_bases;
        for (const std::string& name : names) {
            std::string base;
            if (!parse_head_name(name, base)) { continue; }
            try {
                const auto head = read_head(base);
                if (head) {
                    valid_heads.emplace(base, *head);
                } else {
                    (void)delete_file_accounted(name);
                }
            } catch (...) {
                // A read error is not proof that the head is invalid. Preserve this session's
                // head and every generation until a later pass can read it or prove it malformed.
                protected_bases.insert(base);
                set_error("io_error");
            }
        }
        for (const std::string& name : names) {
            std::string base;
            std::uint64_t generation = 0;
            if (parse_temporary_name(name, base)) {
                (void)delete_file_accounted(name);
                continue;
            }
            if (!parse_snapshot_name(name, base, generation)) { continue; }
            if (protected_bases.contains(base)) { continue; }
            const auto found = valid_heads.find(base);
            if (found == valid_heads.end() ||
                (generation != found->second.current_generation &&
                 generation != found->second.previous_generation)) {
                (void)delete_file_accounted(name);
            }
        }
        refresh_disk_stats_locked();
    }

    void cleanup_orphans_best_effort() noexcept {
        try { cleanup_orphans_locked(); } catch (...) {}
    }

    mutable std::mutex stats_mutex_;
    mutable SessionCacheStoreStats stats_;
    std::mutex operation_mutex_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_;
    std::optional<SaveJob> pending_job_;
    bool stopping_ = false;
    bool capture_in_progress_ = false;
    // The capture caller owns its session through the call; the worker's SaveJob owns the active
    // session through the write. These views avoid allocation in the noexcept writer loop.
    std::string_view capture_session_;
    std::string_view active_session_;
    std::uint64_t active_bytes_ = 0;
    std::uint64_t pending_bytes_ = 0;
    std::uint64_t capture_reserved_bytes_ = 0;
    std::uint64_t coalesced_snapshots_ = 0;
    std::uint64_t cancelled_snapshots_ = 0;
    std::uint64_t epoch_ = 0;
    std::string exclusive_operation_;
    SessionCacheStoreOptions options_;
    SessionCacheIdentity identity_;
    std::filesystem::path root_;
    Digest artifact_digest_{};
    Digest namespace_digest_{};
    std::uint64_t used_bytes_ = 0;
};

SessionCacheStore::SessionCacheStore(SessionCacheStoreOptions options,
                                     SessionCacheIdentity identity)
    : impl_(std::make_unique<Impl>(std::move(options), std::move(identity))) {}

SessionCacheStore::~SessionCacheStore() = default;
SessionCacheStore::SessionCacheStore(SessionCacheStore&&) noexcept = default;
SessionCacheStore& SessionCacheStore::operator=(SessionCacheStore&&) noexcept = default;

SessionCacheCaptureResult SessionCacheStore::capture_and_enqueue(
    ninfer::Engine& engine, std::string_view session,
    std::function<void(std::string_view)> failure_observer) {
    if (!impl_) { return SessionCacheCaptureResult::Skipped; }
    return impl_->capture_and_enqueue(engine, session, std::move(failure_observer));
}

bool SessionCacheStore::restore(ninfer::Engine& engine, std::string_view session,
                                std::function<void()> checkpoint, bool replace_in_memory) {
    if (!impl_) { return false; }
    return impl_->restore(engine, session, checkpoint, replace_in_memory);
}

void SessionCacheStore::clear() {
    if (impl_) { impl_->clear(); }
}

SessionCacheStoreStats SessionCacheStore::snapshot_stats() const {
    return impl_ ? impl_->snapshot_stats() : SessionCacheStoreStats{};
}

} // namespace ninfer::serve

#endif // defined(NINFER_WINDOWS_SERVE)
