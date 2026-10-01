// Windows/V100 implementation by taotuotu, 2026; see NOTICE.
#include "product/media_acquire/acquire.h"

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <winhttp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace ninfer::product::media_acquire {
namespace {

using Clock = std::chrono::steady_clock;

class Control {
public:
    explicit Control(const Policy& policy) : policy_(policy) {}

    void check() const {
        if (policy_.is_cancelled && policy_.is_cancelled()) {
            throw Error(ErrorKind::Cancelled, "media acquisition was cancelled");
        }
        if (policy_.deadline != Clock::time_point{} && Clock::now() >= policy_.deadline) {
            throw Error(ErrorKind::DeadlineExceeded, "media acquisition exceeded request deadline");
        }
    }

    void check_remote() const {
        check();
        if (Clock::now() >= remote_deadline_) {
            throw Error(ErrorKind::RemoteTimeout, "media URL fetch exceeded its time limit");
        }
    }

    [[nodiscard]] int bounded_timeout_ms(int configured, bool remote) const {
        if (configured <= 0) { return 1; }
        const Clock::time_point deadline = remote ? remote_deadline_ : policy_.deadline;
        if (deadline == Clock::time_point{}) { return configured; }
        if (remote) { check_remote(); }
        else { check(); }
        const auto remaining =
            std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
        return std::min<int>(configured, static_cast<int>(std::clamp<std::int64_t>(
                                             remaining, 1, std::numeric_limits<int>::max())));
    }

private:
    static Clock::time_point make_remote_deadline(const Policy& policy) {
        const Clock::time_point timeout_deadline =
            Clock::now() + std::chrono::milliseconds(std::max(1, policy.timeout_ms));
        if (policy.deadline == Clock::time_point{}) { return timeout_deadline; }
        return std::min(timeout_deadline, policy.deadline);
    }

    const Policy& policy_;
    const Clock::time_point remote_deadline_ = make_remote_deadline(policy_);
};

std::wstring utf8_to_wide(std::string_view value, const char* label) {
    if (value.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        value.find('\0') != std::string_view::npos) {
        throw std::invalid_argument(std::string(label) + " is invalid UTF-8 text");
    }
    if (value.empty()) { return {}; }
    const int input_bytes = static_cast<int>(value.size());
    const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(),
                                              input_bytes, nullptr, 0);
    if (required <= 0) { throw std::invalid_argument(std::string(label) + " is invalid UTF-8"); }
    std::wstring out(static_cast<std::size_t>(required), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, value.data(), input_bytes, out.data(),
                            required) != required) {
        throw std::invalid_argument(std::string(label) + " is invalid UTF-8");
    }
    return out;
}

std::string win32_message(const char* operation, DWORD code) {
    return std::string(operation) + " (Windows error " + std::to_string(code) + ")";
}

[[noreturn]] void throw_http_error(const char* operation, DWORD code) {
    if (code == ERROR_WINHTTP_TIMEOUT) {
        throw Error(ErrorKind::RemoteTimeout, win32_message(operation, code));
    }
    throw Error(ErrorKind::RemoteUnavailable, win32_message(operation, code));
}

class InternetHandle {
public:
    InternetHandle() = default;
    explicit InternetHandle(HINTERNET handle) noexcept : handle_(handle) {}
    ~InternetHandle() { reset(); }

    InternetHandle(InternetHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    InternetHandle& operator=(InternetHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    InternetHandle(const InternetHandle&)            = delete;
    InternetHandle& operator=(const InternetHandle&) = delete;

    [[nodiscard]] HINTERNET get() const noexcept { return handle_; }
    [[nodiscard]] explicit operator bool() const noexcept { return handle_ != nullptr; }
    [[nodiscard]] HINTERNET release() noexcept { return std::exchange(handle_, nullptr); }

    void reset(HINTERNET handle = nullptr) noexcept {
        if (handle_ != nullptr) { (void)WinHttpCloseHandle(handle_); }
        handle_ = handle;
    }

private:
    HINTERNET handle_ = nullptr;
};

class KernelHandle {
public:
    KernelHandle() = default;
    explicit KernelHandle(HANDLE handle) noexcept : handle_(handle) {}
    ~KernelHandle() { reset(); }

    KernelHandle(KernelHandle&& other) noexcept : handle_(std::exchange(other.handle_, nullptr)) {}
    KernelHandle& operator=(KernelHandle&& other) noexcept {
        if (this != &other) {
            reset();
            handle_ = std::exchange(other.handle_, nullptr);
        }
        return *this;
    }

    KernelHandle(const KernelHandle&)            = delete;
    KernelHandle& operator=(const KernelHandle&) = delete;

    [[nodiscard]] HANDLE get() const noexcept { return handle_; }

    void reset(HANDLE handle = nullptr) noexcept {
        if (handle_ != nullptr && handle_ != INVALID_HANDLE_VALUE) { (void)CloseHandle(handle_); }
        handle_ = handle;
    }

private:
    HANDLE handle_ = nullptr;
};

enum class AsyncOperation : std::uint8_t {
    None,
    Send,
    Receive,
    Read,
};

struct AsyncCompletion {
    DWORD error = ERROR_SUCCESS;
    DWORD bytes = 0;
};

class AsyncRequest {
public:
    static constexpr std::size_t kReadCapacity = 64U * 1024U;

    AsyncRequest() = default;
    ~AsyncRequest() { close_noexcept(); }

    AsyncRequest(const AsyncRequest&)            = delete;
    AsyncRequest& operator=(const AsyncRequest&) = delete;

    [[nodiscard]] DWORD_PTR context() noexcept {
        return reinterpret_cast<DWORD_PTR>(this);
    }

    void attach(HINTERNET handle) noexcept { handle_ = handle; }
    [[nodiscard]] HINTERNET get() const noexcept { return handle_; }
    [[nodiscard]] std::uint8_t* read_buffer() noexcept { return read_buffer_.data(); }
    [[nodiscard]] DWORD read_capacity() const noexcept {
        return static_cast<DWORD>(read_buffer_.size());
    }

    void begin(AsyncOperation operation) {
        std::lock_guard lock(mutex_);
        operation_ = operation;
        completed_ = false;
        completion_ = {};
    }

    void fail_immediate(DWORD error) {
        std::lock_guard lock(mutex_);
        if (!completed_) {
            completion_.error = error;
            completed_      = true;
        }
        cv_.notify_all();
    }

    AsyncCompletion wait(Control& control) {
        try {
            for (;;) {
                {
                    std::unique_lock lock(mutex_);
                    if (completed_) { return completion_; }
                    cv_.wait_for(lock, std::chrono::milliseconds(50));
                    if (completed_) { return completion_; }
                }
                control.check_remote();
            }
        } catch (...) {
            close_and_wait();
            throw;
        }
    }

    void close_and_wait() {
        HINTERNET handle = std::exchange(handle_, nullptr);
        DWORD close_error = ERROR_SUCCESS;
        if (handle != nullptr && !WinHttpCloseHandle(handle)) { close_error = GetLastError(); }
        std::unique_lock lock(mutex_);
        if (handle == nullptr && !close_started_) { handle_closed_ = true; }
        close_started_ = true;
        // The callback owns this context until HANDLE_CLOSING, even if WinHttpCloseHandle
        // reported an error. Do not return and destroy the context before that final callback.
        cv_.wait(lock, [&] { return handle_closed_; });
        lock.unlock();
        if (close_error != ERROR_SUCCESS) {
            throw_http_error("failed to close media request", close_error);
        }
    }

    void on_status(DWORD status, LPVOID information, DWORD information_length) noexcept {
        std::lock_guard lock(mutex_);
        switch (status) {
        case WINHTTP_CALLBACK_STATUS_HANDLE_CLOSING:
            handle_closed_ = true;
            cv_.notify_all();
            return;
        case WINHTTP_CALLBACK_STATUS_REQUEST_ERROR: {
            if (information == nullptr || information_length < sizeof(WINHTTP_ASYNC_RESULT)) {
                completion_.error = ERROR_WINHTTP_INTERNAL_ERROR;
            } else {
                completion_.error =
                    static_cast<const WINHTTP_ASYNC_RESULT*>(information)->dwError;
            }
            completed_ = true;
            cv_.notify_all();
            return;
        }
        case WINHTTP_CALLBACK_STATUS_SENDREQUEST_COMPLETE:
            complete_if(AsyncOperation::Send, ERROR_SUCCESS, 0);
            return;
        case WINHTTP_CALLBACK_STATUS_HEADERS_AVAILABLE:
            complete_if(AsyncOperation::Receive, ERROR_SUCCESS, 0);
            return;
        case WINHTTP_CALLBACK_STATUS_READ_COMPLETE:
            complete_if(AsyncOperation::Read, ERROR_SUCCESS, information_length);
            return;
        default:
            return;
        }
    }

private:
    void complete_if(AsyncOperation expected, DWORD error, DWORD bytes) noexcept {
        if (operation_ != expected || completed_) { return; }
        completion_.error = error;
        completion_.bytes = bytes;
        completed_        = true;
        cv_.notify_all();
    }

    void close_noexcept() noexcept {
        try {
            close_and_wait();
        } catch (...) {}
    }

    HINTERNET handle_ = nullptr;
    std::array<std::uint8_t, kReadCapacity> read_buffer_{};
    std::mutex mutex_;
    std::condition_variable cv_;
    AsyncOperation operation_ = AsyncOperation::None;
    AsyncCompletion completion_;
    bool completed_    = false;
    bool close_started_ = false;
    bool handle_closed_ = false;
};

VOID CALLBACK winhttp_status_callback(HINTERNET, DWORD_PTR context, DWORD status,
                                      LPVOID information, DWORD information_length) noexcept {
    auto* request = reinterpret_cast<AsyncRequest*>(context);
    if (request != nullptr) { request->on_status(status, information, information_length); }
}

void set_winhttp_option(HINTERNET handle, DWORD option, const void* value, DWORD bytes,
                        const char* label) {
    if (!WinHttpSetOption(handle, option, const_cast<void*>(value), bytes)) {
        throw_http_error(label, GetLastError());
    }
}

AsyncCompletion finish_async_call(AsyncRequest& request, BOOL started, Control& control,
                                  const char* label) {
    if (!started) {
        const DWORD error = GetLastError();
        if (error != ERROR_IO_PENDING) { request.fail_immediate(error); }
    }
    const AsyncCompletion completion = request.wait(control);
    if (completion.error != ERROR_SUCCESS) {
        control.check_remote();
        throw_http_error(label, completion.error);
    }
    return completion;
}

bool ipv4_in(std::uint32_t address, std::uint32_t network, std::uint32_t mask) noexcept {
    return (address & mask) == network;
}

bool private_ipv4(std::uint32_t network_order) noexcept {
    const std::uint32_t address = ntohl(network_order);
    return ipv4_in(address, 0x00000000U, 0xff000000U) || // this network
           ipv4_in(address, 0x0a000000U, 0xff000000U) || // private
           ipv4_in(address, 0x64400000U, 0xffc00000U) || // shared address space
           ipv4_in(address, 0x7f000000U, 0xff000000U) || // loopback
           ipv4_in(address, 0xa9fe0000U, 0xffff0000U) || // link local
           ipv4_in(address, 0xac100000U, 0xfff00000U) || // private
           ipv4_in(address, 0xc0000000U, 0xffffff00U) || // protocol assignments
           ipv4_in(address, 0xc0000200U, 0xffffff00U) || // documentation
           ipv4_in(address, 0xc0586300U, 0xffffff00U) || // deprecated relay anycast
           ipv4_in(address, 0xc0a80000U, 0xffff0000U) || // private
           ipv4_in(address, 0xc6120000U, 0xfffe0000U) || // benchmarking
           ipv4_in(address, 0xc6336400U, 0xffffff00U) || // documentation
           ipv4_in(address, 0xcb007100U, 0xffffff00U) || // documentation
           ipv4_in(address, 0xe0000000U, 0xf0000000U) || // multicast
           ipv4_in(address, 0xf0000000U, 0xf0000000U);   // reserved/broadcast
}

bool private_ipv6(const IN6_ADDR& address) noexcept {
    if (IN6_IS_ADDR_UNSPECIFIED(&address) || IN6_IS_ADDR_LOOPBACK(&address) ||
        IN6_IS_ADDR_LINKLOCAL(&address) || IN6_IS_ADDR_SITELOCAL(&address) ||
        IN6_IS_ADDR_MULTICAST(&address) || (address.u.Byte[0] & 0xfeU) == 0xfcU) {
        return true;
    }
    if (IN6_IS_ADDR_V4MAPPED(&address)) {
        std::uint32_t ipv4 = 0;
        std::memcpy(&ipv4, &address.u.Byte[12], sizeof(ipv4));
        return private_ipv4(ipv4);
    }
    const bool first_96_zero =
        std::all_of(address.u.Byte, address.u.Byte + 12, [](std::uint8_t byte) { return byte == 0; });
    if (first_96_zero) { return true; } // deprecated IPv4-compatible form
    if (address.u.Byte[0] == 0x20U && address.u.Byte[1] == 0x01U &&
        address.u.Byte[2] == 0x0dU && address.u.Byte[3] == 0xb8U) {
        return true; // documentation range
    }
    if (address.u.Byte[0] == 0x00U && address.u.Byte[1] == 0x64U &&
        address.u.Byte[2] == 0xffU && address.u.Byte[3] == 0x9bU) {
        return true; // well-known NAT64 prefix
    }
    return false;
}

struct UrlParts {
    INTERNET_SCHEME scheme_id{};
    std::wstring scheme;
    std::wstring host;
    std::wstring path_query;
    INTERNET_PORT port = 0;

    [[nodiscard]] bool secure() const noexcept { return scheme_id == INTERNET_SCHEME_HTTPS; }

    [[nodiscard]] std::wstring origin() const {
        std::wstring out = scheme + L"://";
        if (host.find(L':') != std::wstring::npos) { out += L"[" + host + L"]"; }
        else { out += host; }
        out += L":" + std::to_wstring(port);
        return out;
    }
};

UrlParts parse_url(std::wstring_view input) {
    if (input.empty() || input.size() > std::numeric_limits<DWORD>::max() ||
        input.find(L'\0') != std::wstring_view::npos) {
        throw std::invalid_argument("invalid media URL");
    }
    const std::size_t authority_begin = input.find(L"://");
    if (authority_begin == std::wstring_view::npos) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }
    const std::size_t authority_start = authority_begin + 3;
    const std::size_t authority_end = input.find_first_of(L"/?#", authority_start);
    const std::wstring_view authority = input.substr(
        authority_start, authority_end == std::wstring_view::npos
                             ? input.size() - authority_start
                             : authority_end - authority_start);
    if (authority.empty() || authority.find(L'@') != std::wstring_view::npos) {
        throw std::invalid_argument("media URL must not contain credentials");
    }

    std::wstring owned(input);
    URL_COMPONENTS components{};
    components.dwStructSize       = sizeof(components);
    components.dwSchemeLength     = static_cast<DWORD>(-1);
    components.dwHostNameLength   = static_cast<DWORD>(-1);
    components.dwUserNameLength   = static_cast<DWORD>(-1);
    components.dwPasswordLength   = static_cast<DWORD>(-1);
    components.dwUrlPathLength    = static_cast<DWORD>(-1);
    components.dwExtraInfoLength  = static_cast<DWORD>(-1);
    if (!WinHttpCrackUrl(owned.c_str(), static_cast<DWORD>(owned.size()), 0, &components)) {
        throw std::invalid_argument("invalid media URL");
    }
    if ((components.nScheme != INTERNET_SCHEME_HTTP &&
         components.nScheme != INTERNET_SCHEME_HTTPS) ||
        components.dwHostNameLength == 0 || components.nPort == 0 ||
        components.dwUserNameLength != 0 || components.dwPasswordLength != 0) {
        throw std::invalid_argument("media URL must be credential-free HTTP(S)");
    }

    UrlParts out;
    out.scheme_id = components.nScheme;
    out.scheme.assign(components.lpszScheme, components.dwSchemeLength);
    out.host.assign(components.lpszHostName, components.dwHostNameLength);
    if (out.host.size() >= 2 && out.host.front() == L'[' && out.host.back() == L']') {
        out.host = out.host.substr(1, out.host.size() - 2);
    }
    out.port = components.nPort;
    if (components.dwUrlPathLength > 0) {
        out.path_query.assign(components.lpszUrlPath, components.dwUrlPathLength);
    } else {
        out.path_query = L"/";
    }
    if (components.dwExtraInfoLength > 0) {
        out.path_query.append(components.lpszExtraInfo, components.dwExtraInfoLength);
    }
    if (const std::size_t fragment = out.path_query.find(L'#');
        fragment != std::wstring::npos) {
        out.path_query.erase(fragment);
    }
    if (out.path_query.empty()) { out.path_query = L"/"; }
    if (out.path_query.front() != L'/') { out.path_query.insert(out.path_query.begin(), L'/'); }
    return out;
}

std::wstring normalize_relative_path(std::wstring path) {
    const std::size_t query_index = path.find(L'?');
    std::wstring query;
    if (query_index != std::wstring::npos) {
        query = path.substr(query_index);
        path.erase(query_index);
    }
    const bool trailing = path.empty() || path.back() == L'/' || path.ends_with(L"/.") ||
                          path.ends_with(L"/..");
    std::vector<std::wstring> segments;
    std::size_t cursor = 0;
    while (cursor <= path.size()) {
        const std::size_t slash = path.find(L'/', cursor);
        const std::size_t end   = slash == std::wstring::npos ? path.size() : slash;
        const std::wstring_view segment(path.data() + cursor, end - cursor);
        if (segment == L"..") {
            if (!segments.empty()) { segments.pop_back(); }
        } else if (segment != L"." && !(cursor == 0 && segment.empty())) {
            segments.emplace_back(segment);
        }
        if (slash == std::wstring::npos) { break; }
        cursor = slash + 1;
    }
    std::wstring out = L"/";
    for (std::size_t i = 0; i < segments.size(); ++i) {
        if (i != 0) { out.push_back(L'/'); }
        out += segments[i];
    }
    if (trailing && !out.ends_with(L'/')) { out.push_back(L'/'); }
    out += query;
    return out;
}

std::wstring resolve_redirect(const UrlParts& current, std::wstring location) {
    while (!location.empty() && (location.front() == L' ' || location.front() == L'\t' ||
                                 location.front() == L'\r' || location.front() == L'\n')) {
        location.erase(location.begin());
    }
    while (!location.empty() && (location.back() == L' ' || location.back() == L'\t' ||
                                 location.back() == L'\r' || location.back() == L'\n')) {
        location.pop_back();
    }
    if (const std::size_t fragment = location.find(L'#'); fragment != std::wstring::npos) {
        location.erase(fragment);
    }
    if (location.empty()) { throw Error(ErrorKind::RemoteUnavailable, "media redirect has no location"); }

    const std::size_t colon     = location.find(L':');
    const std::size_t delimiter = location.find_first_of(L"/?");
    if (colon != std::wstring::npos &&
        (delimiter == std::wstring::npos || colon < delimiter)) {
        return location;
    }
    if (location.starts_with(L"//")) { return current.scheme + L":" + location; }

    const std::wstring origin = current.origin();
    const std::size_t query  = current.path_query.find(L'?');
    const std::wstring current_path = current.path_query.substr(0, query);
    if (location.front() == L'?') { return origin + current_path + location; }
    if (location.front() == L'/') { return origin + normalize_relative_path(std::move(location)); }

    const std::size_t last_slash = current_path.find_last_of(L'/');
    const std::wstring directory = last_slash == std::wstring::npos
                                       ? L"/"
                                       : current_path.substr(0, last_slash + 1);
    return origin + normalize_relative_path(directory + location);
}

std::string wide_error_text(const wchar_t* operation, DWORD code) {
    std::wstring message(operation);
    message += L" (Windows error ";
    message += std::to_wstring(code);
    message += L")";
    std::string out;
    out.reserve(message.size());
    for (const wchar_t ch : message) { out.push_back(ch <= 0x7f ? static_cast<char>(ch) : '?'); }
    return out;
}

class WinsockStartup {
public:
    WinsockStartup() {
        WSADATA data{};
        const int error = WSAStartup(MAKEWORD(2, 2), &data);
        if (error != 0) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "failed to initialize Windows socket services (error " +
                            std::to_string(error) + ")");
        }
    }
    ~WinsockStartup() { (void)WSACleanup(); }
};

PADDRINFOEXW resolve_host(const std::wstring& host, const Policy& policy, Control& control) {
    static WinsockStartup winsock;
    (void)winsock;
    control.check_remote();

    ADDRINFOEXW hints{};
    hints.ai_family   = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    PADDRINFOEXW results = nullptr;
    OVERLAPPED overlapped{};
    KernelHandle event(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    if (event.get() == nullptr) {
        throw Error(ErrorKind::RemoteUnavailable,
                    win32_message("failed to create DNS event", GetLastError()));
    }
    overlapped.hEvent = event.get();
    HANDLE cancel_handle = nullptr;
    const Clock::time_point connect_deadline =
        Clock::now() + std::chrono::milliseconds(policy.connect_timeout_ms);
    const auto check_connect = [&] {
        control.check_remote();
        if (Clock::now() >= connect_deadline) {
            throw Error(ErrorKind::RemoteTimeout, "media URL DNS resolution timed out");
        }
    };
    check_connect();
    int error = GetAddrInfoExW(host.c_str(), nullptr, NS_DNS, nullptr, &hints, &results, nullptr,
                               &overlapped, nullptr, &cancel_handle);
    if (error == WSA_IO_PENDING) {
        for (;;) {
            const DWORD wait = WaitForSingleObject(event.get(), 50);
            if (wait == WAIT_OBJECT_0) { break; }
            if (wait != WAIT_TIMEOUT) {
                const DWORD wait_error = GetLastError();
                if (cancel_handle != nullptr) { (void)GetAddrInfoExCancel(&cancel_handle); }
                // OVERLAPPED and result storage remain live until DNS signals completion.
                for (;;) {
                    if (WaitForSingleObject(event.get(), INFINITE) == WAIT_OBJECT_0) { break; }
                    Sleep(1);
                }
                (void)GetAddrInfoExOverlappedResult(&overlapped);
                if (results != nullptr) { FreeAddrInfoExW(results); }
                throw Error(ErrorKind::RemoteUnavailable,
                            win32_message("failed while waiting for media DNS", wait_error));
            }
            try {
                check_connect();
            } catch (...) {
                if (cancel_handle != nullptr) { (void)GetAddrInfoExCancel(&cancel_handle); }
                // Cancellation is asynchronous; retain OVERLAPPED/results until completion.
                for (;;) {
                    if (WaitForSingleObject(event.get(), INFINITE) == WAIT_OBJECT_0) { break; }
                    Sleep(1);
                }
                (void)GetAddrInfoExOverlappedResult(&overlapped);
                if (results != nullptr) { FreeAddrInfoExW(results); }
                throw;
            }
        }
        error = GetAddrInfoExOverlappedResult(&overlapped);
    }
    std::unique_ptr<ADDRINFOEXW, decltype(&FreeAddrInfoExW)> owned_results(results,
                                                                            FreeAddrInfoExW);
    check_connect();
    if (error != NO_ERROR) {
        control.check_remote();
        throw Error(ErrorKind::RemoteUnavailable,
                    wide_error_text(L"failed to resolve media URL", static_cast<DWORD>(error)));
    }
    if (results == nullptr) {
        throw Error(ErrorKind::RemoteUnavailable, "media URL host resolved to no addresses");
    }
    (void)policy;
    return owned_results.release();
}

std::wstring resolved_address(const UrlParts& url, const Policy& policy, Control& control) {
    PADDRINFOEXW raw = resolve_host(url.host, policy, control);
    std::unique_ptr<ADDRINFOEXW, decltype(&FreeAddrInfoExW)> addresses(raw, FreeAddrInfoExW);
    std::wstring selected_v6;
    for (const ADDRINFOEXW* it = raw; it != nullptr; it = it->ai_next) {
        if (it->ai_addr == nullptr) { continue; }
        if (!policy.allow_private_network) {
            const bool blocked = it->ai_family == AF_INET
                                     ? private_ipv4(reinterpret_cast<const SOCKADDR_IN*>(it->ai_addr)
                                                        ->sin_addr.S_un.S_addr)
                                 : it->ai_family == AF_INET6
                                     ? private_ipv6(reinterpret_cast<const SOCKADDR_IN6*>(it->ai_addr)
                                                        ->sin6_addr)
                                     : true;
            if (blocked) { continue; }
        }
        std::array<wchar_t, INET6_ADDRSTRLEN> text{};
        const void* address =
            it->ai_family == AF_INET
                ? static_cast<const void*>(
                      &reinterpret_cast<const SOCKADDR_IN*>(it->ai_addr)->sin_addr)
                : it->ai_family == AF_INET6
                      ? static_cast<const void*>(
                            &reinterpret_cast<const SOCKADDR_IN6*>(it->ai_addr)->sin6_addr)
                      : nullptr;
        if (address == nullptr) { continue; }
        if (InetNtopW(it->ai_family, const_cast<void*>(address), text.data(),
                      static_cast<DWORD>(text.size())) == nullptr) {
            continue;
        }
        if (it->ai_family == AF_INET) { return text.data(); }
        if (selected_v6.empty()) { selected_v6 = text.data(); }
    }
    if (!selected_v6.empty()) { return selected_v6; }
    if (policy.allow_private_network) {
        throw Error(ErrorKind::RemoteUnavailable, "media URL host resolved to no usable address");
    }
    throw std::invalid_argument("media URL resolves only to disallowed network addresses");
}

std::optional<std::wstring> query_header(HINTERNET request, DWORD query) {
    DWORD bytes = 0;
    if (WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, nullptr, &bytes, nullptr)) {
        return std::wstring{};
    }
    const DWORD first_error = GetLastError();
    if (first_error == ERROR_WINHTTP_HEADER_NOT_FOUND) { return std::nullopt; }
    if (first_error != ERROR_INSUFFICIENT_BUFFER || bytes == 0) {
        throw_http_error("failed to inspect media response header", first_error);
    }
    std::vector<wchar_t> buffer((static_cast<std::size_t>(bytes) + sizeof(wchar_t) - 1) /
                                sizeof(wchar_t) + 1,
                                L'\0');
    DWORD capacity = static_cast<DWORD>(buffer.size() * sizeof(wchar_t));
    if (!WinHttpQueryHeaders(request, query, WINHTTP_HEADER_NAME_BY_INDEX, buffer.data(),
                             &capacity, nullptr)) {
        throw_http_error("failed to inspect media response header", GetLastError());
    }
    return std::wstring(buffer.data());
}

DWORD query_status(HINTERNET request) {
    DWORD status = 0;
    DWORD bytes  = sizeof(status);
    if (!WinHttpQueryHeaders(request, WINHTTP_QUERY_STATUS_CODE | WINHTTP_QUERY_FLAG_NUMBER,
                             WINHTTP_HEADER_NAME_BY_INDEX, &status, &bytes, nullptr)) {
        throw_http_error("failed to inspect media HTTP status", GetLastError());
    }
    return status;
}

std::optional<std::uint64_t> query_content_length(HINTERNET request) {
    const std::optional<std::wstring> value = query_header(request, WINHTTP_QUERY_CONTENT_LENGTH);
    if (!value || value->empty()) { return std::nullopt; }
    std::uint64_t result = 0;
    for (const wchar_t ch : *value) {
        if (ch < L'0' || ch > L'9') { return std::nullopt; }
        const std::uint64_t digit = static_cast<std::uint64_t>(ch - L'0');
        if (result > (std::numeric_limits<std::uint64_t>::max() - digit) / 10U) {
            return std::nullopt;
        }
        result = result * 10U + digit;
    }
    return result;
}

void async_send(AsyncRequest& request, Control& control) {
    request.begin(AsyncOperation::Send);
    const BOOL started = WinHttpSendRequest(request.get(), WINHTTP_NO_ADDITIONAL_HEADERS, 0,
                                             WINHTTP_NO_REQUEST_DATA, 0, 0, request.context());
    (void)finish_async_call(request, started, control, "failed to send media request");
}

void async_receive(AsyncRequest& request, Control& control) {
    request.begin(AsyncOperation::Receive);
    const BOOL started = WinHttpReceiveResponse(request.get(), nullptr);
    (void)finish_async_call(request, started, control, "failed to receive media response");
}

std::vector<std::uint8_t> async_read_body(AsyncRequest& request, const Policy& policy,
                                          Control& control) {
    std::vector<std::uint8_t> bytes;
    control.check_remote();
    if (const std::optional<std::uint64_t> declared = query_content_length(request.get());
        declared && *declared > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
    }
    for (;;) {
        control.check_remote();
        request.begin(AsyncOperation::Read);
        const BOOL started = WinHttpReadData(request.get(), request.read_buffer(),
                                              request.read_capacity(), nullptr);
        const AsyncCompletion completion = finish_async_call(
            request, started, control, "failed while reading media response");
        control.check_remote();
        const DWORD downloaded = completion.bytes;
        if (downloaded == 0) { break; }
        if (downloaded > policy.max_bytes - std::min(policy.max_bytes, bytes.size())) {
            throw Error(ErrorKind::BudgetExceeded, "media URL exceeds byte limit");
        }
        bytes.insert(bytes.end(), request.read_buffer(), request.read_buffer() + downloaded);
    }
    control.check_remote();
    if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
    return bytes;
}

std::vector<std::uint8_t> fetch_url(std::wstring url, const Policy& policy, Control& control) {
    if (!policy.allow_remote) { throw std::invalid_argument("remote media URLs are disabled"); }
    control.check_remote();

    InternetHandle session(WinHttpOpen(L"ninfer/vision", WINHTTP_ACCESS_TYPE_NO_PROXY,
                                       WINHTTP_NO_PROXY_NAME, WINHTTP_NO_PROXY_BYPASS,
                                       WINHTTP_FLAG_ASYNC));
    if (!session) { throw_http_error("failed to create WinHTTP session", GetLastError()); }
    const WINHTTP_STATUS_CALLBACK previous = WinHttpSetStatusCallback(
        session.get(), winhttp_status_callback,
        WINHTTP_CALLBACK_FLAG_ALL_COMPLETIONS | WINHTTP_CALLBACK_FLAG_HANDLES, 0);
    if (previous == WINHTTP_INVALID_STATUS_CALLBACK) {
        throw_http_error("failed to configure WinHTTP callbacks", GetLastError());
    }
    const int resolve_timeout = control.bounded_timeout_ms(policy.connect_timeout_ms, true);
    const int connect_timeout = control.bounded_timeout_ms(policy.connect_timeout_ms, true);
    const int transfer_timeout = control.bounded_timeout_ms(policy.timeout_ms, true);
    if (!WinHttpSetTimeouts(session.get(), resolve_timeout, connect_timeout, transfer_timeout,
                            transfer_timeout)) {
        throw_http_error("failed to configure WinHTTP timeouts", GetLastError());
    }

    for (int redirect = 0; redirect <= policy.max_redirects; ++redirect) {
        control.check_remote();
        const UrlParts parts = parse_url(url);
        const std::wstring pin = resolved_address(parts, policy, control);
        control.check_remote();

        InternetHandle connection(WinHttpConnect(session.get(), parts.host.c_str(), parts.port, 0));
        if (!connection) { throw_http_error("failed to open media host", GetLastError()); }
        AsyncRequest request_state;
        const DWORD flags = parts.secure() ? WINHTTP_FLAG_SECURE : 0;
        InternetHandle unopened_request(WinHttpOpenRequest(
            connection.get(), L"GET", parts.path_query.c_str(), nullptr, WINHTTP_NO_REFERER,
            WINHTTP_DEFAULT_ACCEPT_TYPES, flags));
        if (!unopened_request) { throw_http_error("failed to open media request", GetLastError()); }
        const DWORD_PTR context = request_state.context();
        set_winhttp_option(unopened_request.get(), WINHTTP_OPTION_CONTEXT_VALUE, &context,
                           sizeof(context), "failed to configure WinHTTP request context");
        request_state.attach(unopened_request.release());
        HINTERNET request = request_state.get();

        const DWORD disable_features = WINHTTP_DISABLE_AUTHENTICATION | WINHTTP_DISABLE_COOKIES |
                                       WINHTTP_DISABLE_REDIRECTS;
        set_winhttp_option(request, WINHTTP_OPTION_DISABLE_FEATURE, &disable_features,
                           sizeof(disable_features), "failed to disable WinHTTP features");
        const DWORD redirect_policy = WINHTTP_OPTION_REDIRECT_POLICY_NEVER;
        set_winhttp_option(request, WINHTTP_OPTION_REDIRECT_POLICY, &redirect_policy,
                           sizeof(redirect_policy), "failed to disable automatic redirects");
        const DWORD reject_user_password = TRUE;
        set_winhttp_option(request, WINHTTP_OPTION_REJECT_USERPWD_IN_URL,
                           &reject_user_password, sizeof(reject_user_password),
                           "failed to reject URL credentials");
        const DWORD pin_bytes = static_cast<DWORD>((pin.size() + 1) * sizeof(wchar_t));
        if (!WinHttpSetOption(request, WINHTTP_OPTION_RESOLUTION_HOSTNAME,
                              const_cast<wchar_t*>(pin.c_str()), pin_bytes)) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "WinHTTP DNS pinning is unavailable; use a base64 data URI instead");
        }
        static constexpr wchar_t kAcceptIdentity[] = L"Accept-Encoding: identity\r\n";
        if (!WinHttpAddRequestHeaders(request, kAcceptIdentity, static_cast<DWORD>(-1),
                                      WINHTTP_ADDREQ_FLAG_ADD | WINHTTP_ADDREQ_FLAG_REPLACE)) {
            throw_http_error("failed to configure media request headers", GetLastError());
        }

        async_send(request_state, control);
        async_receive(request_state, control);
        const DWORD status = query_status(request);
        if (status >= 200 && status < 300) {
            std::vector<std::uint8_t> bytes = async_read_body(request_state, policy, control);
            control.check_remote();
            request_state.close_and_wait();
            return bytes;
        }
        if (status < 300 || status >= 400 || redirect == policy.max_redirects) {
            throw Error(ErrorKind::RemoteUnavailable,
                        "media URL returned HTTP " + std::to_string(status));
        }
        const std::optional<std::wstring> location =
            query_header(request, WINHTTP_QUERY_LOCATION);
        if (!location || location->empty()) {
            throw Error(ErrorKind::RemoteUnavailable, "media redirect has no location");
        }
        const std::wstring next = resolve_redirect(parts, *location);
        request_state.close_and_wait();
        connection.reset();
        url = next;
    }
    throw Error(ErrorKind::RemoteUnavailable, "too many media URL redirects");
}

std::wstring normalize_final_path(std::wstring path) {
    constexpr std::wstring_view kUncPrefix = L"\\\\?\\UNC\\";
    constexpr std::wstring_view kDevicePrefix = L"\\\\?\\";
    if (path.starts_with(kUncPrefix)) {
        path = L"\\\\" + path.substr(kUncPrefix.size());
    } else if (path.starts_with(kDevicePrefix)) {
        path.erase(0, kDevicePrefix.size());
    }
    return path;
}

bool path_within_root(const std::filesystem::path& path,
                      const std::filesystem::path& root) {
    std::wstring file = normalize_final_path(path.native());
    std::wstring base = normalize_final_path(root.native());
    while (base.size() > 1 && (base.back() == L'\\' || base.back() == L'/')) {
        base.pop_back();
    }
    if (base.empty()) { return false; }
    base.push_back(L'\\');
    if (file.size() < base.size()) { return false; }
    return CompareStringOrdinal(file.data(), static_cast<int>(base.size()), base.data(),
                                static_cast<int>(base.size()), TRUE) == CSTR_EQUAL;
}

std::wstring final_path_from_handle(HANDLE file) {
    const DWORD required = GetFinalPathNameByHandleW(file, nullptr, 0,
                                                       FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (required == 0) {
        throw std::invalid_argument(win32_message("failed to resolve media file path", GetLastError()));
    }
    std::wstring path(static_cast<std::size_t>(required) + 1, L'\0');
    const DWORD written = GetFinalPathNameByHandleW(file, path.data(),
                                                     static_cast<DWORD>(path.size()),
                                                     FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
    if (written == 0 || written >= path.size()) {
        throw std::invalid_argument(win32_message("failed to resolve media file path", GetLastError()));
    }
    path.resize(written);
    return normalize_final_path(std::move(path));
}

std::vector<std::uint8_t> read_path(const Source& source, const Policy& policy, Control& control) {
    control.check();
    const std::filesystem::path source_path(utf8_to_wide(source.value, "media path"));
    std::error_code error;
    const std::filesystem::path path = std::filesystem::weakly_canonical(source_path, error);
    if (error) { throw std::invalid_argument("media path cannot be resolved"); }

    std::filesystem::path root;
    if (!policy.media_root.empty()) {
        root = std::filesystem::weakly_canonical(policy.media_root, error);
        if (error || !std::filesystem::is_directory(root, error) || error) {
            throw std::invalid_argument("configured media root is not a directory");
        }
        if (!path_within_root(path, root)) {
            throw std::invalid_argument("media path is outside configured media root");
        }
    }

    KernelHandle file(CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                  OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_SEQUENTIAL_SCAN,
                                  nullptr));
    if (file.get() == INVALID_HANDLE_VALUE) {
        throw std::invalid_argument("failed to open media path");
    }
    if (!root.empty() && !path_within_root(std::filesystem::path(final_path_from_handle(file.get())),
                                           root)) {
        throw std::invalid_argument("media path resolves outside configured media root");
    }
    FILE_STANDARD_INFO info{};
    if (!GetFileInformationByHandleEx(file.get(), FileStandardInfo, &info, sizeof(info)) ||
        info.Directory || info.EndOfFile.QuadPart < 0) {
        throw std::invalid_argument("media path is not a regular file");
    }
    const auto file_size = static_cast<std::uint64_t>(info.EndOfFile.QuadPart);
    if (file_size > policy.max_bytes) {
        throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
    }

    std::vector<std::uint8_t> bytes;
    bytes.reserve(static_cast<std::size_t>(file_size));
    std::array<std::uint8_t, 64U * 1024U> chunk{};
    for (;;) {
        control.check();
        DWORD read = 0;
        if (!ReadFile(file.get(), chunk.data(), static_cast<DWORD>(chunk.size()), &read, nullptr)) {
            throw std::invalid_argument(win32_message("failed to read media path", GetLastError()));
        }
        if (read == 0) { break; }
        if (read > policy.max_bytes - std::min(policy.max_bytes, bytes.size())) {
            throw Error(ErrorKind::BudgetExceeded, "media file exceeds byte limit");
        }
        bytes.insert(bytes.end(), chunk.data(), chunk.data() + read);
    }
    control.check();
    if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
    return bytes;
}

std::vector<std::uint8_t> decode_base64(std::string_view text, const Policy& policy,
                                        Control& control) {
    static constexpr std::array<std::int8_t, 256> table = [] {
        std::array<std::int8_t, 256> out{};
        out.fill(-1);
        for (int i = 0; i < 26; ++i) {
            out[static_cast<std::size_t>('A' + i)] = static_cast<std::int8_t>(i);
            out[static_cast<std::size_t>('a' + i)] = static_cast<std::int8_t>(26 + i);
        }
        for (int i = 0; i < 10; ++i) { out[static_cast<std::size_t>('0' + i)] = i + 52; }
        out[static_cast<std::size_t>('+')] = 62;
        out[static_cast<std::size_t>('/')] = 63;
        return out;
    }();

    std::vector<std::uint8_t> out;
    std::array<std::uint8_t, 4> quartet{};
    std::size_t count = 0;
    bool padded      = false;
    std::size_t input_offset = 0;
    for (const unsigned char c : text) {
        if ((input_offset++ & 0xffffU) == 0) { control.check(); }
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') { continue; }
        if (c == '=') {
            if (count < 2 || count > 3) { throw std::invalid_argument("malformed base64 media data"); }
            padded = true;
            quartet[count++] = 0xffU;
        } else {
            if (padded || table[c] < 0) { throw std::invalid_argument("malformed base64 media data"); }
            quartet[count++] = static_cast<std::uint8_t>(table[c]);
        }
        if (count != 4) { continue; }
        const std::size_t padding = quartet[3] == 0xffU ? (quartet[2] == 0xffU ? 2U : 1U) : 0U;
        if ((quartet[2] == 0xffU && quartet[3] != 0xffU) ||
            (padding == 2U && (quartet[1] & 0x0fU) != 0) ||
            (padding == 1U && (quartet[2] & 0x03U) != 0)) {
            throw std::invalid_argument("malformed base64 media padding");
        }
        const std::uint32_t bits = (static_cast<std::uint32_t>(quartet[0]) << 18U) |
                                   (static_cast<std::uint32_t>(quartet[1]) << 12U) |
                                   (static_cast<std::uint32_t>(quartet[2] & 0x3fU) << 6U) |
                                   static_cast<std::uint32_t>(quartet[3] & 0x3fU);
        out.push_back(static_cast<std::uint8_t>(bits >> 16U));
        if (padding < 2) { out.push_back(static_cast<std::uint8_t>(bits >> 8U)); }
        if (padding == 0) { out.push_back(static_cast<std::uint8_t>(bits)); }
        if (out.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
        }
        if (padding != 0) {
            padded = true;
            // Padding is valid only at the end; reject any subsequent non-whitespace byte.
            // The loop-level padded check enforces this.
        }
        count = 0;
    }
    if (count != 0) { throw std::invalid_argument("malformed base64 media padding"); }
    control.check();
    return out;
}

std::vector<std::uint8_t> decode_data_uri(const Source& source, const Policy& policy,
                                          Control& control) {
    const std::size_t comma = source.value.find(',');
    if (!source.value.starts_with("data:") || comma == std::string::npos ||
        source.value.substr(0, comma).find(";base64") == std::string::npos) {
        throw std::invalid_argument("media data source must be a base64 data URI");
    }
    const std::size_t encoded = source.value.size() - comma - 1;
    constexpr std::size_t kWhitespaceAllowance = 4096;
    const std::size_t max_size = std::numeric_limits<std::size_t>::max();
    const std::size_t groups = policy.max_bytes / 3;
    const std::size_t remainder = policy.max_bytes % 3;
    const std::size_t max_encoded = groups > (max_size - (remainder == 0 ? 0U : 4U)) / 4U
                                        ? max_size
                                        : groups * 4U + (remainder == 0 ? 0U : 4U);
    const std::size_t allowed_encoded =
        max_encoded > max_size - kWhitespaceAllowance ? max_size
                                                       : max_encoded + kWhitespaceAllowance;
    if (encoded > allowed_encoded) {
        throw Error(ErrorKind::BudgetExceeded, "media data exceeds byte limit");
    }
    std::vector<std::uint8_t> bytes = decode_base64(
        std::string_view(source.value).substr(comma + 1), policy, control);
    if (bytes.empty()) { throw std::invalid_argument("media source contains no data"); }
    return bytes;
}

} // namespace

std::vector<std::uint8_t> acquire_bytes(const Source& source, const Policy& policy) {
    if (policy.max_bytes == 0 || policy.connect_timeout_ms <= 0 || policy.timeout_ms <= 0 ||
        policy.max_redirects < 0) {
        throw std::invalid_argument("media acquisition policy is invalid");
    }
    Control control(policy);
    control.check();
    switch (source.kind) {
    case SourceKind::Bytes: {
        if (source.bytes.empty()) { throw std::invalid_argument("media source is empty"); }
        if (source.bytes.size() > policy.max_bytes) {
            throw Error(ErrorKind::BudgetExceeded, "media bytes exceed byte limit");
        }
        std::vector<std::uint8_t> out;
        out.reserve(source.bytes.size());
        constexpr std::size_t kChunk = 64U * 1024U;
        for (std::size_t offset = 0; offset < source.bytes.size(); offset += kChunk) {
            control.check();
            const std::size_t size = std::min(kChunk, source.bytes.size() - offset);
            out.insert(out.end(), source.bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                       source.bytes.begin() + static_cast<std::ptrdiff_t>(offset + size));
        }
        control.check();
        return out;
    }
    case SourceKind::Data:
        if (source.value.empty()) { throw std::invalid_argument("media source is empty"); }
        return decode_data_uri(source, policy, control);
    case SourceKind::Path:
        if (source.value.empty()) { throw std::invalid_argument("media source is empty"); }
        return read_path(source, policy, control);
    case SourceKind::Url:
        if (source.value.empty()) { throw std::invalid_argument("media source is empty"); }
        return fetch_url(utf8_to_wide(source.value, "media URL"), policy, control);
    }
    throw std::invalid_argument("unsupported media source kind");
}

} // namespace ninfer::product::media_acquire
