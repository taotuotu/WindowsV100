#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/operational_log.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    ifndef WIN32_LEAN_AND_MEAN
#        define WIN32_LEAN_AND_MEAN
#    endif
#    include <windows.h>
#    include <cuda.h>
#endif

namespace {

constexpr char kDefaultArtifact[] = "models/qwen3_8_27b_nvfp4_v2.ninfer";

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load(std::memory_order_acquire);
    if (server != nullptr) { server->stop(); }
}

#if defined(_WIN32)
BOOL WINAPI handle_console_event(DWORD event) {
    switch (event) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        handle_signal(SIGINT);
        return TRUE;
    default:
        return FALSE;
    }
}

std::string wide_to_utf8(const wchar_t* value) {
    const int required = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, nullptr,
                                                0, nullptr, nullptr);
    if (required <= 0) { throw std::runtime_error("failed to convert argv from UTF-16 to UTF-8"); }
    std::string result(static_cast<std::size_t>(required), '\0');
    if (::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1, result.data(), required,
                              nullptr, nullptr) <= 0) {
        throw std::runtime_error("failed to convert argv from UTF-16 to UTF-8");
    }
    result.resize(static_cast<std::size_t>(required - 1));
    return result;
}

[[noreturn]] void throw_no_registered_gpu() {
    throw std::runtime_error(
        "registered 27B profile requires CUDA SM70/Volta32GB; pass --device N for explicit selection");
}

std::string cuda_driver_error_text(const std::string& operation, CUresult status) {
    const char* description = nullptr;
    if (cuGetErrorString(status, &description) != CUDA_SUCCESS || description == nullptr) {
        const char* name = nullptr;
        if (cuGetErrorName(status, &name) == CUDA_SUCCESS && name != nullptr) {
            description = name;
        } else {
            description = "unknown CUDA driver error";
        }
    }
    return std::string(operation) + ": " + description;
}

void resolve_default_cuda_device(ninfer::serve::ServeOptions& options) {
    if (options.device >= 0) { return; }

    const CUresult init_status = cuInit(0);
    if (init_status == CUDA_ERROR_NO_DEVICE) { throw_no_registered_gpu(); }
    if (init_status != CUDA_SUCCESS) {
        throw std::runtime_error(cuda_driver_error_text("CUDA driver initialization failed",
                                                       init_status));
    }

    int device_count = 0;
    const CUresult count_status = cuDeviceGetCount(&device_count);
    if (count_status == CUDA_ERROR_NO_DEVICE ||
        (count_status == CUDA_SUCCESS && device_count == 0)) {
        throw_no_registered_gpu();
    }
    if (count_status != CUDA_SUCCESS) {
        throw std::runtime_error(cuda_driver_error_text("CUDA device enumeration failed",
                                                       count_status));
    }

    constexpr std::uint64_t kMinimumVoltaMemoryBytes = 30ULL * 1024ULL * 1024ULL * 1024ULL;
    for (int ordinal = 0; ordinal < device_count; ++ordinal) {
        CUdevice device{};
        CUresult status = cuDeviceGet(&device, ordinal);
        if (status != CUDA_SUCCESS) {
            throw std::runtime_error(cuda_driver_error_text(
                std::string("CUDA device lookup failed for ordinal ") + std::to_string(ordinal),
                status));
        }

        int major = 0;
        int minor = 0;
        status = cuDeviceComputeCapability(&major, &minor, device);
        if (status != CUDA_SUCCESS) {
            throw std::runtime_error(cuda_driver_error_text(
                std::string("CUDA compute capability query failed for ordinal ") +
                    std::to_string(ordinal),
                status));
        }

        std::size_t total_memory = 0;
        status = cuDeviceTotalMem(&total_memory, device);
        if (status != CUDA_SUCCESS) {
            throw std::runtime_error(cuda_driver_error_text(
                std::string("CUDA memory query failed for ordinal ") + std::to_string(ordinal),
                status));
        }

        if (major == 7 && minor == 0 &&
            static_cast<std::uint64_t>(total_memory) >= kMinimumVoltaMemoryBytes) {
            options.device = ordinal;
            return;
        }
    }

    throw_no_registered_gpu();
}
#endif

std::vector<std::string> effective_arguments(const std::vector<std::string>& argv) {
    const std::string program = argv.empty() ? "ninfer-windows-serve.exe" : argv.front();
    for (std::size_t i = 1; i < argv.size(); ++i) {
        if (argv[i] == "--help" || argv[i] == "-h") { return {program, "--help"}; }
    }

    std::string model = kDefaultArtifact;
    std::size_t first_user_option = argv.empty() ? 0 : 1;
    if (first_user_option < argv.size() && !argv[first_user_option].empty() &&
        argv[first_user_option].front() != '-') {
        model = argv[first_user_option++];
    }

    std::vector<std::string> effective;
    effective.reserve(24 + argv.size());
    effective.push_back(program);
    effective.push_back(std::move(model));
    const std::vector<std::string> defaults{
        "--host", "127.0.0.1", "--port", "8110", "--model-id", "qwen3.8-27b",
        "--max-context", "8192", "--max-concurrency", "1",
        "--prefill-chunk", "512", "--kv-dtype", "bf16", "--spec", "mtp",
        "--draft-tokens", "3", "--lm-head-draft", "--no-thinking", "--temperature", "0",
        "--seed", "123",
    };
    effective.insert(effective.end(), defaults.begin(), defaults.end());
    effective.insert(effective.end(), argv.begin() + static_cast<std::ptrdiff_t>(first_user_option),
                     argv.end());
    return effective;
}

int run(const std::vector<std::string>& raw_arguments) {
    std::vector<std::string> arguments;
    std::vector<char*> mutable_arguments;
    try {
        arguments = effective_arguments(raw_arguments);
        mutable_arguments.reserve(arguments.size());
        for (std::string& argument : arguments) { mutable_arguments.push_back(argument.data()); }
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-windows-serve: " << exception.what() << '\n';
        return 1;
    }

    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(static_cast<int>(mutable_arguments.size()),
                                                     mutable_arguments.data());
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-windows-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(arguments.front().c_str());
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-windows-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(arguments.front().c_str());
        return 0;
    }
#if defined(_WIN32)
    try {
        resolve_default_cuda_device(options);
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-windows-serve: " << exception.what() << '\n';
        return 1;
    }
#endif

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-windows-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        ninfer::serve::GenerationService service(options, startup_log.observer());
        startup_log.engine_ready(service.load_summary());
        operational_log.engine_capacity(service);

        using Clock = std::chrono::steady_clock;
        const Clock::time_point warmup_started = Clock::now();
        operational_log.warmup_started();
        try {
            service.warmup();
        } catch (const std::exception& exception) {
            const double seconds =
                std::chrono::duration<double>(Clock::now() - warmup_started).count();
            operational_log.warmup_failure(seconds, exception.what());
            return 1;
        }
        operational_log.warmup_complete(
            std::chrono::duration<double>(Clock::now() - warmup_started).count());
        server.attach(service);

        g_server.store(&server, std::memory_order_release);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
#if defined(_WIN32)
        if (!::SetConsoleCtrlHandler(handle_console_event, TRUE)) {
            logger->warn("could not register Windows console shutdown handler");
        }
#endif

        serving = true;
        operational_log.server_ready(options.host, options.port, server.public_model_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr, std::memory_order_release);
#if defined(_WIN32)
        (void)::SetConsoleCtrlHandler(handle_console_event, FALSE);
#endif
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr, std::memory_order_release);
#if defined(_WIN32)
        (void)::SetConsoleCtrlHandler(handle_console_event, FALSE);
#endif
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
    ::SetConsoleCP(CP_UTF8);
    ::SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    try {
        for (int i = 0; i < argc; ++i) { arguments.push_back(wide_to_utf8(argv[i])); }
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-windows-serve: " << exception.what() << '\n';
        return 1;
    }
    return run(arguments);
}
#else
int main(int argc, char* argv[]) {
    std::vector<std::string> arguments;
    arguments.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) { arguments.emplace_back(argv[i]); }
    return run(arguments);
}
#endif
