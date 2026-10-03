// Modified for the Windows/V100 port by taotuotu, 2026; see NOTICE.
#include "serve/serve_options.h"
#include "product/speculative_options.h"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace ninfer::serve {
namespace {

int parse_nonnegative_int(const char* text, const char* label) {
    char* end        = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' || value < 0 ||
        value > static_cast<long>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<int>(value);
}

float parse_float_in(const char* text, const char* label, float lo, float hi) {
    char* end          = nullptr;
    const double value = std::strtod(text, &end);
    if (end == text || *end != '\0' || !(value >= lo) || !(value <= hi)) {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<float>(value);
}

std::uint64_t parse_u64(const char* text, const char* label) {
    if (text == nullptr || *text == '\0' || *text == '-') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " +
                                    (text == nullptr ? "" : text));
    }
    errno                          = 0;
    char* end                      = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0') {
        throw std::invalid_argument(std::string("invalid ") + label + ": " + text);
    }
    return static_cast<std::uint64_t>(value);
}

KvCacheStorage parse_kv_dtype(const char* text) {
    const std::string value(text);
    if (value == "bf16") { return KvCacheStorage::BFloat16; }
    if (value == "int8") { return KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return KvCacheStorage::Fp8E4M3Row256; }
    if (value == "nvfp4") { return KvCacheStorage::Nvfp4Group16; }
    if (value == "k8v4") { return KvCacheStorage::Fp8KeyNvfp4Value; }
    throw std::invalid_argument("invalid kv-dtype: " + value);
}

KvCapacityPolicy parse_kv_capacity(const char* text) {
    if (std::string_view(text) == "auto") { return KvCapacityPolicy::automatic(); }
    const int value = parse_nonnegative_int(text, "kv-capacity");
    if (value == 0) { throw std::invalid_argument("--kv-capacity must be positive"); }
    return KvCapacityPolicy::explicit_capacity(static_cast<std::uint32_t>(value));
}

} // namespace

std::string serve_usage_text(const char* argv0) {
    return std::string("usage: ") + argv0 +
#if defined(NINFER_WINDOWS_SERVE)
           " [model.ninfer] [--model PATH] "
#else
           " <model.ninfer> "
#endif
           "[--host H] [--port N] [--api-key KEY] "
           "[--model-id ID] [--max-context N] [--kv-capacity N|auto] [--max-concurrency N] "
           "[--max-pending-requests N] [--pending-timeout-ms N] "
           "[--prefill-chunk N] [--log-stats-interval-ms N] "
#if defined(NINFER_WINDOWS_SERVE)
           "[--device auto|N] "
           "[--context-disk-cache DIR | --no-context-disk-cache] "
           "[--context-disk-cache-mib N] [--context-disk-cache-sessions N] "
#else
           "[--device N] "
#endif
           "[--context-cost-presets FILE] "
           "[--max-request-mib N] [--media-cache-mib N] [--media-live-mib N] "
           "[--media-preprocess-threads N] "
           "[--device-state-slots N] [--host-state-slots N] [--host-kv-mib N] "
           "[--max-private-continuations N] [--max-shared-prefixes N] "
           "[--max-long-anchors-per-continuation N] "
           "[--request-log-jsonl FILE] "
           "[--response-store-max-records N] [--response-store-max-mib N] "
           "[--kv-dtype bf16|int8|fp8|nvfp4|k8v4] [--spec mtp|dflash|dflash2 --draft-tokens N] "
           "[--default-max-tokens N] [--default-thinking-budget N] "
           "[--vision] [--no-cuda-graph] [--no-prefix-reuse] "
           "[--lm-head-draft] [--no-thinking] [--preserve-thinking] [--cors] "
           "[--temperature F] [--top-p F] [--top-k N] [--min-p F] [--presence-penalty F] "
           "[--frequency-penalty F] [--seed N] [--greedy]\n"
           "       [--log-level trace|debug|info|warning|error|critical|off]\n"
           "       serves OpenAI Responses/Chat Completions and Anthropic Messages endpoints\n"
           "       --default-max-tokens defaults to " +
           std::to_string(kDefaultMaxTokens) +
           " when omitted\n"
           "       --max-request-mib defaults to 384 and is enforced before JSON parsing\n"
           "       --media-cache-mib defaults to 1024; 0 disables retained media reuse\n"
           "       --media-live-mib defaults to 2048 and bounds all live BF16 patch payloads\n"
           "       --media-preprocess-threads defaults to 0 (auto, at most 16 workers)\n"
           "       --request-log-jsonl appends full-precision server/request records\n"
           "       --model-id overrides the artifact identity.model_id reported by the server\n"
           "       Responses state is process-local and bounded to 1024 records / 256 MiB by "
           "default\n"
           "       --log-stats-interval-ms defaults to 5000; 0 disables periodic throughput logs\n"
#if defined(NINFER_WINDOWS_SERVE)
           "       --vision enables PNG/JPEG/BMP image input and fixed Vision GPU allocations; "
           "video input is unavailable on Windows\n"
           "       disk context snapshots default to .local/context-cache, 32768 MiB, 8 sessions; "
           "--no-context-disk-cache disables persistence\n"
#else
           "       --vision enables media and loads the fixed Vision GPU allocations\n"
#endif
           "       --kv-capacity auto leaves " +
           std::to_string(kDefaultKvCapacityHeadroomBytes / (1024ULL * 1024ULL)) +
           " MiB of sizing headroom\n"
#if defined(NINFER_WINDOWS_SERVE)
           "       compatible-prefix reuse is enabled by default; hits require a matching prompt prefix "
           "and saved state\n"
           "       exclusive-session cache defaults: extra Device StateImages=1, Host StateImages=8, Host KV=0 MiB, "
           "private continuations=2x concurrency, shared prefixes=max(concurrency,4), anchors=1\n"
           "       Windows serves one generation request at a time; a concurrent request receives HTTP 429. "
           "Switching session clears RAM checkpoints and restores its saved disk snapshot when enabled.\n"
           "       --no-prefix-reuse disables compatible-prefix caching\n"
#else
           "       --no-prefix-reuse disables compatible-prefix caching (enabled by default)\n"
           "       context cache defaults: device-state=max-concurrency, private=2x concurrency, "
           "shared=max(max-concurrency,4), anchors=2; Host state=8 slots, Host KV=8192 MiB\n"
#endif
           "       --device-state-slots is extra checkpoint capacity beyond active lanes; "
           "--host-kv-mib uses MiB\n"
           "       --default-thinking-budget caps model-origin thinking for enabled requests; "
           "control tokens count toward the request output limit\n"
           "       --preserve-thinking retains closed-turn assistant reasoning in later prompts\n"
           "       sampler defaults come from the loaded model and resolved thinking mode; "
           "server flags and request fields override individual values.\n"
           "       --greedy forces temperature 0 (exact argmax).\n"
#if defined(NINFER_WINDOWS_SERVE)
           "       Windows defaults: model "
           "models/qwen3_8_27b_nvfp4_v2.ninfer, "
           "host 127.0.0.1, port 8110, alias qwen3.8-27b, CUDA device auto (SM70, >=30 GiB), "
           "context/KV 143600, "
           "concurrency 1, prefill chunk 2048, bf16 KV, MTP draft width 6 with optimized head, "
           "native thinking on, prefix reuse on, temperature 0.8, seed 123. Request sampling fields "
           "may override process defaults.\n"
           "       --device auto selects the first CUDA SM70 device with at least 30 GiB total memory; "
           "pass --device N to select a device explicitly.\n"
           "       supply the artifact positionally or with --model/-m.\n"
#endif
        ;
}

ServeOptions parse_serve_options(int argc, char** argv) {
    ServeOptions options;
#if defined(NINFER_WINDOWS_SERVE)
    options.max_context                                 = 143600;
    options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    options.prefill_chunk                               = 2048;
    options.speculative.draft_tokens                     = 6;
    options.device                                      = -1;
    options.context_cache.device_state_slots                = 1;
    options.context_cache.host_state_slots                  = 8;
    options.context_cache.host_kv_capacity_bytes            = 0;
    options.context_cache.max_long_anchors_per_continuation = 1;
#endif
    options.startup_argv.reserve(static_cast<std::size_t>(argc));
    bool redact_next = false;
    for (int i = 0; i < argc; ++i) {
        if (redact_next) {
            options.startup_argv.emplace_back("<redacted>");
            redact_next = false;
            continue;
        }
        options.startup_argv.emplace_back(argv[i] == nullptr ? "" : argv[i]);
        redact_next = options.startup_argv.back() == "--api-key";
    }
    bool default_max_tokens_explicit = false;
    bool kv_capacity_explicit        = false;
    bool context_capacity_explicit   = false;
    if (argc >= 2 && (std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h")) {
        options.help_requested = true;
        return options;
    }
    if (argc < 2) { throw std::invalid_argument("artifact path is required"); }
    options.artifact_path = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg    = argv[i];
        const auto require_value = [&](const char* flag) -> const char* {
            if (++i >= argc) { throw std::invalid_argument(std::string(flag) + " needs a value"); }
            return argv[i];
        };
#if defined(NINFER_WINDOWS_SERVE)
        if (arg == "--model" || arg == "-m") {
            options.artifact_path = require_value(arg.c_str());
            if (options.artifact_path.empty()) {
                throw std::invalid_argument(arg + " model path must not be empty");
            }
        } else
#endif
        if (arg == "--host") {
            options.host = require_value("--host");
        } else if (arg == "--port") {
            options.port = parse_nonnegative_int(require_value("--port"), "port");
        } else if (arg == "--api-key") {
            options.api_key = require_value("--api-key");
        } else if (arg == "--model-id") {
            options.model_id_override = require_value("--model-id");
            if (options.model_id_override->empty()) {
                throw std::invalid_argument("--model-id must not be empty");
            }
        } else if (arg == "--max-context") {
            options.max_context = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-context"), "max-context"));
        } else if (arg == "--kv-capacity") {
            options.kv_capacity  = parse_kv_capacity(require_value("--kv-capacity"));
            kv_capacity_explicit = true;
        } else if (arg == "--max-concurrency") {
            options.max_concurrency = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-concurrency"), "max-concurrency"));
        } else if (arg == "--max-pending-requests") {
            options.max_pending_requests = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--max-pending-requests"), "max-pending-requests"));
        } else if (arg == "--pending-timeout-ms") {
            options.pending_timeout_ms = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--pending-timeout-ms"), "pending-timeout-ms"));
        } else if (arg == "--prefill-chunk") {
            options.prefill_chunk = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--prefill-chunk"), "prefill-chunk"));
        } else if (arg == "--context-cost-presets") {
            options.context_cost_presets = require_value("--context-cost-presets");
            if (options.context_cost_presets.empty()) {
                throw std::invalid_argument("--context-cost-presets must not be empty");
            }
        } else if (arg == "--log-stats-interval-ms") {
            options.log_stats_interval_ms = static_cast<std::uint32_t>(parse_nonnegative_int(
                require_value("--log-stats-interval-ms"), "log-stats-interval-ms"));
#if defined(NINFER_WINDOWS_SERVE)
        } else if (arg == "--context-disk-cache") {
            const std::string directory = require_value("--context-disk-cache");
            if (directory.empty()) {
                throw std::invalid_argument("--context-disk-cache must not be empty");
            }
            options.context_disk_cache_directory = std::filesystem::u8path(directory);
        } else if (arg == "--no-context-disk-cache") {
            options.context_disk_cache_directory.clear();
        } else if (arg == "--context-disk-cache-mib") {
            const auto mib = parse_u64(require_value("--context-disk-cache-mib"),
                                       "context-disk-cache-mib");
            if (mib == 0 || mib > std::numeric_limits<std::uint64_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--context-disk-cache-mib is out of range");
            }
            options.context_disk_cache_bytes = mib << 20;
        } else if (arg == "--context-disk-cache-sessions") {
            const auto count = parse_u64(require_value("--context-disk-cache-sessions"),
                                         "context-disk-cache-sessions");
            if (count == 0 || count > 128) {
                throw std::invalid_argument("--context-disk-cache-sessions must be in [1,128]");
            }
            options.context_disk_cache_sessions = static_cast<std::uint32_t>(count);
#endif
        } else if (arg == "--max-request-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--max-request-mib"), "max-request-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--max-request-mib is out of range");
            }
            options.max_request_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-cache-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-cache-mib"), "media-cache-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-cache-mib is out of range");
            }
            options.media_cache_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-live-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--media-live-mib"), "media-live-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--media-live-mib is out of range");
            }
            options.media_live_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--media-preprocess-threads") {
            const int threads = parse_nonnegative_int(require_value("--media-preprocess-threads"),
                                                      "media-preprocess-threads");
            if (threads > 64) {
                throw std::invalid_argument("--media-preprocess-threads must be in [0,64]");
            }
            options.media_preprocess_threads = static_cast<std::uint32_t>(threads);
        } else if (arg == "--device-state-slots") {
            options.context_cache.device_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--device-state-slots"), "device-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-state-slots") {
            options.context_cache.host_state_slots = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--host-state-slots"), "host-state-slots"));
            context_capacity_explicit = true;
        } else if (arg == "--host-kv-mib") {
            const std::uint64_t mib = parse_u64(require_value("--host-kv-mib"), "host-kv-mib");
            if (mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--host-kv-mib is out of range");
            }
            options.context_cache.host_kv_capacity_bytes = static_cast<std::size_t>(mib << 20);
            context_capacity_explicit                    = true;
        } else if (arg == "--max-private-continuations") {
            options.context_cache.max_private_continuations =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-private-continuations"), "max-private-continuations"));
            context_capacity_explicit = true;
        } else if (arg == "--max-shared-prefixes") {
            options.context_cache.max_shared_prefixes =
                static_cast<std::uint32_t>(parse_nonnegative_int(
                    require_value("--max-shared-prefixes"), "max-shared-prefixes"));
            context_capacity_explicit = true;
        } else if (arg == "--max-long-anchors-per-continuation") {
            options.context_cache.max_long_anchors_per_continuation = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--max-long-anchors-per-continuation"),
                                      "max-long-anchors-per-continuation"));
            context_capacity_explicit = true;
        } else if (arg == "--request-log-jsonl") {
            options.request_log_jsonl = require_value("--request-log-jsonl");
            if (options.request_log_jsonl.empty()) {
                throw std::invalid_argument("--request-log-jsonl must not be empty");
            }
        } else if (arg == "--response-store-max-records") {
            const int records = parse_nonnegative_int(require_value("--response-store-max-records"),
                                                      "response-store-max-records");
            if (records == 0) {
                throw std::invalid_argument("--response-store-max-records must be positive");
            }
            options.response_store_max_records = static_cast<std::size_t>(records);
        } else if (arg == "--response-store-max-mib") {
            const std::uint64_t mib =
                parse_u64(require_value("--response-store-max-mib"), "response-store-max-mib");
            if (mib == 0 || mib > std::numeric_limits<std::size_t>::max() / (1ULL << 20)) {
                throw std::invalid_argument("--response-store-max-mib is out of range");
            }
            options.response_store_max_bytes = static_cast<std::size_t>(mib << 20);
        } else if (arg == "--device") {
            const char* value = require_value("--device");
#if defined(NINFER_WINDOWS_SERVE)
            if (std::string_view(value) == "auto") {
                options.device = -1;
            } else {
                options.device = parse_nonnegative_int(value, "device");
            }
#else
            options.device = parse_nonnegative_int(value, "device");
#endif
        } else if (arg == "--kv-dtype") {
            options.kv_cache = parse_kv_dtype(require_value("--kv-dtype"));
        } else if (arg == "--spec") {
            options.speculative.backend =
                product::parse_speculative_backend(require_value("--spec"));
        } else if (arg == "--draft-tokens") {
            options.speculative.draft_tokens = static_cast<std::uint32_t>(
                parse_nonnegative_int(require_value("--draft-tokens"), "draft-tokens"));
        } else if (arg == "--default-max-tokens") {
            options.default_max_tokens =
                parse_nonnegative_int(require_value("--default-max-tokens"), "default-max-tokens");
            default_max_tokens_explicit = true;
        } else if (arg == "--default-thinking-budget") {
            const std::uint64_t budget =
                parse_u64(require_value("--default-thinking-budget"), "default-thinking-budget");
            if (budget == 0 || budget > std::numeric_limits<std::uint32_t>::max()) {
                throw std::invalid_argument("--default-thinking-budget is out of range");
            }
            options.default_thinking_budget = static_cast<std::uint32_t>(budget);
        } else if (arg == "--vision") {
            options.enable_vision = true;
        } else if (arg == "--no-cuda-graph") {
            options.use_cuda_graph = false;
        } else if (arg == "--no-prefix-reuse") {
            options.allow_prefix_reuse = false;
        } else if (arg == "--lm-head-draft") {
            options.speculative.proposal_head = ProposalHead::Optimized;
        } else if (arg == "--no-thinking") {
            options.enable_thinking = false;
        } else if (arg == "--preserve-thinking") {
            options.preserve_thinking = true;
        } else if (arg == "--cors") {
            options.enable_cors = true;
        } else if (arg == "--temperature") {
            options.sampling_overrides.temperature =
                parse_float_in(require_value("--temperature"), "temperature", 0.0f, 2.0f);
        } else if (arg == "--top-p") {
            options.sampling_overrides.top_p =
                parse_float_in(require_value("--top-p"), "top-p", 0.0f, 1.0f);
        } else if (arg == "--top-k") {
            const int top_k = parse_nonnegative_int(require_value("--top-k"), "top-k");
            if (top_k > 20) { throw std::invalid_argument("top-k must be in [0,20]"); }
            options.sampling_overrides.top_k = top_k;
        } else if (arg == "--min-p") {
            options.sampling_overrides.min_p =
                parse_float_in(require_value("--min-p"), "min-p", 0.0f, 1.0f);
        } else if (arg == "--presence-penalty") {
            options.sampling_overrides.presence_penalty = parse_float_in(
                require_value("--presence-penalty"), "presence-penalty", -2.0f, 2.0f);
        } else if (arg == "--frequency-penalty") {
            options.sampling_overrides.frequency_penalty = parse_float_in(
                require_value("--frequency-penalty"), "frequency-penalty", -2.0f, 2.0f);
        } else if (arg == "--seed") {
            options.sampling_overrides.seed = parse_u64(require_value("--seed"), "seed");
        } else if (arg == "--greedy") {
            options.greedy = true;
        } else if (arg == "--log-level") {
            options.log_level = product::parse_log_level(require_value("--log-level"));
        } else {
            throw std::invalid_argument("unknown argument: " + arg);
        }
    }
    if (!kv_capacity_explicit) {
        options.kv_capacity = KvCapacityPolicy::explicit_capacity(options.max_context);
    }
    if (!options.allow_prefix_reuse) {
        if (context_capacity_explicit) {
            throw std::invalid_argument(
                "--no-prefix-reuse cannot be combined with context-cache capacity options");
        }
        options.context_cache = ContextCacheOptions{
            .enabled = false,
            .host_state_slots = 0,
            .host_kv_capacity_bytes = 0,
        };
    }
    if (options.port <= 0 || options.port > 65535) {
        throw std::invalid_argument("--port must be in [1,65535]");
    }
    if (options.max_context == 0) { throw std::invalid_argument("--max-context must be positive"); }
    if (options.kv_capacity.mode == KvCapacityMode::Explicit &&
        options.kv_capacity.explicit_tokens < options.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (options.max_concurrency == 0 || options.max_concurrency > kMaximumConcurrency) {
        throw std::invalid_argument("--max-concurrency must be in [1,8]");
    }
#if defined(NINFER_WINDOWS_SERVE)
    if (options.max_concurrency != 1) {
        throw std::invalid_argument("the Windows exclusive-session server requires --max-concurrency 1");
    }
#endif
    if (options.max_pending_requests == 0) {
        throw std::invalid_argument("--max-pending-requests must be positive");
    }
    if (options.pending_timeout_ms == 0) {
        throw std::invalid_argument("--pending-timeout-ms must be positive");
    }
    if (options.max_request_bytes == 0) {
        throw std::invalid_argument("--max-request-mib must be positive");
    }
    if (options.prefill_chunk == 0 || options.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    product::validate_speculative_cli_options(options.speculative);
    if (default_max_tokens_explicit) {
        if (options.default_max_tokens <= 0) {
            throw std::invalid_argument("--default-max-tokens must be positive");
        }
    }
    return options;
}

std::string resolve_public_model_id(const ServeOptions& options,
                                    std::string_view artifact_model_id) {
    if (options.model_id_override.has_value()) { return *options.model_id_override; }
    if (artifact_model_id.empty()) {
        throw std::logic_error("loaded artifact model_id must not be empty");
    }
    return std::string(artifact_model_id);
}

} // namespace ninfer::serve
