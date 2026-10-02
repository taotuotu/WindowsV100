// Modified for the Windows/V100 port by taotuotu, 2026; see NOTICE.
#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/translate.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <iterator>
#include <iomanip>
#include <mutex>
#include <stdexcept>
#include <sstream>
#include <string>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
    std::atomic<bool> stopping{false};
#if defined(NINFER_WINDOWS_SERVE)
    std::string cache_session;
    std::string cache_owner = "none";
    std::uint64_t cache_switch_count = 0;
#endif
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::ThinkingBudgetCapacityInsufficient:
        error.param.clear();
        error.status = 400;
        error.code   = "thinking_budget_capacity_insufficient";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::InvalidMedia:
        error.status = 400;
        error.code   = "invalid_media";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.param.clear();
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

#if defined(NINFER_WINDOWS_SERVE)
// An ownership hint for clients without a session header; never a proof of prefix identity.
std::string automatic_session_key(const GenerationRequest& request) {
    std::uint64_t digest = 14695981039346656037ULL;
    const auto add = [&](std::string_view value) {
        for (const unsigned char byte : value) { digest = (digest ^ byte) * 1099511628211ULL; }
        digest = (digest ^ 0xffU) * 1099511628211ULL;
    };
    for (const ChatTurn& turn : request.messages) {
        if (turn.role != ChatRole::User) { continue; }
        for (const ContentPart& part : turn.content) {
            add(std::to_string(static_cast<int>(part.kind)));
            add(part.kind == ContentKind::Text ? part.text : part.source.value);
        }
        break;
    }
    std::ostringstream key;
    key << "api-auto-" << std::hex << std::setw(16) << std::setfill('0') << digest;
    return key.str();
}
#endif

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception, const char* code) {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_media");
    }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes               = std::move(source_bytes);
    media.image_resize_policy = part.image_resize_policy;
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    explicit ServiceOutputSink(const StreamSink& sink) : sink_(&sink) {}

    void start(ninfer::GenerationStart start) override {
        if (sink_->on_start) { sink_->on_start(start); }
    }

    void progress(ninfer::PromptProgress progress) override {
        if (sink_->on_progress) { sink_->on_progress(progress); }
    }

    void timing(ninfer::GenerationTimingObservation timing) override {
        if (sink_->on_timing) { sink_->on_timing(timing); }
    }

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            if (sink_->on_content) { sink_->on_content(delta.text); }
        }
    }

private:
    const StreamSink* sink_ = nullptr;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, StartupObserver startup_observer,
                                     std::function<void(std::string_view)> cache_failure_observer)
    : options_(std::move(options)),
      cache_failure_observer_(std::move(cache_failure_observer)) {
#if defined(NINFER_WINDOWS_SERVE)
    if (options_.max_concurrency != 1) {
        throw std::invalid_argument("the Windows server accepts one inference request at a time");
    }
#endif
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = std::filesystem::u8path(options_.artifact_path);
    engine_options.device                   = options_.device;
    engine_options.max_context              = options_.max_context;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.kv_cache                 = options_.kv_cache;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.speculative              = options_.speculative;
    engine_options.context_cache            = options_.context_cache;
    engine_options.context_cost.preset_path = options_.context_cost_presets;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.startup_observer         = std::move(startup_observer);
    engine_              = std::make_unique<ninfer::Engine>(std::move(engine_options));
    prompt_capabilities_ = engine_->prompt_capabilities();
    request_capacity_    = std::make_shared<RequestCapacity>(
#if defined(NINFER_WINDOWS_SERVE)
        1);
#else
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
#endif
#if defined(NINFER_WINDOWS_SERVE)
    const auto& actual = engine_->options();
    const auto load = engine_->load_summary();
    // Disk snapshots currently have a physical implementation for the registered dense 27B
    // Program. Other targets keep their existing in-memory cache route.
    const bool supported_target = load.model_id == "qwen3.6-27b" || load.model_id == "qwen3.8-27b";
    const bool supported_backend = actual.speculative.backend == SpeculativeBackend::None ||
                                   actual.speculative.backend == SpeculativeBackend::Mtp;
    const bool supported_kv = actual.kv_cache == KvCacheStorage::BFloat16 ||
                              actual.kv_cache == KvCacheStorage::Int8Group64 ||
                              actual.kv_cache == KvCacheStorage::Fp8E4M3Row256;
    if (!options_.context_disk_cache_directory.empty() && options_.allow_prefix_reuse &&
        actual.context_cache.enabled &&
        supported_target && supported_backend && supported_kv) {
        disk_cache_ = std::make_unique<SessionCacheStore>(
            SessionCacheStoreOptions{.directory = options_.context_disk_cache_directory,
                                     .max_bytes = options_.context_disk_cache_bytes,
                                     .max_sessions = options_.context_disk_cache_sessions},
            SessionCacheIdentity{.artifact_path = actual.artifact_path,
                                 .model_id = load.model_id,
                                 .weights_id = load.weights_id,
                                 .vision_enabled = actual.enable_vision,
                                 .kv_dtype = actual.kv_cache,
                                 .speculative_backend = actual.speculative.backend,
                                 .draft_tokens = actual.speculative.draft_tokens,
                                 .proposal_head = actual.speculative.proposal_head});
    }
#endif
}

std::shared_ptr<RequestLifetime>
GenerationService::acquire_request_lifetime(DeadlinePolicy deadline_policy) const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (request_capacity_->stopping.load(std::memory_order_acquire)) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Unavailable,
                                                     "inference service is shutting down"));
        }
        if (request_capacity_->active >= request_capacity_->maximum) {
#if defined(NINFER_WINDOWS_SERVE)
            ApiError error;
            error.status = 429;
            error.type = "rate_limit_error";
            error.code = "inference_busy";
            error.message = "another inference request is active; retry after it finishes";
            throw ApiException(std::move(error));
#else
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
#endif
        }
        ++request_capacity_->active;
    }
    try {
        const Clock::time_point deadline =
            deadline_policy == DeadlinePolicy::UnboundedStartup
                ? Clock::time_point::max()
                : started + std::chrono::milliseconds(options_.pending_timeout_ms);
        return std::make_shared<RequestLifetime>(request_capacity_, started, deadline);
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           GenerationConsumerMode consumer_mode,
                                           ninfer::GenerationObservationOptions observation,
                                           std::function<bool()> is_cancelled,
                                           ContextCacheHints context_cache) const {
    return prepare_impl(
        request, consumer_mode, observation, std::move(is_cancelled), std::move(context_cache),
        options_.allow_prefix_reuse ? CacheParticipation::ReadWrite : CacheParticipation::Disabled,
        DeadlinePolicy::ClientPendingTimeout);
}

PreparedRequest GenerationService::prepare_impl(const GenerationRequest& request,
                                                GenerationConsumerMode consumer_mode,
                                                ninfer::GenerationObservationOptions observation,
                                                std::function<bool()> is_cancelled,
                                                ContextCacheHints context_cache,
                                                CacheParticipation cache_participation,
                                                DeadlinePolicy deadline_policy) const {
    PreparedRequest prepared;
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    ninfer::RequestOptions request_options = to_request_options(
        request, options_, semantics, cache_participation == CacheParticipation::ReadWrite);
    prepared.enable_thinking            = semantics.enable_thinking;
    prepared.thinking_budget            = request_options.execution.thinking.budget;
    prepared.effective_reasoning_effort = semantics.effective_reasoning_effort;
    prepared.preserve_thinking          = semantics.preserve_thinking;
    const bool request_has_media        = request.media_item_count() != 0;
#if defined(NINFER_WINDOWS_SERVE)
    for (const ChatTurn& turn : request.messages) {
        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Video) {
                const std::invalid_argument error("video input is unavailable in this Windows image server");
                throw_invalid_input(error, "video_not_supported");
            }
        }
    }
#endif
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime(deadline_policy);

    const auto client_cancelled = std::move(is_cancelled);
    is_cancelled = [capacity = request_capacity_, client_cancelled] {
        return capacity->stopping.load(std::memory_order_acquire) ||
               (client_cancelled && client_cancelled());
    };

    try {
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
#if defined(NINFER_WINDOWS_SERVE)
        if (cache_participation == CacheParticipation::ReadWrite) {
            if (!context_cache.session_key) {
                context_cache.session_key = automatic_session_key(request);
            }
        }
        const auto requested_session = context_cache.session_key;
#endif
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        std::vector<PromptCacheMarker> protocol_markers = std::move(input.context_cache.markers);
        const bool protocol_allows_engine_automatic =
            input.context_cache.allow_engine_automatic_shared_prefixes;
        input.context_cache = std::move(context_cache);
        input.context_cache.markers.insert(input.context_cache.markers.end(),
                                           std::make_move_iterator(protocol_markers.begin()),
                                           std::make_move_iterator(protocol_markers.end()));
        input.context_cache.allow_engine_automatic_shared_prefixes =
            input.context_cache.allow_engine_automatic_shared_prefixes &&
            protocol_allows_engine_automatic;
#if defined(NINFER_WINDOWS_SERVE)
        // Actively maintain this owner's shared frontier. Explicit-disabled writes remain
        // absent; exact matching and physical feasibility remain Engine policy.
        for (PromptCacheMarker& marker : input.context_cache.markers) {
            if (has_shared_candidate_evidence(marker.evidence,
                                               SharedCandidateEvidence::DefaultAutomatic)) {
                marker.evidence |= SharedCandidateEvidence::RequestedAutomatic;
            }
        }
#endif
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
#if defined(NINFER_WINDOWS_SERVE)
        if (cache_participation == CacheParticipation::ReadWrite && requested_session) {
            bool changed = false;
            {
                std::lock_guard lock(request_capacity_->mutex);
                changed = request_capacity_->cache_session != *requested_session;
            }
            if (changed) {
                // A fully prepared/validated request may replace the current owner's cache.
                engine_->clear_context_cache();
                if (disk_cache_) {
                    (void)disk_cache_->restore(*engine_, *requested_session, [&] {
                        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
                    });
                }
                std::lock_guard lock(request_capacity_->mutex);
                request_capacity_->cache_session = *requested_session;
                request_capacity_->cache_owner = requested_session->starts_with("client:web-")
                                                     ? "web" : "api";
                ++request_capacity_->cache_switch_count;
            }
            prepared.cache_session = *requested_session;
        }
#endif
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              consumer_mode == GenerationConsumerMode::Streaming
                                                  ? ninfer::OutputConsumerMode::Streaming
                                                  : ninfer::OutputConsumerMode::Aggregate,
                                              observation, prepared.lifetime->deadline);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const bool request_has_media = request.media_item_count() != 0;
#if defined(NINFER_WINDOWS_SERVE)
    for (const ChatTurn& turn : request.messages) {
        for (const ContentPart& part : turn.content) {
            if (part.kind == ContentKind::Video) {
                const std::invalid_argument error("video input is unavailable in this Windows image server");
                throw_invalid_input(error, "video_not_supported");
            }
        }
    }
#endif
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) {
        throw_invalid_input(exception, "invalid_prompt");
    }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) { output_sink = std::make_unique<ServiceOutputSink>(*sink); }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation([capacity = request_capacity_,
                                             external = std::move(is_cancelled), sink]() {
        return capacity->stopping.load(std::memory_order_acquire) || (external && external()) ||
               (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
    });

    ninfer::GenerationResult result;
#if defined(NINFER_WINDOWS_SERVE)
    const auto save_terminal_cache = [&] {
        if (!disk_cache_ || !prepared.cache_session) { return; }
        // wait() also settles the Engine's terminal state before rethrowing a stream consumer
        // exception. Freeze and enqueue in that path too: a disconnected socket must not bypass
        // persistence, while the background writer remains independent of the HTTP stream.
        try {
            const SessionCacheCaptureResult captured = disk_cache_->capture_and_enqueue(
                *engine_, *prepared.cache_session, cache_failure_observer_);
            if (captured == SessionCacheCaptureResult::NoCheckpoint &&
                engine_->is_available()) {
                // An unsafe provisional cancellation may have consumed the in-memory endpoint.
                // Restore the previous complete session snapshot now, so a same-owner next turn
                // does not need a session switch to recover it. A missing file keeps RAM intact.
                (void)disk_cache_->restore(*engine_, *prepared.cache_session, {}, true);
            }
        }
        catch (const std::exception& exception) {
            if (cache_failure_observer_) {
                try { cache_failure_observer_(exception.what()); } catch (...) {}
            }
            // Persistence diagnostics remain visible without replacing the answer/transport
            // error. The previous atomically committed generation is retained on failure.
        }
    };
#endif
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) {
#if defined(NINFER_WINDOWS_SERVE)
        save_terminal_cache();
        prepared.lifetime.reset();
#endif
        throw_request_error(exception);
    } catch (...) {
#if defined(NINFER_WINDOWS_SERVE)
        save_terminal_cache();
        prepared.lifetime.reset();
#endif
        throw;
    }
    GenerationOutcome outcome;
    outcome.text                = std::move(result.content);
    outcome.reasoning           = std::move(result.reasoning);
    outcome.prompt_tokens       = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens   = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens    = static_cast<int>(result.reasoning_tokens);
    outcome.thinking            = result.thinking;
    outcome.finish_reason       = result.finish_reason;
    outcome.matched_stop_string = std::move(result.matched_stop_string);

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds          = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds         = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds          = result.timings.decode_seconds;
    outcome.metrics.prompt_wall_seconds     = result.timings.prompt_wall_seconds;
    outcome.metrics.generation_wall_seconds = result.timings.generation_wall_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.engine_timing               = result.engine_timing;
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.materialization             = result.materialization;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    outcome.tool_calls      = std::move(result.tool_calls);
    outcome.tool_call_parse = result.tool_call_parse;
#if defined(NINFER_WINDOWS_SERVE)
    // Keep the request slot through stable capture. Background disk commit owns only host bytes;
    // release the slot before the terminal HTTP frame so the next turn can use the RAM cache.
    save_terminal_cache();
    prepared.lifetime.reset();
#endif
    return outcome;
}

void GenerationService::request_shutdown() noexcept {
    request_capacity_->stopping.store(true, std::memory_order_release);
}

#if defined(NINFER_WINDOWS_SERVE)
ContextSessionInfo GenerationService::context_session_info() const {
    std::lock_guard lock(request_capacity_->mutex);
    return {.busy = request_capacity_->active != 0,
            .owner = request_capacity_->cache_owner,
            .switch_count = request_capacity_->cache_switch_count};
}

SessionCacheStoreStats GenerationService::disk_cache_stats() const {
    if (disk_cache_) { return disk_cache_->snapshot_stats(); }
    SessionCacheStoreStats stats;
    stats.last_error_code = "disabled";
    return stats;
}

void GenerationService::clear_context_cache() const {
    auto reservation = acquire_request_lifetime(DeadlinePolicy::ClientPendingTimeout);
    engine_->clear_context_cache();
    if (disk_cache_) { disk_cache_->clear(); }
    std::lock_guard lock(request_capacity_->mutex);
    request_capacity_->cache_session.clear();
    request_capacity_->cache_owner = "none";
}
#endif

void GenerationService::warmup() {
    GenerationRequest request;
    ChatTurn turn;
    turn.role = ChatRole::User;
    ContentPart content;
    content.kind     = ContentKind::Text;
    content.text     = "hi";
    content.type_raw = "text";
    turn.content.push_back(std::move(content));
    request.messages.push_back(std::move(turn));
    request.max_tokens = 4;
    PreparedRequest prepared =
        prepare_impl(request, GenerationConsumerMode::Aggregate, {}, {}, {},
                     CacheParticipation::Disabled, DeadlinePolicy::UnboundedStartup);
    run(prepared, nullptr);
}

} // namespace ninfer::serve
