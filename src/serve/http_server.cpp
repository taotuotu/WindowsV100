// Modified for the Windows/V100 port by taotuotu, 2026; see NOTICE.
#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
#include "product/speculative_options.h"
#include "serve/web/chat_page.h"
#endif

#include <nlohmann/json.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <filesystem>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
const char* kv_dtype_name(ninfer::KvCacheStorage storage) noexcept {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

const char* proposal_head_name(ninfer::ProposalHead proposal_head) noexcept {
    switch (proposal_head) {
    case ninfer::ProposalHead::Full:
        return "full";
    case ninfer::ProposalHead::Optimized:
        return "optimized";
    }
    return "unknown";
}

std::string utf8_basename(const std::filesystem::path& path) {
    const std::u8string basename = path.filename().u8string();
    return std::string(reinterpret_cast<const char*>(basename.data()), basename.size());
}
#endif

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    ui_server_instance_id_ = "serve-ui-" + new_openai_request_id();
#endif
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
std::uint64_t HttpServer::ui_uptime_ms() const noexcept {
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - ui_server_started_at_);
    return static_cast<std::uint64_t>(std::max<std::int64_t>(elapsed.count(), 0));
}

void HttpServer::trim_ui_request_history_locked() {
    while (ui_recent_requests_.size() > kUiRecentRequestLimit) {
        const auto newest = std::prev(ui_recent_requests_.end());
        const auto oldest_terminal = std::find_if(
            ui_recent_requests_.begin(), newest,
            [](const UiRecentRequest& request) { return request.status != "running"; });
        if (oldest_terminal != newest) {
            ui_recent_requests_.erase(oldest_terminal);
        } else {
            ui_recent_requests_.pop_front();
        }
    }
}
#endif

void HttpServer::record_request_start(const RequestLogContext& context) {
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    {
        std::lock_guard lock(ui_metrics_mutex_);
        ++ui_requests_started_;
        UiRecentRequest request;
        request.id             = context.id;
        request.protocol       = context.protocol;
        request.status         = "running";
        request.started_at_ms  = ui_uptime_ms();
        ui_recent_requests_.push_back(std::move(request));
        trim_ui_request_history_locked();
    }
#endif
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    {
        std::lock_guard lock(ui_metrics_mutex_);
        ++ui_requests_started_;
        ++ui_requests_rejected_;
        const std::uint64_t now = ui_uptime_ms();
        UiRecentRequest request;
        request.id             = context.id;
        request.protocol       = context.protocol;
        request.status         = "rejected";
        request.finished_at_ms = now;
        ui_recent_requests_.push_back(std::move(request));
        trim_ui_request_history_locked();
    }
#endif
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    const int prompt_tokens     = std::max(0, outcome.prompt_tokens);
    const int completion_tokens = std::max(0, outcome.completion_tokens);
    const std::uint64_t cached_tokens = std::min<std::uint64_t>(
        outcome.metrics.prefix_cache_hit_tokens, static_cast<std::uint64_t>(prompt_tokens));
    const int computed_prefill_tokens =
        prompt_tokens - static_cast<int>(cached_tokens);
    const auto measured_ms = [](double seconds) -> std::optional<double> {
        if (!std::isfinite(seconds) || seconds < 0.0) { return std::nullopt; }
        return seconds * 1000.0;
    };
    const auto measured_rate = [](double tokens, double seconds) -> std::optional<double> {
        if (!std::isfinite(seconds) || seconds <= 0.0) { return std::nullopt; }
        const double rate = tokens / seconds;
        return std::isfinite(rate) && rate >= 0.0 ? std::optional<double>(rate) : std::nullopt;
    };
    const std::optional<double> decode_tok_s = measured_rate(
        static_cast<double>(completion_tokens > 0 ? completion_tokens - 1 : 0),
        outcome.metrics.generation_wall_seconds);
    const std::optional<double> prefill_tok_s = measured_rate(
        static_cast<double>(computed_prefill_tokens), outcome.metrics.prompt_wall_seconds);
    const std::uint64_t finished_at_ms = ui_uptime_ms();
    {
        std::lock_guard lock(ui_metrics_mutex_);
        ++ui_requests_completed_;
        ui_cached_prompt_tokens_ += cached_tokens;
        const auto request = std::find_if(
            ui_recent_requests_.begin(), ui_recent_requests_.end(),
            [&context](const UiRecentRequest& candidate) { return candidate.id == context.id; });
        UiRecentRequest terminal_request;
        terminal_request.id               = context.id;
        terminal_request.protocol         = context.protocol;
        terminal_request.status           = "completed";
        terminal_request.prompt_tokens    = prompt_tokens;
        terminal_request.completion_tokens = completion_tokens;
        terminal_request.cached_tokens    = cached_tokens;
        terminal_request.ttft_ms          = measured_ms(outcome.metrics.ttft_seconds);
        terminal_request.total_ms         = measured_ms(outcome.metrics.total_seconds);
        terminal_request.decode_tok_s     = decode_tok_s;
        terminal_request.prefill_tok_s    = prefill_tok_s;
        terminal_request.finished_at_ms  = finished_at_ms;
        if (request != ui_recent_requests_.end()) {
            terminal_request.started_at_ms = request->started_at_ms;
            *request = std::move(terminal_request);
        } else {
            ui_recent_requests_.push_back(std::move(terminal_request));
            trim_ui_request_history_locked();
        }
    }
#endif
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    {
        std::lock_guard lock(ui_metrics_mutex_);
        ++ui_requests_failed_;
        const auto request = std::find_if(
            ui_recent_requests_.begin(), ui_recent_requests_.end(),
            [&context](const UiRecentRequest& candidate) { return candidate.id == context.id; });
        if (request != ui_recent_requests_.end()) {
            request->status          = "failed";
            request->finished_at_ms = ui_uptime_ms();
        } else {
            UiRecentRequest terminal_request;
            terminal_request.id             = context.id;
            terminal_request.protocol       = context.protocol;
            terminal_request.status         = "failed";
            terminal_request.finished_at_ms = ui_uptime_ms();
            ui_recent_requests_.push_back(std::move(terminal_request));
            trim_ui_request_history_locked();
        }
    }
#endif
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
        if (options_.api_key.empty() || (req.path == "/" && req.method == "GET") ||
            req.path == "/health" || req.method == "OPTIONS") {
#else
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
#endif
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    server_.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Cache-Control", "no-store");
        res.set_content(web::kChatPageHtml.data(), web::kChatPageHtml.size(),
                        "text/html; charset=utf-8");
    });
    server_.Get("/ui/model-info", [this](const httplib::Request&, httplib::Response& res) {
        if (service_ == nullptr || !service_->is_available()) {
            res.status = 503;
            res.set_content(nlohmann::json{{"status", "unavailable"}}.dump(),
                            "application/json");
            return;
        }

        const ninfer::LoadSummary load = service_->load_summary();
        const ninfer::EngineOptions& engine = service_->engine_options();
        const std::string artifact_file = utf8_basename(engine.artifact_path);
        const ninfer::SpeculativeOptions& speculative = engine.speculative;
        const nlohmann::json info{
            {"api_model_id", public_model_id_},
            {"model_id", load.model_id},
            {"weights_id", load.weights_id},
            {"artifact_file", artifact_file},
            {"kv_dtype", kv_dtype_name(engine.kv_cache)},
            {"max_context", engine.max_context},
            {"device", engine.device},
            {"speculative_backend", product::speculative_backend_name(speculative.backend)},
            {"draft_tokens", speculative.draft_tokens},
            {"proposal_head", proposal_head_name(speculative.proposal_head)},
            {"prefix_reuse", service_->options().allow_prefix_reuse},
        };
        res.set_header("Cache-Control", "no-store");
        res.set_content(info.dump(), "application/json");
    });
    server_.Get("/ui/metrics", [this](const httplib::Request& req, httplib::Response& res) {
        handle_ui_metrics(req, res);
    });
#endif
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    res.set_content(make_model_object(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
void HttpServer::handle_ui_metrics(const httplib::Request&, httplib::Response& res) const {
    using Json = nlohmann::json;

    const bool attached = service_ != nullptr;
    const bool available = attached && service_->is_available();
    const ninfer::RuntimeStats runtime = attached ? service_->runtime_stats()
                                                  : ninfer::RuntimeStats{};

    std::uint64_t decode_baseline = 0;
    std::uint64_t prefill_baseline = 0;
    std::uint64_t requests_started = 0;
    std::uint64_t requests_completed = 0;
    std::uint64_t requests_failed = 0;
    std::uint64_t requests_rejected = 0;
    std::uint64_t cached_prompt_tokens = 0;
    std::deque<UiRecentRequest> recent_requests;
    {
        std::lock_guard lock(ui_metrics_mutex_);
        decode_baseline       = ui_runtime_decode_baseline_;
        prefill_baseline      = ui_runtime_prefill_baseline_;
        requests_started      = ui_requests_started_;
        requests_completed    = ui_requests_completed_;
        requests_failed       = ui_requests_failed_;
        requests_rejected     = ui_requests_rejected_;
        cached_prompt_tokens  = ui_cached_prompt_tokens_;
        recent_requests       = ui_recent_requests_;
    }

    const auto counter_delta = [](std::uint64_t current, std::uint64_t baseline) {
        return current >= baseline ? current - baseline : std::uint64_t{0};
    };
    const auto optional_json = [](const auto& value) -> Json {
        return value ? Json(*value) : Json(nullptr);
    };
    Json recent = Json::array();
    for (const UiRecentRequest& request : recent_requests) {
        recent.push_back(Json{
            {"id", request.id},
            {"protocol", request.protocol},
            {"status", request.status},
            {"prompt_tokens", optional_json(request.prompt_tokens)},
            {"completion_tokens", optional_json(request.completion_tokens)},
            {"cached_tokens", optional_json(request.cached_tokens)},
            {"ttft_ms", optional_json(request.ttft_ms)},
            {"total_ms", optional_json(request.total_ms)},
            {"decode_tok_s", optional_json(request.decode_tok_s)},
            {"prefill_tok_s", optional_json(request.prefill_tok_s)},
            {"started_at_ms", optional_json(request.started_at_ms)},
            {"finished_at_ms", optional_json(request.finished_at_ms)},
        });
    }

    const Json response{
        {"schema_version", 1},
        {"server_instance_id", ui_server_instance_id_},
        {"uptime_ms", ui_uptime_ms()},
        {"available", available},
        {"totals",
         Json{{"committed_decode_tokens",
               counter_delta(runtime.committed_decode_tokens, decode_baseline)},
              {"computed_prefill_tokens",
               counter_delta(runtime.computed_prefill_tokens, prefill_baseline)},
              {"requests_started", requests_started},
              {"requests_completed", requests_completed},
              {"requests_failed", requests_failed},
              {"requests_rejected", requests_rejected},
              {"cached_prompt_tokens", cached_prompt_tokens}}},
        {"scheduler",
         Json{{"running", runtime.running_requests},
              {"waiting", runtime.waiting_requests},
              {"prefilling", runtime.prefilling_requests},
              {"decode_ready", runtime.decode_ready_requests},
              {"materializing", runtime.materializing_requests}}},
        {"recent_requests", std::move(recent)},
    };

    res.status = available ? 200 : 503;
    res.set_header("Cache-Control", "no-store");
    res.set_content(response.dump(), "application/json");
}
#endif

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_id);
#if defined(NINFER_WINDOWS_TEXT_ONLY_SERVE)
    const ninfer::RuntimeStats ui_baseline = service.runtime_stats();
    {
        std::lock_guard lock(ui_metrics_mutex_);
        ui_runtime_decode_baseline_  = ui_baseline.committed_decode_tokens;
        ui_runtime_prefill_baseline_ = ui_baseline.computed_prefill_tokens;
    }
#endif
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
