#include "ninfer/engine.h"

#include <algorithm>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace {

struct Options {
    std::filesystem::path artifact_path;
    std::filesystem::path prompt_file;
    std::string prompt;
    std::string system;
    std::uint32_t max_new = 128;
    std::uint32_t max_context = 2048;
    std::uint32_t prefill_chunk = 1024;
    std::uint32_t draft_tokens = 1;
    int device = 1;
    ninfer::KvCacheStorage kv_cache = ninfer::KvCacheStorage::BFloat16;
    ninfer::KvCapacityPolicy kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(2048);
    ninfer::SamplingOverrides sampling;
    std::vector<ninfer::StopString> stop_strings;
    std::optional<std::uint32_t> thinking_budget;
    bool no_mtp = false;
    bool lm_head_draft = false;
    bool enable_thinking = true;
    bool use_cuda_graph = true;
    bool raw_output = false;
    bool print_token_ids = false;
    bool help = false;
};

std::string usage(std::string_view program) {
    return "Usage: " + std::string(program) +
           " <model.ninfer> --prompt TEXT [options]\n"
           "\n"
           "Text-only Windows/V100 frontend for the public NInfer Engine. MTP is enabled by\n"
           "default with one draft token; no FFmpeg or curl media path is linked.\n"
           "\n"
           "  --model PATH             .ninfer artifact (or first positional argument)\n"
           "  --prompt TEXT            user message (mutually exclusive with --prompt-file)\n"
           "  --prompt-file PATH       UTF-8 text file, read without newline normalization\n"
           "  --system TEXT            optional system message\n"
           "  --device N               CUDA device ordinal (default: 1, Tesla V100 here)\n"
           "  --max-context N          context capacity (default: 2048)\n"
           "  --max-new N              output-token limit (default: 128)\n"
           "  --prefill-chunk N        prefill chunk size, multiple of 128 (default: 1024)\n"
           "  --draft-tokens N         MTP draft width 1..7 (default: 1)\n"
           "  --lm-head-draft          use the optimized proposal head with MTP\n"
           "  --no-mtp                 disable MTP and load only target weights\n"
           "  --kv-dtype TYPE          bf16|int8|fp8 (default: bf16 on V100)\n"
           "  --kv-capacity N|auto     KV capacity policy (default: max-context)\n"
           "  --temperature F          model-default sampling override\n"
           "  --top-k N                top-k sampling override, 0 disables\n"
           "  --top-p F                top-p sampling override\n"
           "  --min-p F                min-p sampling override\n"
           "  --presence-penalty F     presence penalty override\n"
           "  --frequency-penalty F    frequency penalty override\n"
           "  --seed N                 deterministic sampling seed\n"
           "  --greedy                 set temperature to zero\n"
           "  --stop TEXT              content stop string (repeatable)\n"
           "  --reasoning-stop TEXT    reasoning stop string (repeatable)\n"
           "  --thinking-budget N      optional thinking-token budget\n"
           "  --no-thinking            disable thinking template mode\n"
           "  --raw-output             bypass output post-processing\n"
           "  --print-token-ids        print generated token IDs to stderr\n"
           "  --no-cuda-graph          disable CUDA Graph execution\n"
           "  Throughput reports decode_tok_s (N-1 / Program decode),\n"
           "  wall_decode_tok_s (N-1 / first-to-last output wall), and\n"
           "  overall_tok_s (N / request wall, including prompt work) separately.\n"
           "  -h, --help               show this help\n";
}

std::string_view value_after(const std::vector<std::string>& args, std::size_t& index,
                             std::string_view flag) {
    if (++index >= args.size()) {
        throw std::invalid_argument(std::string(flag) + " requires a value");
    }
    return args[index];
}

template <typename T>
T parse_integer(std::string_view text, std::string_view name) {
    T value{};
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument("invalid " + std::string(name) + ": " + std::string(text));
    }
    return value;
}

float parse_float(std::string_view text, std::string_view name, float minimum, float maximum) {
    const std::string value(text);
    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(value.c_str(), &end);
    if (errno == ERANGE || end != value.c_str() + value.size() || !std::isfinite(parsed) ||
        parsed < minimum || parsed > maximum) {
        throw std::invalid_argument("invalid " + std::string(name) + ": " + value);
    }
    return parsed;
}

ninfer::KvCacheStorage parse_kv_cache(std::string_view value) {
    if (value == "bf16") { return ninfer::KvCacheStorage::BFloat16; }
    if (value == "int8") { return ninfer::KvCacheStorage::Int8Group64; }
    if (value == "fp8") { return ninfer::KvCacheStorage::Fp8E4M3Row256; }
    throw std::invalid_argument("invalid --kv-dtype: " + std::string(value));
}

std::string path_for_error(const std::filesystem::path& path) {
#if defined(_WIN32)
    const auto utf8 = path.u8string();
    return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
    return path.string();
#endif
}

std::string read_utf8_prompt_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open prompt file: " + path_for_error(path));
    }
    std::string text;
    text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if (input.bad()) {
        throw std::runtime_error("failed to read prompt file: " + path_for_error(path));
    }
    if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEFU &&
        static_cast<unsigned char>(text[1]) == 0xBBU &&
        static_cast<unsigned char>(text[2]) == 0xBFU) {
        text.erase(0, 3);
    }
    return text;
}

Options parse_options(const std::vector<std::string>& args) {
    Options result;
    bool have_model = false;
    bool have_prompt = false;
    bool have_prompt_file = false;
    bool kv_capacity_set = false;
    bool draft_width_set = false;

    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view arg(args[i]);
        if (arg == "--help" || arg == "-h") {
            result.help = true;
            return result;
        }
        if (arg == "--model") {
            if (have_model) { throw std::invalid_argument("model path was provided more than once"); }
            result.artifact_path =
                std::filesystem::u8path(std::string(value_after(args, i, arg)));
            have_model = true;
        } else if (arg == "--prompt") {
            if (have_prompt || have_prompt_file) {
                throw std::invalid_argument("pass exactly one of --prompt or --prompt-file");
            }
            result.prompt = std::string(value_after(args, i, arg));
            have_prompt = true;
        } else if (arg == "--prompt-file") {
            if (have_prompt || have_prompt_file) {
                throw std::invalid_argument("pass exactly one of --prompt or --prompt-file");
            }
            result.prompt_file =
                std::filesystem::u8path(std::string(value_after(args, i, arg)));
            have_prompt_file = true;
        } else if (arg == "--system") {
            result.system = std::string(value_after(args, i, arg));
        } else if (arg == "--device") {
            const auto value = parse_integer<std::int32_t>(value_after(args, i, arg), "--device");
            if (value < 0) { throw std::invalid_argument("--device must be nonnegative"); }
            result.device = value;
        } else if (arg == "--max-context") {
            result.max_context = parse_integer<std::uint32_t>(value_after(args, i, arg), arg);
        } else if (arg == "--max-new") {
            result.max_new = parse_integer<std::uint32_t>(value_after(args, i, arg), arg);
        } else if (arg == "--prefill-chunk") {
            result.prefill_chunk = parse_integer<std::uint32_t>(value_after(args, i, arg), arg);
        } else if (arg == "--draft-tokens") {
            result.draft_tokens = parse_integer<std::uint32_t>(value_after(args, i, arg), arg);
            draft_width_set = true;
        } else if (arg == "--no-mtp") {
            result.no_mtp = true;
        } else if (arg == "--lm-head-draft") {
            result.lm_head_draft = true;
        } else if (arg == "--kv-dtype") {
            result.kv_cache = parse_kv_cache(value_after(args, i, arg));
        } else if (arg == "--kv-capacity") {
            const std::string_view value = value_after(args, i, arg);
            if (value == "auto") {
                result.kv_capacity = ninfer::KvCapacityPolicy::automatic();
            } else {
                result.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
                    parse_integer<std::uint32_t>(value, arg));
            }
            kv_capacity_set = true;
        } else if (arg == "--temperature") {
            result.sampling.temperature = parse_float(value_after(args, i, arg), arg, 0.0F, 2.0F);
        } else if (arg == "--top-p") {
            result.sampling.top_p = parse_float(value_after(args, i, arg), arg, 0.0F, 1.0F);
        } else if (arg == "--min-p") {
            result.sampling.min_p = parse_float(value_after(args, i, arg), arg, 0.0F, 1.0F);
        } else if (arg == "--presence-penalty") {
            result.sampling.presence_penalty =
                parse_float(value_after(args, i, arg), arg, -2.0F, 2.0F);
        } else if (arg == "--frequency-penalty") {
            result.sampling.frequency_penalty =
                parse_float(value_after(args, i, arg), arg, -2.0F, 2.0F);
        } else if (arg == "--top-k") {
            const auto value = parse_integer<std::uint32_t>(value_after(args, i, arg), arg);
            if (value > 20) { throw std::invalid_argument("--top-k must be in [0, 20]"); }
            result.sampling.top_k = static_cast<std::int32_t>(value);
        } else if (arg == "--seed") {
            result.sampling.seed = parse_integer<std::uint64_t>(value_after(args, i, arg), arg);
        } else if (arg == "--greedy") {
            result.sampling.temperature = 0.0F;
        } else if (arg == "--stop" || arg == "--reasoning-stop") {
            const std::string text(value_after(args, i, arg));
            if (text.empty()) { throw std::invalid_argument(std::string(arg) + " cannot be empty"); }
            result.stop_strings.push_back(ninfer::StopString{
                .text = text,
                .channel = arg == "--stop" ? ninfer::OutputChannel::Content
                                             : ninfer::OutputChannel::Reasoning,
            });
        } else if (arg == "--thinking-budget") {
            result.thinking_budget = parse_integer<std::uint32_t>(value_after(args, i, arg), arg);
        } else if (arg == "--no-thinking") {
            result.enable_thinking = false;
        } else if (arg == "--raw-output") {
            result.raw_output = true;
        } else if (arg == "--print-token-ids") {
            result.print_token_ids = true;
        } else if (arg == "--no-cuda-graph") {
            result.use_cuda_graph = false;
        } else if (!arg.empty() && arg.front() == '-') {
            throw std::invalid_argument("unknown option: " + std::string(arg));
        } else {
            if (have_model) { throw std::invalid_argument("unexpected positional argument"); }
            result.artifact_path = std::filesystem::u8path(std::string(arg));
            have_model = true;
        }
    }

    if (!have_model || result.artifact_path.empty()) {
        throw std::invalid_argument("a .ninfer model path is required");
    }
    if (!have_prompt && !have_prompt_file) {
        throw std::invalid_argument("pass exactly one of --prompt or --prompt-file");
    }
    if (have_prompt_file) { result.prompt = read_utf8_prompt_file(result.prompt_file); }
    if (result.prompt.empty()) { throw std::invalid_argument("prompt text must be nonempty"); }
    if (result.max_context == 0 || result.max_new == 0) {
        throw std::invalid_argument("--max-context and --max-new must be positive");
    }
    if (result.prefill_chunk == 0 || result.prefill_chunk % 128 != 0) {
        throw std::invalid_argument("--prefill-chunk must be a positive multiple of 128");
    }
    if (!kv_capacity_set) {
        result.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(result.max_context);
    }
    if (result.kv_capacity.mode == ninfer::KvCapacityMode::Explicit &&
        result.kv_capacity.explicit_tokens < result.max_context) {
        throw std::invalid_argument("--kv-capacity must be at least --max-context");
    }
    if (result.no_mtp && draft_width_set) {
        throw std::invalid_argument("--draft-tokens cannot be combined with --no-mtp");
    }
    if (result.no_mtp && result.lm_head_draft) {
        throw std::invalid_argument("--lm-head-draft cannot be combined with --no-mtp");
    }
    if (!result.no_mtp && (result.draft_tokens < 1 || result.draft_tokens > 7)) {
        throw std::invalid_argument("MTP --draft-tokens must be in [1, 7]");
    }
    if (!result.enable_thinking && result.thinking_budget) {
        throw std::invalid_argument("--thinking-budget cannot be combined with --no-thinking");
    }
    return result;
}

class StreamingSink final : public ninfer::OutputSink {
public:
    void start(ninfer::GenerationStart) override {}
    void progress(ninfer::PromptProgress) override {}
    void timing(ninfer::GenerationTimingObservation) override {}

    void publish(ninfer::OutputDelta delta) override {
        std::ostream& output = delta.channel == ninfer::OutputChannel::Reasoning
                                   ? std::cerr
                                   : std::cout;
        output << delta.text;
        output.flush();
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            reasoning_seen_ = reasoning_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { reasoning_ends_newline_ = delta.text.back() == '\n'; }
        } else {
            content_seen_ = content_seen_ || !delta.text.empty();
            if (!delta.text.empty()) { content_ends_newline_ = delta.text.back() == '\n'; }
        }
    }

    void finish(bool successful) {
        if (finished_) { return; }
        finished_ = true;
        if ((successful && !content_seen_) || (content_seen_ && !content_ends_newline_)) {
            std::cout << '\n';
        }
        std::cout.flush();
        if (reasoning_seen_ && !reasoning_ends_newline_) { std::cerr << '\n'; }
    }

private:
    bool content_seen_ = false;
    bool content_ends_newline_ = false;
    bool reasoning_seen_ = false;
    bool reasoning_ends_newline_ = false;
    bool finished_ = false;
};

std::string_view finish_reason(ninfer::FinishReason reason) noexcept {
    switch (reason) {
    case ninfer::FinishReason::None: return "none";
    case ninfer::FinishReason::OutputLimit: return "output-limit";
    case ninfer::FinishReason::ContextCapacity: return "context-capacity";
    case ninfer::FinishReason::StopToken: return "stop-token";
    case ninfer::FinishReason::StopString: return "stop-string";
    case ninfer::FinishReason::Cancelled: return "cancelled";
    }
    return "unknown";
}

std::string_view startup_phase(ninfer::StartupPhase phase) noexcept {
    switch (phase) {
    case ninfer::StartupPhase::EngineStartup: return "engine";
    case ninfer::StartupPhase::CudaInitialize: return "cuda";
    case ninfer::StartupPhase::ArtifactInspect: return "artifact";
    case ninfer::StartupPhase::TargetPlan: return "target-plan";
    case ninfer::StartupPhase::WeightsMaterialize: return "weights";
    case ninfer::StartupPhase::WeightsStagingPin: return "staging";
    case ninfer::StartupPhase::TargetFinalize: return "target-finalize";
    case ninfer::StartupPhase::FrontendInitialize: return "frontend";
    case ninfer::StartupPhase::ProgramInitialize: return "program";
    case ninfer::StartupPhase::HostStatePin: return "host-state";
    case ninfer::StartupPhase::HostKvPin: return "host-kv";
    case ninfer::StartupPhase::CudaGraphPrepare: return "cuda-graph";
    case ninfer::StartupPhase::EngineFinalize: return "engine-finalize";
    }
    return "unknown";
}

int run(const std::vector<std::string>& args) {
    Options cli;
    try {
        cli = parse_options(args);
    } catch (const std::exception& error) {
        std::cerr << "error: " << error.what() << '\n' << usage(args.empty() ? "ninfer-windows-text" : args[0]);
        return 2;
    }
    if (cli.help) {
        std::cout << usage(args.empty() ? "ninfer-windows-text" : args[0]);
        return 0;
    }

    try {
        ninfer::PromptInput input;
        input.options.enable_thinking = cli.enable_thinking;
        if (!cli.system.empty()) {
            ninfer::ChatMessage system;
            system.role = ninfer::ChatRole::System;
            ninfer::MessagePart part;
            part.kind = ninfer::MessagePartKind::Text;
            part.text = cli.system;
            system.parts.push_back(std::move(part));
            input.messages.push_back(std::move(system));
        }
        ninfer::ChatMessage user;
        user.role = ninfer::ChatRole::User;
        ninfer::MessagePart user_part;
        user_part.kind = ninfer::MessagePartKind::Text;
        user_part.text = cli.prompt;
        user.parts.push_back(std::move(user_part));
        input.messages.push_back(std::move(user));

        ninfer::EngineOptions engine_options;
        engine_options.artifact_path = cli.artifact_path;
        engine_options.device = cli.device;
        engine_options.max_context = cli.max_context;
        engine_options.kv_capacity = cli.kv_capacity;
        engine_options.prefill_chunk = cli.prefill_chunk;
        engine_options.kv_cache = cli.kv_cache;
        engine_options.enable_vision = false;
        engine_options.use_cuda_graph = cli.use_cuda_graph;
        engine_options.context_cache.enabled = false;
        engine_options.context_cache.host_state_slots = 0;
        engine_options.context_cache.host_kv_capacity_bytes = 0;
        std::uint32_t last_weight_percent = 0;
        engine_options.startup_observer.callback = [&last_weight_percent](const ninfer::StartupEvent& event) {
            if (event.status == ninfer::StartupStatus::Begin) {
                std::cerr << "startup " << startup_phase(event.phase) << " begin\n";
            } else if (event.phase == ninfer::StartupPhase::WeightsMaterialize &&
                       event.status == ninfer::StartupStatus::Progress && event.total != 0) {
                const std::uint32_t percent = static_cast<std::uint32_t>(
                    (static_cast<long double>(event.current) * 100.0L) / event.total);
                if (percent >= last_weight_percent + 10U || percent == 100U) {
                    last_weight_percent = percent;
                    std::cerr << "startup weights " << percent << "%\n";
                }
            } else if (event.status == ninfer::StartupStatus::Complete) {
                std::cerr << "startup " << startup_phase(event.phase) << " complete"
                          << " elapsed_s=" << std::fixed << std::setprecision(3)
                          << static_cast<double>(event.elapsed_ns) / 1.0e9 << '\n';
            }
        };
        if (!cli.no_mtp) {
            engine_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            engine_options.speculative.draft_tokens = cli.draft_tokens;
            engine_options.speculative.proposal_head =
                cli.lm_head_draft ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full;
        }

        ninfer::Engine engine(std::move(engine_options));
        ninfer::PreparedPrompt prompt = engine.prepare(std::move(input));
        ninfer::RequestOptions request;
        request.execution.requested_output_tokens = cli.max_new;
        request.execution.sampling = cli.sampling;
        request.execution.thinking.budget = cli.thinking_budget;
        request.stop.strings = cli.stop_strings;
        request.output.raw = cli.raw_output;

        ninfer::GenerationHandle generation = engine.submit(
            std::move(prompt), std::move(request), ninfer::OutputConsumerMode::Streaming,
            ninfer::GenerationObservationOptions{.phase_timings = true});
        const ninfer::ResolvedSamplingParameters sampling = generation.resolved_sampling();
        StreamingSink sink;
        ninfer::GenerationResult result;
        try {
            result = generation.wait(&sink);
            sink.finish(true);
        } catch (...) {
            sink.finish(false);
            throw;
        }

        std::cerr << std::fixed << std::setprecision(3)
                  << "prompt_tokens=" << result.prompt.prompt_tokens
                  << " generated_tokens=" << result.generated_token_ids.size()
                  << " finish=" << finish_reason(result.finish_reason)
                  << " prefill_s=" << result.timings.prefill_seconds
                  << " decode_s=" << result.timings.decode_seconds
                  << " decode_tok_s="
                  << (result.timings.decode_seconds > 0.0 && result.generated_token_ids.size() > 1
                          ? static_cast<double>(result.generated_token_ids.size() - 1) /
                                result.timings.decode_seconds
                          : 0.0)
                  << " generation_wall_s=" << result.timings.generation_wall_seconds
                  << " wall_decode_tok_s="
                  << (result.timings.generation_wall_seconds > 0.0 &&
                              result.generated_token_ids.size() > 1
                          ? static_cast<double>(result.generated_token_ids.size() - 1) /
                                result.timings.generation_wall_seconds
                          : 0.0)
                  << " overall_tok_s="
                  << (result.timings.total_seconds > 0.0
                          ? static_cast<double>(result.generated_token_ids.size()) /
                                result.timings.total_seconds
                          : 0.0)
                  << " total_s=" << result.timings.total_seconds
                  << " sampling="
                  << (sampling.temperature <= 0.0F ? "greedy" : "sampled")
                  << " temp=" << sampling.temperature << " seed=" << sampling.seed << '\n';
        if (cli.print_token_ids) {
            std::cerr << "generated_token_ids=";
            for (std::size_t i = 0; i < result.generated_token_ids.size(); ++i) {
                if (i != 0) { std::cerr << ' '; }
                std::cerr << result.generated_token_ids[i];
            }
            std::cerr << '\n';
        }
        if (result.speculative.enabled) {
            const double accepted = result.speculative.drafted_tokens == 0
                                        ? 0.0
                                        : 100.0 * static_cast<double>(result.speculative.accepted_tokens) /
                                              static_cast<double>(result.speculative.drafted_tokens);
            std::cerr << "MTP proposal_head=" << (cli.lm_head_draft ? "optimized" : "full")
                      << " rounds=" << result.speculative.rounds
                      << " drafted=" << result.speculative.drafted_tokens
                      << " accepted=" << result.speculative.accepted_tokens
                      << " acceptance=" << accepted << "%\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ninfer-windows-text: " << error.what() << '\n';
        return 1;
    }
}

#if defined(_WIN32)
std::string wide_to_utf8(const wchar_t* value) {
    const int required = ::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                                                nullptr, 0, nullptr, nullptr);
    if (required <= 0) { throw std::runtime_error("failed to convert command line to UTF-8"); }
    std::string result(static_cast<std::size_t>(required), '\0');
    if (::WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, value, -1,
                              result.data(), required, nullptr, nullptr) <= 0) {
        throw std::runtime_error("failed to convert command line to UTF-8");
    }
    result.resize(static_cast<std::size_t>(required - 1));
    return result;
}
#endif

} // namespace

#if defined(_WIN32)
int wmain(int argc, wchar_t* argv[]) {
    ::SetConsoleCP(CP_UTF8);
    ::SetConsoleOutputCP(CP_UTF8);
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    try {
        for (int i = 0; i < argc; ++i) { args.push_back(wide_to_utf8(argv[i])); }
        return run(args);
    } catch (const std::exception& error) {
        std::cerr << "ninfer-windows-text: " << error.what() << '\n';
        return 1;
    }
}
#else
int main(int argc, char* argv[]) {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) { args.emplace_back(argv[i]); }
    return run(args);
}
#endif
