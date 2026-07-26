#include "toyllm/runtime/profiling.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>

namespace toyllm {

namespace {

using SteadyClock = std::chrono::steady_clock;
using SystemClock = std::chrono::system_clock;

std::string json_escape(std::string_view text) {
  std::string output;
  output.reserve(text.size() + 8U);
  for (const char ch : text) {
    switch (ch) {
      case '"':
        output += "\\\"";
        break;
      case '\\':
        output += "\\\\";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20U) {
          output += "\\u00";
          constexpr char kHex[] = "0123456789abcdef";
          output.push_back(kHex[(static_cast<unsigned char>(ch) >> 4U) & 0xFU]);
          output.push_back(kHex[static_cast<unsigned char>(ch) & 0xFU]);
        } else {
          output.push_back(ch);
        }
    }
  }
  return output;
}

std::string make_random_base36(std::size_t count) {
  static thread_local std::mt19937_64 rng{std::random_device{}()};
  static constexpr char kAlphabet[] = "0123456789abcdefghijklmnopqrstuvwxyz";
  std::uniform_int_distribution<std::size_t> dist(0, 35U);
  std::string output;
  output.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    output.push_back(kAlphabet[dist(rng)]);
  }
  return output;
}

std::string format_timestamp(std::chrono::system_clock::time_point tp) {
  const auto time = SystemClock::to_time_t(tp);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif
  std::ostringstream output;
  output << std::put_time(&tm, "%Y-%m-%dT%H:%M:%S");
  return output.str();
}

std::string format_request_stamp(std::chrono::system_clock::time_point tp) {
  const auto time = SystemClock::to_time_t(tp);
  std::tm tm{};
#if defined(_WIN32)
  localtime_s(&tm, &time);
#else
  localtime_r(&time, &tm);
#endif
  std::ostringstream output;
  output << std::put_time(&tm, "%d%m%Y-%H%M%S");
  return output.str();
}

std::string make_request_id(std::string_view prefix) {
  const auto now = SystemClock::now();
  std::ostringstream output;
  output << prefix << '-' << format_request_stamp(now) << '-' << make_random_base36(6);
  return output.str();
}

std::string span_label(const std::string& name, const std::vector<ProfileField>& fields) {
  if (name == "layer") {
    for (const auto& field : fields) {
      if (field.key == "layer") {
        return "layer_" + field.value;
      }
    }
  }
  if (name == "forward_token") {
    for (const auto& field : fields) {
      if (field.key == "position") {
        return "forward_token_" + field.value;
      }
    }
  }
  if (name == "decode_step") {
    for (const auto& field : fields) {
      if (field.key == "step") {
        return "decode_step_" + field.value;
      }
    }
  }
  return name;
}

std::filesystem::path ensure_output_dir(const ObservabilityConfig& config,
                                        const std::string& request_id) {
  auto root = config.profile_output_dir;
  if (root.empty()) {
    root = std::filesystem::path{"build"} / "profiles";
  }
  return root / request_id;
}

void write_file(const std::filesystem::path& path, std::string_view content) {
  std::filesystem::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  if (!output) {
    throw std::runtime_error("failed to open profile output file: " + path.string());
  }
  output << content;
  if (!output) {
    throw std::runtime_error("failed to write profile output file: " + path.string());
  }
}

std::string shell_quote(std::string_view value) {
  std::string output{"'"};
  for (const char ch : value) {
    if (ch == '\'') {
      output += "'\\''";
    } else {
      output.push_back(ch);
    }
  }
  output.push_back('\'');
  return output;
}

struct SpanRecord {
  std::uint32_t id{0};
  std::uint32_t parent_id{0};
  std::string name;
  std::uint64_t start_ns{0};
  std::uint64_t end_ns{0};
  std::uint32_t depth{0};
  std::vector<ProfileField> fields;
};

struct SpanSummary {
  std::uint64_t total_ns{0};
  std::uint64_t self_ns{0};
};

std::string profile_mode_name(ProfileMode mode) {
  switch (mode) {
    case ProfileMode::off:
      return "off";
    case ProfileMode::summary:
      return "summary";
    case ProfileMode::trace:
      return "trace";
    case ProfileMode::flamegraph:
      return "flamegraph";
    case ProfileMode::all:
      return "all";
  }
  return "off";
}

std::vector<const SpanRecord*> build_span_stack(const std::vector<SpanRecord>& spans,
                                                const SpanRecord& span) {
  std::vector<const SpanRecord*> stack;
  stack.push_back(&span);
  auto parent_id = span.parent_id;
  while (parent_id != 0U) {
    const auto index = static_cast<std::size_t>(parent_id - 1U);
    if (index >= spans.size()) {
      break;
    }
    const auto& parent = spans[index];
    stack.push_back(&parent);
    parent_id = parent.parent_id;
  }
  std::reverse(stack.begin(), stack.end());
  return stack;
}

std::string stack_name(const std::vector<const SpanRecord*>& stack) {
  std::ostringstream output;
  bool first = true;
  for (const auto* span : stack) {
    if (!first) {
      output << ';';
    }
    first = false;
    output << span_label(span->name, span->fields);
  }
  return output.str();
}

}  // namespace

struct RequestProfiler::Impl {
  explicit Impl(const ObservabilityConfig& config)
      : config(config), started_wall(SystemClock::now()), started_steady(SteadyClock::now()) {}

  ObservabilityConfig config;
  std::chrono::system_clock::time_point started_wall;
  std::chrono::steady_clock::time_point started_steady;
  std::vector<SpanRecord> spans;
  std::vector<std::uint32_t> stack;
  std::map<std::string, std::string> metadata;
  std::string status{"ok"};
  bool finalized{false};
};

RequestProfiler::RequestProfiler(ObservabilityConfig config) : config_(std::move(config)) {
  if (config_.request_id.empty()) {
    config_.request_id = make_cli_request_id();
  }
  if (config_.profile_output_dir.empty() && config_.profile_mode != ProfileMode::off) {
    config_.profile_output_dir = std::filesystem::path{"build"} / "profiles";
  }
  request_id_ = config_.request_id;
  impl_ = new Impl(config_);
}

RequestProfiler::~RequestProfiler() { delete impl_; }

bool RequestProfiler::enabled() const { return impl_ != nullptr && config_.profile_mode != ProfileMode::off; }

const std::string& RequestProfiler::request_id() const { return request_id_; }

const ObservabilityConfig& RequestProfiler::config() const { return config_; }

ScopedProfileSpan RequestProfiler::scoped(std::string_view name) {
  return scoped(name, {});
}

ScopedProfileSpan RequestProfiler::scoped(std::string_view name, std::vector<ProfileField> fields) {
  if (!enabled()) {
    return {};
  }
  return ScopedProfileSpan(this, begin_span(std::string{name}, std::move(fields)));
}

void RequestProfiler::set_metadata(std::string key, std::string value) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->metadata[std::move(key)] = std::move(value);
}

void RequestProfiler::set_metadata(std::string key, std::size_t value) {
  set_metadata(std::move(key), std::to_string(value));
}

void RequestProfiler::set_status(std::string status) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->status = std::move(status);
}

std::uint32_t RequestProfiler::begin_span(std::string name, std::vector<ProfileField> fields) {
  if (impl_ == nullptr) {
    return 0U;
  }
  const auto id = static_cast<std::uint32_t>(impl_->spans.size() + 1U);
  const auto parent_id = impl_->stack.empty() ? 0U : impl_->stack.back();
  const auto depth = static_cast<std::uint32_t>(impl_->stack.size());
  impl_->spans.push_back(SpanRecord{
    id,
    parent_id,
    std::move(name),
    static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() -
                                                           impl_->started_steady)
        .count()),
    0U,
    depth,
    std::move(fields),
  });
  impl_->stack.push_back(id);
  return id;
}

void RequestProfiler::end_span(std::uint32_t span_id) {
  if (impl_ == nullptr || span_id == 0U) {
    return;
  }
  if (!impl_->stack.empty() && impl_->stack.back() == span_id) {
    impl_->stack.pop_back();
  } else {
    auto it = std::find(impl_->stack.begin(), impl_->stack.end(), span_id);
    if (it != impl_->stack.end()) {
      impl_->stack.erase(it);
    }
  }
  const auto index = static_cast<std::size_t>(span_id - 1U);
  if (index < impl_->spans.size()) {
    impl_->spans[index].end_ns = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(SteadyClock::now() -
                                                           impl_->started_steady)
        .count());
  }
}

ProfileArtifacts RequestProfiler::write_artifacts() {
  ProfileArtifacts artifacts;
  if (!enabled() || impl_ == nullptr || impl_->finalized) {
    return artifacts;
  }
  impl_->finalized = true;

  const auto output_dir = ensure_output_dir(config_, request_id_);
  std::filesystem::create_directories(output_dir);
  artifacts.output_dir = output_dir;

  std::vector<SpanSummary> summary(impl_->spans.size() + 1U);
  for (const auto& span : impl_->spans) {
    if (span.id == 0U) {
      continue;
    }
    const auto total = span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0U;
    summary[span.id].total_ns = total;
  }
  for (const auto& span : impl_->spans) {
    if (span.parent_id != 0U) {
      summary[span.parent_id].self_ns += summary[span.id].total_ns;
    }
  }
  for (std::size_t index = 1; index < summary.size(); ++index) {
    if (summary[index].total_ns >= summary[index].self_ns) {
      summary[index].self_ns = summary[index].total_ns - summary[index].self_ns;
    } else {
      summary[index].self_ns = 0U;
    }
  }

  const auto now_wall = SystemClock::now();
  const auto total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now_wall -
                                                                             impl_->started_wall)
                          .count();

  std::string prompt_tokens = "0";
  std::string generated_tokens = "0";
  if (const auto it = impl_->metadata.find("prompt_tokens"); it != impl_->metadata.end()) {
    prompt_tokens = it->second;
  }
  if (const auto it = impl_->metadata.find("generated_tokens"); it != impl_->metadata.end()) {
    generated_tokens = it->second;
  }
  std::string device = "cpu";
  if (const auto it = impl_->metadata.find("device"); it != impl_->metadata.end()) {
    device = it->second;
  }
  std::string model_dir;
  if (const auto it = impl_->metadata.find("model_dir"); it != impl_->metadata.end()) {
    model_dir = it->second;
  }

  const auto total_ms = static_cast<double>(total_ns) / 1'000'000.0;
  const auto gen_tokens_value = std::stoull(generated_tokens);
  const auto decode_ns = [&]() -> std::uint64_t {
    for (const auto& span : impl_->spans) {
      if (span.name == "request.decode") {
        return span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0U;
      }
    }
    return 0U;
  }();
  const auto prefill_ns = [&]() -> std::uint64_t {
    std::uint64_t request_total = 0;
    std::uint64_t qwen35_commit_total = 0;
    for (const auto& span : impl_->spans) {
      const auto duration = span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0U;
      if (span.name == "request.prefill") {
        request_total += duration;
      } else if (span.name.rfind("qwen35.prefill.commit", 0) == 0) {
        qwen35_commit_total += duration;
      }
    }
    return qwen35_commit_total != 0U ? qwen35_commit_total : request_total;
  }();
  const auto tokenize_ns = [&]() -> std::uint64_t {
    std::uint64_t total = 0;
    for (const auto& span : impl_->spans) {
      if (span.name == "request.tokenize") {
        total += span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0U;
      }
      if (span.name == "request.tokenize.load") {
        total += span.end_ns > span.start_ns ? span.end_ns - span.start_ns : 0U;
      }
    }
    return total;
  }();
  const auto decode_ms = static_cast<double>(decode_ns) / 1'000'000.0;
  const auto prefill_ms = static_cast<double>(prefill_ns) / 1'000'000.0;
  const auto tokenize_ms = static_cast<double>(tokenize_ns) / 1'000'000.0;
  const auto tok_s =
    decode_ns == 0U ? 0.0 : static_cast<double>(gen_tokens_value) /
      (static_cast<double>(decode_ns) / 1'000'000'000.0);

  std::ostringstream summary_text;
  summary_text << "request_id: " << request_id_ << '\n';
  if (const auto it = impl_->metadata.find("client_request_id"); it != impl_->metadata.end()) {
    summary_text << "client_request_id: " << it->second << '\n';
  }
  summary_text << "device: " << device << '\n';
  summary_text << "model_dir: " << model_dir << '\n';
  summary_text << "prompt_tokens: " << prompt_tokens << '\n';
  summary_text << "generated_tokens: " << generated_tokens << '\n';
  summary_text << "total_ms: " << total_ms << '\n';
  summary_text << "tokenize_ms: " << tokenize_ms << '\n';
  summary_text << "prefill_ms: " << prefill_ms << '\n';
  summary_text << "decode_ms: " << decode_ms << '\n';
  summary_text << "tok_s: " << tok_s << '\n';
  summary_text << '\n';
  summary_text << "top operators by self time:\n";

  std::map<std::string, std::uint64_t> op_self_ns;
  std::map<std::string, std::uint64_t> layer_self_ns;
  for (const auto& span : impl_->spans) {
    const auto index = static_cast<std::size_t>(span.id);
    if (index >= summary.size()) {
      continue;
    }
    const auto label = span_label(span.name, span.fields);
    op_self_ns[label] += summary[index].self_ns;
    for (const auto& field : span.fields) {
      if (field.key == "layer") {
        layer_self_ns["layer_" + field.value] += summary[index].self_ns;
      }
    }
  }
  std::vector<std::pair<std::string, std::uint64_t>> sorted_ops(op_self_ns.begin(),
                                                                op_self_ns.end());
  std::sort(sorted_ops.begin(), sorted_ops.end(),
            [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
  std::size_t rank = 0;
  for (const auto& entry : sorted_ops) {
    if (rank >= 10U) {
      break;
    }
    summary_text << rank + 1U << ". " << entry.first << ' ' << static_cast<double>(entry.second) /
                      1'000'000.0 << " ms\n";
    ++rank;
  }
  if (config_.per_layer && !layer_self_ns.empty()) {
    std::vector<std::pair<std::string, std::uint64_t>> sorted_layers(layer_self_ns.begin(),
                                                                     layer_self_ns.end());
    std::sort(sorted_layers.begin(), sorted_layers.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
    summary_text << '\n' << "top layers by total time:\n";
    rank = 0;
    for (const auto& entry : sorted_layers) {
      if (rank >= 10U) {
        break;
      }
      summary_text << rank + 1U << ". " << entry.first << ' '
                   << static_cast<double>(entry.second) / 1'000'000.0 << " ms\n";
      ++rank;
    }
  }

  write_file(output_dir / "summary.txt", summary_text.str());
  artifacts.wrote_summary = true;

  if (config_.emit_summary_json) {
    std::ostringstream summary_json;
    summary_json << '{';
    summary_json << "\"request_id\":\"" << json_escape(request_id_) << "\",";
    summary_json << "\"device\":\"" << json_escape(device) << "\",";
    summary_json << "\"model_dir\":\"" << json_escape(model_dir) << "\",";
    summary_json << "\"prompt_tokens\":" << prompt_tokens << ',';
    summary_json << "\"generated_tokens\":" << generated_tokens << ',';
    summary_json << "\"total_ms\":" << total_ms << ',';
    summary_json << "\"tokenize_ms\":" << tokenize_ms << ',';
    summary_json << "\"prefill_ms\":" << prefill_ms << ',';
    summary_json << "\"decode_ms\":" << decode_ms << ',';
    summary_json << "\"tok_s\":" << tok_s << ',';
    summary_json << "\"status\":\"" << json_escape(impl_->status) << "\",";
    summary_json << "\"metadata\":{";
    bool first = true;
    for (const auto& entry : impl_->metadata) {
      if (!first) {
        summary_json << ',';
      }
      first = false;
      summary_json << "\"" << json_escape(entry.first) << "\":\""
                   << json_escape(entry.second) << "\"";
    }
    summary_json << "},";
    summary_json << "\"operators\":[";
    first = true;
    for (const auto& entry : sorted_ops) {
      if (!first) {
        summary_json << ',';
      }
      first = false;
      summary_json << "{\"name\":\"" << json_escape(entry.first) << "\",\"self_ms\":"
                   << static_cast<double>(entry.second) / 1'000'000.0 << '}';
    }
    summary_json << "]}";
    write_file(output_dir / "summary.json", summary_json.str());
  }

  const auto want_trace = config_.profile_mode == ProfileMode::trace ||
                          config_.profile_mode == ProfileMode::all;
  const auto want_flamegraph = config_.profile_mode == ProfileMode::flamegraph ||
                               config_.profile_mode == ProfileMode::all;

  if (want_trace) {
    std::ostringstream trace;
    trace << "{\"traceEvents\":[";
    bool first = true;
    for (const auto& span : impl_->spans) {
      if (span.end_ns <= span.start_ns) {
        continue;
      }
      if (!first) {
        trace << ',';
      }
      first = false;
      trace << "{\"name\":\"" << json_escape(span_label(span.name, span.fields))
            << "\",\"cat\":\"operator\",\"ph\":\"X\",\"ts\":"
            << (span.start_ns / 1000U) << ",\"dur\":" << ((span.end_ns - span.start_ns) / 1000U)
            << ",\"pid\":1,\"tid\":1,\"args\":{";
      bool first_arg = true;
      for (const auto& field : span.fields) {
        if (!first_arg) {
          trace << ',';
        }
        first_arg = false;
        trace << "\"" << json_escape(field.key) << "\":\"" << json_escape(field.value) << "\"";
      }
      trace << "}}";
    }
    trace << "]}";
    write_file(output_dir / "trace.json", trace.str());
    artifacts.wrote_trace = true;
  }

  if (want_flamegraph) {
    std::ostringstream folded;
    for (const auto& span : impl_->spans) {
      if (span.end_ns <= span.start_ns) {
        continue;
      }
      const auto self_ns = span.id < summary.size() ? summary[span.id].self_ns : 0U;
      if (self_ns / 1000U < config_.min_duration_us) {
        continue;
      }
      const auto stack = build_span_stack(impl_->spans, span);
      folded << stack_name(stack) << ' ' << (self_ns / 1000U) << '\n';
    }
    write_file(output_dir / "profile.folded", folded.str());
    artifacts.wrote_folded = true;

    const auto folded_path = output_dir / "profile.folded";
    const auto svg_path = output_dir / "profile.svg";
    const auto command = "command -v flamegraph.pl >/dev/null 2>&1 && flamegraph.pl " +
                         shell_quote(folded_path.string()) + " > " + shell_quote(svg_path.string());
    if (std::system(command.c_str()) == 0) {
      artifacts.wrote_flamegraph = std::filesystem::is_regular_file(svg_path);
    }
  }

  std::ostringstream manifest;
  manifest << '{';
  manifest << "\"request_id\":\"" << json_escape(request_id_) << "\",";
  manifest << "\"client_request_id\":\""
           << json_escape(impl_->metadata.count("client_request_id") != 0U
                            ? impl_->metadata.at("client_request_id")
                            : std::string{}) << "\",";
  manifest << "\"created_at\":\"" << json_escape(format_timestamp(impl_->started_wall)) << "\",";
  manifest << "\"device\":\"" << json_escape(device) << "\",";
  manifest << "\"model_dir\":\"" << json_escape(model_dir) << "\",";
  manifest << "\"profile_mode\":\"" << json_escape(profile_mode_name(config_.profile_mode))
           << "\",";
  manifest << "\"has_trace\":" << (artifacts.wrote_trace ? "true" : "false") << ',';
  manifest << "\"has_flamegraph\":" << (artifacts.wrote_flamegraph ? "true" : "false") << ',';
  manifest << "\"status\":\"" << json_escape(impl_->status) << "\"}";
  write_file(output_dir / "manifest.json", manifest.str());

  return artifacts;
}

ScopedProfileSpan::ScopedProfileSpan(RequestProfiler* profiler, std::uint32_t span_id)
    : profiler_(profiler), span_id_(span_id) {}

ScopedProfileSpan::~ScopedProfileSpan() {
  if (profiler_ != nullptr) {
    profiler_->end_span(span_id_);
  }
}

ScopedProfileSpan::ScopedProfileSpan(ScopedProfileSpan&& other) noexcept
    : profiler_(other.profiler_), span_id_(other.span_id_) {
  other.profiler_ = nullptr;
  other.span_id_ = 0U;
}

ScopedProfileSpan& ScopedProfileSpan::operator=(ScopedProfileSpan&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  if (profiler_ != nullptr) {
    profiler_->end_span(span_id_);
  }
  profiler_ = other.profiler_;
  span_id_ = other.span_id_;
  other.profiler_ = nullptr;
  other.span_id_ = 0U;
  return *this;
}

std::string profile_mode_to_string(ProfileMode mode) { return profile_mode_name(mode); }

std::optional<ProfileMode> parse_profile_mode(std::string_view value) {
  if (value == "off") {
    return ProfileMode::off;
  }
  if (value == "summary") {
    return ProfileMode::summary;
  }
  if (value == "trace") {
    return ProfileMode::trace;
  }
  if (value == "flamegraph") {
    return ProfileMode::flamegraph;
  }
  if (value == "all") {
    return ProfileMode::all;
  }
  return std::nullopt;
}

std::string make_cli_request_id() { return make_request_id("cli"); }

std::string make_gateway_request_id() { return make_request_id("req"); }

}  // namespace toyllm
