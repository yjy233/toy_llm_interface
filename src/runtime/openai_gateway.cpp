#include "toyllm/runtime/openai_gateway.hpp"

#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/qwen35_multimodal.hpp"
#include "toyllm/runtime/reasoning_parser.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <string_view>
#include <utility>
#include <vector>

namespace toyllm {

namespace {

struct JsonValue {
  enum class Type {
    null,
    boolean,
    number,
    string,
    array,
    object,
  };

  Type type{Type::null};
  bool boolean{false};
  double number{0.0};
  std::string string;
  std::vector<JsonValue> array;
  std::vector<std::pair<std::string, JsonValue>> object;
};

class JsonParser {
 public:
  explicit JsonParser(std::string_view input) : input_(input) {}

  JsonValue parse() {
    auto value = parse_value();
    skip_ws();
    if (position_ != input_.size()) {
      fail("unexpected trailing JSON input");
    }
    return value;
  }

 private:
  JsonValue parse_value() {
    skip_ws();
    if (position_ >= input_.size()) {
      fail("unexpected end of JSON");
    }
    const char ch = input_[position_];
    if (ch == '"') {
      JsonValue value;
      value.type = JsonValue::Type::string;
      value.string = parse_string();
      return value;
    }
    if (ch == '{') {
      return parse_object();
    }
    if (ch == '[') {
      return parse_array();
    }
    if (ch == '-' || (ch >= '0' && ch <= '9')) {
      return parse_number();
    }
    if (consume_literal("true")) {
      JsonValue value;
      value.type = JsonValue::Type::boolean;
      value.boolean = true;
      return value;
    }
    if (consume_literal("false")) {
      JsonValue value;
      value.type = JsonValue::Type::boolean;
      value.boolean = false;
      return value;
    }
    if (consume_literal("null")) {
      return JsonValue{};
    }
    fail("unsupported JSON value");
  }

  JsonValue parse_object() {
    expect('{');
    JsonValue value;
    value.type = JsonValue::Type::object;
    if (consume('}')) {
      return value;
    }
    while (true) {
      const auto key = parse_string();
      expect(':');
      value.object.emplace_back(key, parse_value());
      if (consume('}')) {
        return value;
      }
      expect(',');
    }
  }

  JsonValue parse_array() {
    expect('[');
    JsonValue value;
    value.type = JsonValue::Type::array;
    if (consume(']')) {
      return value;
    }
    while (true) {
      value.array.push_back(parse_value());
      if (consume(']')) {
        return value;
      }
      expect(',');
    }
  }

  JsonValue parse_number() {
    const auto start = position_;
    if (position_ < input_.size() && input_[position_] == '-') {
      ++position_;
    }
    while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
      ++position_;
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      while (position_ < input_.size() &&
             std::isdigit(static_cast<unsigned char>(input_[position_])) != 0) {
        ++position_;
      }
    }

    JsonValue value;
    value.type = JsonValue::Type::number;
    value.number = std::stod(std::string{input_.substr(start, position_ - start)});
    return value;
  }

  std::string parse_string() {
    expect('"');
    std::string output;
    while (position_ < input_.size()) {
      const char ch = input_[position_++];
      if (ch == '"') {
        return output;
      }
      if (ch != '\\') {
        output.push_back(ch);
        continue;
      }
      if (position_ >= input_.size()) {
        fail("unterminated JSON escape");
      }
      const char escaped = input_[position_++];
      switch (escaped) {
        case '"':
        case '\\':
        case '/':
          output.push_back(escaped);
          break;
        case 'b':
          output.push_back('\b');
          break;
        case 'f':
          output.push_back('\f');
          break;
        case 'n':
          output.push_back('\n');
          break;
        case 'r':
          output.push_back('\r');
          break;
        case 't':
          output.push_back('\t');
          break;
        case 'u':
          append_unicode_escape(output);
          break;
        default:
          fail("unsupported JSON escape");
      }
    }
    fail("unterminated JSON string");
  }

  void append_unicode_escape(std::string& output) {
    if (position_ + 4U > input_.size()) {
      fail("invalid JSON unicode escape");
    }
    unsigned int codepoint = 0;
    for (int index = 0; index < 4; ++index) {
      const char ch = input_[position_++];
      codepoint <<= 4U;
      if (ch >= '0' && ch <= '9') {
        codepoint += static_cast<unsigned int>(ch - '0');
      } else if (ch >= 'a' && ch <= 'f') {
        codepoint += static_cast<unsigned int>(ch - 'a' + 10);
      } else if (ch >= 'A' && ch <= 'F') {
        codepoint += static_cast<unsigned int>(ch - 'A' + 10);
      } else {
        fail("invalid JSON unicode escape");
      }
    }
    append_utf8(output, codepoint);
  }

  static void append_utf8(std::string& output, unsigned int codepoint) {
    if (codepoint <= 0x7FU) {
      output.push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
      output.push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      output.push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      output.push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      output.push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
  }

  void skip_ws() {
    while (position_ < input_.size()) {
      const auto ch = static_cast<unsigned char>(input_[position_]);
      if (ch != ' ' && ch != '\n' && ch != '\r' && ch != '\t') {
        return;
      }
      ++position_;
    }
  }

  bool consume(char expected) {
    skip_ws();
    if (position_ < input_.size() && input_[position_] == expected) {
      ++position_;
      return true;
    }
    return false;
  }

  void expect(char expected) {
    skip_ws();
    if (position_ >= input_.size() || input_[position_] != expected) {
      std::string message = "expected JSON character ";
      message.push_back(expected);
      fail(message);
    }
    ++position_;
  }

  bool consume_literal(std::string_view literal) {
    if (input_.substr(position_, literal.size()) != literal) {
      return false;
    }
    position_ += literal.size();
    return true;
  }

  [[noreturn]] void fail(std::string_view message) const {
    std::ostringstream output;
    output << message << " at byte " << position_;
    throw std::runtime_error(output.str());
  }

  std::string_view input_;
  std::size_t position_{0};
};

const JsonValue* object_get(const JsonValue& value, std::string_view key) {
  if (value.type != JsonValue::Type::object) {
    return nullptr;
  }
  for (const auto& entry : value.object) {
    if (entry.first == key) {
      return &entry.second;
    }
  }
  return nullptr;
}

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

std::int64_t unix_time() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration_cast<std::chrono::seconds>(now).count();
}

std::string completion_id() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return "chatcmpl-" +
         std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::string text_completion_id() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return "cmpl-" +
         std::to_string(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

std::string string_value(const JsonValue* value, std::string_view fallback = {}) {
  if (value == nullptr || value->type != JsonValue::Type::string) {
    return std::string{fallback};
  }
  return value->string;
}

bool bool_value(const JsonValue* value, bool fallback = false) {
  if (value == nullptr || value->type != JsonValue::Type::boolean) {
    return fallback;
  }
  return value->boolean;
}

bool required_bool_value(const JsonValue* value, std::string_view name,
                         bool fallback = false) {
  if (value == nullptr || value->type == JsonValue::Type::null) {
    return fallback;
  }
  if (value->type != JsonValue::Type::boolean) {
    throw std::runtime_error(std::string{name} + " must be a boolean");
  }
  return value->boolean;
}

std::optional<std::size_t> size_value(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::number || value->number < 0.0) {
    return std::nullopt;
  }
  if (value->number > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::size_t>(value->number);
}

std::optional<std::uint64_t> u64_value(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::number || value->number < 0.0) {
    return std::nullopt;
  }
  if (value->number > static_cast<double>(std::numeric_limits<std::uint64_t>::max())) {
    return std::nullopt;
  }
  return static_cast<std::uint64_t>(value->number);
}

std::optional<double> double_value(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::number) {
    return std::nullopt;
  }
  return value->number;
}

struct ParsedMessageContent {
  std::string text;
  std::vector<ChatContentPart> parts;
};

ParsedMessageContent parse_message_content(const JsonValue* value) {
  if (value == nullptr || value->type == JsonValue::Type::null) {
    return {};
  }
  if (value->type == JsonValue::Type::string) {
    return ParsedMessageContent{value->string, {}};
  }
  if (value->type == JsonValue::Type::array) {
    ParsedMessageContent output;
    for (const auto& item : value->array) {
      if (item.type != JsonValue::Type::object) {
        throw std::runtime_error("message content array entries must be objects");
      }
      const auto type = string_value(object_get(item, "type"));
      if (type == "text") {
        const auto* text = object_get(item, "text");
        if (text == nullptr || text->type != JsonValue::Type::string) {
          throw std::runtime_error("message content text part requires string text");
        }
        output.text += text->string;
        output.parts.push_back(ChatContentPart{
          ChatContentPartKind::text,
          text->string,
          {},
          {},
        });
        continue;
      }
      if (type == "image_url") {
        const auto* image_url = object_get(item, "image_url");
        std::string url;
        std::string detail;
        if (image_url != nullptr && image_url->type == JsonValue::Type::string) {
          url = image_url->string;
        } else if (image_url != nullptr && image_url->type == JsonValue::Type::object) {
          const auto* url_value = object_get(*image_url, "url");
          if (url_value == nullptr || url_value->type != JsonValue::Type::string) {
            throw std::runtime_error("message content image_url part requires image_url.url");
          }
          url = url_value->string;
          detail = string_value(object_get(*image_url, "detail"));
        } else {
          throw std::runtime_error("message content image_url part requires image_url");
        }
        if (url.empty()) {
          throw std::runtime_error("message content image_url.url must not be empty");
        }
        ChatContentPart part{
          ChatContentPartKind::image_url,
          {},
          url,
          detail,
        };
        if (qwen35_image_url_is_data_url(url)) {
          auto parsed = parse_qwen35_image_data_url(url);
          if (!parsed.is_ok()) {
            throw std::runtime_error(parsed.status().message());
          }
          part.image_mime_type = std::move(parsed.value().mime_type);
          part.image_bytes = std::move(parsed.value().bytes);
          part.image_width = parsed.value().width;
          part.image_height = parsed.value().height;
        }
        part.image_fingerprint = qwen35_image_content_fingerprint(
          part.image_url, part.image_mime_type, part.detail, part.image_bytes);
        output.parts.push_back(std::move(part));
        continue;
      }
      throw std::runtime_error("unsupported message content part type: " + type);
    }
    return output;
  }
  throw std::runtime_error("message content must be a string, null, or content array");
}

struct ToolSpec {
  std::string name;
};

struct ChatRequest {
  std::string model;
  std::vector<ChatMessage> messages;
  std::size_t max_tokens{16};
  std::size_t prefill_chunk_tokens{0};
  bool stream{false};
  bool stream_include_usage{false};
  bool enable_thinking{false};
  bool enable_mtp{true};
  std::size_t mtp_draft_tokens{3};
  double mtp_p_min{0.30};
  bool cache_prompt{false};
  std::size_t cache_reuse_min_tokens{0};
  std::size_t cache_block_tokens{0};
  std::size_t cache_capacity_blocks{0};
  ReasoningFormat reasoning_format{ReasoningFormat::none};
  CpuSamplingConfig sampling;
  Device compute_device{Device::cpu()};
  std::vector<ToolSpec> tools;
  std::optional<std::string> forced_tool;
};

struct CompletionRequest {
  std::string model;
  std::string prompt;
  std::size_t max_tokens{16};
  std::size_t prefill_chunk_tokens{0};
  bool stream{false};
  bool stream_include_usage{false};
  bool enable_thinking{false};
  bool enable_mtp{true};
  std::size_t mtp_draft_tokens{3};
  double mtp_p_min{0.30};
  bool cache_prompt{false};
  std::size_t cache_reuse_min_tokens{0};
  std::size_t cache_block_tokens{0};
  std::size_t cache_capacity_blocks{0};
  CpuSamplingConfig sampling;
  Device compute_device{Device::cpu()};
};

std::vector<ToolSpec> parse_tools(const JsonValue* value) {
  std::vector<ToolSpec> tools;
  if (value == nullptr || value->type != JsonValue::Type::array) {
    return tools;
  }
  for (const auto& item : value->array) {
    const auto* function = object_get(item, "function");
    const auto* name = function == nullptr ? nullptr : object_get(*function, "name");
    if (name != nullptr && name->type == JsonValue::Type::string && !name->string.empty()) {
      tools.push_back(ToolSpec{name->string});
    }
  }
  return tools;
}

std::optional<std::string> parse_tool_choice(const JsonValue* value,
                                             const std::vector<ToolSpec>& tools) {
  if (value == nullptr || tools.empty()) {
    return std::nullopt;
  }
  if (value->type == JsonValue::Type::string) {
    if (value->string == "required") {
      return tools.front().name;
    }
    return std::nullopt;
  }
  const auto* function = object_get(*value, "function");
  const auto* name = function == nullptr ? nullptr : object_get(*function, "name");
  if (name != nullptr && name->type == JsonValue::Type::string && !name->string.empty()) {
    return name->string;
  }
  return std::nullopt;
}

std::vector<ChatMessage> parse_messages(const JsonValue* value) {
  if (value == nullptr || value->type != JsonValue::Type::array) {
    throw std::runtime_error("messages must be an array");
  }
  std::vector<ChatMessage> messages;
  for (const auto& item : value->array) {
    if (item.type != JsonValue::Type::object) {
      throw std::runtime_error("message entries must be objects");
    }
    const auto role = string_value(object_get(item, "role"));
    const auto content = parse_message_content(object_get(item, "content"));
    if (role == "system" || role == "user" || role == "assistant") {
      messages.push_back(ChatMessage{role, content.text, content.parts});
      continue;
    }
    if (role == "tool") {
      const auto tool_id = string_value(object_get(item, "tool_call_id"), "tool");
      messages.push_back(ChatMessage{"user", "Tool result " + tool_id + ":\n" + content.text});
      continue;
    }
    throw std::runtime_error("unsupported message role: " + role);
  }
  if (messages.empty()) {
    throw std::runtime_error("messages must not be empty");
  }
  return messages;
}

void parse_common_generation_options(const JsonValue& root, const OpenAIGatewayConfig& config,
                                     std::size_t& max_tokens,
                                     std::size_t& prefill_chunk_tokens,
                                     bool& stream, bool& stream_include_usage,
                                     bool& enable_thinking,
                                     bool& enable_mtp,
                                     std::size_t& mtp_draft_tokens,
                                     double& mtp_p_min,
                                     bool& cache_prompt,
                                     std::size_t& cache_reuse_min_tokens,
                                     std::size_t& cache_block_tokens,
                                     std::size_t& cache_capacity_blocks,
                                     Device& compute_device, CpuSamplingConfig& sampling) {
  max_tokens = config.default_max_tokens;
  if (const auto parsed = size_value(object_get(root, "max_tokens")); parsed.has_value()) {
    max_tokens = *parsed;
  }
  if (const auto parsed = size_value(object_get(root, "max_completion_tokens"));
      parsed.has_value()) {
    max_tokens = *parsed;
  }
  prefill_chunk_tokens = config.prefill_chunk_tokens;
  if (const auto parsed = size_value(object_get(root, "prefill_chunk_tokens"));
      parsed.has_value()) {
    prefill_chunk_tokens = *parsed;
  }
  stream = bool_value(object_get(root, "stream"), false);
  stream_include_usage = false;
  if (const auto* stream_options = object_get(root, "stream_options");
      stream_options != nullptr && stream_options->type == JsonValue::Type::object) {
    stream_include_usage =
      required_bool_value(object_get(*stream_options, "include_usage"),
                          "stream_options.include_usage", false);
  }
  enable_thinking =
    required_bool_value(object_get(root, "enable_thinking"), "enable_thinking", false);
  if (const auto* chat_template_kwargs = object_get(root, "chat_template_kwargs");
      chat_template_kwargs != nullptr) {
    if (chat_template_kwargs->type != JsonValue::Type::object) {
      throw std::runtime_error("chat_template_kwargs must be an object");
    }
    enable_thinking =
      required_bool_value(object_get(*chat_template_kwargs, "enable_thinking"),
                          "chat_template_kwargs.enable_thinking", enable_thinking);
  }
  enable_mtp = required_bool_value(object_get(root, "mtp"), "mtp",
                                   config.enable_mtp);
  mtp_draft_tokens = config.mtp_draft_tokens;
  if (const auto parsed = size_value(object_get(root, "mtp_draft_tokens"));
      parsed.has_value()) {
    mtp_draft_tokens = *parsed;
  }
  mtp_p_min = config.mtp_p_min;
  if (const auto parsed = double_value(object_get(root, "mtp_p_min"));
      parsed.has_value()) {
    mtp_p_min = *parsed;
  }
  if (!(mtp_p_min >= 0.0 && mtp_p_min <= 1.0)) {
    throw std::runtime_error("mtp_p_min must be in [0, 1]");
  }
  compute_device = config.compute_device;
  const auto device = string_value(object_get(root, "device"));
  if (device == "cpu") {
    compute_device = Device::cpu();
  } else if (device == "mps" || device == "mps:0") {
    compute_device = Device::mps();
  } else if (!device.empty()) {
    throw std::runtime_error("device must be cpu, mps, or mps:0");
  }
  if (const auto temperature = double_value(object_get(root, "temperature"));
      temperature.has_value()) {
    sampling.do_sample = true;
    sampling.temperature_set = true;
    sampling.temperature = *temperature;
  }
  if (const auto top_p = double_value(object_get(root, "top_p")); top_p.has_value()) {
    sampling.do_sample = true;
    sampling.top_p_set = true;
    sampling.top_p = *top_p;
  }
  if (const auto top_k = size_value(object_get(root, "top_k")); top_k.has_value()) {
    sampling.do_sample = true;
    sampling.top_k_set = true;
    sampling.top_k = *top_k;
  }
  if (const auto seed = u64_value(object_get(root, "seed")); seed.has_value()) {
    sampling.do_sample = true;
    sampling.seed_set = true;
    sampling.seed = *seed;
  }
  cache_prompt = required_bool_value(object_get(root, "cache_prompt"),
                                     "cache_prompt", config.cache_prompt);
  cache_reuse_min_tokens = config.cache_reuse_min_tokens;
  if (const auto parsed = size_value(object_get(root, "n_cache_reuse"));
      parsed.has_value()) {
    cache_reuse_min_tokens = *parsed;
  }
  cache_block_tokens = config.cache_block_tokens;
  if (const auto parsed = size_value(object_get(root, "cache_block_tokens"));
      parsed.has_value()) {
    cache_block_tokens = *parsed;
  }
  cache_capacity_blocks = config.cache_capacity_blocks;
  if (const auto parsed = size_value(object_get(root, "cache_capacity_blocks"));
      parsed.has_value()) {
    cache_capacity_blocks = *parsed;
  }
}

ReasoningFormat parse_request_reasoning_format(const JsonValue& root) {
  const auto* value = object_get(root, "reasoning_format");
  if (value == nullptr || value->type == JsonValue::Type::null) {
    return ReasoningFormat::none;
  }
  if (value->type != JsonValue::Type::string) {
    throw std::runtime_error("reasoning_format must be a string");
  }
  const auto parsed = parse_reasoning_format(value->string);
  if (!parsed.is_ok()) {
    throw std::runtime_error(parsed.status().message());
  }
  return parsed.value();
}

std::string parse_prompt(const JsonValue* value) {
  if (value == nullptr) {
    throw std::runtime_error("prompt is required");
  }
  if (value->type == JsonValue::Type::string) {
    if (value->string.empty()) {
      throw std::runtime_error("prompt must not be empty");
    }
    return value->string;
  }
  if (value->type == JsonValue::Type::array) {
    std::string output;
    for (const auto& item : value->array) {
      if (item.type != JsonValue::Type::string) {
        throw std::runtime_error("prompt array entries must be strings");
      }
      if (!output.empty()) {
        output.push_back('\n');
      }
      output += item.string;
    }
    if (output.empty()) {
      throw std::runtime_error("prompt must not be empty");
    }
    return output;
  }
  throw std::runtime_error("prompt must be a string or string array");
}

JsonValue parse_request_object(std::string_view body) {
  const auto root = JsonParser(body).parse();
  if (root.type != JsonValue::Type::object) {
    throw std::runtime_error("request body must be a JSON object");
  }
  return root;
}

ChatRequest parse_chat_request(std::string_view body, const OpenAIGatewayConfig& config) {
  const auto root = parse_request_object(body);
  ChatRequest request;
  request.model = string_value(object_get(root, "model"), config.model_id);
  request.messages = parse_messages(object_get(root, "messages"));
  parse_common_generation_options(root, config, request.max_tokens,
                                  request.prefill_chunk_tokens, request.stream,
                                  request.stream_include_usage,
                                  request.enable_thinking, request.enable_mtp,
                                  request.mtp_draft_tokens, request.mtp_p_min,
                                  request.cache_prompt,
                                  request.cache_reuse_min_tokens,
                                  request.cache_block_tokens,
                                  request.cache_capacity_blocks,
                                  request.compute_device,
                                  request.sampling);
  request.reasoning_format = parse_request_reasoning_format(root);
  request.tools = parse_tools(object_get(root, "tools"));
  request.forced_tool = parse_tool_choice(object_get(root, "tool_choice"), request.tools);
  return request;
}

CompletionRequest parse_completion_request(std::string_view body,
                                           const OpenAIGatewayConfig& config) {
  const auto root = parse_request_object(body);
  CompletionRequest request;
  request.model = string_value(object_get(root, "model"), config.model_id);
  request.prompt = parse_prompt(object_get(root, "prompt"));
  parse_common_generation_options(root, config, request.max_tokens,
                                  request.prefill_chunk_tokens, request.stream,
                                  request.stream_include_usage,
                                  request.enable_thinking, request.enable_mtp,
                                  request.mtp_draft_tokens, request.mtp_p_min,
                                  request.cache_prompt,
                                  request.cache_reuse_min_tokens,
                                  request.cache_block_tokens,
                                  request.cache_capacity_blocks,
                                  request.compute_device,
                                  request.sampling);
  return request;
}

bool write_all(int fd, std::string_view data) {
  std::size_t written = 0;
  while (written < data.size()) {
    const auto result = ::send(fd, data.data() + written, data.size() - written, 0);
    if (result <= 0) {
      return false;
    }
    written += static_cast<std::size_t>(result);
  }
  return true;
}

std::string http_status_text(int status) {
  switch (status) {
    case 200:
      return "OK";
    case 400:
      return "Bad Request";
    case 404:
      return "Not Found";
    case 405:
      return "Method Not Allowed";
    case 429:
      return "Too Many Requests";
    case 501:
      return "Not Implemented";
    case 500:
      return "Internal Server Error";
    default:
      return "Error";
  }
}

void send_json_response(
  int fd, int status, std::string_view body,
  const std::vector<std::pair<std::string, std::string>>& extra_headers = {}) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << ' ' << http_status_text(status) << "\r\n";
  response << "Content-Type: application/json\r\n";
  for (const auto& header : extra_headers) {
    response << header.first << ": " << header.second << "\r\n";
  }
  response << "Content-Length: " << body.size() << "\r\n";
  response << "Connection: close\r\n\r\n";
  response << body;
  (void)write_all(fd, response.str());
}

std::vector<std::pair<std::string, std::string>> request_headers(
  std::string_view request_id) {
  return {{"X-Request-Id", std::string{request_id}}};
}

std::vector<std::pair<std::string, std::string>> request_headers(
  std::string_view request_id, const CpuPromptCacheReport& cache) {
  auto headers = request_headers(request_id);
  if (cache.enabled) {
    headers.push_back({"X-Kraken-Prompt-Cache-Hit-Tokens",
                       std::to_string(cache.hit_tokens)});
    headers.push_back({"X-Kraken-Prompt-Cache-Miss-Tokens",
                       std::to_string(cache.miss_tokens)});
    headers.push_back({"X-Kraken-Prompt-Cache-Committed-Tokens",
                       std::to_string(cache.committed_tokens)});
  }
  return headers;
}

std::string join_size_values(const std::vector<std::size_t>& values) {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

std::vector<std::pair<std::string, std::string>> request_headers(
  std::string_view request_id, const CpuPromptCacheReport& cache,
  const CpuMtpReport& mtp) {
  auto headers = request_headers(request_id, cache);
  if (mtp.available) {
    headers.push_back({"X-Kraken-MTP-Enabled", mtp.enabled ? "1" : "0"});
    headers.push_back({"X-Kraken-MTP-Layers", std::to_string(mtp.layers)});
    headers.push_back({"X-Kraken-MTP-Draft-Tokens", std::to_string(mtp.draft_tokens)});
    headers.push_back({"X-Kraken-MTP-P-Min", std::to_string(mtp.p_min)});
    headers.push_back({"X-Kraken-MTP-Drafted-Tokens", std::to_string(mtp.drafted_tokens)});
    headers.push_back({"X-Kraken-MTP-Accepted-Tokens", std::to_string(mtp.accepted_tokens)});
    headers.push_back({"X-Kraken-MTP-Verify-Steps", std::to_string(mtp.verify_steps)});
    headers.push_back({"X-Kraken-MTP-Confidence-Stops",
                       std::to_string(mtp.confidence_stops)});
    headers.push_back({"X-Kraken-MTP-Adaptive-Budget",
                       std::to_string(mtp.adaptive_budget)});
    headers.push_back({"X-Kraken-MTP-Adaptive-Changes",
                       std::to_string(mtp.adaptive_changes)});
    if (!mtp.verified_by_position.empty()) {
      headers.push_back({"X-Kraken-MTP-Verified-By-Position",
                         join_size_values(mtp.verified_by_position)});
    }
    if (!mtp.accepted_by_position.empty()) {
      headers.push_back({"X-Kraken-MTP-Accepted-By-Position",
                         join_size_values(mtp.accepted_by_position)});
    }
    if (!mtp.enabled && !mtp.disabled_reason.empty()) {
      headers.push_back({"X-Kraken-MTP-Disabled-Reason", mtp.disabled_reason});
    }
  }
  return headers;
}

void send_text_response(int fd, int status, std::string_view content_type,
                        std::string_view body,
                        const std::vector<std::pair<std::string, std::string>>& extra_headers = {}) {
  std::ostringstream response;
  response << "HTTP/1.1 " << status << ' ' << http_status_text(status) << "\r\n";
  response << "Content-Type: " << content_type << "\r\n";
  for (const auto& header : extra_headers) {
    response << header.first << ": " << header.second << "\r\n";
  }
  response << "Content-Length: " << body.size() << "\r\n";
  response << "Connection: close\r\n\r\n";
  response << body;
  (void)write_all(fd, response.str());
}

void send_sse_headers(int fd,
                     const std::vector<std::pair<std::string, std::string>>& extra_headers = {}) {
  const std::string headers =
    [&]() {
      std::ostringstream output;
      output << "HTTP/1.1 200 OK\r\n"
             << "Content-Type: text/event-stream\r\n"
             << "Cache-Control: no-cache\r\n";
      for (const auto& header : extra_headers) {
        output << header.first << ": " << header.second << "\r\n";
      }
      output << "Connection: close\r\n\r\n";
      return output.str();
    }();
  (void)write_all(fd, headers);
}

void send_sse(int fd, std::string_view json) {
  (void)write_all(fd, "data: ");
  (void)write_all(fd, json);
  (void)write_all(fd, "\n\n");
}

std::string error_body(std::string_view message, std::string_view type = "invalid_request_error") {
  std::ostringstream output;
  output << "{\"error\":{\"message\":\"" << json_escape(message) << "\",\"type\":\""
         << json_escape(type) << "\",\"param\":null,\"code\":null}}";
  return output.str();
}

std::atomic_bool& gateway_inference_busy() {
  static std::atomic_bool busy{false};
  return busy;
}

class GatewayInferencePermit {
 public:
  GatewayInferencePermit() {
    bool expected = false;
    acquired_ = gateway_inference_busy().compare_exchange_strong(expected, true);
  }

  GatewayInferencePermit(const GatewayInferencePermit&) = delete;
  GatewayInferencePermit& operator=(const GatewayInferencePermit&) = delete;

  ~GatewayInferencePermit() {
    if (acquired_) {
      gateway_inference_busy().store(false);
    }
  }

  [[nodiscard]] bool acquired() const { return acquired_; }

 private:
  bool acquired_{false};
};

void send_inference_busy_response(int fd, std::string_view request_id) {
  send_json_response(
    fd, 429,
    error_body("another inference request is already running; retry later",
               "rate_limit_error"),
    {{"X-Request-Id", std::string{request_id}}, {"Retry-After", "1"}});
}

std::optional<std::pair<int, std::string>> chat_multimodal_error(
  const OpenAIGatewayConfig& config, const ChatRequest& request) {
  if (!chat_messages_have_image_content(request.messages)) {
    return std::nullopt;
  }
  if (config.mmproj_path.empty()) {
    return std::pair<int, std::string>{
      400,
      "image input requires starting the gateway with --mmproj pointing to the "
      "Qwen3.5 conditional mmproj GGUF",
    };
  }
  const auto metadata = load_qwen35_mmproj_metadata(config.mmproj_path);
  if (!metadata.is_ok()) {
    return std::pair<int, std::string>{
      400,
      "failed to load --mmproj: " + metadata.status().message(),
    };
  }
  if (!qwen35_mmproj_is_qwen3vl_merger(metadata.value())) {
    return std::pair<int, std::string>{
      400,
      "Qwen3.5 image input requires a qwen3vl_merger mmproj; got projector_type=" +
        metadata.value().projector_type + " vision_projector_type=" +
        metadata.value().vision_projector_type,
    };
  }
  const auto prompt_plan = plan_qwen35_multimodal_prompt(
    metadata.value(), request.messages, true, request.enable_thinking);
  if (!prompt_plan.is_ok()) {
    return std::pair<int, std::string>{
      400,
      "failed to plan Qwen3.5 multimodal prompt: " +
        prompt_plan.status().message(),
    };
  }
  for (const auto& message : request.messages) {
    for (const auto& part : message.content_parts) {
      if (part.kind != ChatContentPartKind::image_url || part.image_bytes.empty()) {
        continue;
      }
      Qwen35ImageDataUrl image;
      image.mime_type = part.image_mime_type;
      image.bytes = part.image_bytes;
      image.width = part.image_width;
      image.height = part.image_height;
      const auto preprocessed =
        preprocess_qwen35_image_for_vision(metadata.value(), image);
      if (!preprocessed.is_ok()) {
        return std::pair<int, std::string>{
          400,
          "failed to preprocess Qwen3.5 image input: " +
            preprocessed.status().message(),
        };
      }
    }
  }
  return std::nullopt;
}

std::string models_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output << "{\"object\":\"list\",\"data\":[{\"id\":\"" << json_escape(config.model_id)
         << "\",\"object\":\"model\",\"created\":0,\"owned_by\":\"kraken-infer\"}]}";
  return output.str();
}

std::string chat_page_config_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output << "{\"model\":\"" << json_escape(config.model_id) << "\",\"device\":\""
         << json_escape(config.compute_device.to_string()) << "\",\"max_new_tokens\":"
         << config.default_max_tokens << ",\"has_mmproj\":"
         << (config.mmproj_path.empty() ? "false" : "true") << '}';
  return output.str();
}

std::filesystem::path profile_root_dir(const OpenAIGatewayConfig& config);

std::string profile_page_config_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output << "{\"profile_dir\":\"" << json_escape(profile_root_dir(config).string())
         << "\",\"profile_mode\":\""
         << json_escape(profile_mode_to_string(config.observability.profile_mode)) << "\"}";
  return output.str();
}

std::optional<std::filesystem::path> find_web_asset(std::string_view filename) {
  auto directory = std::filesystem::current_path();
  for (int depth = 0; depth < 8; ++depth) {
    const auto candidate = directory / "web" / std::filesystem::path{std::string{filename}};
    if (std::filesystem::is_regular_file(candidate)) {
      return candidate;
    }
    if (directory == directory.root_path()) {
      break;
    }
    directory = directory.parent_path();
  }
  return std::nullopt;
}

std::optional<std::string> read_web_asset(std::string_view filename) {
  const auto path = find_web_asset(filename);
  if (!path.has_value()) {
    return std::nullopt;
  }

  std::ifstream input(*path, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

void send_web_asset(int fd, std::string_view filename, std::string_view content_type) {
  const auto body = read_web_asset(filename);
  if (!body.has_value()) {
    send_json_response(fd, 500,
                       error_body(std::string{"missing web asset: "} + std::string{filename},
                                  "server_error"));
    return;
  }
  send_text_response(fd, 200, content_type, *body);
}

std::filesystem::path profile_root_dir(const OpenAIGatewayConfig& config) {
  if (!config.observability.profile_output_dir.empty()) {
    return config.observability.profile_output_dir;
  }
  return std::filesystem::path{"build"} / "profiles";
}

std::optional<std::string> read_text_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return std::nullopt;
  }
  std::ostringstream output;
  output << input.rdbuf();
  return output.str();
}

std::optional<JsonValue> read_json_object_file(const std::filesystem::path& path) {
  const auto body = read_text_file(path);
  if (!body.has_value()) {
    return std::nullopt;
  }
  try {
    return JsonParser(*body).parse();
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

std::string profile_index_body(const OpenAIGatewayConfig& config) {
  const auto root = profile_root_dir(config);
  std::vector<std::filesystem::path> request_dirs;
  if (std::filesystem::exists(root)) {
    for (const auto& entry : std::filesystem::directory_iterator(root)) {
      if (entry.is_directory()) {
        request_dirs.push_back(entry.path());
      }
    }
  }
  std::sort(request_dirs.begin(), request_dirs.end(),
            [](const auto& lhs, const auto& rhs) {
              return lhs.filename().string() > rhs.filename().string();
            });

  std::ostringstream output;
  output << "{\"profiles\":[";
  bool first = true;
  for (const auto& dir : request_dirs) {
    const auto manifest = read_json_object_file(dir / "manifest.json");
    if (!manifest.has_value() || manifest->type != JsonValue::Type::object) {
      continue;
    }
    const auto* request_id = object_get(*manifest, "request_id");
    if (request_id == nullptr || request_id->type != JsonValue::Type::string) {
      continue;
    }
    if (!first) {
      output << ',';
    }
    first = false;
    const auto* created_at = object_get(*manifest, "created_at");
    const auto* device = object_get(*manifest, "device");
    const auto* model_dir = object_get(*manifest, "model_dir");
    const auto* profile_mode = object_get(*manifest, "profile_mode");
    const auto* status = object_get(*manifest, "status");
    const auto* has_trace = object_get(*manifest, "has_trace");
    const auto* has_flamegraph = object_get(*manifest, "has_flamegraph");
    output << '{';
    output << "\"request_id\":\"" << json_escape(request_id->string) << "\",";
    output << "\"created_at\":\""
           << json_escape(created_at != nullptr && created_at->type == JsonValue::Type::string
                           ? created_at->string
                           : std::string{}) << "\",";
    output << "\"device\":\""
           << json_escape(device != nullptr && device->type == JsonValue::Type::string
                           ? device->string
                           : std::string{}) << "\",";
    output << "\"model_dir\":\""
           << json_escape(model_dir != nullptr && model_dir->type == JsonValue::Type::string
                           ? model_dir->string
                           : std::string{}) << "\",";
    output << "\"profile_mode\":\""
           << json_escape(profile_mode != nullptr && profile_mode->type == JsonValue::Type::string
                           ? profile_mode->string
                           : std::string{}) << "\",";
    output << "\"status\":\""
           << json_escape(status != nullptr && status->type == JsonValue::Type::string
                           ? status->string
                           : std::string{}) << "\",";
    output << "\"has_trace\":" << (has_trace != nullptr && has_trace->type == JsonValue::Type::boolean &&
                                     has_trace->boolean ? "true" : "false") << ',';
    output << "\"has_flamegraph\":"
           << (has_flamegraph != nullptr && has_flamegraph->type == JsonValue::Type::boolean &&
               has_flamegraph->boolean ? "true" : "false");
    output << '}';
  }
  output << "]}";
  return output.str();
}

std::optional<std::string> profile_artifact_body(const OpenAIGatewayConfig& config,
                                                 std::string_view request_id,
                                                 std::string_view filename) {
  const auto path = profile_root_dir(config) / std::filesystem::path{std::string{request_id}} /
                    std::filesystem::path{std::string{filename}};
  return read_text_file(path);
}

std::string openapi_body(const OpenAIGatewayConfig& config) {
  std::ostringstream output;
  output
    << "{\"openapi\":\"3.0.3\",\"info\":{\"title\":\"kraken-infer OpenAI-compatible "
       "Gateway\",\"version\":\"0.1.0\"},\"servers\":[{\"url\":\"http://"
    << json_escape(config.host) << ':' << config.port
    << "\"}],\"paths\":{"
       "\"/health\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Gateway health\"}}}},"
       "\"/chat_page\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Browser chat "
       "page\",\"content\":{\"text/html\":{}}}}}},"
       "\"/chat_page/config\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Browser chat "
       "page config\"}}}},"
       "\"/profile_page\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Profile viewer "
       "page\",\"content\":{\"text/html\":{}}}}}},"
       "\"/profile_page/config\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Profile "
       "viewer config\"}}}},"
       "\"/profiles/index.json\":{\"get\":{\"responses\":{\"200\":{\"description\":\"Recent "
       "profile request index\"}}}},"
       "\"/profiles/{request_id}/{artifact}\":{\"get\":{\"parameters\":[{\"name\":\"request_id\","
       "\"in\":\"path\",\"required\":true,\"schema\":{\"type\":\"string\"}},{\"name\":\"artifact\","
       "\"in\":\"path\",\"required\":true,\"schema\":{\"type\":\"string\",\"enum\":[\"manifest.json\","
       "\"summary.json\",\"summary.txt\",\"trace.json\",\"profile.folded\",\"profile.svg\"]}}],"
       "\"responses\":{\"200\":{\"description\":\"Profile artifact\"},\"404\":{\"description\":"
       "\"Profile artifact not found\"}}}},"
       "\"/v1/models\":{\"get\":{\"responses\":{\"200\":{\"description\":\"List models\"}}}},"
       "\"/v1/completions\":{\"post\":{\"requestBody\":{\"required\":true,\"content\":{"
       "\"application/json\":{\"schema\":{\"type\":\"object\",\"properties\":{\"model\":"
       "{\"type\":\"string\"},\"prompt\":{\"oneOf\":[{\"type\":\"string\"},{\"type\":\"array\","
       "\"items\":{\"type\":\"string\"}}]},\"max_tokens\":{\"type\":\"integer\"},"
       "\"prefill_chunk_tokens\":{\"type\":\"integer\"},"
       "\"temperature\":{\"type\":\"number\"},\"top_k\":{\"type\":\"integer\"},"
       "\"top_p\":{\"type\":\"number\"},"
       "\"stream\":{\"type\":\"boolean\"},\"stream_options\":{\"type\":\"object\","
       "\"properties\":{\"include_usage\":{\"type\":\"boolean\"}},"
       "\"additionalProperties\":true},\"enable_thinking\":{\"type\":\"boolean\"},"
       "\"chat_template_kwargs\":{\"type\":\"object\"},"
       "\"mtp\":{\"type\":\"boolean\"},\"mtp_draft_tokens\":{\"type\":\"integer\"},"
       "\"mtp_p_min\":{\"type\":\"number\"},"
       "\"cache_prompt\":{\"type\":\"boolean\"},\"n_cache_reuse\":{\"type\":\"integer\"},"
       "\"cache_block_tokens\":{\"type\":\"integer\"},\"cache_capacity_blocks\":{\"type\":\"integer\"},"
       "\"seed\":{\"type\":\"integer\"},\"device\":{\"type\":\"string\",\"enum\":[\"cpu\","
       "\"mps\",\"mps:0\"]}},\"required\":[\"prompt\"]}}}},\"responses\":{\"200\":"
       "{\"description\":\"Text completion or SSE stream\",\"content\":{\"application/json\":"
       "{\"schema\":{\"type\":\"object\",\"properties\":{\"usage\":{\"$ref\":"
       "\"#/components/schemas/Usage\"}}}}}}}}},"
       "\"/v1/chat/completions\":{\"post\":{\"requestBody\":{\"required\":true,\"content\":{"
       "\"application/json\":{\"schema\":{\"type\":\"object\",\"properties\":{\"model\":"
       "{\"type\":\"string\"},\"messages\":{\"type\":\"array\",\"items\":{\"type\":\"object\","
       "\"properties\":{\"role\":{\"type\":\"string\"},\"content\":{\"oneOf\":[{\"type\":\"string\"},"
       "{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"type\":{\"type\":\"string\","
       "\"enum\":[\"text\",\"image_url\"]},\"text\":{\"type\":\"string\"},\"image_url\":{\"oneOf\":["
       "{\"type\":\"string\"},{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"},"
       "\"detail\":{\"type\":\"string\"}},\"required\":[\"url\"]}]}}}}]}}}},\"tools\":{\"type\":\"array\"},"
       "\"tool_choice\":{},\"max_tokens\":{\"type\":\"integer\"},\"max_completion_tokens\":"
       "{\"type\":\"integer\"},\"prefill_chunk_tokens\":{\"type\":\"integer\"},"
       "\"temperature\":{\"type\":\"number\"},\"top_k\":"
       "{\"type\":\"integer\"},\"top_p\":{\"type\":"
       "\"number\"},\"stream\":{\"type\":\"boolean\"},\"stream_options\":{\"type\":\"object\","
       "\"properties\":{\"include_usage\":{\"type\":\"boolean\"}},"
       "\"additionalProperties\":true},\"enable_thinking\":{\"type\":\"boolean\"},"
       "\"chat_template_kwargs\":{\"type\":\"object\",\"properties\":{\"enable_thinking\":"
       "{\"type\":\"boolean\"}},\"additionalProperties\":true},"
       "\"reasoning_format\":{\"type\":\"string\",\"enum\":[\"none\",\"auto\",\"deepseek\"]},"
       "\"mtp\":{\"type\":\"boolean\"},\"mtp_draft_tokens\":{\"type\":\"integer\"},"
       "\"mtp_p_min\":{\"type\":\"number\"},"
       "\"cache_prompt\":{\"type\":\"boolean\"},\"n_cache_reuse\":{\"type\":\"integer\"},"
       "\"cache_block_tokens\":{\"type\":\"integer\"},\"cache_capacity_blocks\":{\"type\":\"integer\"},"
       "\"seed\":{\"type\":\"integer\"},\"device\":{\"type\":\"string\",\"enum\":[\"cpu\","
       "\"mps\",\"mps:0\"]}},\"required\":[\"messages\"]}}}},\"responses\":{\"200\":"
       "{\"description\":\"Chat completion, tool call, or SSE stream\",\"content\":"
       "{\"application/json\":{\"schema\":{\"type\":\"object\",\"properties\":{\"usage\":"
       "{\"$ref\":\"#/components/schemas/Usage\"}}}}}}}}}},\"components\":{\"schemas\":"
       "{\"Usage\":{\"type\":\"object\",\"properties\":{\"prompt_tokens\":{\"type\":\"integer\"},"
       "\"completion_tokens\":{\"type\":\"integer\"},\"total_tokens\":{\"type\":\"integer\"},"
       "\"prompt_tokens_details\":{\"type\":\"object\",\"properties\":{\"cached_tokens\":"
       "{\"type\":\"integer\",\"description\":\"Number of prompt tokens served from the "
       "cross-request KV cache.\"}}}},\"required\":[\"prompt_tokens\",\"completion_tokens\","
       "\"total_tokens\",\"prompt_tokens_details\"]}}}}";
  return output.str();
}

void append_usage(std::ostringstream& output, std::size_t prompt_tokens,
                  std::size_t completion_tokens, std::size_t cached_tokens) {
  cached_tokens = std::min(cached_tokens, prompt_tokens);
  output << "\"usage\":{\"prompt_tokens\":" << prompt_tokens
         << ",\"completion_tokens\":" << completion_tokens
         << ",\"total_tokens\":" << (prompt_tokens + completion_tokens)
         << ",\"prompt_tokens_details\":{\"cached_tokens\":" << cached_tokens
         << "}}";
}

std::string completion_body(std::string_view id, std::int64_t created, std::string_view model,
                            std::string_view text, std::string_view finish_reason,
                            std::size_t prompt_tokens, std::size_t completion_tokens,
                            std::size_t cached_tokens) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"text_completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(text)
         << "\",\"finish_reason\":\"" << json_escape(finish_reason)
         << "\"}],";
  append_usage(output, prompt_tokens, completion_tokens, cached_tokens);
  output << '}';
  return output.str();
}

std::string chat_completion_body(std::string_view id, std::int64_t created,
                                 std::string_view model,
                                 const ReasoningMessage& message,
                                 std::string_view finish_reason,
                                 std::size_t prompt_tokens,
                                 std::size_t completion_tokens,
                                 std::size_t cached_tokens) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\"";
  if (!message.reasoning_content.empty()) {
    output << ",\"reasoning_content\":\"" << json_escape(message.reasoning_content) << '"';
  }
  output << ",\"content\":\"" << json_escape(message.content)
         << "\"},\"finish_reason\":\"" << json_escape(finish_reason)
         << "\"}],";
  append_usage(output, prompt_tokens, completion_tokens, cached_tokens);
  output << '}';
  return output.str();
}

std::string tool_call_body(std::string_view id, std::int64_t created, std::string_view model,
                           std::string_view tool_name) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\","
            "\"content\":null,\"tool_calls\":[{\"id\":\"call_0\",\"type\":\"function\","
            "\"function\":{\"name\":\""
         << json_escape(tool_name)
         << "\",\"arguments\":\"{}\"}}]},\"finish_reason\":\"tool_calls\"}],";
  append_usage(output, 0, 0, 0);
  output << '}';
  return output.str();
}

std::string completion_stream_chunk(std::string_view id, std::int64_t created,
                                    std::string_view model, std::string_view text,
                                    std::string_view finish_reason) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"text_completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"text\":\"" << json_escape(text)
         << "\",\"finish_reason\":";
  if (finish_reason.empty()) {
    output << "null";
  } else {
    output << '"' << json_escape(finish_reason) << '"';
  }
  output << "}]}";
  return output.str();
}

std::string completion_stream_usage_chunk(std::string_view id,
                                          std::int64_t created,
                                          std::string_view model,
                                          std::size_t prompt_tokens,
                                          std::size_t completion_tokens,
                                          std::size_t cached_tokens) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"text_completion\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[],";
  append_usage(output, prompt_tokens, completion_tokens, cached_tokens);
  output << '}';
  return output.str();
}

std::string stream_chunk(std::string_view id, std::int64_t created, std::string_view model,
                         std::string_view content_delta,
                         std::string_view reasoning_delta, bool role,
                         std::string_view finish_reason) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"delta\":{";
  bool wrote = false;
  if (role) {
    output << "\"role\":\"assistant\"";
    wrote = true;
  }
  if (!reasoning_delta.empty()) {
    if (wrote) {
      output << ',';
    }
    output << "\"reasoning_content\":\"" << json_escape(reasoning_delta) << "\"";
    wrote = true;
  }
  if (!content_delta.empty()) {
    if (wrote) {
      output << ',';
    }
    output << "\"content\":\"" << json_escape(content_delta) << "\"";
  }
  output << "},\"finish_reason\":";
  if (finish_reason.empty()) {
    output << "null";
  } else {
    output << '"' << json_escape(finish_reason) << '"';
  }
  output << "}]}";
  return output.str();
}

std::string stream_usage_chunk(std::string_view id, std::int64_t created,
                               std::string_view model,
                               std::size_t prompt_tokens,
                               std::size_t completion_tokens,
                               std::size_t cached_tokens) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id)
         << "\",\"object\":\"chat.completion.chunk\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[],";
  append_usage(output, prompt_tokens, completion_tokens, cached_tokens);
  output << '}';
  return output.str();
}

std::string tool_stream_chunk(std::string_view id, std::int64_t created, std::string_view model,
                              std::string_view tool_name, bool finish) {
  std::ostringstream output;
  output << "{\"id\":\"" << json_escape(id) << "\",\"object\":\"chat.completion.chunk\",";
  output << "\"created\":" << created << ",\"model\":\"" << json_escape(model)
         << "\",\"choices\":[{\"index\":0,\"delta\":";
  if (finish) {
    output << "{}";
  } else {
    output << "{\"role\":\"assistant\",\"tool_calls\":[{\"index\":0,\"id\":\"call_0\","
              "\"type\":\"function\",\"function\":{\"name\":\""
           << json_escape(tool_name) << "\",\"arguments\":\"{}\"}}]}";
  }
  output << ",\"finish_reason\":";
  output << (finish ? "\"tool_calls\"" : "null") << "}]}";
  return output.str();
}

struct HttpRequest {
  std::string method;
  std::string path;
  std::string query;
  std::string body;
  std::unordered_map<std::string, std::string> headers;

  [[nodiscard]] std::string header(std::string_view key) const {
    auto normalized = std::string{key};
    for (auto& ch : normalized) {
      ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    const auto it = headers.find(normalized);
    if (it == headers.end()) {
      return {};
    }
    return it->second;
  }
};

std::string lower_ascii(std::string value) {
  for (auto& ch : value) {
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

HttpRequest read_http_request(int fd) {
  std::string buffer;
  char chunk[4096];
  while (buffer.find("\r\n\r\n") == std::string::npos) {
    const auto count = ::recv(fd, chunk, sizeof(chunk), 0);
    if (count <= 0) {
      throw std::runtime_error("failed to read HTTP request");
    }
    buffer.append(chunk, static_cast<std::size_t>(count));
    if (buffer.size() > 1024U * 1024U) {
      throw std::runtime_error("HTTP headers exceed limit");
    }
  }

  const auto header_end = buffer.find("\r\n\r\n");
  const auto header_text = buffer.substr(0, header_end);
  std::istringstream header_stream(header_text);
  std::string request_line;
  std::getline(header_stream, request_line);
  if (!request_line.empty() && request_line.back() == '\r') {
    request_line.pop_back();
  }
  std::istringstream line_stream(request_line);
  HttpRequest request;
  line_stream >> request.method >> request.path;
  if (request.method.empty() || request.path.empty()) {
    throw std::runtime_error("invalid HTTP request line");
  }

  std::size_t content_length = 0;
  std::string line;
  while (std::getline(header_stream, line)) {
    if (!line.empty() && line.back() == '\r') {
      line.pop_back();
    }
    const auto colon = line.find(':');
    if (colon == std::string::npos) {
      continue;
    }
    auto name = lower_ascii(line.substr(0, colon));
    auto value = line.substr(colon + 1U);
    while (!value.empty() && value.front() == ' ') {
      value.erase(value.begin());
    }
    if (name == "content-length") {
      content_length = static_cast<std::size_t>(std::stoull(value));
    }
    request.headers.emplace(std::move(name), std::move(value));
  }

  const auto body_start = header_end + 4U;
  while (buffer.size() < body_start + content_length) {
    const auto count = ::recv(fd, chunk, sizeof(chunk), 0);
    if (count <= 0) {
      throw std::runtime_error("failed to read HTTP body");
    }
    buffer.append(chunk, static_cast<std::size_t>(count));
    if (buffer.size() > 8U * 1024U * 1024U) {
      throw std::runtime_error("HTTP body exceeds limit");
    }
  }
  request.body = buffer.substr(body_start, content_length);
  const auto query = request.path.find('?');
  if (query != std::string::npos) {
    request.query = request.path.substr(query + 1U);
    request.path.resize(query);
  }
  return request;
}

void send_forced_tool_response(int fd, const ChatRequest& request, std::string_view request_id) {
  const auto id = completion_id();
  const auto created = unix_time();
  const auto tool_name = *request.forced_tool;
  if (!request.stream) {
    send_json_response(fd, 200, tool_call_body(id, created, request.model, tool_name),
                       {{"X-Request-Id", std::string{request_id}}});
    return;
  }
  send_sse_headers(fd, {{"X-Request-Id", std::string{request_id}}});
  send_sse(fd, tool_stream_chunk(id, created, request.model, tool_name, false));
  send_sse(fd, tool_stream_chunk(id, created, request.model, tool_name, true));
  if (request.stream_include_usage) {
    send_sse(fd, stream_usage_chunk(id, created, request.model, 0, 0, 0));
  }
  send_sse(fd, "[DONE]");
}

void send_completion_response(int fd, const OpenAIGatewayConfig& config,
                              const CompletionRequest& request, std::string_view request_id,
                              std::string_view client_request_id,
                              const ObservabilityConfig& observability) {
  CpuGenerationRequest generation;
  generation.model_dir = config.model_dir;
  generation.prompt = request.prompt;
  generation.max_new_tokens = request.max_tokens;
  generation.prefill_chunk_tokens = request.prefill_chunk_tokens;
  generation.enable_thinking = request.enable_thinking;
  generation.enable_mtp = request.enable_mtp;
  generation.mtp_draft_tokens = request.mtp_draft_tokens;
  generation.mtp_p_min = request.mtp_p_min;
  generation.cache_prompt = request.cache_prompt;
  generation.cache_reuse_min_tokens = request.cache_reuse_min_tokens;
  generation.cache_block_tokens = request.cache_block_tokens;
  generation.cache_capacity_blocks = request.cache_capacity_blocks;
  generation.compute_device = request.compute_device;
  generation.sampling = request.sampling;
  generation.observability = observability;
  generation.observability.request_id = std::string{request_id};
  generation.observability.client_request_id = std::string{client_request_id};

  const auto id = text_completion_id();
  const auto created = unix_time();
  if (!request.stream) {
    const auto result = generate_cpu(generation);
    if (!result.is_ok()) {
      send_json_response(fd, 500, error_body(result.status().message(), "server_error"),
                         {{"X-Request-Id", std::string{request_id}}});
      return;
    }
    send_json_response(fd, 200,
                       completion_body(id, created, request.model, result.value().text,
                                       result.value().finish_reason,
                                       result.value().prompt_tokens,
                                       result.value().generated_tokens,
                                       result.value().prompt_cache.hit_tokens),
                       request_headers(request_id, result.value().prompt_cache,
                                       result.value().mtp));
    return;
  }

  send_sse_headers(fd, {{"X-Request-Id", std::string{request_id}}});
  generation.stream_token = [&](std::string_view token) {
    send_sse(fd, completion_stream_chunk(id, created, request.model, token, {}));
  };
  const auto result = generate_cpu(generation);
  if (!result.is_ok()) {
    send_sse(fd, error_body(result.status().message(), "server_error"));
    send_sse(fd, "[DONE]");
    return;
  }
  send_sse(fd, completion_stream_chunk(id, created, request.model, {},
                                       result.value().finish_reason));
  if (request.stream_include_usage) {
    send_sse(fd, completion_stream_usage_chunk(
                   id, created, request.model, result.value().prompt_tokens,
                   result.value().generated_tokens,
                   result.value().prompt_cache.hit_tokens));
  }
  send_sse(fd, "[DONE]");
}

void send_chat_response(int fd, const OpenAIGatewayConfig& config, const ChatRequest& request,
                        std::string_view request_id, std::string_view client_request_id,
                        const ObservabilityConfig& observability) {
  if (const auto multimodal_error = chat_multimodal_error(config, request);
      multimodal_error.has_value()) {
    const auto error_type = multimodal_error->first == 501 ? "not_implemented_error"
                                                           : "invalid_request_error";
    send_json_response(fd, multimodal_error->first,
                       error_body(multimodal_error->second, error_type),
                       {{"X-Request-Id", std::string{request_id}}});
    return;
  }
  if (request.forced_tool.has_value()) {
    send_forced_tool_response(fd, request, request_id);
    return;
  }

  CpuGenerationRequest generation;
  generation.model_dir = config.model_dir;
  generation.messages = request.messages;
  generation.mmproj_path = config.mmproj_path;
  generation.max_new_tokens = request.max_tokens;
  generation.prefill_chunk_tokens = request.prefill_chunk_tokens;
  generation.enable_thinking = request.enable_thinking;
  generation.enable_mtp = request.enable_mtp;
  generation.mtp_draft_tokens = request.mtp_draft_tokens;
  generation.mtp_p_min = request.mtp_p_min;
  generation.cache_prompt = request.cache_prompt;
  generation.cache_reuse_min_tokens = request.cache_reuse_min_tokens;
  generation.cache_block_tokens = request.cache_block_tokens;
  generation.cache_capacity_blocks = request.cache_capacity_blocks;
  generation.compute_device = request.compute_device;
  generation.sampling = request.sampling;
  generation.observability = observability;
  generation.observability.request_id = std::string{request_id};
  generation.observability.client_request_id = std::string{client_request_id};

  const auto id = completion_id();
  const auto created = unix_time();
  if (!request.stream) {
    const auto result = generate_cpu(generation);
    if (!result.is_ok()) {
      send_json_response(fd, 500, error_body(result.status().message(), "server_error"),
                         {{"X-Request-Id", std::string{request_id}}});
      return;
    }
    const auto message = split_reasoning_content(result.value().text,
                                                 request.enable_thinking,
                                                 request.reasoning_format);
    send_json_response(fd, 200,
                       chat_completion_body(id, created, request.model, message,
                                            result.value().finish_reason,
                                            result.value().prompt_tokens,
                                            result.value().generated_tokens,
                                            result.value().prompt_cache.hit_tokens),
                       request_headers(request_id, result.value().prompt_cache,
                                       result.value().mtp));
    return;
  }

  send_sse_headers(fd, {{"X-Request-Id", std::string{request_id}}});
  send_sse(fd, stream_chunk(id, created, request.model, {}, {}, true, {}));
  ReasoningStreamParser reasoning_parser{request.enable_thinking,
                                         request.reasoning_format};
  generation.stream_token = [&](std::string_view token) {
    for (const auto& delta : reasoning_parser.push(token)) {
      if (delta.kind == ReasoningDelta::Kind::reasoning) {
        send_sse(fd, stream_chunk(id, created, request.model, {}, delta.text,
                                  false, {}));
      } else {
        send_sse(fd, stream_chunk(id, created, request.model, delta.text, {},
                                  false, {}));
      }
    }
  };
  const auto result = generate_cpu(generation);
  if (!result.is_ok()) {
    send_sse(fd, error_body(result.status().message(), "server_error"));
    send_sse(fd, "[DONE]");
    return;
  }
  for (const auto& delta : reasoning_parser.finish()) {
    if (delta.kind == ReasoningDelta::Kind::reasoning) {
      send_sse(fd, stream_chunk(id, created, request.model, {}, delta.text,
                                false, {}));
    } else {
      send_sse(fd, stream_chunk(id, created, request.model, delta.text, {},
                                false, {}));
    }
  }
  send_sse(fd, stream_chunk(id, created, request.model, {}, {}, false,
                            result.value().finish_reason));
  if (request.stream_include_usage) {
    send_sse(fd, stream_usage_chunk(id, created, request.model,
                                    result.value().prompt_tokens,
                                    result.value().generated_tokens,
                                    result.value().prompt_cache.hit_tokens));
  }
  send_sse(fd, "[DONE]");
}

void handle_client(int fd, const OpenAIGatewayConfig& config) {
  try {
    const auto request = read_http_request(fd);
    if (request.path == "/health" || request.path == "/v1/health") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, "{\"status\":\"ok\"}");
      return;
    }
    if (request.path == "/v1/models") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, models_body(config));
      return;
    }
    if (request.path == "/profile_page" || request.path == "/profile_page.html") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_web_asset(fd, "profile_page.html", "text/html; charset=utf-8");
      return;
    }
    if (request.path == "/profile_page.css") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_web_asset(fd, "profile_page.css", "text/css; charset=utf-8");
      return;
    }
    if (request.path == "/profile_page.js") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_web_asset(fd, "profile_page.js", "application/javascript; charset=utf-8");
      return;
    }
    if (request.path == "/profile_page/config") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, profile_page_config_body(config));
      return;
    }
    if (request.path == "/profiles/index.json") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, profile_index_body(config));
      return;
    }
    if (request.path.rfind("/profiles/", 0) == 0) {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      const auto suffix = request.path.substr(std::string{"/profiles/"}.size());
      const auto slash = suffix.find('/');
      if (slash == std::string::npos) {
        send_json_response(fd, 404, error_body("route not found"));
        return;
      }
      const auto request_id = suffix.substr(0, slash);
      const auto file = suffix.substr(slash + 1U);
      if (file.empty()) {
        send_json_response(fd, 404, error_body("route not found"));
        return;
      }
      std::optional<std::string> body;
      std::string content_type = "text/plain; charset=utf-8";
      if (file == "summary.json" || file == "trace.json" || file == "manifest.json") {
        body = profile_artifact_body(config, request_id, file);
        content_type = "application/json";
      } else if (file == "summary.txt" || file == "profile.folded") {
        body = profile_artifact_body(config, request_id, file);
        content_type = "text/plain; charset=utf-8";
      } else if (file == "profile.svg") {
        body = profile_artifact_body(config, request_id, file);
        content_type = "image/svg+xml";
      }
      if (!body.has_value()) {
        send_json_response(fd, 404, error_body("profile artifact not found"));
        return;
      }
      send_text_response(fd, 200, content_type, *body);
      return;
    }
    if (request.path == "/chat_page/config") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, chat_page_config_body(config));
      return;
    }
    if (request.path == "/chat_page") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_web_asset(fd, "chat_page.html", "text/html; charset=utf-8");
      return;
    }
    if (request.path == "/chat_page.css") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_web_asset(fd, "chat_page.css", "text/css; charset=utf-8");
      return;
    }
    if (request.path == "/chat_page.js") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_web_asset(fd, "chat_page.js", "application/javascript; charset=utf-8");
      return;
    }
    if (request.path == "/openapi.json" || request.path == "/v1/openapi.json") {
      if (request.method != "GET") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      send_json_response(fd, 200, openapi_body(config));
      return;
    }
    if (request.path == "/v1/completions") {
      if (request.method != "POST") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      const auto request_id = make_gateway_request_id();
      GatewayInferencePermit inference_permit;
      if (!inference_permit.acquired()) {
        send_inference_busy_response(fd, request_id);
        return;
      }
      const auto completion_request = parse_completion_request(request.body, config);
      auto observability = config.observability;
      if (const auto profile_mode = parse_profile_mode(request.header("x-kraken-profile"));
          profile_mode.has_value()) {
        observability.profile_mode = *profile_mode;
      }
      send_completion_response(fd, config, completion_request, request_id, request.header("x-request-id"),
                               observability);
      return;
    }
    if (request.path == "/v1/chat/completions") {
      if (request.method != "POST") {
        send_json_response(fd, 405, error_body("method not allowed"));
        return;
      }
      const auto request_id = make_gateway_request_id();
      GatewayInferencePermit inference_permit;
      if (!inference_permit.acquired()) {
        send_inference_busy_response(fd, request_id);
        return;
      }
      const auto chat_request = parse_chat_request(request.body, config);
      auto observability = config.observability;
      if (const auto profile_mode = parse_profile_mode(request.header("x-kraken-profile"));
          profile_mode.has_value()) {
        observability.profile_mode = *profile_mode;
      }
      send_chat_response(fd, config, chat_request, request_id, request.header("x-request-id"),
                         observability);
      return;
    }
    send_json_response(fd, 404, error_body("route not found"));
  } catch (const std::exception& error) {
    send_json_response(fd, 400, error_body(error.what()));
  }
}

Status errno_status(std::string_view operation) {
  return Status::internal_error(std::string{operation} + ": " + std::strerror(errno));
}

}  // namespace

// openapi gateway server 
Status serve_openai_gateway(const OpenAIGatewayConfig& config) {
  if (config.port <= 0 || config.port > 65535) {
    return Status::invalid_argument("gateway port must be in 1..65535");
  }
  std::optional<Qwen35MmprojMetadata> mmproj_metadata;
  if (!config.mmproj_path.empty()) {
    /*
      After metadata.value() is assigned below, mmproj_metadata is logically
      equivalent to the following JSON object (the C++ value is a
      Qwen35MmprojMetadata struct, not a JSON string).

      Values were read from the real Qwen3.5 0.8B mmproj at:
      models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf

      This is JSONC because it includes explanatory comments. Some fields are
      read directly from GGUF metadata, while the loader derives image pixel
      limits and tensor-dependent fields.

      {
        // Source mmproj file; it contains the vision encoder and projector.
        "path": "models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf",
        // GGUF architecture. An mmproj must identify itself as CLIP.
        "architecture": "clip",
        // Selects the Qwen3-VL merger that maps vision features to text embeddings.
        "projector_type": "qwen3vl_merger",
        // Alternate metadata key for projector_type; empty in this file.
        "vision_projector_type": "",
        // Merge each 2x2 group of vision patches into one image token.
        "spatial_merge_size": 2,
        // Width and height, in pixels, of one vision patch.
        "patch_size": 16,
        // Accepted image area bounds; 8192..4194304 pixels becomes 8..4096 tokens.
        "image_min_pixels": 8192,
        "image_max_pixels": 4194304,
        // Per-channel RGB normalization: normalized = (pixel - mean) / std.
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        // True when both normalization arrays were present in the GGUF.
        "image_mean_std_present": true,
        // Nominal vision input size recorded by the model; inputs may be dynamic.
        "image_size": 768,
        // Vision projection dimension before/within the merger.
        "projection_dim": 1024,
        // Hidden width of each vision transformer's feed-forward network.
        "vision_feed_forward_length": 3072,
        // Number of attention heads in each vision transformer block.
        "vision_attention_head_count": 12,
        // Epsilon used by vision layer normalization.
        "vision_attention_layer_norm_epsilon": 0.000001,
        // One flag per vision block; true would expose that block via deepstack.
        "deepstack_layer_flags": [
          false, false, false, false, false, false,
          false, false, false, false, false, false
        ],
        // Number of transformer blocks and their input embedding width.
        "vision_block_count": 12,
        "vision_embedding_length": 768,
        // Raw GGUF inventory and on-disk size, useful for diagnostics.
        "tensor_count": 154,
        "metadata_count": 23,
        "file_size": 207345952,
        // Derived count of blocks marked true in deepstack_layer_flags.
        "deepstack_layer_count": 0,
        // Derived from projector tensor shapes; must equal text-model hidden_size.
        "projector_output_width": 1024,
        // Loader validation result for all tensors required by qwen3vl_merger.
        "qwen3vl_required_tensors_present": true,
        // Names would be listed here when required projector tensors are absent.
        "missing_required_tensors": []
      }

      image_min_pixels/image_max_pixels are derived by the loader when the GGUF
      omits clip.vision.image_min_pixels and clip.vision.image_max_pixels.
    */
    const auto metadata = load_qwen35_mmproj_metadata(config.mmproj_path);
    if (!metadata.is_ok()) {
      return Status::invalid_argument("failed to load --mmproj: " +
                                      metadata.status().message());
    }
    mmproj_metadata = metadata.value();
    const auto bundle = load_model_bundle(config.model_dir);
    if (!bundle.is_ok()) {
      return Status::invalid_argument("failed to load model metadata for --mmproj: " +
                                      bundle.status().message());
    }
    if (!bundle.value().model.gguf || bundle.value().model.architecture != "qwen35") {
      return Status::invalid_argument(
        "--mmproj image input requires a qwen35 GGUF text model");
    }
    const auto compatibility =
      validate_qwen35_mmproj_text_embedding_compatibility(
        *mmproj_metadata, bundle.value().model.hidden_size);
    if (!compatibility.is_ok()) {
      return compatibility;
    }
  }
  (void)::signal(SIGPIPE, SIG_IGN);

  const int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server_fd < 0) {
    return errno_status("socket failed");
  }

  const int reuse = 1;
  (void)::setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse));
#ifdef SO_NOSIGPIPE
  const int no_sigpipe = 1;
  (void)::setsockopt(server_fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(static_cast<std::uint16_t>(config.port));
  if (::inet_pton(AF_INET, config.host.c_str(), &address.sin_addr) != 1) {
    ::close(server_fd);
    return Status::invalid_argument("gateway host must be an IPv4 address");
  }
  if (::bind(server_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    const auto status = errno_status("bind failed");
    ::close(server_fd);
    return status;
  }
  if (::listen(server_fd, 16) != 0) {
    const auto status = errno_status("listen failed");
    ::close(server_fd);
    return status;
  }

  std::cout << "OpenAI-compatible gateway listening on http://" << config.host << ':'
            << config.port << '\n';
  std::cout << "model: " << config.model_id << '\n';
  std::cout << "model path: " << config.model_dir.string() << '\n';
  std::cout << "device: " << config.compute_device.to_string() << '\n';
  if (resolve_gguf_model_path(config.model_dir).is_ok()) {
    std::cout << "gguf backend: native Qwen3.5 Metal runtime\n";
    if (!config.mmproj_path.empty()) {
      std::cout << "mmproj: " << config.mmproj_path.string() << '\n';
      if (mmproj_metadata.has_value()) {
        std::cout << "mmproj projector_type: "
                  << mmproj_metadata->projector_type << '\n';
        std::cout << "mmproj vision_projector_type: "
                  << mmproj_metadata->vision_projector_type << '\n';
        std::cout << "mmproj qwen3vl_required_tensors_present: "
                  << (mmproj_metadata->qwen3vl_required_tensors_present ? "true"
                                                                         : "false")
                  << '\n';
        std::cout << "mmproj deepstack_layer_count: "
                  << mmproj_metadata->deepstack_layer_count << '\n';
        std::cout << "mmproj projector_output_width: "
                  << mmproj_metadata->projector_output_width << '\n';
        std::cout << "mmproj native_image_min_pixels: "
                  << mmproj_metadata->image_min_pixels << '\n';
        std::cout << "mmproj native_image_max_pixels: "
                  << mmproj_metadata->image_max_pixels << '\n';
        std::cout << "mmproj native_image_mean: ["
                  << mmproj_metadata->image_mean[0] << ", "
                  << mmproj_metadata->image_mean[1] << ", "
                  << mmproj_metadata->image_mean[2] << "]\n";
        std::cout << "mmproj native_image_std: ["
                  << mmproj_metadata->image_std[0] << ", "
                  << mmproj_metadata->image_std[1] << ", "
                  << mmproj_metadata->image_std[2] << "]\n";
        const auto min_plan = plan_qwen35_image_embeddings(*mmproj_metadata, 28, 28);
        if (min_plan.is_ok()) {
          std::cout << "mmproj native_image_patch_size: "
                    << min_plan.value().patch_size << '\n';
          std::cout << "mmproj native_image_spatial_merge_size: "
                    << min_plan.value().spatial_merge_size << '\n';
          std::cout << "mmproj native_image_token_limits: "
                    << min_plan.value().min_image_tokens << ".."
                    << min_plan.value().max_image_tokens << '\n';
          std::cout << "mmproj native_image_28x28_plan_tokens: "
                    << min_plan.value().image_tokens << '\n';
        }
      }
    }
    std::cout << "mtp: " << (config.enable_mtp ? "enabled" : "disabled") << '\n';
  }

  while (true) {
    sockaddr_in client_address{};
    socklen_t client_length = sizeof(client_address);
    const int client_fd =
      ::accept(server_fd, reinterpret_cast<sockaddr*>(&client_address), &client_length);
    if (client_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      const auto status = errno_status("accept failed");
      ::close(server_fd);
      return status;
    }
#ifdef SO_NOSIGPIPE
    (void)::setsockopt(client_fd, SOL_SOCKET, SO_NOSIGPIPE, &no_sigpipe, sizeof(no_sigpipe));
#endif
    try {
      std::thread{
        [client_fd, &config]() {
          handle_client(client_fd, config);
          ::close(client_fd);
        },
      }.detach();
    } catch (const std::exception&) {
      ::close(client_fd);
    }
  }
}

}  // namespace toyllm
