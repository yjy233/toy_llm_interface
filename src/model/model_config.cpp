#include "toyllm/model/model_config.hpp"

#include "toyllm/runtime/gguf_reader.hpp"

#include <filesystem>
#include <sstream>
#include <string_view>
#include <utility>

namespace toyllm {

namespace {

std::string bool_to_string(bool value) {
  return value ? "true" : "false";
}

std::string join_ints(const std::vector<std::int64_t>& values) {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << values[index];
  }
  return output.str();
}

std::string gguf_arch_key(const GgufFile& file, std::string_view suffix) {
  auto arch = gguf_get_string(file, "general.architecture");
  if (!arch.is_ok()) {
    return {};
  }
  std::string key = arch.value();
  key.push_back('.');
  key.append(suffix);
  return key;
}

std::int64_t gguf_optional_i64(const GgufFile& file, std::string_view suffix,
                               std::int64_t fallback) {
  const auto key = gguf_arch_key(file, suffix);
  if (key.empty()) {
    return fallback;
  }
  auto value = gguf_get_i64(file, key);
  return value.is_ok() ? value.value() : fallback;
}

double gguf_optional_f64(const GgufFile& file, std::string_view suffix,
                         double fallback) {
  const auto key = gguf_arch_key(file, suffix);
  if (key.empty()) {
    return fallback;
  }
  auto value = gguf_get_f64(file, key);
  return value.is_ok() ? value.value() : fallback;
}

std::vector<std::int64_t> gguf_optional_i64_array(
  const GgufFile& file, std::string_view suffix,
  std::vector<std::int64_t> fallback) {
  const auto key = gguf_arch_key(file, suffix);
  if (key.empty()) {
    return fallback;
  }
  auto value = gguf_get_i64_array(file, key);
  return value.is_ok() ? value.value() : fallback;
}

std::uint64_t gguf_optional_array_size(const GgufFile& file,
                                       const std::string& key,
                                       std::uint64_t fallback) {
  auto value = gguf_get_array_size(file, key);
  return value.is_ok() ? value.value() : fallback;
}

Status validate_positive(std::int64_t value, std::string_view name) {
  if (value <= 0) {
    return Status::invalid_argument(std::string{name} + " must be positive");
  }
  return Status::ok();
}

Result<ModelBundle> load_gguf_bundle(const std::filesystem::path& input_path) {
  auto gguf_path = resolve_gguf_model_path(input_path);
  if (!gguf_path.is_ok()) {
    return gguf_path.status();
  }
  auto gguf = read_gguf_file(gguf_path.value());
  if (!gguf.is_ok()) {
    return gguf.status();
  }
  auto architecture = gguf_get_string(gguf.value(), "general.architecture");
  if (!architecture.is_ok()) {
    return architecture.status();
  }
  if (architecture.value() != "qwen35" && architecture.value() != "clip") {
    return Status::invalid_argument(
      "only Qwen3.5 text GGUF and its CLIP mmproj are supported; got " +
      architecture.value());
  }

  ModelBundle bundle;
  bundle.model_dir = std::filesystem::is_directory(input_path)
                       ? input_path
                       : gguf_path.value().parent_path();
  bundle.model_file = gguf_path.value();
  bundle.gguf_version = gguf.value().version;
  bundle.gguf_tensor_count = gguf.value().tensor_count;
  bundle.gguf_metadata_count = gguf.value().metadata_count;
  bundle.gguf_file_size = gguf.value().file_size;

  ModelConfig model;
  model.gguf = true;
  model.architecture = architecture.value();
  if (model.architecture == "qwen35") {
    model.hidden_size = gguf_optional_i64(gguf.value(), "embedding_length", 0);
    model.intermediate_size =
      gguf_optional_i64(gguf.value(), "feed_forward_length", 0);
    model.vocab_size = gguf_optional_i64(gguf.value(), "vocab_size", 0);
    if (model.vocab_size == 0) {
      model.vocab_size = static_cast<std::int64_t>(
        gguf_optional_array_size(gguf.value(), "tokenizer.ggml.tokens", 0));
    }
    model.max_position_embeddings =
      gguf_optional_i64(gguf.value(), "context_length", 0);
    model.num_attention_heads =
      gguf_optional_i64(gguf.value(), "attention.head_count", 0);
    model.num_key_value_heads =
      gguf_optional_i64(gguf.value(), "attention.head_count_kv", 0);
    model.num_hidden_layers =
      gguf_optional_i64(gguf.value(), "block_count", 0);
    model.total_layer_count = model.num_hidden_layers;
    model.rms_norm_eps =
      gguf_optional_f64(gguf.value(), "attention.layer_norm_rms_epsilon", 0.0);
    model.rope_theta = gguf_optional_f64(gguf.value(), "rope.freq_base", 0.0);
    model.head_dim =
      gguf_optional_i64(gguf.value(), "attention.key_length", 0);
    if (model.head_dim == 0 && model.num_attention_heads > 0) {
      model.head_dim = model.hidden_size / model.num_attention_heads;
    }
    model.full_attention_interval =
      gguf_optional_i64(gguf.value(), "full_attention_interval", 4);
    model.linear_conv_kernel_dim =
      gguf_optional_i64(gguf.value(), "ssm.conv_kernel", 0);
    model.linear_key_head_dim =
      gguf_optional_i64(gguf.value(), "ssm.state_size", 0);
    model.linear_value_head_dim = model.linear_key_head_dim;
    model.linear_num_key_heads =
      gguf_optional_i64(gguf.value(), "ssm.group_count", 0);
    model.linear_num_value_heads =
      gguf_optional_i64(gguf.value(), "ssm.time_step_rank", 0);
    model.linear_inner_size =
      gguf_optional_i64(gguf.value(), "ssm.inner_size", 0);
    model.mtp_num_hidden_layers =
      gguf_optional_i64(gguf.value(), "nextn_predict_layers", 0);
    model.main_layer_count =
      model.num_hidden_layers - model.mtp_num_hidden_layers;
    if (model.main_layer_count < 0) {
      return Status::invalid_argument(
        "GGUF nextn_predict_layers exceeds block_count");
    }
    model.expert_count =
      gguf_optional_i64(gguf.value(), "expert_count", 0);
    model.expert_used_count =
      gguf_optional_i64(gguf.value(), "expert_used_count", 0);
    model.expert_feed_forward_length =
      gguf_optional_i64(gguf.value(), "expert_feed_forward_length", 0);
    model.expert_shared_feed_forward_length =
      gguf_optional_i64(gguf.value(), "expert_shared_feed_forward_length", 0);
    model.rope_dimension_sections = gguf_optional_i64_array(
      gguf.value(), "rope.dimension_sections", {11, 11, 10, 0});
    model.attention_recurrent_layers = gguf_optional_i64_array(
      gguf.value(), "attention.recurrent_layers", {});
    if (model.attention_recurrent_layers.empty() &&
        model.main_layer_count > 0) {
      model.attention_recurrent_layers.reserve(
        static_cast<std::size_t>(model.main_layer_count));
      for (std::int64_t layer = 0; layer < model.main_layer_count; ++layer) {
        model.attention_recurrent_layers.push_back(
          ((layer + 1) % model.full_attention_interval) != 0 ? 1 : 0);
      }
    }
  }

  GenerationConfig generation;
  if (auto bos = gguf_get_i64(
        gguf.value(), "tokenizer.ggml.bos_token_id");
      bos.is_ok()) {
    generation.bos_token_id = bos.value();
    model.bos_token_id = bos.value();
  }
  if (auto eos = gguf_get_i64(
        gguf.value(), "tokenizer.ggml.eos_token_id");
      eos.is_ok()) {
    generation.eos_token_ids.push_back(eos.value());
    generation.pad_token_id = eos.value();
    model.eos_token_id = eos.value();
  }
  if (model.architecture == "qwen35" &&
      generation.eos_token_ids.empty()) {
    generation.eos_token_ids.push_back(model.eos_token_id);
  }

  TokenizerInfo tokenizer;
  tokenizer.available =
    find_gguf_metadata(gguf.value(), "tokenizer.ggml.tokens") != nullptr;
  if (auto value = gguf_get_string(
        gguf.value(), "tokenizer.ggml.model");
      value.is_ok()) {
    tokenizer.model = value.value();
  }
  if (auto value = gguf_get_string(
        gguf.value(), "tokenizer.ggml.pre");
      value.is_ok()) {
    tokenizer.pre = value.value();
  }
  tokenizer.base_vocab_size =
    gguf_optional_array_size(gguf.value(), "tokenizer.ggml.tokens", 0);
  tokenizer.total_vocab_size = tokenizer.base_vocab_size;
  tokenizer.max_token_id =
    tokenizer.base_vocab_size == 0 ? 0 : tokenizer.base_vocab_size - 1;

  bundle.model = std::move(model);
  bundle.generation = std::move(generation);
  bundle.tokenizer = std::move(tokenizer);
  auto validation = validate_model_bundle(bundle);
  if (!validation.is_ok()) {
    return validation;
  }
  return bundle;
}

}  // namespace

Result<ModelBundle> load_model_bundle(
  const std::filesystem::path& model_path) {
  if (!std::filesystem::exists(model_path)) {
    return Status::unavailable(
      "model path does not exist: " + model_path.string());
  }
  return load_gguf_bundle(model_path);
}

Status validate_model_bundle(const ModelBundle& bundle) {
  const auto& model = bundle.model;
  if (!model.gguf) {
    return Status::invalid_argument(
      "Qwen3.5 runtime only supports GGUF models");
  }
  if (model.architecture == "clip") {
    return Status::ok();
  }
  if (model.architecture != "qwen35") {
    return Status::invalid_argument(
      "unsupported Qwen3.5 architecture: " + model.architecture);
  }
  if (auto status = validate_positive(
        model.hidden_size, "embedding_length");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(
        model.intermediate_size, "feed_forward_length");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(
        model.num_attention_heads, "attention.head_count");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(
        model.num_key_value_heads, "attention.head_count_kv");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(
        model.main_layer_count, "main layer count");
      !status.is_ok()) {
    return status;
  }
  if (auto status = validate_positive(model.vocab_size, "vocab_size");
      !status.is_ok()) {
    return status;
  }
  if (model.num_attention_heads % model.num_key_value_heads != 0) {
    return Status::invalid_argument(
      "attention.head_count must be divisible by head_count_kv");
  }
  if (model.rms_norm_eps <= 0.0) {
    return Status::invalid_argument(
      "attention.layer_norm_rms_epsilon must be positive");
  }
  if (model.rope_theta <= 0.0) {
    return Status::invalid_argument("rope.freq_base must be positive");
  }
  if (model.rope_dimension_sections.empty()) {
    return Status::invalid_argument(
      "rope.dimension_sections must not be empty");
  }
  if (model.attention_recurrent_layers.size() <
      static_cast<std::size_t>(model.main_layer_count)) {
    return Status::invalid_argument(
      "attention_recurrent_layers shorter than main layers");
  }
  return Status::ok();
}

std::string format_model_summary(const ModelBundle& bundle) {
  const auto& model = bundle.model;
  const auto& generation = bundle.generation;
  const auto& tokenizer = bundle.tokenizer;
  const auto gqa_group_size =
    model.num_key_value_heads == 0
      ? 0
      : model.num_attention_heads / model.num_key_value_heads;

  std::ostringstream output;
  output << "Model path: " << bundle.model_dir.string() << '\n';
  output << "Model file: " << bundle.model_file.string() << '\n';
  output << "Format: GGUF v" << bundle.gguf_version << '\n';
  output << "File size: " << bundle.gguf_file_size << " bytes\n";
  output << "Architecture: " << model.architecture << '\n';
  output << "Tensors: " << bundle.gguf_tensor_count << '\n';
  output << "Metadata entries: " << bundle.gguf_metadata_count << '\n';
  if (model.architecture == "qwen35") {
    output << "Layers total: " << model.total_layer_count << '\n';
    output << "Layers main: " << model.main_layer_count << '\n';
    output << "MTP/NextN layers: " << model.mtp_num_hidden_layers << '\n';
    output << "Hidden size: " << model.hidden_size << '\n';
    output << "Attention heads: " << model.num_attention_heads << '\n';
    output << "KV heads: " << model.num_key_value_heads << '\n';
    output << "GQA group size: " << gqa_group_size << '\n';
    output << "Head dim: " << model.head_dim << '\n';
    output << "Intermediate size: " << model.intermediate_size << '\n';
    output << "Context length: " << model.max_position_embeddings << '\n';
    output << "Full attention interval: "
           << model.full_attention_interval << '\n';
    output << "Linear conv kernel: "
           << model.linear_conv_kernel_dim << '\n';
    output << "Linear state/head dim: "
           << model.linear_key_head_dim << '\n';
    output << "Linear key heads: "
           << model.linear_num_key_heads << '\n';
    output << "Linear value heads: "
           << model.linear_num_value_heads << '\n';
    output << "Linear inner size: " << model.linear_inner_size << '\n';
    output << "MRoPE sections: "
           << join_ints(model.rope_dimension_sections) << '\n';
    output << "Recurrent layer flags: "
           << model.attention_recurrent_layers.size() << '\n';
    output << "RMSNorm eps: " << model.rms_norm_eps << '\n';
    output << "RoPE theta: "
           << static_cast<std::int64_t>(model.rope_theta) << '\n';
    output << "Vocab size: " << model.vocab_size << '\n';
    output << "Tokenizer available: "
           << bool_to_string(tokenizer.available) << '\n';
    output << "Tokenizer model: " << tokenizer.model << '\n';
    output << "Tokenizer pre: " << tokenizer.pre << '\n';
    output << "Tokenizer tokens: "
           << tokenizer.base_vocab_size << '\n';
    output << "BOS token id: " << model.bos_token_id << '\n';
    output << "EOS token id: " << model.eos_token_id << '\n';
    output << "Generation eos ids: "
           << join_ints(generation.eos_token_ids) << '\n';
  }
  output << "Validation: ok\n";
  return output.str();
}

}  // namespace toyllm
