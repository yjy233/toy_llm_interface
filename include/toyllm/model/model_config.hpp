#pragma once

#include "toyllm/core/status.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace toyllm {

struct ModelConfig {
  std::string architecture;
  double rms_norm_eps{0.0};
  double rope_theta{0.0};

  std::int64_t bos_token_id{0};
  std::int64_t eos_token_id{0};
  std::int64_t head_dim{0};
  std::int64_t hidden_size{0};
  std::int64_t intermediate_size{0};
  std::int64_t max_position_embeddings{0};
  std::int64_t num_attention_heads{0};
  std::int64_t num_hidden_layers{0};
  std::int64_t num_key_value_heads{0};
  std::int64_t vocab_size{0};

  bool gguf{false};
  std::int64_t total_layer_count{0};
  std::int64_t main_layer_count{0};
  std::int64_t full_attention_interval{4};
  std::int64_t linear_conv_kernel_dim{0};
  std::int64_t linear_key_head_dim{0};
  std::int64_t linear_value_head_dim{0};
  std::int64_t linear_num_key_heads{0};
  std::int64_t linear_num_value_heads{0};
  std::int64_t linear_inner_size{0};
  std::int64_t mtp_num_hidden_layers{0};
  std::int64_t expert_count{0};
  std::int64_t expert_used_count{0};
  std::int64_t expert_feed_forward_length{0};
  std::int64_t expert_shared_feed_forward_length{0};
  std::vector<std::int64_t> rope_dimension_sections;
  std::vector<std::int64_t> attention_recurrent_layers;
};

struct GenerationConfig {
  double temperature{1.0};
  double top_p{1.0};
  std::int64_t bos_token_id{0};
  std::int64_t pad_token_id{0};
  std::int64_t top_k{0};
  std::vector<std::int64_t> eos_token_ids;
};

struct TokenizerInfo {
  bool available{false};
  std::string model;
  std::string pre;
  std::uint64_t base_vocab_size{0};
  std::uint64_t total_vocab_size{0};
  std::uint64_t max_token_id{0};
};

struct ModelBundle {
  std::filesystem::path model_dir;
  std::filesystem::path model_file;
  ModelConfig model;
  GenerationConfig generation;
  TokenizerInfo tokenizer;
  std::uint64_t gguf_version{0};
  std::uint64_t gguf_tensor_count{0};
  std::uint64_t gguf_metadata_count{0};
  std::uint64_t gguf_file_size{0};
};

[[nodiscard]] Result<ModelBundle> load_model_bundle(const std::filesystem::path& model_dir);
[[nodiscard]] Status validate_model_bundle(const ModelBundle& bundle);
[[nodiscard]] std::string format_model_summary(const ModelBundle& bundle);

}  // namespace toyllm
