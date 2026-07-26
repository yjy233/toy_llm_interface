#include "toyllm/runtime/qwen35_runtime.hpp"

#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/gguf_tokenizer.hpp"
#include "toyllm/runtime/qwen35_multimodal.hpp"
#include "toyllm/runtime/qwen35_vl_adapter.hpp"
#include "toyllm/runtime/qwen35_prefix_cache.hpp"
#include "toyllm/runtime/qwen35_weight_map.hpp"

#include "cpu/debug_dump.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string_view>
#include <unordered_map>
#include <utility>

namespace toyllm {

namespace {

struct Qwen35MetalCache {
  mps::MpsBuffer key_cache;
  mps::MpsBuffer value_cache;
  mps::MpsBuffer recurrent_r_cache;
  mps::MpsBuffer recurrent_s_cache;
  bool kv_cache_f16{true};
  std::uint64_t key_cache_bytes{0};
  std::uint64_t value_cache_bytes{0};
  std::uint64_t recurrent_r_cache_bytes{0};
  std::uint64_t recurrent_s_cache_bytes{0};
};

struct Qwen35MetalWeight {
  std::string name;
  std::uint32_t type{0};
  std::vector<std::uint64_t> shape;
  std::uint64_t byte_size{0};
  mps::MpsBuffer buffer;
  mps::MpsBuffer dense_f32_transposed;
  std::uint64_t dense_f32_transposed_bytes{0};
};

struct Qwen35MetalWeightStore {
  std::vector<Qwen35MetalWeight> weights;
  std::unordered_map<std::string, std::size_t> index_by_name;
  std::uint64_t total_bytes{0};
  std::uint64_t dense_f32_transposed_bytes{0};
  std::size_t f32_tensors{0};
  std::size_t q4_k_tensors{0};
  std::size_t q5_k_tensors{0};
  std::size_t q6_k_tensors{0};
  std::size_t dense_f32_transposed_tensors{0};
};

struct Qwen35LinearLayerProbe {
  mps::MpsBuffer normed_hidden;
  mps::MpsBuffer qkv;
  mps::MpsBuffer gate;
  mps::MpsBuffer beta;
  mps::MpsBuffer alpha;
  mps::MpsBuffer recurrent_gate;
  mps::MpsBuffer conv_output;
  mps::MpsBuffer q_conv;
  mps::MpsBuffer k_conv;
  mps::MpsBuffer v_conv;
  mps::MpsBuffer attn_output;
  mps::MpsBuffer linear_attn_output;
  mps::MpsBuffer post_attn_norm;
  mps::MpsBuffer ffn_up;
  mps::MpsBuffer ffn_gate;
  mps::MpsBuffer layer_output;
  std::size_t qkv_values{0};
  std::size_t gate_values{0};
  std::size_t beta_values{0};
  std::size_t alpha_values{0};
  std::size_t recurrent_gate_values{0};
  std::size_t conv_output_values{0};
  std::size_t q_conv_values{0};
  std::size_t k_conv_values{0};
  std::size_t v_conv_values{0};
  std::size_t attn_output_values{0};
  std::size_t norm_gated_values{0};
  std::size_t linear_attn_output_values{0};
  std::size_t attn_residual_values{0};
  std::size_t post_attn_norm_values{0};
  std::size_t ffn_up_values{0};
  std::size_t ffn_gate_values{0};
  std::size_t ffn_swiglu_values{0};
  std::size_t layer_output_values{0};
};

struct Qwen35LinearLayerStats {
  std::size_t qkv_values{0};
  std::size_t gate_values{0};
  std::size_t beta_values{0};
  std::size_t alpha_values{0};
  std::size_t recurrent_gate_values{0};
  std::size_t conv_output_values{0};
  std::size_t q_conv_values{0};
  std::size_t k_conv_values{0};
  std::size_t v_conv_values{0};
  std::size_t attn_output_values{0};
  std::size_t norm_gated_values{0};
  std::size_t linear_attn_output_values{0};
  std::size_t attn_residual_values{0};
  std::size_t post_attn_norm_values{0};
  std::size_t ffn_up_values{0};
  std::size_t ffn_gate_values{0};
  std::size_t ffn_swiglu_values{0};
  std::size_t layer_output_values{0};
};

struct Qwen35FullAttentionLayerStats {
  std::size_t q_full_values{0};
  std::size_t query_values{0};
  std::size_t gate_values{0};
  std::size_t key_values{0};
  std::size_t value_values{0};
  std::size_t attention_values{0};
  std::size_t attention_output_values{0};
  std::size_t layer_output_values{0};
};

struct Qwen35LogitsOutput {
  mps::MpsBuffer h_nextn;
  mps::MpsBuffer logits;
  mps::MpsBuffer argmax_output;
  mps::MpsBuffer top1_probability;
  std::size_t logits_values{0};
  std::size_t rows{1};
};

struct Qwen35MtpForwardOutput {
  mps::MpsBuffer h_nextn;
  mps::MpsBuffer logits;
  mps::MpsBuffer argmax_output;
  mps::MpsBuffer top1_probability;
  std::size_t logits_values{0};
  std::size_t rows{0};
};

struct Qwen35MainForwardOutput {
  mps::MpsBuffer hidden;
  std::size_t rows{0};
};

struct Qwen35MtpRuntimeState {
  bool available{false};
  bool enabled{false};
  std::string disabled_reason;
  std::size_t draft_tokens{0};
  double p_min{0.0};
  std::size_t drafted_tokens{0};
  std::size_t accepted_tokens{0};
  std::size_t verify_steps{0};
  std::size_t confidence_stops{0};
  std::size_t adaptive_budget{0};
  std::size_t adaptive_changes{0};
  std::size_t adaptive_window_drafted{0};
  std::size_t adaptive_window_accepted{0};
  std::vector<std::size_t> verified_by_position;
  std::vector<std::size_t> accepted_by_position;
  Qwen35MetalCache cache;
  mps::MpsBuffer pending_h;
  mps::MpsBuffer recurrent_r_snapshot;
  mps::MpsBuffer recurrent_s_snapshot;
};

struct Qwen35EffectiveSamplingConfig {
  bool do_sample{false};
  double temperature{1.0};
  std::size_t top_k{0};
  double top_p{1.0};
  std::uint64_t seed{0};
};

class MpsGraphGuard {
 public:
  explicit MpsGraphGuard(const mps::MpsContext& context) : context_(&context) {}

  MpsGraphGuard(const MpsGraphGuard&) = delete;
  MpsGraphGuard& operator=(const MpsGraphGuard&) = delete;

  ~MpsGraphGuard() {
    if (active_) {
      context_->abort_graph();
    }
  }

  [[nodiscard]] Status begin() {
    auto status = context_->begin_graph();
    active_ = status.is_ok();
    return status;
  }

  [[nodiscard]] Status commit() {
    if (!active_) {
      return Status::ok();
    }
    active_ = false;
    return context_->commit_graph();
  }

 private:
  const mps::MpsContext* context_;
  bool active_{false};
};

enum class Qwen35PrefillCommitMode {
  off,
  chunk,
  layer,
};

std::string qwen35_position_tensor_name(std::size_t position, std::string_view suffix) {
  std::ostringstream output;
  output << "position." << position << '.' << suffix;
  return output.str();
}

std::string qwen35_layer_tensor_name(std::size_t chunk_start, std::size_t layer,
                                     std::string_view suffix) {
  std::ostringstream output;
  output << "chunk." << chunk_start << ".layer." << layer << '.' << suffix;
  return output.str();
}

std::string qwen35_step_tensor_name(std::string_view prefix, std::string_view suffix) {
  if (prefix.empty()) {
    return std::string{suffix};
  }
  std::ostringstream output;
  output << prefix << '.' << suffix;
  return output.str();
}

Status dump_qwen35_mps_f32(const mps::MpsContext& context, cpu::DebugDumper* dumper,
                           std::string_view name,
                           const std::vector<std::uint64_t>& shape,
                           const mps::MpsBuffer& buffer, std::size_t values) {
  if (dumper == nullptr) {
    return Status::ok();
  }
  std::vector<float> host(values, 0.0F);
  auto status = context.copy_from_buffer(buffer, host.data(), values * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  dumper->write_f32(name, shape, host);
  return Status::ok();
}

[[maybe_unused]] Qwen35LinearLayerStats qwen35_linear_layer_stats(
  const Qwen35LinearLayerProbe& probe) {
  return Qwen35LinearLayerStats{
    probe.qkv_values,
    probe.gate_values,
    probe.beta_values,
    probe.alpha_values,
    probe.recurrent_gate_values,
    probe.conv_output_values,
    probe.q_conv_values,
    probe.k_conv_values,
    probe.v_conv_values,
    probe.attn_output_values,
    probe.norm_gated_values,
    probe.linear_attn_output_values,
    probe.attn_residual_values,
    probe.post_attn_norm_values,
    probe.ffn_up_values,
    probe.ffn_gate_values,
    probe.ffn_swiglu_values,
    probe.layer_output_values,
  };
}

Result<std::array<std::size_t, 4>> qwen35_mrope_sections(const ModelConfig& config) {
  if (config.rope_dimension_sections.size() < 4U) {
    return Status::invalid_argument("Qwen3.5 MRoPE requires four dimension sections");
  }
  std::array<std::size_t, 4> sections{};
  for (std::size_t index = 0; index < sections.size(); ++index) {
    if (config.rope_dimension_sections[index] < 0) {
      return Status::invalid_argument("Qwen3.5 MRoPE section must be non-negative");
    }
    sections[index] = static_cast<std::size_t>(config.rope_dimension_sections[index]);
  }
  if (sections[0] + sections[1] + sections[2] + sections[3] == 0) {
    return Status::invalid_argument("Qwen3.5 MRoPE sections must not all be zero");
  }
  return sections;
}

const Qwen35MetalWeight* find_metal_weight(const Qwen35MetalWeightStore& store,
                                           const std::string& name) {
  const auto iterator = store.index_by_name.find(name);
  if (iterator == store.index_by_name.end()) {
    return nullptr;
  }
  if (iterator->second >= store.weights.size()) {
    return nullptr;
  }
  return &store.weights[iterator->second];
}

std::size_t qwen35_mul_mv_ext_r1ptg(std::size_t tokens) {
  switch (tokens) {
    case 3:
    case 6:
      return 3;
    case 5:
      return 5;
    default:
      return 4;
  }
}

std::string qwen35_quant_matmul_dispatch_name(std::uint32_t type,
                                              std::size_t tokens) {
  if (type == 0U || type == 8U) {
    return "dense_f32_transposed";
  }
  if (type != 12U && type != 13U && type != 14U) {
    return "unsupported";
  }
  if (tokens >= 4U && tokens <= 8U) {
    return "mul_mv_ext_r1_" +
           std::to_string(qwen35_mul_mv_ext_r1ptg(tokens));
  }
  if (tokens > 8U) {
    return "mul_mm_simd_64x32";
  }
  return "row_reduce_matmul";
}

std::string qwen35_gdn_dispatch_name(std::size_t key_heads,
                                     std::size_t value_heads,
                                     std::size_t head_dim) {
  if (head_dim == 128U && key_heads == value_heads) {
    return "qwen35_simdgroup_4rows";
  }
  return "generic_batched_gdn";
}

enum class Qwen35EnvFlag {
  unset,
  enabled,
  disabled,
};

Qwen35EnvFlag qwen35_env_flag(const char* name) {
  const char* value = std::getenv(name);
  if (value == nullptr) {
    return Qwen35EnvFlag::unset;
  }
  const std::string_view setting{value};
  if (setting == "1" || setting == "true" || setting == "TRUE" ||
      setting == "on" || setting == "ON" || setting == "yes" ||
      setting == "YES") {
    return Qwen35EnvFlag::enabled;
  }
  if (setting == "0" || setting == "false" || setting == "FALSE" ||
      setting == "off" || setting == "OFF" || setting == "no" ||
      setting == "NO") {
    return Qwen35EnvFlag::disabled;
  }
  return Qwen35EnvFlag::unset;
}

bool qwen35_attention_uses_flash(std::size_t tokens, std::size_t head_dim) {
  if (head_dim != 256U) {
    return false;
  }
  const auto env_flag = qwen35_env_flag("KRAKEN_QWEN35_FLASH_ATTENTION");
  if (env_flag == Qwen35EnvFlag::disabled) {
    return false;
  }
  return env_flag == Qwen35EnvFlag::enabled || tokens >= 64U;
}

bool qwen35_attention_uses_tiled_f32(std::size_t tokens, std::size_t head_dim) {
  if (head_dim > 256U) {
    return false;
  }
  const auto env_flag = qwen35_env_flag("KRAKEN_QWEN35_TILED_ATTENTION");
  if (env_flag == Qwen35EnvFlag::disabled) {
    return false;
  }
  if (env_flag == Qwen35EnvFlag::enabled) {
    return tokens >= 16U;
  }
  return tokens >= 1024U;
}

std::size_t qwen35_requested_tiled_attention_cache_tile() {
  const char* value = std::getenv("KRAKEN_QWEN35_TILED_ATTENTION_TILE");
  if (value == nullptr) {
    return 16U;
  }
  const std::string_view setting{value};
  if (setting == "32") {
    return 32U;
  }
  return 16U;
}

std::string qwen35_attention_dispatch_name(std::size_t tokens,
                                           std::size_t key_count,
                                           std::size_t head_dim,
                                           bool f16_kv) {
  if (qwen35_attention_uses_flash(tokens, head_dim)) {
    std::string name = f16_kv ? "flash256_f16_kv" : "flash256_f32";
    if (key_count % 64U != 0U) {
      name += "+tail";
    }
    return name;
  }
  if (f16_kv) {
    return "online_f16_kv";
  }
  if (qwen35_attention_uses_tiled_f32(tokens, head_dim)) {
    return "tiled_f32_" +
           std::to_string(qwen35_requested_tiled_attention_cache_tile());
  }
  return "online_f32";
}

Result<std::size_t> positive_size(std::int64_t value, const char* name) {
  if (value <= 0) {
    return Status::invalid_argument(std::string{name} + " must be positive");
  }
  const auto unsigned_value = static_cast<std::uint64_t>(value);
  if (unsigned_value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument(std::string{name} + " exceeds size_t range");
  }
  return static_cast<std::size_t>(unsigned_value);
}

bool checked_mul_u64(std::uint64_t lhs, std::uint64_t rhs, std::uint64_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

bool checked_mul_size(std::size_t lhs, std::size_t rhs, std::size_t& result) {
  if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

Status checked_mul_assign(std::uint64_t& value, std::uint64_t multiplier,
                          const char* name) {
  std::uint64_t result = 0;
  if (!checked_mul_u64(value, multiplier, result)) {
    return Status::invalid_argument(std::string{name} + " byte count overflow");
  }
  value = result;
  return Status::ok();
}

Result<std::uint64_t> cache_bytes(std::size_t a, std::size_t b, std::size_t c,
                                  std::size_t d, const char* name) {
  std::uint64_t bytes = static_cast<std::uint64_t>(a);
  auto status = checked_mul_assign(bytes, static_cast<std::uint64_t>(b), name);
  if (!status.is_ok()) {
    return status;
  }
  status = checked_mul_assign(bytes, static_cast<std::uint64_t>(c), name);
  if (!status.is_ok()) {
    return status;
  }
  status = checked_mul_assign(bytes, static_cast<std::uint64_t>(d), name);
  if (!status.is_ok()) {
    return status;
  }
  return bytes;
}

Result<std::uint64_t> add_cache_bytes(std::uint64_t lhs, std::uint64_t rhs,
                                      const char* name) {
  if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
    return Status::invalid_argument(std::string{name} + " byte count overflow");
  }
  return lhs + rhs;
}

std::size_t ceil_div(std::size_t value, std::size_t divisor) {
  return value == 0 ? 0 : 1U + ((value - 1U) / divisor);
}

Result<std::size_t> checked_size_t(std::uint64_t value, const char* name) {
  if (value > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument(std::string{name} + " exceeds size_t range");
  }
  return static_cast<std::size_t>(value);
}

struct Qwen35HostPrefixCacheLayout {
  std::filesystem::path model_dir;
  std::filesystem::path model_file;
  std::uint64_t gguf_version{0};
  std::uint64_t gguf_tensor_count{0};
  std::uint64_t gguf_metadata_count{0};
  std::uint64_t gguf_file_size{0};
  std::uint64_t tokenizer_fingerprint{0};
  std::size_t block_tokens{0};
  std::size_t full_attention_layers{0};
  std::size_t linear_attention_layers{0};
  std::size_t kv_dim{0};
  std::size_t kv_element_bytes{0};
  std::size_t hidden_size{0};
  std::uint64_t recurrent_r_cache_bytes{0};
  std::uint64_t recurrent_s_cache_bytes{0};
  bool kv_cache_f16{true};
};

bool operator==(const Qwen35HostPrefixCacheLayout& lhs,
                const Qwen35HostPrefixCacheLayout& rhs) {
  return lhs.model_dir == rhs.model_dir &&
         lhs.model_file == rhs.model_file &&
         lhs.gguf_version == rhs.gguf_version &&
         lhs.gguf_tensor_count == rhs.gguf_tensor_count &&
         lhs.gguf_metadata_count == rhs.gguf_metadata_count &&
         lhs.gguf_file_size == rhs.gguf_file_size &&
         lhs.tokenizer_fingerprint == rhs.tokenizer_fingerprint &&
         lhs.block_tokens == rhs.block_tokens &&
         lhs.full_attention_layers == rhs.full_attention_layers &&
         lhs.linear_attention_layers == rhs.linear_attention_layers &&
         lhs.kv_dim == rhs.kv_dim &&
         lhs.kv_element_bytes == rhs.kv_element_bytes &&
         lhs.hidden_size == rhs.hidden_size &&
         lhs.recurrent_r_cache_bytes == rhs.recurrent_r_cache_bytes &&
         lhs.recurrent_s_cache_bytes == rhs.recurrent_s_cache_bytes &&
         lhs.kv_cache_f16 == rhs.kv_cache_f16;
}

std::uint64_t qwen35_prefix_mix_u64(std::uint64_t hash,
                                    std::uint64_t value) {
  constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
  hash ^= value;
  hash *= kFnvPrime;
  return hash;
}

std::uint64_t qwen35_prefix_mix_bytes(std::uint64_t hash,
                                      std::string_view value) {
  hash = qwen35_prefix_mix_u64(hash,
                               static_cast<std::uint64_t>(value.size()));
  for (const char ch : value) {
    hash = qwen35_prefix_mix_u64(
      hash, static_cast<std::uint64_t>(static_cast<unsigned char>(ch)));
  }
  return hash;
}

std::uint64_t qwen35_tokenizer_fingerprint(const GgufTokenizer& tokenizer) {
  std::uint64_t hash = 1469598103934665603ULL;
  hash = qwen35_prefix_mix_bytes(hash, tokenizer.model);
  hash = qwen35_prefix_mix_bytes(hash, tokenizer.pre);
  hash = qwen35_prefix_mix_bytes(hash, tokenizer.chat_template);
  hash = qwen35_prefix_mix_u64(
    hash, static_cast<std::uint64_t>(tokenizer.tokens.size()));
  for (const auto& token : tokenizer.tokens) {
    hash = qwen35_prefix_mix_bytes(hash, token);
  }
  hash = qwen35_prefix_mix_u64(
    hash, static_cast<std::uint64_t>(tokenizer.token_types.size()));
  for (const auto token_type : tokenizer.token_types) {
    hash = qwen35_prefix_mix_u64(
      hash, static_cast<std::uint64_t>(token_type));
  }
  hash = qwen35_prefix_mix_u64(
    hash, static_cast<std::uint64_t>(tokenizer.merges.size()));
  for (const auto& merge : tokenizer.merges) {
    hash = qwen35_prefix_mix_bytes(hash, merge);
  }
  hash = qwen35_prefix_mix_u64(
    hash, static_cast<std::uint64_t>(tokenizer.bos_token_id));
  hash = qwen35_prefix_mix_u64(
    hash, static_cast<std::uint64_t>(tokenizer.eos_token_id));
  hash = qwen35_prefix_mix_u64(
    hash, static_cast<std::uint64_t>(tokenizer.pad_token_id));
  hash = qwen35_prefix_mix_u64(hash, tokenizer.add_bos_token ? 1U : 0U);
  hash = qwen35_prefix_mix_u64(hash, tokenizer.add_eos_token ? 1U : 0U);
  return hash == 0 ? 1 : hash;
}

struct Qwen35HostPrefixPayload {
  std::vector<std::uint8_t> key_cache;
  std::vector<std::uint8_t> value_cache;
  std::vector<std::uint8_t> recurrent_r_cache;
  std::vector<std::uint8_t> recurrent_s_cache;
  std::vector<std::uint8_t> last_hidden;
};

struct Qwen35HostPrefixCommitResult {
  bool committed{false};
  bool evicted{false};
};

class Qwen35HostPrefixCache {
 public:
  void configure(const Qwen35HostPrefixCacheLayout& layout,
                 Qwen35PrefixCacheConfig config) {
    if (!layout_.has_value() || !(*layout_ == layout) ||
        config.block_tokens != index_.config().block_tokens ||
        config.capacity_blocks != index_.config().capacity_blocks ||
        config.min_reuse_tokens != index_.config().min_reuse_tokens ||
        config.enabled != index_.config().enabled) {
      layout_ = layout;
      index_ = Qwen35PrefixCacheIndex(config);
      payloads_.clear();
    }
  }

  [[nodiscard]] Qwen35PrefixCacheLookup lookup(
    const std::vector<std::int64_t>& tokens) {
    return index_.lookup(tokens);
  }

  [[nodiscard]] Result<Qwen35HostPrefixCommitResult> commit_block(
    const mps::MpsContext& context, const Qwen35MetalCache& cache,
    const Qwen35ExecutionPlan& plan, const mps::MpsBuffer& chunk_hidden,
    const std::vector<std::int64_t>& tokens, std::size_t token_start,
    std::size_t chunk_tokens) {
    if (!layout_.has_value() || chunk_tokens != layout_->block_tokens ||
        token_start + layout_->block_tokens > tokens.size()) {
      return Qwen35HostPrefixCommitResult{};
    }
    auto hash = index_.hash_for_block(tokens, token_start);
    if (!hash.has_value()) {
      return Qwen35HostPrefixCommitResult{};
    }
    if (payloads_.find(*hash) != payloads_.end()) {
      return Qwen35HostPrefixCommitResult{};
    }

    auto payload = read_payload(context, cache, plan, chunk_hidden, token_start,
                                chunk_tokens);
    if (!payload.is_ok()) {
      return payload.status();
    }

    auto commit = index_.commit_block(tokens, token_start);
    Qwen35HostPrefixCommitResult result;
    if (!commit.inserted) {
      return result;
    }
    if (commit.evicted) {
      payloads_.erase(commit.evicted_hash);
      result.evicted = true;
    }
    payloads_[commit.hash] = std::move(payload.value());
    result.committed = true;
    return result;
  }

  [[nodiscard]] Status restore(
    const mps::MpsContext& context, Qwen35MetalCache& cache,
    const Qwen35ExecutionPlan& plan,
    const std::vector<std::uint64_t>& block_hashes) const {
    if (!layout_.has_value()) {
      return Status::ok();
    }
    for (std::size_t block_index = 0; block_index < block_hashes.size();
         ++block_index) {
      const auto it = payloads_.find(block_hashes[block_index]);
      if (it == payloads_.end()) {
        return Status::invalid_argument("Qwen3.5 prefix cache payload is missing");
      }
      auto status = restore_kv_block(context, cache, plan, it->second,
                                     block_index * layout_->block_tokens);
      if (!status.is_ok()) {
        return status;
      }
    }
    if (!block_hashes.empty()) {
      const auto it = payloads_.find(block_hashes.back());
      if (it == payloads_.end()) {
        return Status::invalid_argument("Qwen3.5 prefix cache payload is missing");
      }
      auto status = restore_recurrent(context, cache, it->second);
      if (!status.is_ok()) {
        return status;
      }
    }
    return Status::ok();
  }

  [[nodiscard]] const std::vector<std::uint8_t>* last_hidden(
    std::uint64_t hash) const {
    const auto it = payloads_.find(hash);
    if (it == payloads_.end()) {
      return nullptr;
    }
    return &it->second.last_hidden;
  }

  [[nodiscard]] Qwen35PrefixCacheStats stats() const {
    return index_.stats();
  }

 private:
  Qwen35PrefixCacheIndex index_{Qwen35PrefixCacheConfig{}};
  std::optional<Qwen35HostPrefixCacheLayout> layout_;
  std::unordered_map<std::uint64_t, Qwen35HostPrefixPayload> payloads_;

  [[nodiscard]] Result<std::size_t> kv_block_layer_bytes() const {
    if (!layout_.has_value()) {
      return Status::invalid_argument("Qwen3.5 prefix cache layout is not configured");
    }
    std::size_t elements = 0;
    if (!checked_mul_size(layout_->block_tokens, layout_->kv_dim, elements)) {
      return Status::invalid_argument("Qwen3.5 prefix cache KV block size overflow");
    }
    std::size_t bytes = 0;
    if (!checked_mul_size(elements, layout_->kv_element_bytes, bytes)) {
      return Status::invalid_argument("Qwen3.5 prefix cache KV block bytes overflow");
    }
    return bytes;
  }

  [[nodiscard]] Result<std::size_t> kv_layer_payload_offset(
    std::size_t layer) const {
    auto block_bytes = kv_block_layer_bytes();
    if (!block_bytes.is_ok()) {
      return block_bytes.status();
    }
    std::size_t offset = 0;
    if (!checked_mul_size(layer, block_bytes.value(), offset)) {
      return Status::invalid_argument("Qwen3.5 prefix cache payload offset overflow");
    }
    return offset;
  }

  [[nodiscard]] Result<std::size_t> active_kv_byte_offset(
    const Qwen35ExecutionPlan& plan, std::size_t layer,
    std::size_t token_start) const {
    std::size_t row = 0;
    if (!checked_mul_size(layer, plan.cache.attention_capacity_tokens, row)) {
      return Status::invalid_argument("Qwen3.5 active KV layer offset overflow");
    }
    if (token_start > std::numeric_limits<std::size_t>::max() - row) {
      return Status::invalid_argument("Qwen3.5 active KV token offset overflow");
    }
    row += token_start;
    std::size_t elements = 0;
    if (!checked_mul_size(row, plan.cache.kv_dim, elements)) {
      return Status::invalid_argument("Qwen3.5 active KV element offset overflow");
    }
    std::size_t bytes = 0;
    if (!checked_mul_size(elements, plan.cache.kv_cache_element_bytes, bytes)) {
      return Status::invalid_argument("Qwen3.5 active KV byte offset overflow");
    }
    return bytes;
  }

  [[nodiscard]] Result<Qwen35HostPrefixPayload> read_payload(
    const mps::MpsContext& context, const Qwen35MetalCache& cache,
    const Qwen35ExecutionPlan& plan, const mps::MpsBuffer& chunk_hidden,
    std::size_t token_start, std::size_t chunk_tokens) const {
    auto block_bytes = kv_block_layer_bytes();
    if (!block_bytes.is_ok()) {
      return block_bytes.status();
    }
    std::size_t kv_payload_bytes = 0;
    if (!checked_mul_size(layout_->full_attention_layers, block_bytes.value(),
                          kv_payload_bytes)) {
      return Status::invalid_argument("Qwen3.5 prefix cache KV payload overflow");
    }

    Qwen35HostPrefixPayload payload;
    payload.key_cache.resize(kv_payload_bytes);
    payload.value_cache.resize(kv_payload_bytes);
    for (std::size_t layer = 0; layer < layout_->full_attention_layers; ++layer) {
      auto source_offset = active_kv_byte_offset(plan, layer, token_start);
      if (!source_offset.is_ok()) {
        return source_offset.status();
      }
      auto payload_offset = kv_layer_payload_offset(layer);
      if (!payload_offset.is_ok()) {
        return payload_offset.status();
      }
      auto status = context.copy_from_buffer_at(
        cache.key_cache, source_offset.value(),
        payload.key_cache.data() + payload_offset.value(), block_bytes.value());
      if (!status.is_ok()) {
        return status;
      }
      status = context.copy_from_buffer_at(
        cache.value_cache, source_offset.value(),
        payload.value_cache.data() + payload_offset.value(), block_bytes.value());
      if (!status.is_ok()) {
        return status;
      }
    }

    auto recurrent_r_bytes =
      checked_size_t(cache.recurrent_r_cache_bytes, "Qwen3.5 recurrent R cache");
    if (!recurrent_r_bytes.is_ok()) {
      return recurrent_r_bytes.status();
    }
    payload.recurrent_r_cache.resize(recurrent_r_bytes.value());
    if (!payload.recurrent_r_cache.empty()) {
      auto status = context.copy_from_buffer_at(
        cache.recurrent_r_cache, 0, payload.recurrent_r_cache.data(),
        payload.recurrent_r_cache.size());
      if (!status.is_ok()) {
        return status;
      }
    }

    auto recurrent_s_bytes =
      checked_size_t(cache.recurrent_s_cache_bytes, "Qwen3.5 recurrent S cache");
    if (!recurrent_s_bytes.is_ok()) {
      return recurrent_s_bytes.status();
    }
    payload.recurrent_s_cache.resize(recurrent_s_bytes.value());
    if (!payload.recurrent_s_cache.empty()) {
      auto status = context.copy_from_buffer_at(
        cache.recurrent_s_cache, 0, payload.recurrent_s_cache.data(),
        payload.recurrent_s_cache.size());
      if (!status.is_ok()) {
        return status;
      }
    }

    std::size_t hidden_bytes = 0;
    if (!checked_mul_size(layout_->hidden_size, sizeof(float), hidden_bytes)) {
      return Status::invalid_argument("Qwen3.5 prefix last hidden bytes overflow");
    }
    std::size_t hidden_row = 0;
    if (!checked_mul_size(chunk_tokens - 1U, layout_->hidden_size, hidden_row)) {
      return Status::invalid_argument("Qwen3.5 prefix last hidden offset overflow");
    }
    std::size_t hidden_offset = 0;
    if (!checked_mul_size(hidden_row, sizeof(float), hidden_offset)) {
      return Status::invalid_argument("Qwen3.5 prefix last hidden byte offset overflow");
    }
    payload.last_hidden.resize(hidden_bytes);
    auto status = context.copy_from_buffer_at(chunk_hidden, hidden_offset,
                                              payload.last_hidden.data(),
                                              payload.last_hidden.size());
    if (!status.is_ok()) {
      return status;
    }
    return payload;
  }

  [[nodiscard]] Status restore_kv_block(
    const mps::MpsContext& context, Qwen35MetalCache& cache,
    const Qwen35ExecutionPlan& plan, const Qwen35HostPrefixPayload& payload,
    std::size_t token_start) const {
    auto block_bytes = kv_block_layer_bytes();
    if (!block_bytes.is_ok()) {
      return block_bytes.status();
    }
    for (std::size_t layer = 0; layer < layout_->full_attention_layers; ++layer) {
      auto destination_offset = active_kv_byte_offset(plan, layer, token_start);
      if (!destination_offset.is_ok()) {
        return destination_offset.status();
      }
      auto payload_offset = kv_layer_payload_offset(layer);
      if (!payload_offset.is_ok()) {
        return payload_offset.status();
      }
      auto status = context.copy_to_buffer_at(
        cache.key_cache, destination_offset.value(),
        payload.key_cache.data() + payload_offset.value(), block_bytes.value());
      if (!status.is_ok()) {
        return status;
      }
      status = context.copy_to_buffer_at(
        cache.value_cache, destination_offset.value(),
        payload.value_cache.data() + payload_offset.value(), block_bytes.value());
      if (!status.is_ok()) {
        return status;
      }
    }
    return Status::ok();
  }

  [[nodiscard]] Status restore_recurrent(
    const mps::MpsContext& context, Qwen35MetalCache& cache,
    const Qwen35HostPrefixPayload& payload) const {
    if (!payload.recurrent_r_cache.empty()) {
      auto status = context.copy_to_buffer_at(
        cache.recurrent_r_cache, 0, payload.recurrent_r_cache.data(),
        payload.recurrent_r_cache.size());
      if (!status.is_ok()) {
        return status;
      }
    }
    if (!payload.recurrent_s_cache.empty()) {
      auto status = context.copy_to_buffer_at(
        cache.recurrent_s_cache, 0, payload.recurrent_s_cache.data(),
        payload.recurrent_s_cache.size());
      if (!status.is_ok()) {
        return status;
      }
    }
    return Status::ok();
  }
};

Qwen35HostPrefixCache& qwen35_host_prefix_cache() {
  static Qwen35HostPrefixCache cache;
  return cache;
}

std::mutex& qwen35_host_prefix_cache_mutex() {
  static std::mutex mutex;
  return mutex;
}

float f16_to_float_cpu(const std::uint8_t* data) {
  const auto value = static_cast<std::uint16_t>(
    static_cast<std::uint16_t>(data[0]) |
    (static_cast<std::uint16_t>(data[1]) << 8U));
  const auto sign = static_cast<std::uint32_t>((value >> 15U) & 1U);
  const auto exponent = static_cast<std::uint32_t>((value >> 10U) & 31U);
  const auto mantissa = static_cast<std::uint32_t>(value & 1023U);
  float result = 0.0F;
  if (exponent == 0U) {
    result = std::ldexp(static_cast<float>(mantissa), -24);
  } else if (exponent == 31U) {
    result = mantissa == 0U ? std::numeric_limits<float>::infinity()
                            : std::numeric_limits<float>::quiet_NaN();
  } else {
    result = std::ldexp(1.0F + static_cast<float>(mantissa) / 1024.0F,
                        static_cast<int>(exponent) - 15);
  }
  return sign == 0U ? result : -result;
}

std::uint32_t q4_k_scale_min_cpu(std::uint32_t index,
                                 const std::uint8_t* scales) {
  std::uint32_t d = 0;
  std::uint32_t m = 0;
  if (index < 4U) {
    d = scales[index] & 63U;
    m = scales[index + 4U] & 63U;
  } else {
    d = static_cast<std::uint32_t>(scales[index + 4U] & 0xFU) |
        (static_cast<std::uint32_t>(scales[index - 4U] >> 6U) << 4U);
    m = static_cast<std::uint32_t>(scales[index + 4U] >> 4U) |
        (static_cast<std::uint32_t>(scales[index] >> 6U) << 4U);
  }
  return d | (m << 8U);
}

bool should_build_dense_prefill_weight(const GgufTensorInfo& tensor) {
  return tensor.name.rfind("blk.", 0) == 0 && tensor.shape.size() == 2U &&
         (tensor.type == 12U || tensor.type == 13U || tensor.type == 14U);
}

bool should_build_dense_transposed_weight(const GgufTensorInfo& tensor,
                                          bool build_dense_prefill_weights) {
  if (tensor.shape.size() != 2U) {
    return false;
  }
  if (tensor.type == 0U && tensor.name.rfind("blk.", 0) == 0) {
    return true;
  }
  if (tensor.type == 8U) {
    return true;
  }
  return build_dense_prefill_weights && should_build_dense_prefill_weight(tensor);
}

bool dense_prefill_weights_enabled() {
  const char* value = std::getenv("KRAKEN_QWEN35_DENSE_PREFILL");
  if (value == nullptr) {
    return false;
  }
  const std::string_view setting{value};
  return setting == "1" || setting == "true" || setting == "TRUE" ||
         setting == "on" || setting == "ON" || setting == "yes" ||
         setting == "YES";
}

bool qwen35_f16_kv_cache_enabled() {
  const char* value = std::getenv("KRAKEN_QWEN35_F16_KV");
  if (value == nullptr) {
    return true;
  }
  const std::string_view setting{value};
  if (setting == "0" || setting == "false" || setting == "FALSE" ||
      setting == "off" || setting == "OFF" || setting == "no" ||
      setting == "NO") {
    return false;
  }
  return true;
}

bool qwen35_fused_norm_gated_enabled() {
  const char* value = std::getenv("KRAKEN_QWEN35_FUSED_NORM_GATED");
  if (value == nullptr) {
    return false;
  }
  const std::string_view setting{value};
  return setting == "1" || setting == "true" || setting == "TRUE" ||
         setting == "on" || setting == "ON" || setting == "yes" ||
         setting == "YES";
}

Qwen35PrefillCommitMode qwen35_prefill_commit_mode() {
  const char* value = std::getenv("KRAKEN_QWEN35_PREFILL_COMMIT");
  if (value == nullptr) {
    value = std::getenv("KRAKEN_QWEN35_PROFILE_COMMIT");
  }
  if (value == nullptr) {
    return Qwen35PrefillCommitMode::chunk;
  }
  const std::string_view setting{value};
  if (setting == "single" || setting == "SINGLE" || setting == "off" ||
      setting == "OFF" || setting == "0") {
    return Qwen35PrefillCommitMode::off;
  }
  if (setting == "chunk" || setting == "CHUNK") {
    return Qwen35PrefillCommitMode::chunk;
  }
  if (setting == "layer" || setting == "LAYER") {
    return Qwen35PrefillCommitMode::layer;
  }
  return Qwen35PrefillCommitMode::chunk;
}

std::vector<ProfileField> qwen35_chunk_profile_fields(std::size_t chunk_start,
                                                       std::size_t tokens) {
  return {
    ProfileField{"chunk_start", std::to_string(chunk_start)},
    ProfileField{"tokens", std::to_string(tokens)},
  };
}

std::vector<ProfileField> qwen35_layer_profile_fields(std::size_t layer,
                                                       std::size_t chunk_start,
                                                       std::size_t tokens,
                                                       std::string kind) {
  return {
    ProfileField{"layer", std::to_string(layer)},
    ProfileField{"chunk_start", std::to_string(chunk_start)},
    ProfileField{"tokens", std::to_string(tokens)},
    ProfileField{"kind", std::move(kind)},
  };
}

Status qwen35_prefill_commit_and_rebegin(MpsGraphGuard& graph_guard,
                                         RequestProfiler& profiler,
                                         std::string_view name,
                                         std::vector<ProfileField> fields) {
  {
    auto commit_span = profiler.scoped(name, fields);
    auto status = graph_guard.commit();
    if (!status.is_ok()) {
      return status;
    }
  }
  {
    auto begin_span = profiler.scoped("qwen35.prefill.graph_begin");
    return graph_guard.begin();
  }
}

Result<std::vector<float>> dequantize_quantized_transposed_f32(
  const GgufTensorInfo& tensor, const GgufTensorBytes& bytes) {
  if (tensor.shape.size() != 2U) {
    return Status::invalid_argument("dense dequant requires a 2D tensor: " +
                                    tensor.name);
  }
  const auto cols_u64 = tensor.shape[0];
  const auto rows_u64 = tensor.shape[1];
  const std::uint64_t block_size =
    tensor.type == 0U ? 1U : (tensor.type == 8U ? 32U : 256U);
  if (cols_u64 == 0 || rows_u64 == 0 || cols_u64 % block_size != 0) {
    return Status::invalid_argument("dense dequant tensor shape is invalid: " +
                                    tensor.name);
  }
  auto cols_result = checked_size_t(cols_u64, tensor.name.c_str());
  auto rows_result = checked_size_t(rows_u64, tensor.name.c_str());
  if (!cols_result.is_ok()) {
    return cols_result.status();
  }
  if (!rows_result.is_ok()) {
    return rows_result.status();
  }
  const auto cols = cols_result.value();
  const auto rows = rows_result.value();
  const auto blocks_per_row = cols / static_cast<std::size_t>(block_size);
  const std::size_t block_bytes =
    tensor.type == 0U ? sizeof(float)
                      : (tensor.type == 8U ? 34U
                      : (tensor.type == 12U ? 144U
                                            : (tensor.type == 13U ? 176U : 210U)));
  std::size_t expected_bytes = 0;
  if (!checked_mul_size(rows, blocks_per_row, expected_bytes) ||
      !checked_mul_size(expected_bytes, block_bytes, expected_bytes)) {
    return Status::invalid_argument("dense dequant byte count overflow: " +
                                    tensor.name);
  }
  if (bytes.data == nullptr || bytes.size < expected_bytes) {
    return Status::invalid_argument("dense dequant tensor bytes are too small: " +
                                    tensor.name);
  }
  std::size_t values = 0;
  if (!checked_mul_size(rows, cols, values)) {
    return Status::invalid_argument("dense dequant value count overflow: " +
                                    tensor.name);
  }

  std::vector<float> transposed(values, 0.0F);
  const auto* data = reinterpret_cast<const std::uint8_t*>(bytes.data);
  for (std::size_t row = 0; row < rows; ++row) {
    const auto row_base = row * blocks_per_row * block_bytes;
    for (std::size_t block = 0; block < blocks_per_row; ++block) {
      const auto* block_data = data + row_base + block * block_bytes;
      const auto col_block = block * static_cast<std::size_t>(block_size);
      if (tensor.type == 0U) {
        float value = 0.0F;
        std::memcpy(&value, block_data, sizeof(float));
        transposed[col_block * rows + row] = value;
      } else if (tensor.type == 8U) {
        const float d = f16_to_float_cpu(block_data);
        const auto* qs = reinterpret_cast<const std::int8_t*>(block_data + 2U);
        for (std::uint32_t i = 0; i < 32U; ++i) {
          transposed[(col_block + i) * rows + row] =
            d * static_cast<float>(qs[i]);
        }
      } else if (tensor.type == 12U) {
        const float d = f16_to_float_cpu(block_data);
        const float dmin = f16_to_float_cpu(block_data + 2U);
        const auto* scales = block_data + 4U;
        const auto* qs = block_data + 16U;
        for (std::uint32_t segment = 0; segment < 4U; ++segment) {
          const auto packed1 = q4_k_scale_min_cpu(segment * 2U, scales);
          const auto packed2 = q4_k_scale_min_cpu(segment * 2U + 1U, scales);
          const float d1 = d * static_cast<float>(packed1 & 0xFFU);
          const float m1 = dmin * static_cast<float>((packed1 >> 8U) & 0xFFU);
          const float d2 = d * static_cast<float>(packed2 & 0xFFU);
          const float m2 = dmin * static_cast<float>((packed2 >> 8U) & 0xFFU);
          const auto* q = qs + segment * 32U;
          const auto col_base = col_block + static_cast<std::size_t>(segment) * 64U;
          for (std::uint32_t i = 0; i < 32U; ++i) {
            const auto packed = q[i];
            transposed[(col_base + i) * rows + row] =
              d1 * static_cast<float>(packed & 0xFU) - m1;
            transposed[(col_base + 32U + i) * rows + row] =
              d2 * static_cast<float>(packed >> 4U) - m2;
          }
        }
      } else if (tensor.type == 13U) {
        const float d = f16_to_float_cpu(block_data);
        const float dmin = f16_to_float_cpu(block_data + 2U);
        const auto* scales = block_data + 4U;
        const auto* qh = block_data + 16U;
        const auto* qs = block_data + 48U;
        std::uint32_t u1 = 1U;
        std::uint32_t u2 = 2U;
        for (std::uint32_t segment = 0; segment < 4U; ++segment) {
          const auto packed1 = q4_k_scale_min_cpu(segment * 2U, scales);
          const auto packed2 = q4_k_scale_min_cpu(segment * 2U + 1U, scales);
          const float d1 = d * static_cast<float>(packed1 & 0xFFU);
          const float m1 = dmin * static_cast<float>((packed1 >> 8U) & 0xFFU);
          const float d2 = d * static_cast<float>(packed2 & 0xFFU);
          const float m2 = dmin * static_cast<float>((packed2 >> 8U) & 0xFFU);
          const auto* q = qs + segment * 32U;
          const auto col_base = col_block + static_cast<std::size_t>(segment) * 64U;
          for (std::uint32_t i = 0; i < 32U; ++i) {
            const auto packed = q[i];
            const auto high1 = (qh[i] & u1) != 0U ? 16U : 0U;
            const auto high2 = (qh[i] & u2) != 0U ? 16U : 0U;
            transposed[(col_base + i) * rows + row] =
              d1 * static_cast<float>((packed & 0xFU) + high1) - m1;
            transposed[(col_base + 32U + i) * rows + row] =
              d2 * static_cast<float>((packed >> 4U) + high2) - m2;
          }
          u1 <<= 2U;
          u2 <<= 2U;
        }
      } else if (tensor.type == 14U) {
        const auto* ql_base = block_data;
        const auto* qh_base = block_data + 128U;
        const auto* scales_base =
          reinterpret_cast<const std::int8_t*>(block_data + 192U);
        const float d = f16_to_float_cpu(block_data + 208U);
        for (std::uint32_t chunk = 0; chunk < 2U; ++chunk) {
          const auto* ql = ql_base + chunk * 64U;
          const auto* qh = qh_base + chunk * 32U;
          const auto* scales = scales_base + chunk * 8U;
          const auto col_base = col_block + static_cast<std::size_t>(chunk) * 128U;
          for (std::uint32_t i = 0; i < 32U; ++i) {
            const auto scale_index = i / 16U;
            const int q1 =
              static_cast<int>((ql[i] & 0xFU) | (((qh[i] >> 0U) & 3U) << 4U)) - 32;
            const int q2 =
              static_cast<int>((ql[i + 32U] & 0xFU) |
                               (((qh[i] >> 2U) & 3U) << 4U)) - 32;
            const int q3 =
              static_cast<int>((ql[i] >> 4U) |
                               (((qh[i] >> 4U) & 3U) << 4U)) - 32;
            const int q4 =
              static_cast<int>((ql[i + 32U] >> 4U) |
                               (((qh[i] >> 6U) & 3U) << 4U)) - 32;
            transposed[(col_base + i) * rows + row] =
              d * static_cast<float>(scales[scale_index + 0U]) *
              static_cast<float>(q1);
            transposed[(col_base + 32U + i) * rows + row] =
              d * static_cast<float>(scales[scale_index + 2U]) *
              static_cast<float>(q2);
            transposed[(col_base + 64U + i) * rows + row] =
              d * static_cast<float>(scales[scale_index + 4U]) *
              static_cast<float>(q3);
            transposed[(col_base + 96U + i) * rows + row] =
              d * static_cast<float>(scales[scale_index + 6U]) *
              static_cast<float>(q4);
          }
        }
      }
    }
  }
  return transposed;
}

Status zero_allocated_buffer(const mps::MpsContext& context, mps::MpsBuffer& buffer,
                             std::uint64_t bytes, const char* name) {
  auto size = checked_size_t(bytes, name);
  if (!size.is_ok()) {
    return size.status();
  }
  return context.zero_buffer(buffer, size.value());
}

Result<Qwen35MetalCache> allocate_qwen35_metal_cache(const mps::MpsContext& context,
                                                     const Qwen35ExecutionPlan& plan) {
  if (plan.cache.kv_cache_bytes % 2U != 0) {
    return Status::invalid_argument("Qwen3.5 KV cache byte count must split evenly");
  }

  Qwen35MetalCache cache;
  cache.kv_cache_f16 = plan.cache.kv_cache_f16;
  cache.key_cache_bytes = plan.cache.kv_cache_bytes / 2U;
  cache.value_cache_bytes = plan.cache.kv_cache_bytes / 2U;
  cache.recurrent_r_cache_bytes = plan.cache.recurrent_r_cache_bytes;
  cache.recurrent_s_cache_bytes = plan.cache.recurrent_s_cache_bytes;

  auto key_size = checked_size_t(cache.key_cache_bytes, "Qwen3.5 key cache");
  if (!key_size.is_ok()) {
    return key_size.status();
  }
  auto key = context.make_buffer(key_size.value());
  if (!key.is_ok()) {
    return key.status();
  }
  cache.key_cache = std::move(key.value());
  auto status = zero_allocated_buffer(context, cache.key_cache, cache.key_cache_bytes,
                                      "Qwen3.5 key cache");
  if (!status.is_ok()) {
    return status;
  }

  auto value_size = checked_size_t(cache.value_cache_bytes, "Qwen3.5 value cache");
  if (!value_size.is_ok()) {
    return value_size.status();
  }
  auto value = context.make_buffer(value_size.value());
  if (!value.is_ok()) {
    return value.status();
  }
  cache.value_cache = std::move(value.value());
  status = zero_allocated_buffer(context, cache.value_cache, cache.value_cache_bytes,
                                 "Qwen3.5 value cache");
  if (!status.is_ok()) {
    return status;
  }

  auto recurrent_r_size =
    checked_size_t(cache.recurrent_r_cache_bytes, "Qwen3.5 recurrent R cache");
  if (!recurrent_r_size.is_ok()) {
    return recurrent_r_size.status();
  }
  auto recurrent_r = context.make_buffer(recurrent_r_size.value());
  if (!recurrent_r.is_ok()) {
    return recurrent_r.status();
  }
  cache.recurrent_r_cache = std::move(recurrent_r.value());
  status = zero_allocated_buffer(context, cache.recurrent_r_cache,
                                 cache.recurrent_r_cache_bytes,
                                 "Qwen3.5 recurrent R cache");
  if (!status.is_ok()) {
    return status;
  }

  auto recurrent_s_size =
    checked_size_t(cache.recurrent_s_cache_bytes, "Qwen3.5 recurrent S cache");
  if (!recurrent_s_size.is_ok()) {
    return recurrent_s_size.status();
  }
  auto recurrent_s = context.make_buffer(recurrent_s_size.value());
  if (!recurrent_s.is_ok()) {
    return recurrent_s.status();
  }
  cache.recurrent_s_cache = std::move(recurrent_s.value());
  status = zero_allocated_buffer(context, cache.recurrent_s_cache,
                                 cache.recurrent_s_cache_bytes,
                                 "Qwen3.5 recurrent S cache");
  if (!status.is_ok()) {
    return status;
  }

  return cache;
}

Result<Qwen35MetalCache> allocate_qwen35_mtp_metal_cache(
  const mps::MpsContext& context, const Qwen35ExecutionPlan& plan,
  std::size_t mtp_layers) {
  if (mtp_layers == 0) {
    return Status::invalid_argument("Qwen3.5 MTP cache requires at least one layer");
  }
  if (plan.cache.attention_capacity_tokens == 0 || plan.cache.kv_dim == 0) {
    return Status::invalid_argument("Qwen3.5 MTP cache dimensions must be positive");
  }

  Qwen35MetalCache cache;
  cache.kv_cache_f16 = plan.cache.kv_cache_f16;
  const auto key_bytes = cache_bytes(mtp_layers, plan.cache.attention_capacity_tokens,
                                     plan.cache.kv_dim, plan.cache.kv_cache_element_bytes,
                                     "Qwen3.5 MTP key cache");
  if (!key_bytes.is_ok()) {
    return key_bytes.status();
  }
  const auto value_bytes = cache_bytes(mtp_layers, plan.cache.attention_capacity_tokens,
                                       plan.cache.kv_dim, plan.cache.kv_cache_element_bytes,
                                       "Qwen3.5 MTP value cache");
  if (!value_bytes.is_ok()) {
    return value_bytes.status();
  }
  cache.key_cache_bytes = key_bytes.value();
  cache.value_cache_bytes = value_bytes.value();

  auto key_size = checked_size_t(cache.key_cache_bytes, "Qwen3.5 MTP key cache");
  if (!key_size.is_ok()) {
    return key_size.status();
  }
  auto key = context.make_buffer(key_size.value());
  if (!key.is_ok()) {
    return key.status();
  }
  cache.key_cache = std::move(key.value());
  auto status = zero_allocated_buffer(context, cache.key_cache, cache.key_cache_bytes,
                                      "Qwen3.5 MTP key cache");
  if (!status.is_ok()) {
    return status;
  }

  auto value_size = checked_size_t(cache.value_cache_bytes, "Qwen3.5 MTP value cache");
  if (!value_size.is_ok()) {
    return value_size.status();
  }
  auto value = context.make_buffer(value_size.value());
  if (!value.is_ok()) {
    return value.status();
  }
  cache.value_cache = std::move(value.value());
  status = zero_allocated_buffer(context, cache.value_cache, cache.value_cache_bytes,
                                 "Qwen3.5 MTP value cache");
  if (!status.is_ok()) {
    return status;
  }
  return cache;
}

Status require_supported_qwen35_weight_type(std::uint32_t type, const std::string& name) {
  switch (type) {
    case 0:   // F32
    case 8:   // Q8_0, uploaded with a dense F32 fallback for MTP projectors.
    case 12:  // Q4_K
    case 13:  // Q5_K
    case 14:  // Q6_K
      return Status::ok();
    default:
      return Status::invalid_argument("unsupported Qwen3.5 GGUF tensor type for Metal upload: " +
                                      name + " type=" + std::to_string(type));
  }
}

void count_uploaded_tensor_type(Qwen35MetalWeightStore& store, std::uint32_t type) {
  switch (type) {
    case 0:
      ++store.f32_tensors;
      break;
    case 8:
      break;
    case 12:
      ++store.q4_k_tensors;
      break;
    case 13:
      ++store.q5_k_tensors;
      break;
    case 14:
      ++store.q6_k_tensors;
      break;
    default:
      break;
  }
}

Result<Qwen35MetalWeightStore> upload_qwen35_metal_weights(
  const mps::MpsContext& context, const GgufFile& gguf,
  bool build_dense_prefill_weights) {
  auto mapped = GgufMappedData::open(gguf);
  if (!mapped.is_ok()) {
    return mapped.status();
  }

  Qwen35MetalWeightStore store;
  store.weights.reserve(gguf.tensors.size());
  store.index_by_name.reserve(gguf.tensors.size());

  for (const auto& tensor : gguf.tensors) {
    auto type_status = require_supported_qwen35_weight_type(tensor.type, tensor.name);
    if (!type_status.is_ok()) {
      return type_status;
    }
    if (tensor.byte_size == 0) {
      return Status::invalid_argument("Qwen3.5 GGUF tensor has zero byte size: " +
                                      tensor.name);
    }
    auto size = checked_size_t(tensor.byte_size, tensor.name.c_str());
    if (!size.is_ok()) {
      return size.status();
    }
    auto bytes = mapped.value().tensor_bytes(tensor);
    if (!bytes.is_ok()) {
      return bytes.status();
    }
    if (bytes.value().data == nullptr || bytes.value().size != size.value()) {
      return Status::internal_error("Qwen3.5 GGUF mapped bytes mismatch for tensor: " +
                                    tensor.name);
    }

    auto buffer = context.make_buffer(size.value());
    if (!buffer.is_ok()) {
      return buffer.status();
    }
    auto copy_status =
      context.copy_to_buffer(buffer.value(), bytes.value().data, bytes.value().size);
    if (!copy_status.is_ok()) {
      return copy_status;
    }

    auto total = add_cache_bytes(store.total_bytes, tensor.byte_size,
                                 "Qwen3.5 Metal weight store");
    if (!total.is_ok()) {
      return total.status();
    }
    store.total_bytes = total.value();
    count_uploaded_tensor_type(store, tensor.type);

    Qwen35MetalWeight uploaded;
    uploaded.name = tensor.name;
    uploaded.type = tensor.type;
    uploaded.shape = tensor.shape;
    uploaded.byte_size = tensor.byte_size;
    uploaded.buffer = std::move(buffer.value());

    if (should_build_dense_transposed_weight(tensor, build_dense_prefill_weights)) {
      auto dense = dequantize_quantized_transposed_f32(tensor, bytes.value());
      if (!dense.is_ok()) {
        return dense.status();
      }
      std::uint64_t dense_bytes_u64 = static_cast<std::uint64_t>(dense.value().size());
      auto dense_byte_status = checked_mul_assign(dense_bytes_u64, sizeof(float),
                                                  tensor.name.c_str());
      if (!dense_byte_status.is_ok()) {
        return dense_byte_status;
      }
      auto dense_bytes = checked_size_t(dense_bytes_u64, tensor.name.c_str());
      if (!dense_bytes.is_ok()) {
        return dense_bytes.status();
      }
      auto dense_buffer = context.make_buffer(dense_bytes.value());
      if (!dense_buffer.is_ok()) {
        return dense_buffer.status();
      }
      auto dense_copy_status =
        context.copy_to_buffer(dense_buffer.value(), dense.value().data(),
                               dense_bytes.value());
      if (!dense_copy_status.is_ok()) {
        return dense_copy_status;
      }
      uploaded.dense_f32_transposed = std::move(dense_buffer.value());
      uploaded.dense_f32_transposed_bytes = dense_bytes_u64;
      auto dense_total = add_cache_bytes(store.dense_f32_transposed_bytes,
                                         dense_bytes_u64,
                                         "Qwen3.5 dense prefill weight store");
      if (!dense_total.is_ok()) {
        return dense_total.status();
      }
      store.dense_f32_transposed_bytes = dense_total.value();
      ++store.dense_f32_transposed_tensors;
    }

    const auto index = store.weights.size();
    store.index_by_name.emplace(uploaded.name, index);
    store.weights.push_back(std::move(uploaded));
  }

  return store;
}

Result<mps::MpsBuffer> make_qwen35_token_id_buffer(
  const mps::MpsContext& context, const std::vector<std::int64_t>& token_ids) {
  if (token_ids.empty()) {
    return Status::invalid_argument("Qwen3.5 token id buffer must not be empty");
  }
  std::vector<std::int32_t> device_ids;
  device_ids.reserve(token_ids.size());
  for (const auto token_id : token_ids) {
    if (token_id < 0 ||
        token_id > static_cast<std::int64_t>(std::numeric_limits<std::int32_t>::max())) {
      return Status::invalid_argument("Qwen3.5 token id exceeds int32 range");
    }
    device_ids.push_back(static_cast<std::int32_t>(token_id));
  }
  auto buffer = context.make_buffer(device_ids.size() * sizeof(std::int32_t));
  if (!buffer.is_ok()) {
    return buffer.status();
  }
  auto status =
    context.copy_to_buffer(buffer.value(), device_ids.data(),
                           device_ids.size() * sizeof(std::int32_t));
  if (!status.is_ok()) {
    return status;
  }
  return std::move(buffer.value());
}

Result<mps::MpsBuffer> run_qwen35_token_embeddings_from_weight(
  const mps::MpsContext& context, const Qwen35MetalWeight& embedding,
  const std::vector<std::int64_t>& token_ids, const mps::MpsBuffer& token_id_buffer,
  std::size_t hidden_size) {
  if (token_ids.empty()) {
    return Status::invalid_argument("Qwen3.5 token embedding requires tokens");
  }
  if (hidden_size == 0) {
    return Status::invalid_argument("Qwen3.5 hidden size must be positive");
  }
  if (embedding.shape.size() != 2 || embedding.shape[0] != hidden_size ||
      embedding.shape[1] == 0) {
    return Status::invalid_argument("Qwen3.5 token embedding shape does not match plan");
  }
  const auto embedding_rows = embedding.shape[1];
  if (embedding_rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument("Qwen3.5 token embedding row count exceeds size_t range");
  }
  for (const auto token_id : token_ids) {
    if (token_id < 0 || static_cast<std::uint64_t>(token_id) >= embedding_rows) {
      return Status::invalid_argument("Qwen3.5 token id is outside embedding rows");
    }
  }
  if (token_ids.size() > std::numeric_limits<std::size_t>::max() / hidden_size) {
    return Status::invalid_argument("Qwen3.5 token embedding output size overflow");
  }

  const auto output_values = token_ids.size() * hidden_size;
  std::uint64_t hidden_bytes_u64 = static_cast<std::uint64_t>(output_values);
  auto byte_status = checked_mul_assign(hidden_bytes_u64, sizeof(float),
                                        "Qwen3.5 token embedding output");
  if (!byte_status.is_ok()) {
    return byte_status;
  }
  auto hidden_bytes = checked_size_t(hidden_bytes_u64, "Qwen3.5 token embedding output");
  if (!hidden_bytes.is_ok()) {
    return hidden_bytes.status();
  }
  auto output = context.make_buffer(hidden_bytes.value());
  if (!output.is_ok()) {
    return output.status();
  }

  const auto rows = static_cast<std::size_t>(embedding_rows);
  Status status = Status::ok();
  switch (embedding.type) {
    case 0: {
      for (std::size_t token_index = 0; token_index < token_ids.size(); ++token_index) {
        const auto row = static_cast<std::size_t>(token_ids[token_index]);
        if (row > std::numeric_limits<std::size_t>::max() / hidden_size) {
          return Status::invalid_argument("Qwen3.5 token embedding source offset overflow");
        }
        status = context.copy_f32_region(embedding.buffer, output.value(),
                                         row * hidden_size,
                                         token_index * hidden_size, hidden_size);
        if (!status.is_ok()) {
          return status;
        }
      }
      break;
    }
    case 12:
      status = context.dequantize_rows_q4_k_f32(embedding.buffer, rows,
                                                token_id_buffer, token_ids.size(),
                                                hidden_size, output.value());
      break;
    case 13:
      status = context.dequantize_rows_q5_k_f32(embedding.buffer, rows,
                                                token_id_buffer, token_ids.size(),
                                                hidden_size, output.value());
      break;
    case 14:
      status = context.dequantize_rows_q6_k_f32(embedding.buffer, rows,
                                                token_id_buffer, token_ids.size(),
                                                hidden_size, output.value());
      break;
    default:
      status = Status::invalid_argument("unsupported Qwen3.5 token embedding tensor type");
      break;
  }
  if (!status.is_ok()) {
    return status;
  }

  return std::move(output.value());
}

Result<mps::MpsBuffer> run_qwen35_token_embeddings(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  const std::vector<std::int64_t>& token_ids, const mps::MpsBuffer& token_id_buffer,
  std::size_t hidden_size) {
  const auto* embedding = find_metal_weight(weights, "token_embd.weight");
  if (embedding == nullptr) {
    return Status::invalid_argument("missing Qwen3.5 Metal weight: token_embd.weight");
  }
  return run_qwen35_token_embeddings_from_weight(
    context, *embedding, token_ids, token_id_buffer, hidden_size);
}

Result<mps::MpsBuffer> run_qwen35_token_embeddings_from_id_buffer(
  const mps::MpsContext& context, const Qwen35MetalWeight& embedding,
  std::size_t token_count, const mps::MpsBuffer& token_id_buffer,
  std::size_t hidden_size) {
  if (token_count == 0) {
    return Status::invalid_argument("Qwen3.5 token embedding requires tokens");
  }
  if (hidden_size == 0) {
    return Status::invalid_argument("Qwen3.5 hidden size must be positive");
  }
  if (embedding.shape.size() != 2 || embedding.shape[0] != hidden_size ||
      embedding.shape[1] == 0) {
    return Status::invalid_argument("Qwen3.5 token embedding shape does not match plan");
  }
  const auto embedding_rows = embedding.shape[1];
  if (embedding_rows > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument("Qwen3.5 token embedding row count exceeds size_t range");
  }
  if (token_count > std::numeric_limits<std::size_t>::max() / hidden_size) {
    return Status::invalid_argument("Qwen3.5 token embedding output size overflow");
  }
  const auto output_values = token_count * hidden_size;
  std::uint64_t hidden_bytes_u64 = static_cast<std::uint64_t>(output_values);
  auto byte_status = checked_mul_assign(hidden_bytes_u64, sizeof(float),
                                        "Qwen3.5 token embedding output");
  if (!byte_status.is_ok()) {
    return byte_status;
  }
  auto hidden_bytes = checked_size_t(hidden_bytes_u64, "Qwen3.5 token embedding output");
  if (!hidden_bytes.is_ok()) {
    return hidden_bytes.status();
  }
  auto output = context.make_buffer(hidden_bytes.value());
  if (!output.is_ok()) {
    return output.status();
  }

  const auto rows = static_cast<std::size_t>(embedding_rows);
  Status status = Status::ok();
  switch (embedding.type) {
    case 12:
      status = context.dequantize_rows_q4_k_f32(embedding.buffer, rows,
                                                token_id_buffer, token_count,
                                                hidden_size, output.value());
      break;
    case 13:
      status = context.dequantize_rows_q5_k_f32(embedding.buffer, rows,
                                                token_id_buffer, token_count,
                                                hidden_size, output.value());
      break;
    case 14:
      status = context.dequantize_rows_q6_k_f32(embedding.buffer, rows,
                                                token_id_buffer, token_count,
                                                hidden_size, output.value());
      break;
    default:
      return Status::invalid_argument(
        "Qwen3.5 dynamic token embedding requires a K-quant embedding: " +
        embedding.name);
  }
  if (!status.is_ok()) {
    return status;
  }
  return std::move(output.value());
}

Result<mps::MpsBuffer> make_qwen35_f32_buffer(const mps::MpsContext& context,
                                              std::size_t values,
                                              const char* name) {
  std::uint64_t bytes_u64 = static_cast<std::uint64_t>(values);
  auto status = checked_mul_assign(bytes_u64, sizeof(float), name);
  if (!status.is_ok()) {
    return status;
  }
  auto bytes = checked_size_t(bytes_u64, name);
  if (!bytes.is_ok()) {
    return bytes.status();
  }
  return context.make_buffer(bytes.value());
}

Result<mps::MpsBuffer> make_qwen35_f32_buffer(const mps::MpsContext& context,
                                              const std::vector<float>& values,
                                              const char* name) {
  auto buffer = make_qwen35_f32_buffer(context, values.size(), name);
  if (!buffer.is_ok()) {
    return buffer.status();
  }
  auto status = context.copy_to_buffer(buffer.value(), values.data(),
                                       values.size() * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  return std::move(buffer.value());
}

Result<mps::MpsBuffer> make_qwen35_mixed_prefill_hidden(
  const mps::MpsContext& context, const Qwen35MetalWeight& token_embedding,
  const Qwen35MixedPrefillPlan& mixed, std::size_t hidden_size) {
  if (hidden_size == 0 || mixed.total_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 mixed prefill hidden requires tokens");
  }
  if (mixed.embedding_width != hidden_size) {
    return Status::invalid_argument(
      "Qwen3.5 mixed prefill embedding width does not match text hidden size");
  }
  if (mixed.total_tokens > std::numeric_limits<std::size_t>::max() / hidden_size) {
    return Status::invalid_argument("Qwen3.5 mixed prefill hidden size overflow");
  }

  auto output = make_qwen35_f32_buffer(
    context, mixed.total_tokens * hidden_size, "Qwen3.5 mixed prefill hidden");
  if (!output.is_ok()) {
    return output.status();
  }

  std::size_t rows_written = 0;
  for (const auto& chunk : mixed.chunks) {
    if (chunk.start_token > mixed.total_tokens ||
        chunk.token_count > mixed.total_tokens - chunk.start_token) {
      return Status::invalid_argument("Qwen3.5 mixed prefill chunk is out of range");
    }
    if (chunk.token_count == 0) {
      continue;
    }
    std::size_t chunk_values = 0;
    if (!checked_mul_size(chunk.token_count, hidden_size, chunk_values)) {
      return Status::invalid_argument("Qwen3.5 mixed prefill chunk size overflow");
    }
    std::size_t destination_values = 0;
    if (!checked_mul_size(chunk.start_token, hidden_size, destination_values)) {
      return Status::invalid_argument(
        "Qwen3.5 mixed prefill destination offset overflow");
    }

    Status status;
    if (chunk.kind == Qwen35MultimodalPromptChunkKind::text) {
      if (chunk.text_tokens.size() != chunk.token_count) {
        return Status::invalid_argument(
          "Qwen3.5 mixed prefill text chunk token count mismatch");
      }
      auto token_buffer = make_qwen35_token_id_buffer(context, chunk.text_tokens);
      if (!token_buffer.is_ok()) {
        return token_buffer.status();
      }
      auto text_hidden = run_qwen35_token_embeddings_from_weight(
        context, token_embedding, chunk.text_tokens, token_buffer.value(), hidden_size);
      if (!text_hidden.is_ok()) {
        return text_hidden.status();
      }
      status = context.copy_f32_region(text_hidden.value(), output.value(), 0,
                                       destination_values, chunk_values);
    } else {
      if (chunk.image_embeddings.size() != chunk_values) {
        return Status::invalid_argument(
          "Qwen3.5 mixed prefill image embedding count mismatch");
      }
      status = context.copy_to_buffer_at(
        output.value(), destination_values * sizeof(float),
        chunk.image_embeddings.data(), chunk_values * sizeof(float));
    }
    if (!status.is_ok()) {
      return status;
    }
    rows_written += chunk.token_count;
  }
  if (rows_written != mixed.total_tokens) {
    return Status::invalid_argument(
      "Qwen3.5 mixed prefill chunks do not cover all tokens");
  }
  return std::move(output.value());
}

Result<mps::MpsBuffer> make_qwen35_mrope_positions(const mps::MpsContext& context,
                                                   std::size_t position) {
  if (position > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max())) {
    return Status::invalid_argument("Qwen3.5 MRoPE position exceeds int32 range");
  }
  const std::int32_t value = static_cast<std::int32_t>(position);
  const std::array<std::int32_t, 4> positions{value, value, value, value};
  auto buffer = context.make_buffer(positions.size() * sizeof(std::int32_t));
  if (!buffer.is_ok()) {
    return buffer.status();
  }
  auto status = context.copy_to_buffer(buffer.value(), positions.data(),
                                       positions.size() * sizeof(std::int32_t));
  if (!status.is_ok()) {
    return status;
  }
  return std::move(buffer.value());
}

Result<mps::MpsBuffer> make_qwen35_mrope_positions(const mps::MpsContext& context,
                                                   std::size_t start_position,
                                                   std::size_t tokens) {
  if (tokens == 0) {
    return Status::invalid_argument("Qwen3.5 batched MRoPE positions require tokens");
  }
  if (start_position > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) ||
      tokens - 1U >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()) - start_position) {
    return Status::invalid_argument("Qwen3.5 batched MRoPE position exceeds int32 range");
  }

  std::vector<std::int32_t> positions(tokens * 4U, 0);
  for (std::size_t section = 0; section < 4U; ++section) {
    for (std::size_t token = 0; token < tokens; ++token) {
      positions[token + tokens * section] =
        static_cast<std::int32_t>(start_position + token);
    }
  }
  auto buffer = context.make_buffer(positions.size() * sizeof(std::int32_t));
  if (!buffer.is_ok()) {
    return buffer.status();
  }
  auto status = context.copy_to_buffer(buffer.value(), positions.data(),
                                       positions.size() * sizeof(std::int32_t));
  if (!status.is_ok()) {
    return status;
  }
  return std::move(buffer.value());
}

Result<mps::MpsBuffer> make_qwen35_mrope_positions(
  const mps::MpsContext& context, const std::vector<std::int32_t>& positions,
  std::size_t total_tokens, std::size_t start_token, std::size_t tokens) {
  if (tokens == 0 || total_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 MRoPE position slice requires tokens");
  }
  if (start_token > total_tokens || tokens > total_tokens - start_token) {
    return Status::invalid_argument("Qwen3.5 MRoPE position slice is out of range");
  }
  if (total_tokens > std::numeric_limits<std::size_t>::max() / 4U ||
      positions.size() != total_tokens * 4U) {
    return Status::invalid_argument("Qwen3.5 MRoPE position buffer shape mismatch");
  }

  std::vector<std::int32_t> chunk_positions(tokens * 4U, 0);
  for (std::size_t section = 0; section < 4U; ++section) {
    const auto source_offset = section * total_tokens + start_token;
    const auto destination_offset = section * tokens;
    std::copy_n(positions.begin() +
                  static_cast<std::ptrdiff_t>(source_offset),
                tokens,
                chunk_positions.begin() +
                  static_cast<std::ptrdiff_t>(destination_offset));
  }
  auto buffer = context.make_buffer(chunk_positions.size() * sizeof(std::int32_t));
  if (!buffer.is_ok()) {
    return buffer.status();
  }
  auto status = context.copy_to_buffer(
    buffer.value(), chunk_positions.data(),
    chunk_positions.size() * sizeof(std::int32_t));
  if (!status.is_ok()) {
    return status;
  }
  return std::move(buffer.value());
}

Result<mps::MpsBuffer> run_qwen35_quant_matvec(const mps::MpsContext& context,
                                               const Qwen35MetalWeight& weight,
                                               const mps::MpsBuffer& input,
                                               std::size_t input_values) {
  if (weight.shape.size() != 2) {
    return Status::invalid_argument("Qwen3.5 matvec weight must be 2D: " + weight.name);
  }
  if (weight.shape[0] != input_values) {
    return Status::invalid_argument("Qwen3.5 matvec input width mismatch for " +
                                    weight.name);
  }
  if (weight.shape[1] >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument("Qwen3.5 matvec output row count exceeds size_t");
  }
  const auto rows = static_cast<std::size_t>(weight.shape[1]);
  auto output = make_qwen35_f32_buffer(context, rows, weight.name.c_str());
  if (!output.is_ok()) {
    return output.status();
  }

  if (weight.dense_f32_transposed.valid()) {
    auto status = context.matmul_f32_f32_device(
      input, weight.dense_f32_transposed, 1, input_values, rows,
      output.value());
    if (!status.is_ok()) {
      return status;
    }
    return std::move(output.value());
  }

  Status status = Status::ok();
  switch (weight.type) {
    case 12:
      status = context.matvec_q4_k_f32_device(weight.buffer, rows, input_values,
                                              input, output.value());
      break;
    case 13:
      status = context.matvec_q5_k_f32_device(weight.buffer, rows, input_values,
                                              input, output.value());
      break;
    case 14:
      status = context.matvec_q6_k_f32_device(weight.buffer, rows, input_values,
                                              input, output.value());
      break;
    default:
      status = Status::invalid_argument("Qwen3.5 matvec weight type is not a K-quant type: " +
                                        weight.name);
      break;
  }
  if (!status.is_ok()) {
    return status;
  }
  return std::move(output.value());
}

Result<mps::MpsBuffer> run_qwen35_quant_matmul(const mps::MpsContext& context,
                                               const Qwen35MetalWeight& weight,
                                               const mps::MpsBuffer& input,
                                               std::size_t tokens,
                                               std::size_t input_values,
                                               RequestProfiler* profiler = nullptr,
                                               std::string_view tag = {}) {
  if (tokens == 0) {
    return Status::invalid_argument("Qwen3.5 matmul requires tokens");
  }
  if (weight.shape.size() != 2) {
    return Status::invalid_argument("Qwen3.5 matmul weight must be 2D: " + weight.name);
  }
  if (weight.shape[0] != input_values) {
    return Status::invalid_argument("Qwen3.5 matmul input width mismatch for " +
                                    weight.name);
  }
  if (weight.shape[1] >
      static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument("Qwen3.5 matmul output row count exceeds size_t");
  }
  const auto rows = static_cast<std::size_t>(weight.shape[1]);
  if (tokens > std::numeric_limits<std::size_t>::max() / rows) {
    return Status::invalid_argument("Qwen3.5 matmul output value count overflow");
  }
  const auto span_name =
    tag.empty() ? std::string{"qwen35.matmul"} : std::string{"qwen35.matmul."} +
                                                   std::string{tag};
  auto span = profiler == nullptr ? ScopedProfileSpan{} : profiler->scoped(
    span_name,
    {
      {"tensor", weight.name},
      {"type", std::to_string(weight.type)},
      {"dispatch", qwen35_quant_matmul_dispatch_name(weight.type, tokens)},
      {"tokens", std::to_string(tokens)},
      {"rows", std::to_string(rows)},
      {"cols", std::to_string(input_values)},
    });
  auto output = make_qwen35_f32_buffer(context, tokens * rows, weight.name.c_str());
  if (!output.is_ok()) {
    return output.status();
  }

  if (weight.dense_f32_transposed.valid()) {
    auto status = context.matmul_f32_f32_device(
      input, weight.dense_f32_transposed, tokens, input_values, rows,
      output.value());
    if (!status.is_ok()) {
      return status;
    }
    return std::move(output.value());
  }

  Status status = Status::ok();
  switch (weight.type) {
    case 12:
      status = context.matmul_q4_k_f32_device(weight.buffer, rows, input_values,
                                              tokens, input, output.value());
      break;
    case 13:
      status = context.matmul_q5_k_f32_device(weight.buffer, rows, input_values,
                                              tokens, input, output.value());
      break;
    case 14:
      status = context.matmul_q6_k_f32_device(weight.buffer, rows, input_values,
                                              tokens, input, output.value());
      break;
    default:
      status = Status::invalid_argument("Qwen3.5 matmul weight type is not a K-quant type: " +
                                        weight.name);
      break;
  }
  if (!status.is_ok()) {
    return status;
  }
  return std::move(output.value());
}

[[maybe_unused]] Result<Qwen35LinearLayerProbe> run_qwen35_linear_layer(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache, const Qwen35LayerBindings& layer,
  const mps::MpsBuffer& hidden, std::size_t hidden_size,
  std::size_t conv_kernel, std::size_t conv_channels,
  std::size_t key_heads, std::size_t key_head_dim,
  std::size_t value_heads, std::size_t value_dim,
  std::size_t recurrent_layer_index,
  std::size_t recurrent_r_elements_per_layer,
  std::size_t recurrent_s_elements_per_layer,
  float rms_eps) {
  if (layer.kind != Qwen35LayerKind::linear_attention) {
    return Status::invalid_argument("Qwen3.5 projection probe requires a linear layer");
  }
  if (key_heads == 0 || key_head_dim == 0 || value_heads == 0 || value_dim == 0) {
    return Status::invalid_argument("Qwen3.5 qkv split dimensions must be positive");
  }
  if (key_heads > std::numeric_limits<std::size_t>::max() / key_head_dim) {
    return Status::invalid_argument("Qwen3.5 qkv split key dimension overflow");
  }
  const auto key_dim = key_heads * key_head_dim;
  if (value_heads > std::numeric_limits<std::size_t>::max() / key_head_dim) {
    return Status::invalid_argument("Qwen3.5 qkv split value dimension overflow");
  }
  if (value_dim != value_heads * key_head_dim) {
    return Status::invalid_argument("Qwen3.5 value dimension does not match GDN head size");
  }
  if (key_dim > (std::numeric_limits<std::size_t>::max() - value_dim) / 2U) {
    return Status::invalid_argument("Qwen3.5 qkv split conv dimension overflow");
  }
  if (conv_channels != 2U * key_dim + value_dim) {
    return Status::invalid_argument("Qwen3.5 qkv split dimensions do not match conv width");
  }
  if (recurrent_layer_index >
      std::numeric_limits<std::size_t>::max() / recurrent_r_elements_per_layer) {
    return Status::invalid_argument("Qwen3.5 recurrent R cache offset overflow");
  }
  if (recurrent_layer_index >
      std::numeric_limits<std::size_t>::max() / recurrent_s_elements_per_layer) {
    return Status::invalid_argument("Qwen3.5 recurrent S cache offset overflow");
  }
  const auto recurrent_r_offset =
    recurrent_layer_index * recurrent_r_elements_per_layer;
  const auto recurrent_s_offset =
    recurrent_layer_index * recurrent_s_elements_per_layer;

  const auto* attn_norm = find_metal_weight(weights, layer.attn_norm.name);
  const auto* attn_post_norm = find_metal_weight(weights, layer.attn_post_norm.name);
  const auto* qkv = find_metal_weight(weights, layer.linear_attention.qkv.name);
  const auto* gate = find_metal_weight(weights, layer.linear_attention.gate.name);
  const auto* conv1d = find_metal_weight(weights, layer.linear_attention.conv1d.name);
  const auto* ssm_norm = find_metal_weight(weights, layer.linear_attention.norm.name);
  const auto* ssm_out = find_metal_weight(weights, layer.linear_attention.output.name);
  const auto* ffn_up = find_metal_weight(weights, layer.ffn_up.name);
  const auto* ffn_gate = find_metal_weight(weights, layer.ffn_gate.name);
  const auto* ffn_down = find_metal_weight(weights, layer.ffn_down.name);
  const auto* beta = find_metal_weight(weights, layer.linear_attention.beta.name);
  const auto* alpha = find_metal_weight(weights, layer.linear_attention.alpha.name);
  const auto* dt_bias = find_metal_weight(weights, layer.linear_attention.dt_bias.name);
  const auto* ssm_a = find_metal_weight(weights, layer.linear_attention.a.name);
  if (attn_norm == nullptr || attn_post_norm == nullptr || qkv == nullptr ||
      gate == nullptr || conv1d == nullptr || ssm_norm == nullptr ||
      ssm_out == nullptr || ffn_up == nullptr || ffn_gate == nullptr ||
      ffn_down == nullptr || beta == nullptr || alpha == nullptr ||
      dt_bias == nullptr || ssm_a == nullptr) {
    return Status::invalid_argument("Qwen3.5 projection probe is missing layer weights");
  }
  if (attn_norm->type != 0 || attn_norm->shape != std::vector<std::uint64_t>{hidden_size}) {
    return Status::invalid_argument("Qwen3.5 attention norm weight shape/type mismatch");
  }
  if (attn_post_norm->type != 0 ||
      attn_post_norm->shape != std::vector<std::uint64_t>{hidden_size}) {
    return Status::invalid_argument("Qwen3.5 post attention norm shape/type mismatch");
  }
  if (ssm_norm->type != 0 ||
      ssm_norm->shape != std::vector<std::uint64_t>{key_head_dim}) {
    return Status::invalid_argument("Qwen3.5 SSM norm weight shape/type mismatch");
  }
  const auto dt_value_heads = layer.linear_attention.dt_bias.shape.empty()
                                ? 0U
                                : layer.linear_attention.dt_bias.shape[0];
  if (dt_value_heads != value_heads) {
    return Status::invalid_argument("Qwen3.5 recurrent vector width mismatch");
  }
  if (dt_bias->type != 0 || ssm_a->type != 0 ||
      dt_bias->shape != std::vector<std::uint64_t>{value_heads} ||
      ssm_a->shape != std::vector<std::uint64_t>{value_heads}) {
    return Status::invalid_argument("Qwen3.5 recurrent gate vector shape/type mismatch");
  }
  if (conv1d->type != 0 ||
      conv1d->shape != std::vector<std::uint64_t>{conv_kernel, conv_channels}) {
    return Status::invalid_argument("Qwen3.5 conv1d weight shape/type mismatch");
  }

  Qwen35LinearLayerProbe probe;
  auto normed = make_qwen35_f32_buffer(context, hidden_size, "Qwen3.5 normed hidden");
  if (!normed.is_ok()) {
    return normed.status();
  }
  auto status = context.rms_norm_f32_f32(hidden, attn_norm->buffer, hidden_size,
                                         rms_eps, normed.value());
  if (!status.is_ok()) {
    return status;
  }
  probe.normed_hidden = std::move(normed.value());

  auto qkv_out = run_qwen35_quant_matvec(context, *qkv, probe.normed_hidden, hidden_size);
  if (!qkv_out.is_ok()) {
    return qkv_out.status();
  }
  probe.qkv_values = static_cast<std::size_t>(qkv->shape[1]);
  probe.qkv = std::move(qkv_out.value());
  if (probe.qkv_values != conv_channels) {
    return Status::invalid_argument("Qwen3.5 qkv projection width does not match conv width");
  }

  auto conv_output = make_qwen35_f32_buffer(context, conv_channels,
                                            "Qwen3.5 conv output");
  if (!conv_output.is_ok()) {
    return conv_output.status();
  }
  status = context.ssm_conv1_f32_stateful_at(cache.recurrent_r_cache,
                                             recurrent_r_offset, probe.qkv,
                                             conv1d->buffer, conv_kernel,
                                             conv_channels, conv_output.value());
  if (!status.is_ok()) {
    return status;
  }
  status = context.silu_f32_in_place(conv_output.value(), conv_channels);
  if (!status.is_ok()) {
    return status;
  }
  probe.conv_output_values = conv_channels;
  probe.conv_output = std::move(conv_output.value());

  auto q_conv = make_qwen35_f32_buffer(context, key_dim, "Qwen3.5 q_conv");
  if (!q_conv.is_ok()) {
    return q_conv.status();
  }
  status = context.copy_f32_region(probe.conv_output, q_conv.value(), 0, 0, key_dim);
  if (!status.is_ok()) {
    return status;
  }
  status = context.l2_norm_f32_in_place(q_conv.value(), key_heads, key_head_dim, rms_eps);
  if (!status.is_ok()) {
    return status;
  }
  probe.q_conv_values = key_dim;
  probe.q_conv = std::move(q_conv.value());

  auto k_conv = make_qwen35_f32_buffer(context, key_dim, "Qwen3.5 k_conv");
  if (!k_conv.is_ok()) {
    return k_conv.status();
  }
  status = context.copy_f32_region(probe.conv_output, k_conv.value(), key_dim, 0, key_dim);
  if (!status.is_ok()) {
    return status;
  }
  status = context.l2_norm_f32_in_place(k_conv.value(), key_heads, key_head_dim, rms_eps);
  if (!status.is_ok()) {
    return status;
  }
  probe.k_conv_values = key_dim;
  probe.k_conv = std::move(k_conv.value());

  auto v_conv = make_qwen35_f32_buffer(context, value_dim, "Qwen3.5 v_conv");
  if (!v_conv.is_ok()) {
    return v_conv.status();
  }
  status = context.copy_f32_region(probe.conv_output, v_conv.value(), 2U * key_dim,
                                   0, value_dim);
  if (!status.is_ok()) {
    return status;
  }
  probe.v_conv_values = value_dim;
  probe.v_conv = std::move(v_conv.value());

  auto gate_out = run_qwen35_quant_matvec(context, *gate, probe.normed_hidden, hidden_size);
  if (!gate_out.is_ok()) {
    return gate_out.status();
  }
  probe.gate_values = static_cast<std::size_t>(gate->shape[1]);
  probe.gate = std::move(gate_out.value());

  auto beta_out = run_qwen35_quant_matvec(context, *beta, probe.normed_hidden, hidden_size);
  if (!beta_out.is_ok()) {
    return beta_out.status();
  }
  probe.beta_values = static_cast<std::size_t>(beta->shape[1]);
  probe.beta = std::move(beta_out.value());
  status = context.sigmoid_f32_in_place(probe.beta, probe.beta_values);
  if (!status.is_ok()) {
    return status;
  }

  auto alpha_out = run_qwen35_quant_matvec(context, *alpha, probe.normed_hidden, hidden_size);
  if (!alpha_out.is_ok()) {
    return alpha_out.status();
  }
  probe.alpha_values = static_cast<std::size_t>(alpha->shape[1]);
  probe.alpha = std::move(alpha_out.value());
  status = context.add_f32_in_place(probe.alpha, dt_bias->buffer, probe.alpha_values);
  if (!status.is_ok()) {
    return status;
  }
  status = context.softplus_f32_in_place(probe.alpha, probe.alpha_values);
  if (!status.is_ok()) {
    return status;
  }
  status = context.mul_f32_in_place(probe.alpha, ssm_a->buffer, probe.alpha_values);
  if (!status.is_ok()) {
    return status;
  }
  probe.recurrent_gate_values = probe.alpha_values;
  probe.recurrent_gate = std::move(probe.alpha);

  auto attn_output = make_qwen35_f32_buffer(context, value_dim,
                                            "Qwen3.5 recurrent attention output");
  if (!attn_output.is_ok()) {
    return attn_output.status();
  }
  status = context.gated_delta_net_f32_in_place_at(
    probe.q_conv, probe.k_conv, probe.v_conv, probe.recurrent_gate, probe.beta,
    cache.recurrent_s_cache, recurrent_s_offset, key_heads, value_heads,
    key_head_dim, attn_output.value());
  if (!status.is_ok()) {
    return status;
  }
  probe.attn_output_values = value_dim;
  probe.attn_output = std::move(attn_output.value());

  status = context.qk_norm_f32_f32(probe.attn_output, ssm_norm->buffer,
                                   value_heads, key_head_dim, rms_eps);
  if (!status.is_ok()) {
    return status;
  }
  status = context.silu_f32_in_place(probe.gate, probe.gate_values);
  if (!status.is_ok()) {
    return status;
  }
  status = context.mul_f32_in_place(probe.attn_output, probe.gate, value_dim);
  if (!status.is_ok()) {
    return status;
  }
  probe.norm_gated_values = value_dim;

  auto linear_attn_output =
    run_qwen35_quant_matvec(context, *ssm_out, probe.attn_output, value_dim);
  if (!linear_attn_output.is_ok()) {
    return linear_attn_output.status();
  }
  probe.linear_attn_output_values = static_cast<std::size_t>(ssm_out->shape[1]);
  probe.linear_attn_output = std::move(linear_attn_output.value());

  if (probe.linear_attn_output_values != hidden_size) {
    return Status::invalid_argument("Qwen3.5 linear attention output width mismatch");
  }
  status = context.add_f32_in_place(probe.linear_attn_output, hidden, hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  probe.attn_residual_values = hidden_size;

  auto post_attn_norm =
    make_qwen35_f32_buffer(context, hidden_size, "Qwen3.5 post attention norm");
  if (!post_attn_norm.is_ok()) {
    return post_attn_norm.status();
  }
  status = context.rms_norm_f32_f32(probe.linear_attn_output,
                                    attn_post_norm->buffer, hidden_size,
                                    rms_eps, post_attn_norm.value());
  if (!status.is_ok()) {
    return status;
  }
  probe.post_attn_norm_values = hidden_size;
  probe.post_attn_norm = std::move(post_attn_norm.value());

  auto ffn_up_out =
    run_qwen35_quant_matvec(context, *ffn_up, probe.post_attn_norm, hidden_size);
  if (!ffn_up_out.is_ok()) {
    return ffn_up_out.status();
  }
  probe.ffn_up_values = static_cast<std::size_t>(ffn_up->shape[1]);
  probe.ffn_up = std::move(ffn_up_out.value());

  auto ffn_gate_out =
    run_qwen35_quant_matvec(context, *ffn_gate, probe.post_attn_norm, hidden_size);
  if (!ffn_gate_out.is_ok()) {
    return ffn_gate_out.status();
  }
  probe.ffn_gate_values = static_cast<std::size_t>(ffn_gate->shape[1]);
  if (probe.ffn_gate_values != probe.ffn_up_values) {
    return Status::invalid_argument("Qwen3.5 FFN gate/up width mismatch");
  }
  probe.ffn_gate = std::move(ffn_gate_out.value());
  status = context.silu_mul_f32_in_place(probe.ffn_gate, probe.ffn_up,
                                         probe.ffn_gate_values);
  if (!status.is_ok()) {
    return status;
  }
  probe.ffn_swiglu_values = probe.ffn_gate_values;

  auto ffn_down_out =
    run_qwen35_quant_matvec(context, *ffn_down, probe.ffn_gate, probe.ffn_gate_values);
  if (!ffn_down_out.is_ok()) {
    return ffn_down_out.status();
  }
  probe.layer_output_values = static_cast<std::size_t>(ffn_down->shape[1]);
  if (probe.layer_output_values != hidden_size) {
    return Status::invalid_argument("Qwen3.5 FFN down output width mismatch");
  }
  status = context.add_f32_in_place(ffn_down_out.value(), probe.linear_attn_output,
                                    hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  probe.layer_output = std::move(ffn_down_out.value());

  return probe;
}

[[maybe_unused]] Result<std::pair<mps::MpsBuffer, Qwen35FullAttentionLayerStats>>
run_qwen35_full_attention_layer(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache, const Qwen35LayerBindings& layer,
  const mps::MpsBuffer& hidden, std::size_t hidden_size,
  std::size_t attention_heads, std::size_t kv_heads, std::size_t head_dim,
  std::size_t full_attention_layer_index, std::size_t position,
  std::size_t capacity_tokens, const std::array<std::size_t, 4>& mrope_sections,
  float rope_theta, float rms_eps) {
  if (layer.kind != Qwen35LayerKind::full_attention) {
    return Status::invalid_argument("Qwen3.5 full attention runner requires a full layer");
  }
  if (attention_heads == 0 || kv_heads == 0 || head_dim == 0 ||
      hidden_size == 0 || capacity_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 full attention dimensions must be positive");
  }
  const auto attn_dim = attention_heads * head_dim;
  const auto kv_dim = kv_heads * head_dim;
  const auto mrope_pairs =
    mrope_sections[0] + mrope_sections[1] + mrope_sections[2] + mrope_sections[3];
  const auto mrope_dims = 2U * mrope_pairs;
  if (mrope_dims == 0 || mrope_dims > head_dim) {
    return Status::invalid_argument("Qwen3.5 MRoPE dimensions are invalid");
  }

  const auto* attn_norm = find_metal_weight(weights, layer.attn_norm.name);
  const auto* attn_post_norm = find_metal_weight(weights, layer.attn_post_norm.name);
  const auto* q = find_metal_weight(weights, layer.full_attention.q.name);
  const auto* k = find_metal_weight(weights, layer.full_attention.k.name);
  const auto* v = find_metal_weight(weights, layer.full_attention.v.name);
  const auto* output = find_metal_weight(weights, layer.full_attention.output.name);
  const auto* q_norm = find_metal_weight(weights, layer.full_attention.q_norm.name);
  const auto* k_norm = find_metal_weight(weights, layer.full_attention.k_norm.name);
  const auto* ffn_up = find_metal_weight(weights, layer.ffn_up.name);
  const auto* ffn_gate = find_metal_weight(weights, layer.ffn_gate.name);
  const auto* ffn_down = find_metal_weight(weights, layer.ffn_down.name);
  if (attn_norm == nullptr || attn_post_norm == nullptr || q == nullptr ||
      k == nullptr || v == nullptr || output == nullptr || q_norm == nullptr ||
      k_norm == nullptr || ffn_up == nullptr || ffn_gate == nullptr ||
      ffn_down == nullptr) {
    return Status::invalid_argument("Qwen3.5 full attention layer is missing weights");
  }
  if (attn_norm->type != 0 || attn_post_norm->type != 0 ||
      q_norm->type != 0 || k_norm->type != 0 ||
      attn_norm->shape != std::vector<std::uint64_t>{hidden_size} ||
      attn_post_norm->shape != std::vector<std::uint64_t>{hidden_size} ||
      q_norm->shape != std::vector<std::uint64_t>{head_dim} ||
      k_norm->shape != std::vector<std::uint64_t>{head_dim}) {
    return Status::invalid_argument("Qwen3.5 full attention norm shape/type mismatch");
  }

  Qwen35FullAttentionLayerStats stats;
  auto normed = make_qwen35_f32_buffer(context, hidden_size,
                                       "Qwen3.5 full attention normed hidden");
  if (!normed.is_ok()) {
    return normed.status();
  }
  auto status = context.rms_norm_f32_f32(hidden, attn_norm->buffer, hidden_size,
                                         rms_eps, normed.value());
  if (!status.is_ok()) {
    return status;
  }

  auto q_full = run_qwen35_quant_matvec(context, *q, normed.value(), hidden_size);
  if (!q_full.is_ok()) {
    return q_full.status();
  }
  stats.q_full_values = static_cast<std::size_t>(q->shape[1]);
  if (stats.q_full_values != 2U * attn_dim) {
    return Status::invalid_argument("Qwen3.5 full attention q projection width mismatch");
  }

  auto query = make_qwen35_f32_buffer(context, attn_dim, "Qwen3.5 full query");
  auto gate = make_qwen35_f32_buffer(context, attn_dim, "Qwen3.5 full gate");
  if (!query.is_ok()) {
    return query.status();
  }
  if (!gate.is_ok()) {
    return gate.status();
  }
  // llama.cpp views Qcur_full as [head][q, gate][head_dim], not as
  // [all_q][all_gate]. Preserve that layout when splitting the projection.
  status = context.copy_f32_rows(q_full.value(), query.value(),
                                 attention_heads, head_dim, 2U * head_dim,
                                 0, head_dim, 0);
  if (!status.is_ok()) {
    return status;
  }
  status = context.copy_f32_rows(q_full.value(), gate.value(),
                                 attention_heads, head_dim, 2U * head_dim,
                                 head_dim, head_dim, 0);
  if (!status.is_ok()) {
    return status;
  }
  stats.query_values = attn_dim;
  stats.gate_values = attn_dim;
  status = context.qk_norm_f32_f32(query.value(), q_norm->buffer,
                                   attention_heads, head_dim, rms_eps);
  if (!status.is_ok()) {
    return status;
  }

  auto key = run_qwen35_quant_matvec(context, *k, normed.value(), hidden_size);
  if (!key.is_ok()) {
    return key.status();
  }
  stats.key_values = static_cast<std::size_t>(k->shape[1]);
  if (stats.key_values != kv_dim) {
    return Status::invalid_argument("Qwen3.5 full attention key width mismatch");
  }
  status = context.qk_norm_f32_f32(key.value(), k_norm->buffer,
                                   kv_heads, head_dim, rms_eps);
  if (!status.is_ok()) {
    return status;
  }

  auto value = run_qwen35_quant_matvec(context, *v, normed.value(), hidden_size);
  if (!value.is_ok()) {
    return value.status();
  }
  stats.value_values = static_cast<std::size_t>(v->shape[1]);
  if (stats.value_values != kv_dim) {
    return Status::invalid_argument("Qwen3.5 full attention value width mismatch");
  }

  auto positions = make_qwen35_mrope_positions(context, position);
  if (!positions.is_ok()) {
    return positions.status();
  }
  status = context.mrope_f32_in_place(query.value(), positions.value(), 1,
                                      attention_heads, head_dim, mrope_dims,
                                      mrope_sections[0], mrope_sections[1],
                                      mrope_sections[2], mrope_sections[3],
                                      rope_theta);
  if (!status.is_ok()) {
    return status;
  }
  status = context.mrope_f32_in_place(key.value(), positions.value(), 1,
                                      kv_heads, head_dim, mrope_dims,
                                      mrope_sections[0], mrope_sections[1],
                                      mrope_sections[2], mrope_sections[3],
                                      rope_theta);
  if (!status.is_ok()) {
    return status;
  }

  const auto cache_offset =
    (full_attention_layer_index * capacity_tokens + position) * kv_dim;
  if (cache.kv_cache_f16) {
    status = context.copy_f32_rows_to_f16(key.value(), cache.key_cache, 1,
                                          kv_dim, kv_dim, 0, kv_dim,
                                          cache_offset);
  } else {
    status = context.copy_f32_region(key.value(), cache.key_cache, 0,
                                     cache_offset, kv_dim);
  }
  if (!status.is_ok()) {
    return status;
  }
  if (cache.kv_cache_f16) {
    status = context.copy_f32_rows_to_f16(value.value(), cache.value_cache, 1,
                                          kv_dim, kv_dim, 0, kv_dim,
                                          cache_offset);
  } else {
    status = context.copy_f32_region(value.value(), cache.value_cache, 0,
                                     cache_offset, kv_dim);
  }
  if (!status.is_ok()) {
    return status;
  }

  auto attention = make_qwen35_f32_buffer(context, attn_dim,
                                          "Qwen3.5 full attention output");
  if (!attention.is_ok()) {
    return attention.status();
  }
  if (cache.kv_cache_f16) {
    status = context.attention_f32_f16_kv(
      query.value(), cache.key_cache, cache.value_cache,
      full_attention_layer_index, position, capacity_tokens, attention_heads,
      kv_heads, head_dim, attention.value());
  } else {
    status = context.attention_f32(
      query.value(), cache.key_cache, cache.value_cache,
      full_attention_layer_index, position, capacity_tokens, attention_heads,
      kv_heads, head_dim, attention.value());
  }
  if (!status.is_ok()) {
    return status;
  }
  stats.attention_values = attn_dim;
  status = context.sigmoid_f32_in_place(gate.value(), attn_dim);
  if (!status.is_ok()) {
    return status;
  }
  status = context.mul_f32_in_place(attention.value(), gate.value(), attn_dim);
  if (!status.is_ok()) {
    return status;
  }

  auto attn_output = run_qwen35_quant_matvec(context, *output, attention.value(), attn_dim);
  if (!attn_output.is_ok()) {
    return attn_output.status();
  }
  stats.attention_output_values = static_cast<std::size_t>(output->shape[1]);
  if (stats.attention_output_values != hidden_size) {
    return Status::invalid_argument("Qwen3.5 full attention output width mismatch");
  }
  status = context.add_f32_in_place(attn_output.value(), hidden, hidden_size);
  if (!status.is_ok()) {
    return status;
  }

  auto post_attn_norm = make_qwen35_f32_buffer(context, hidden_size,
                                               "Qwen3.5 full post attention norm");
  if (!post_attn_norm.is_ok()) {
    return post_attn_norm.status();
  }
  status = context.rms_norm_f32_f32(attn_output.value(), attn_post_norm->buffer,
                                    hidden_size, rms_eps, post_attn_norm.value());
  if (!status.is_ok()) {
    return status;
  }

  auto ffn_up_out =
    run_qwen35_quant_matvec(context, *ffn_up, post_attn_norm.value(), hidden_size);
  if (!ffn_up_out.is_ok()) {
    return ffn_up_out.status();
  }
  auto ffn_gate_out =
    run_qwen35_quant_matvec(context, *ffn_gate, post_attn_norm.value(), hidden_size);
  if (!ffn_gate_out.is_ok()) {
    return ffn_gate_out.status();
  }
  const auto ffn_values = static_cast<std::size_t>(ffn_gate->shape[1]);
  if (ffn_values != static_cast<std::size_t>(ffn_up->shape[1])) {
    return Status::invalid_argument("Qwen3.5 full FFN gate/up width mismatch");
  }
  status = context.silu_mul_f32_in_place(ffn_gate_out.value(), ffn_up_out.value(),
                                         ffn_values);
  if (!status.is_ok()) {
    return status;
  }
  auto ffn_down_out = run_qwen35_quant_matvec(context, *ffn_down,
                                              ffn_gate_out.value(), ffn_values);
  if (!ffn_down_out.is_ok()) {
    return ffn_down_out.status();
  }
  stats.layer_output_values = static_cast<std::size_t>(ffn_down->shape[1]);
  if (stats.layer_output_values != hidden_size) {
    return Status::invalid_argument("Qwen3.5 full FFN down output width mismatch");
  }
  status = context.add_f32_in_place(ffn_down_out.value(), attn_output.value(),
                                    hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  return std::pair<mps::MpsBuffer, Qwen35FullAttentionLayerStats>{
    std::move(ffn_down_out.value()),
    stats,
  };
}

Result<mps::MpsBuffer> run_qwen35_linear_layer_chunk(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache, const Qwen35LayerBindings& layer,
  const mps::MpsBuffer& hidden, std::size_t tokens, std::size_t hidden_size,
  std::size_t conv_kernel, std::size_t conv_channels,
  std::size_t key_heads, std::size_t key_head_dim,
  std::size_t value_heads, std::size_t value_dim,
  std::size_t recurrent_layer_index,
  std::size_t recurrent_r_elements_per_layer,
  std::size_t recurrent_s_elements_per_layer,
  cpu::DebugDumper* dumper, std::size_t layer_index, std::size_t chunk_start,
  float rms_eps, RequestProfiler* profiler) {
  if (layer.kind != Qwen35LayerKind::linear_attention) {
    return Status::invalid_argument("Qwen3.5 chunk linear runner requires a linear layer");
  }
  if (tokens == 0 || hidden_size == 0 || key_heads == 0 ||
      key_head_dim == 0 || value_heads == 0 || value_dim == 0) {
    return Status::invalid_argument("Qwen3.5 chunk linear dimensions must be positive");
  }
  const auto key_dim = key_heads * key_head_dim;
  if (value_dim != value_heads * key_head_dim ||
      conv_channels != 2U * key_dim + value_dim) {
    return Status::invalid_argument("Qwen3.5 chunk linear split dimensions mismatch");
  }
  if (recurrent_layer_index >
      std::numeric_limits<std::size_t>::max() / recurrent_r_elements_per_layer ||
      recurrent_layer_index >
      std::numeric_limits<std::size_t>::max() / recurrent_s_elements_per_layer) {
    return Status::invalid_argument("Qwen3.5 chunk recurrent cache offset overflow");
  }
  const auto recurrent_r_offset =
    recurrent_layer_index * recurrent_r_elements_per_layer;
  const auto recurrent_s_offset =
    recurrent_layer_index * recurrent_s_elements_per_layer;

  const auto* attn_norm = find_metal_weight(weights, layer.attn_norm.name);
  const auto* attn_post_norm = find_metal_weight(weights, layer.attn_post_norm.name);
  const auto* qkv = find_metal_weight(weights, layer.linear_attention.qkv.name);
  const auto* gate = find_metal_weight(weights, layer.linear_attention.gate.name);
  const auto* conv1d = find_metal_weight(weights, layer.linear_attention.conv1d.name);
  const auto* ssm_norm = find_metal_weight(weights, layer.linear_attention.norm.name);
  const auto* ssm_out = find_metal_weight(weights, layer.linear_attention.output.name);
  const auto* ffn_up = find_metal_weight(weights, layer.ffn_up.name);
  const auto* ffn_gate = find_metal_weight(weights, layer.ffn_gate.name);
  const auto* ffn_down = find_metal_weight(weights, layer.ffn_down.name);
  const auto* beta = find_metal_weight(weights, layer.linear_attention.beta.name);
  const auto* alpha = find_metal_weight(weights, layer.linear_attention.alpha.name);
  const auto* dt_bias = find_metal_weight(weights, layer.linear_attention.dt_bias.name);
  const auto* ssm_a = find_metal_weight(weights, layer.linear_attention.a.name);
  if (attn_norm == nullptr || attn_post_norm == nullptr || qkv == nullptr ||
      gate == nullptr || conv1d == nullptr || ssm_norm == nullptr ||
      ssm_out == nullptr || ffn_up == nullptr || ffn_gate == nullptr ||
      ffn_down == nullptr || beta == nullptr || alpha == nullptr ||
      dt_bias == nullptr || ssm_a == nullptr) {
    return Status::invalid_argument("Qwen3.5 chunk linear layer is missing weights");
  }
  if (attn_norm->type != 0 || attn_post_norm->type != 0 ||
      ssm_norm->type != 0 || dt_bias->type != 0 || ssm_a->type != 0 ||
      attn_norm->shape != std::vector<std::uint64_t>{hidden_size} ||
      attn_post_norm->shape != std::vector<std::uint64_t>{hidden_size} ||
      ssm_norm->shape != std::vector<std::uint64_t>{key_head_dim} ||
      dt_bias->shape != std::vector<std::uint64_t>{value_heads} ||
      ssm_a->shape != std::vector<std::uint64_t>{value_heads}) {
    return Status::invalid_argument("Qwen3.5 chunk linear F32 vector shape/type mismatch");
  }
  if (conv1d->type != 0 ||
      conv1d->shape != std::vector<std::uint64_t>{conv_kernel, conv_channels}) {
    return Status::invalid_argument("Qwen3.5 chunk linear conv1d shape/type mismatch");
  }

  auto normed = make_qwen35_f32_buffer(context, tokens * hidden_size,
                                       "Qwen3.5 chunk linear normed hidden");
  if (!normed.is_ok()) {
    return normed.status();
  }
  auto status = context.rms_norm_f32_f32_batched(
    hidden, attn_norm->buffer, tokens, hidden_size, rms_eps, normed.value());
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "attn_norm"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(hidden_size)},
    normed.value(), tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }

  auto qkv_out = run_qwen35_quant_matmul(context, *qkv, normed.value(),
                                         tokens, hidden_size, profiler,
                                         "linear.qkv");
  if (!qkv_out.is_ok()) {
    return qkv_out.status();
  }
  if (qkv->shape[1] != conv_channels) {
    return Status::invalid_argument("Qwen3.5 chunk qkv projection width mismatch");
  }
  status = dump_qwen35_mps_f32(
    context, dumper,
    qwen35_layer_tensor_name(chunk_start, layer_index, "linear_attn_qkv_mixed"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(conv_channels)},
    qkv_out.value(), tokens * conv_channels);
  if (!status.is_ok()) {
    return status;
  }

  auto conv_input = make_qwen35_f32_buffer(
    context, conv_channels * (tokens + conv_kernel - 1U),
    "Qwen3.5 chunk SSM conv input");
  if (!conv_input.is_ok()) {
    return conv_input.status();
  }
  status = context.build_ssm_conv_state_f32(
    cache.recurrent_r_cache, recurrent_r_offset, qkv_out.value(), conv_kernel,
    conv_channels, tokens, conv_input.value());
  if (!status.is_ok()) {
    return status;
  }
  auto conv_output = make_qwen35_f32_buffer(
    context, tokens * conv_channels, "Qwen3.5 chunk SSM conv output");
  if (!conv_output.is_ok()) {
    return conv_output.status();
  }
  status = context.ssm_conv_f32(conv_input.value(), conv1d->buffer, conv_kernel,
                                conv_channels, tokens, 1, conv_output.value());
  if (!status.is_ok()) {
    return status;
  }
  status = context.silu_f32_in_place(conv_output.value(), tokens * conv_channels);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper,
    qwen35_layer_tensor_name(chunk_start, layer_index, "conv_output_silu"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(conv_channels)},
    conv_output.value(), tokens * conv_channels);
  if (!status.is_ok()) {
    return status;
  }

  auto q_conv = make_qwen35_f32_buffer(context, tokens * key_dim,
                                       "Qwen3.5 chunk q_conv");
  auto k_conv = make_qwen35_f32_buffer(context, tokens * key_dim,
                                       "Qwen3.5 chunk k_conv");
  auto v_conv = make_qwen35_f32_buffer(context, tokens * value_dim,
                                       "Qwen3.5 chunk v_conv");
  if (!q_conv.is_ok()) {
    return q_conv.status();
  }
  if (!k_conv.is_ok()) {
    return k_conv.status();
  }
  if (!v_conv.is_ok()) {
    return v_conv.status();
  }
  if (dumper == nullptr) {
    status = context.split_qkv_l2_norm_f32_qwen35(
      conv_output.value(), q_conv.value(), k_conv.value(), v_conv.value(),
      tokens, key_heads, value_heads, key_head_dim, rms_eps);
    if (!status.is_ok()) {
      return status;
    }
  } else {
    status = context.copy_f32_rows(conv_output.value(), q_conv.value(),
                                   tokens, key_dim, conv_channels, 0,
                                   key_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    status = context.copy_f32_rows(conv_output.value(), k_conv.value(),
                                   tokens, key_dim, conv_channels, key_dim,
                                   key_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    status = context.copy_f32_rows(conv_output.value(), v_conv.value(),
                                   tokens, value_dim, conv_channels, 2U * key_dim,
                                   value_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "q_conv"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(key_heads),
       static_cast<std::uint64_t>(key_head_dim)},
      q_conv.value(), tokens * key_dim);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "k_conv"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(key_heads),
       static_cast<std::uint64_t>(key_head_dim)},
      k_conv.value(), tokens * key_dim);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "v_conv"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads),
       static_cast<std::uint64_t>(key_head_dim)},
      v_conv.value(), tokens * value_dim);
    if (!status.is_ok()) {
      return status;
    }
    status = context.l2_norm_f32_in_place(q_conv.value(), tokens * key_heads,
                                          key_head_dim, rms_eps);
    if (!status.is_ok()) {
      return status;
    }
    status = context.l2_norm_f32_in_place(k_conv.value(), tokens * key_heads,
                                          key_head_dim, rms_eps);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper,
      qwen35_layer_tensor_name(chunk_start, layer_index, "q_conv_predelta"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(key_heads),
       static_cast<std::uint64_t>(key_head_dim)},
      q_conv.value(), tokens * key_dim);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper,
      qwen35_layer_tensor_name(chunk_start, layer_index, "k_conv_predelta"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(key_heads),
       static_cast<std::uint64_t>(key_head_dim)},
      k_conv.value(), tokens * key_dim);
    if (!status.is_ok()) {
      return status;
    }
  }

  auto gate_out = run_qwen35_quant_matmul(context, *gate, normed.value(),
                                          tokens, hidden_size, profiler,
                                          "linear.z");
  auto beta_out = run_qwen35_quant_matmul(context, *beta, normed.value(),
                                          tokens, hidden_size, profiler,
                                          "linear.beta");
  auto alpha_out = run_qwen35_quant_matmul(context, *alpha, normed.value(),
                                           tokens, hidden_size, profiler,
                                           "linear.alpha");
  if (!gate_out.is_ok()) {
    return gate_out.status();
  }
  if (!beta_out.is_ok()) {
    return beta_out.status();
  }
  if (!alpha_out.is_ok()) {
    return alpha_out.status();
  }
  if (gate->shape[1] != value_dim || beta->shape[1] != value_heads ||
      alpha->shape[1] != value_heads) {
    return Status::invalid_argument("Qwen3.5 chunk recurrent projection width mismatch");
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "z"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_dim)},
    gate_out.value(), tokens * value_dim);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "beta"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads)},
    beta_out.value(), tokens * value_heads);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "alpha"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads)},
    alpha_out.value(), tokens * value_heads);
  if (!status.is_ok()) {
    return status;
  }
  if (dumper == nullptr) {
    status = context.prepare_qwen35_gdn_gate_beta_f32(
      alpha_out.value(), beta_out.value(), dt_bias->buffer, ssm_a->buffer,
      tokens, value_heads);
    if (!status.is_ok()) {
      return status;
    }
  } else {
    status = context.sigmoid_f32_in_place(beta_out.value(), tokens * value_heads);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper,
      qwen35_layer_tensor_name(chunk_start, layer_index, "beta_sigmoid"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads)},
      beta_out.value(), tokens * value_heads);
    if (!status.is_ok()) {
      return status;
    }
    status = context.add_f32_row_in_place(alpha_out.value(), dt_bias->buffer,
                                          tokens, value_heads);
    if (!status.is_ok()) {
      return status;
    }
    status = context.softplus_f32_in_place(alpha_out.value(), tokens * value_heads);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "a_softplus"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads)},
      alpha_out.value(), tokens * value_heads);
    if (!status.is_ok()) {
      return status;
    }
    status = context.mul_f32_row_in_place(alpha_out.value(), ssm_a->buffer,
                                          tokens, value_heads);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "gate"),
      {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads)},
      alpha_out.value(), tokens * value_heads);
    if (!status.is_ok()) {
      return status;
    }
  }

  auto attn_output = make_qwen35_f32_buffer(
    context, tokens * value_dim, "Qwen3.5 chunk recurrent attention output");
  if (!attn_output.is_ok()) {
    return attn_output.status();
  }
  status = context.gated_delta_net_f32_batched_in_place_at(
    q_conv.value(), k_conv.value(), v_conv.value(), alpha_out.value(),
    beta_out.value(), cache.recurrent_s_cache, recurrent_s_offset, tokens,
    key_heads, value_heads, key_head_dim, attn_output.value());
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper,
    qwen35_layer_tensor_name(chunk_start, layer_index, "gated_delta_output"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads),
     static_cast<std::uint64_t>(key_head_dim)},
    attn_output.value(), tokens * value_dim);
  if (!status.is_ok()) {
    return status;
  }

  if (qwen35_fused_norm_gated_enabled()) {
    status = context.qwen35_norm_gated_f32_in_place(
      attn_output.value(), ssm_norm->buffer, gate_out.value(), tokens,
      value_heads, key_head_dim, rms_eps);
    if (!status.is_ok()) {
      return status;
    }
  } else {
    status = context.qk_norm_f32_f32_batched(attn_output.value(), ssm_norm->buffer,
                                             tokens, value_heads, key_head_dim,
                                             rms_eps);
    if (!status.is_ok()) {
      return status;
    }
    status = context.silu_f32_in_place(gate_out.value(), tokens * value_dim);
    if (!status.is_ok()) {
      return status;
    }
    status = context.mul_f32_in_place(attn_output.value(), gate_out.value(),
                                      tokens * value_dim);
    if (!status.is_ok()) {
      return status;
    }
  }
  status = dump_qwen35_mps_f32(
    context, dumper,
    qwen35_layer_tensor_name(chunk_start, layer_index, "attn_out_norm_gated"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(value_heads),
     static_cast<std::uint64_t>(key_head_dim)},
    attn_output.value(), tokens * value_dim);
  if (!status.is_ok()) {
    return status;
  }

  auto linear_attn_output =
    run_qwen35_quant_matmul(context, *ssm_out, attn_output.value(),
                            tokens, value_dim, profiler, "linear.ssm_out");
  if (!linear_attn_output.is_ok()) {
    return linear_attn_output.status();
  }
  if (ssm_out->shape[1] != hidden_size) {
    return Status::invalid_argument("Qwen3.5 chunk linear output width mismatch");
  }
  status = dump_qwen35_mps_f32(
    context, dumper,
    qwen35_layer_tensor_name(chunk_start, layer_index, "linear_attn_out"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(hidden_size)},
    linear_attn_output.value(), tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  status = context.add_f32_in_place(linear_attn_output.value(), hidden,
                                    tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "attn_residual"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(hidden_size)},
    linear_attn_output.value(), tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }

  auto post_attn_norm = make_qwen35_f32_buffer(
    context, tokens * hidden_size, "Qwen3.5 chunk post attention norm");
  if (!post_attn_norm.is_ok()) {
    return post_attn_norm.status();
  }
  status = context.rms_norm_f32_f32_batched(
    linear_attn_output.value(), attn_post_norm->buffer, tokens, hidden_size,
    rms_eps, post_attn_norm.value());
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper,
    qwen35_layer_tensor_name(chunk_start, layer_index, "attn_post_norm"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(hidden_size)},
    post_attn_norm.value(), tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }

  auto ffn_up_out = run_qwen35_quant_matmul(context, *ffn_up,
                                            post_attn_norm.value(),
                                            tokens, hidden_size, profiler,
                                            "linear.ffn_up");
  auto ffn_gate_out = run_qwen35_quant_matmul(context, *ffn_gate,
                                              post_attn_norm.value(),
                                              tokens, hidden_size, profiler,
                                              "linear.ffn_gate");
  if (!ffn_up_out.is_ok()) {
    return ffn_up_out.status();
  }
  if (!ffn_gate_out.is_ok()) {
    return ffn_gate_out.status();
  }
  const auto ffn_values = static_cast<std::size_t>(ffn_gate->shape[1]);
  if (ffn_values != static_cast<std::size_t>(ffn_up->shape[1])) {
    return Status::invalid_argument("Qwen3.5 chunk FFN gate/up width mismatch");
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "ffn_gate"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(ffn_values)},
    ffn_gate_out.value(), tokens * ffn_values);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "ffn_up"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(ffn_values)},
    ffn_up_out.value(), tokens * ffn_values);
  if (!status.is_ok()) {
    return status;
  }
  status = context.silu_mul_f32_in_place(ffn_gate_out.value(), ffn_up_out.value(),
                                         tokens * ffn_values);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "ffn_swiglu"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(ffn_values)},
    ffn_gate_out.value(), tokens * ffn_values);
  if (!status.is_ok()) {
    return status;
  }

  auto ffn_down_out = run_qwen35_quant_matmul(context, *ffn_down,
                                              ffn_gate_out.value(),
                                              tokens, ffn_values, profiler,
                                              "linear.ffn_down");
  if (!ffn_down_out.is_ok()) {
    return ffn_down_out.status();
  }
  if (ffn_down->shape[1] != hidden_size) {
    return Status::invalid_argument("Qwen3.5 chunk FFN down output width mismatch");
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "ffn_out"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(hidden_size)},
    ffn_down_out.value(), tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  status = context.add_f32_in_place(ffn_down_out.value(), linear_attn_output.value(),
                                    tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_layer_tensor_name(chunk_start, layer_index, "layer_output"),
    {static_cast<std::uint64_t>(tokens), static_cast<std::uint64_t>(hidden_size)},
    ffn_down_out.value(), tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  return std::move(ffn_down_out.value());
}

Result<mps::MpsBuffer> run_qwen35_full_attention_layer_chunk(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache, const Qwen35LayerBindings& layer,
  const mps::MpsBuffer& hidden, std::size_t tokens, std::size_t hidden_size,
  std::size_t attention_heads, std::size_t kv_heads, std::size_t head_dim,
  std::size_t full_attention_layer_index, std::size_t start_position,
  std::size_t mrope_start_position, std::size_t capacity_tokens,
  const std::array<std::size_t, 4>& mrope_sections,
  float rope_theta, float rms_eps, RequestProfiler* profiler,
  const mps::MpsBuffer* mrope_position_buffer = nullptr) {
  if (layer.kind != Qwen35LayerKind::full_attention) {
    return Status::invalid_argument("Qwen3.5 chunk full attention runner requires a full layer");
  }
  if (tokens == 0 || attention_heads == 0 || kv_heads == 0 ||
      head_dim == 0 || hidden_size == 0 || capacity_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 chunk full attention dimensions must be positive");
  }
  const auto attn_dim = attention_heads * head_dim;
  const auto kv_dim = kv_heads * head_dim;
  const auto mrope_pairs =
    mrope_sections[0] + mrope_sections[1] + mrope_sections[2] + mrope_sections[3];
  const auto mrope_dims = 2U * mrope_pairs;
  if (mrope_dims == 0 || mrope_dims > head_dim) {
    return Status::invalid_argument("Qwen3.5 chunk MRoPE dimensions are invalid");
  }

  const auto* attn_norm = find_metal_weight(weights, layer.attn_norm.name);
  const auto* attn_post_norm = find_metal_weight(weights, layer.attn_post_norm.name);
  const auto* q = find_metal_weight(weights, layer.full_attention.q.name);
  const auto* k = find_metal_weight(weights, layer.full_attention.k.name);
  const auto* v = find_metal_weight(weights, layer.full_attention.v.name);
  const auto* output = find_metal_weight(weights, layer.full_attention.output.name);
  const auto* q_norm = find_metal_weight(weights, layer.full_attention.q_norm.name);
  const auto* k_norm = find_metal_weight(weights, layer.full_attention.k_norm.name);
  const auto* ffn_up = find_metal_weight(weights, layer.ffn_up.name);
  const auto* ffn_gate = find_metal_weight(weights, layer.ffn_gate.name);
  const auto* ffn_down = find_metal_weight(weights, layer.ffn_down.name);
  if (attn_norm == nullptr || attn_post_norm == nullptr || q == nullptr ||
      k == nullptr || v == nullptr || output == nullptr || q_norm == nullptr ||
      k_norm == nullptr || ffn_up == nullptr || ffn_gate == nullptr ||
      ffn_down == nullptr) {
    return Status::invalid_argument("Qwen3.5 chunk full attention layer is missing weights");
  }
  if (attn_norm->type != 0 || attn_post_norm->type != 0 ||
      q_norm->type != 0 || k_norm->type != 0 ||
      attn_norm->shape != std::vector<std::uint64_t>{hidden_size} ||
      attn_post_norm->shape != std::vector<std::uint64_t>{hidden_size} ||
      q_norm->shape != std::vector<std::uint64_t>{head_dim} ||
      k_norm->shape != std::vector<std::uint64_t>{head_dim}) {
    return Status::invalid_argument("Qwen3.5 chunk full attention norm shape/type mismatch");
  }

  auto normed = make_qwen35_f32_buffer(context, tokens * hidden_size,
                                       "Qwen3.5 chunk full normed hidden");
  if (!normed.is_ok()) {
    return normed.status();
  }
  auto status = context.rms_norm_f32_f32_batched(
    hidden, attn_norm->buffer, tokens, hidden_size, rms_eps, normed.value());
  if (!status.is_ok()) {
    return status;
  }

  auto q_full = run_qwen35_quant_matmul(context, *q, normed.value(),
                                        tokens, hidden_size, profiler,
                                        "full.q_gate");
  if (!q_full.is_ok()) {
    return q_full.status();
  }
  if (q->shape[1] != 2U * attn_dim) {
    return Status::invalid_argument("Qwen3.5 chunk q projection width mismatch");
  }

  auto query = make_qwen35_f32_buffer(context, tokens * attn_dim,
                                      "Qwen3.5 chunk query");
  auto gate = make_qwen35_f32_buffer(context, tokens * attn_dim,
                                     "Qwen3.5 chunk attention gate");
  if (!query.is_ok()) {
    return query.status();
  }
  if (!gate.is_ok()) {
    return gate.status();
  }
  // llama.cpp stores the joint Q/G projection interleaved per head:
  // [q_head0, gate_head0, q_head1, gate_head1, ...].
  status = context.copy_f32_rows(q_full.value(), query.value(),
                                 tokens * attention_heads, head_dim,
                                 2U * head_dim, 0, head_dim, 0);
  if (!status.is_ok()) {
    return status;
  }
  status = context.copy_f32_rows(q_full.value(), gate.value(),
                                 tokens * attention_heads, head_dim,
                                 2U * head_dim, head_dim, head_dim, 0);
  if (!status.is_ok()) {
    return status;
  }
  status = context.qk_norm_f32_f32_batched(query.value(), q_norm->buffer,
                                           tokens, attention_heads, head_dim,
                                           rms_eps);
  if (!status.is_ok()) {
    return status;
  }

  auto key = run_qwen35_quant_matmul(context, *k, normed.value(),
                                     tokens, hidden_size, profiler, "full.k");
  auto value = run_qwen35_quant_matmul(context, *v, normed.value(),
                                       tokens, hidden_size, profiler, "full.v");
  if (!key.is_ok()) {
    return key.status();
  }
  if (!value.is_ok()) {
    return value.status();
  }
  if (k->shape[1] != kv_dim || v->shape[1] != kv_dim) {
    return Status::invalid_argument("Qwen3.5 chunk KV projection width mismatch");
  }
  status = context.qk_norm_f32_f32_batched(key.value(), k_norm->buffer,
                                           tokens, kv_heads, head_dim,
                                           rms_eps);
  if (!status.is_ok()) {
    return status;
  }

  std::optional<mps::MpsBuffer> generated_positions;
  const mps::MpsBuffer* positions = mrope_position_buffer;
  if (positions == nullptr) {
    auto generated =
      make_qwen35_mrope_positions(context, mrope_start_position, tokens);
    if (!generated.is_ok()) {
      return generated.status();
    }
    generated_positions = std::move(generated.value());
    positions = &generated_positions.value();
  } else if (tokens > std::numeric_limits<std::size_t>::max() /
                         (4U * sizeof(std::int32_t)) ||
             positions->byte_size() < tokens * 4U * sizeof(std::int32_t)) {
    return Status::invalid_argument(
      "Qwen3.5 chunk MRoPE position buffer is too small");
  }
  status = context.mrope_f32_in_place(query.value(), *positions, tokens,
                                      attention_heads, head_dim, mrope_dims,
                                      mrope_sections[0], mrope_sections[1],
                                      mrope_sections[2], mrope_sections[3],
                                      rope_theta);
  if (!status.is_ok()) {
    return status;
  }
  status = context.mrope_f32_in_place(key.value(), *positions, tokens,
                                      kv_heads, head_dim, mrope_dims,
                                      mrope_sections[0], mrope_sections[1],
                                      mrope_sections[2], mrope_sections[3],
                                      rope_theta);
  if (!status.is_ok()) {
    return status;
  }

  const auto cache_offset =
    (full_attention_layer_index * capacity_tokens + start_position) * kv_dim;
  if (cache.kv_cache_f16) {
    status = context.copy_f32_rows_to_f16(key.value(), cache.key_cache,
                                          tokens, kv_dim, kv_dim, 0, kv_dim,
                                          cache_offset);
  } else {
    status = context.copy_f32_rows(key.value(), cache.key_cache, tokens, kv_dim,
                                   kv_dim, 0, kv_dim, cache_offset);
  }
  if (!status.is_ok()) {
    return status;
  }
  if (cache.kv_cache_f16) {
    status = context.copy_f32_rows_to_f16(value.value(), cache.value_cache,
                                          tokens, kv_dim, kv_dim, 0, kv_dim,
                                          cache_offset);
  } else {
    status = context.copy_f32_rows(value.value(), cache.value_cache, tokens,
                                   kv_dim, kv_dim, 0, kv_dim, cache_offset);
  }
  if (!status.is_ok()) {
    return status;
  }

  auto attention = make_qwen35_f32_buffer(context, tokens * attn_dim,
                                          "Qwen3.5 chunk attention output");
  if (!attention.is_ok()) {
    return attention.status();
  }
  if (cache.kv_cache_f16) {
    status = context.attention_f32_batched_f16_kv(
      query.value(), cache.key_cache, cache.value_cache,
      full_attention_layer_index, start_position, tokens, capacity_tokens,
      attention_heads, kv_heads, head_dim, attention.value());
  } else {
    status = context.attention_f32_batched(
      query.value(), cache.key_cache, cache.value_cache,
      full_attention_layer_index, start_position, tokens, capacity_tokens,
      attention_heads, kv_heads, head_dim, attention.value());
  }
  if (!status.is_ok()) {
    return status;
  }
  status = context.sigmoid_f32_in_place(gate.value(), tokens * attn_dim);
  if (!status.is_ok()) {
    return status;
  }
  status = context.mul_f32_in_place(attention.value(), gate.value(),
                                    tokens * attn_dim);
  if (!status.is_ok()) {
    return status;
  }

  auto attn_output = run_qwen35_quant_matmul(context, *output, attention.value(),
                                             tokens, attn_dim, profiler,
                                             "full.out");
  if (!attn_output.is_ok()) {
    return attn_output.status();
  }
  if (output->shape[1] != hidden_size) {
    return Status::invalid_argument("Qwen3.5 chunk attention output width mismatch");
  }
  status = context.add_f32_in_place(attn_output.value(), hidden,
                                    tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }

  auto post_attn_norm = make_qwen35_f32_buffer(
    context, tokens * hidden_size, "Qwen3.5 chunk full post attention norm");
  if (!post_attn_norm.is_ok()) {
    return post_attn_norm.status();
  }
  status = context.rms_norm_f32_f32_batched(
    attn_output.value(), attn_post_norm->buffer, tokens, hidden_size, rms_eps,
    post_attn_norm.value());
  if (!status.is_ok()) {
    return status;
  }

  auto ffn_up_out = run_qwen35_quant_matmul(context, *ffn_up,
                                            post_attn_norm.value(),
                                            tokens, hidden_size, profiler,
                                            "full.ffn_up");
  auto ffn_gate_out = run_qwen35_quant_matmul(context, *ffn_gate,
                                              post_attn_norm.value(),
                                              tokens, hidden_size, profiler,
                                              "full.ffn_gate");
  if (!ffn_up_out.is_ok()) {
    return ffn_up_out.status();
  }
  if (!ffn_gate_out.is_ok()) {
    return ffn_gate_out.status();
  }
  const auto ffn_values = static_cast<std::size_t>(ffn_gate->shape[1]);
  if (ffn_values != static_cast<std::size_t>(ffn_up->shape[1])) {
    return Status::invalid_argument("Qwen3.5 chunk full FFN gate/up width mismatch");
  }
  status = context.silu_mul_f32_in_place(ffn_gate_out.value(), ffn_up_out.value(),
                                         tokens * ffn_values);
  if (!status.is_ok()) {
    return status;
  }
  auto ffn_down_out = run_qwen35_quant_matmul(context, *ffn_down,
                                              ffn_gate_out.value(),
                                              tokens, ffn_values, profiler,
                                              "full.ffn_down");
  if (!ffn_down_out.is_ok()) {
    return ffn_down_out.status();
  }
  if (ffn_down->shape[1] != hidden_size) {
    return Status::invalid_argument("Qwen3.5 chunk full FFN down output width mismatch");
  }
  status = context.add_f32_in_place(ffn_down_out.value(), attn_output.value(),
                                    tokens * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  return std::move(ffn_down_out.value());
}

bool is_qwen35_eos_token(std::int64_t token, const ModelConfig& model,
                         const GenerationConfig& generation,
                         const GgufTokenizer& tokenizer) {
  if (token == model.eos_token_id) {
    return true;
  }
  if (tokenizer.eos_token_id >= 0 && token == tokenizer.eos_token_id) {
    return true;
  }
  return std::find(generation.eos_token_ids.begin(), generation.eos_token_ids.end(),
                   token) != generation.eos_token_ids.end();
}

Result<Qwen35EffectiveSamplingConfig> make_qwen35_effective_sampling(
  const GenerationConfig& generation, const CpuSamplingConfig& request) {
  Qwen35EffectiveSamplingConfig result;
  result.do_sample = request.do_sample;
  result.temperature =
    request.temperature_set ? request.temperature : generation.temperature;
  result.top_k = request.top_k_set
                   ? request.top_k
                   : (generation.top_k > 0 ? static_cast<std::size_t>(generation.top_k)
                                           : std::size_t{0});
  result.top_p = request.top_p_set ? request.top_p : generation.top_p;
  result.seed = request.seed_set
                  ? request.seed
                  : (request.do_sample
                       ? static_cast<std::uint64_t>(std::random_device{}())
                       : std::uint64_t{0});

  if (result.top_p <= 0.0 || result.top_p > 1.0) {
    return Status::invalid_argument("sampling top_p must be in (0, 1]");
  }
  return result;
}

Result<mps::MpsBuffer> run_qwen35_output_norm(
  const mps::MpsContext& context, const mps::MpsBuffer& hidden,
  std::size_t rows, std::size_t hidden_size, float rms_eps,
  const Qwen35MetalWeight& output_norm, cpu::DebugDumper* dumper,
  std::string_view debug_prefix) {
  if (hidden_size == 0) {
    return Status::invalid_argument("Qwen3.5 output norm hidden size must be positive");
  }
  if (rows == 0) {
    return Status::invalid_argument("Qwen3.5 output norm rows must be positive");
  }
  if (output_norm.type != 0 ||
      output_norm.shape != std::vector<std::uint64_t>{hidden_size}) {
    return Status::invalid_argument("Qwen3.5 output norm shape/type mismatch");
  }

  auto final_norm = make_qwen35_f32_buffer(context, rows * hidden_size,
                                           "Qwen3.5 final norm");
  if (!final_norm.is_ok()) {
    return final_norm.status();
  }
  Status status = Status::ok();
  if (rows == 1) {
    status = context.rms_norm_f32_f32(hidden, output_norm.buffer, hidden_size,
                                      rms_eps, final_norm.value());
  } else {
    status = context.rms_norm_f32_f32_batched(
      hidden, output_norm.buffer, rows, hidden_size, rms_eps, final_norm.value());
  }
  if (!status.is_ok()) {
    return status;
  }
  status = dump_qwen35_mps_f32(
    context, dumper, qwen35_step_tensor_name(debug_prefix, "output_norm"),
    rows == 1 ? std::vector<std::uint64_t>{static_cast<std::uint64_t>(hidden_size)}
              : std::vector<std::uint64_t>{static_cast<std::uint64_t>(rows),
                                           static_cast<std::uint64_t>(hidden_size)},
    final_norm.value(), rows * hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  return std::move(final_norm.value());
}

Result<Qwen35LogitsOutput> run_qwen35_lm_head_logits(
  const mps::MpsContext& context, const mps::MpsBuffer& h_nextn,
  std::size_t rows, std::size_t hidden_size, const Qwen35MetalWeight& output_weight,
  cpu::DebugDumper* dumper, std::string_view debug_prefix, bool compute_argmax,
  bool compute_top1_probability = false, bool prefer_top1_only = false) {
  if (rows == 0 || hidden_size == 0) {
    return Status::invalid_argument("Qwen3.5 LM head dimensions must be positive");
  }
  if (output_weight.shape.size() != 2 ||
      output_weight.shape[0] != hidden_size ||
      output_weight.shape[1] >
        static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) {
    return Status::invalid_argument("Qwen3.5 output weight shape mismatch");
  }
  const auto logits_values = static_cast<std::size_t>(output_weight.shape[1]);
  if (prefer_top1_only && compute_argmax && rows == 1 && dumper == nullptr &&
      !output_weight.dense_f32_transposed.valid() &&
      (output_weight.type == 12U || output_weight.type == 13U ||
       output_weight.type == 14U)) {
    auto argmax_output = context.make_buffer(sizeof(std::int32_t));
    if (!argmax_output.is_ok()) {
      return argmax_output.status();
    }
    Result<mps::MpsBuffer> top1_probability{Status::ok()};
    if (compute_top1_probability) {
      top1_probability = context.make_buffer(sizeof(float));
      if (!top1_probability.is_ok()) {
        return top1_probability.status();
      }
    }
    Status status = Status::ok();
    if (compute_top1_probability) {
      switch (output_weight.type) {
        case 12:
          status = context.matvec_q4_k_f32_top1(
            output_weight.buffer, logits_values, hidden_size, h_nextn,
            argmax_output.value(), top1_probability.value());
          break;
        case 13:
          status = context.matvec_q5_k_f32_top1(
            output_weight.buffer, logits_values, hidden_size, h_nextn,
            argmax_output.value(), top1_probability.value());
          break;
        case 14:
          status = context.matvec_q6_k_f32_top1(
            output_weight.buffer, logits_values, hidden_size, h_nextn,
            argmax_output.value(), top1_probability.value());
          break;
        default:
          break;
      }
    } else {
      switch (output_weight.type) {
        case 12:
          status = context.matvec_q4_k_f32_argmax(
            output_weight.buffer, logits_values, hidden_size, h_nextn,
            argmax_output.value());
          break;
        case 13:
          status = context.matvec_q5_k_f32_argmax(
            output_weight.buffer, logits_values, hidden_size, h_nextn,
            argmax_output.value());
          break;
        case 14:
          status = context.matvec_q6_k_f32_argmax(
            output_weight.buffer, logits_values, hidden_size, h_nextn,
            argmax_output.value());
          break;
        default:
          break;
      }
    }
    if (!status.is_ok()) {
      return status;
    }
    Qwen35LogitsOutput output;
    output.rows = rows;
    output.argmax_output = std::move(argmax_output.value());
    if (compute_top1_probability) {
      output.top1_probability = std::move(top1_probability.value());
    }
    output.logits_values = logits_values;
    return output;
  }

  Result<mps::MpsBuffer> logits{Status::ok()};
  if (rows == 1) {
    logits = run_qwen35_quant_matvec(context, output_weight, h_nextn, hidden_size);
  } else {
    logits = run_qwen35_quant_matmul(context, output_weight, h_nextn, rows,
                                     hidden_size);
  }
  if (!logits.is_ok()) {
    return logits.status();
  }
  auto status = dump_qwen35_mps_f32(
    context, dumper, qwen35_step_tensor_name(debug_prefix, "logits"),
    rows == 1 ? std::vector<std::uint64_t>{static_cast<std::uint64_t>(logits_values)}
              : std::vector<std::uint64_t>{static_cast<std::uint64_t>(rows),
                                           static_cast<std::uint64_t>(logits_values)},
    logits.value(), rows * logits_values);
  if (!status.is_ok()) {
    return status;
  }

  Qwen35LogitsOutput output;
  output.rows = rows;
  if (compute_argmax) {
    if (rows != 1) {
      return Status::invalid_argument("Qwen3.5 device argmax currently requires one logits row");
    }
    auto argmax_output = context.make_buffer(sizeof(std::int32_t));
    if (!argmax_output.is_ok()) {
      return argmax_output.status();
    }
    Result<mps::MpsBuffer> top1_probability{Status::ok()};
    if (compute_top1_probability) {
      top1_probability = context.make_buffer(sizeof(float));
      if (!top1_probability.is_ok()) {
        return top1_probability.status();
      }
      status = context.argmax_prob_f32_i32(logits.value(), logits_values,
                                           argmax_output.value(),
                                           top1_probability.value());
    } else {
      status = context.argmax_f32_i32(logits.value(), logits_values,
                                      argmax_output.value());
    }
    if (!status.is_ok()) {
      return status;
    }
    output.argmax_output = std::move(argmax_output.value());
    if (compute_top1_probability) {
      output.top1_probability = std::move(top1_probability.value());
    }
  } else if (compute_top1_probability) {
    return Status::invalid_argument(
      "Qwen3.5 top1 probability requires device argmax");
  }
  output.logits = std::move(logits.value());
  output.logits_values = logits_values;
  return output;
}

Result<Qwen35LogitsOutput> run_qwen35_output_logits(
  const mps::MpsContext& context, const mps::MpsBuffer& hidden,
  std::size_t hidden_size, float rms_eps, const Qwen35MetalWeight& output_norm,
  const Qwen35MetalWeight& output_weight, cpu::DebugDumper* dumper,
  std::string_view debug_prefix, bool compute_argmax) {
  auto h_nextn = run_qwen35_output_norm(context, hidden, 1, hidden_size, rms_eps,
                                        output_norm, dumper, debug_prefix);
  if (!h_nextn.is_ok()) {
    return h_nextn.status();
  }
  auto logits = run_qwen35_lm_head_logits(
    context, h_nextn.value(), 1, hidden_size, output_weight, dumper,
    debug_prefix, compute_argmax);
  if (!logits.is_ok()) {
    return logits.status();
  }
  logits.value().h_nextn = std::move(h_nextn.value());
  return logits;
}

Result<std::int64_t> read_qwen35_argmax_token(const mps::MpsContext& context,
                                              const mps::MpsBuffer& argmax_output) {
  std::int32_t next_token_i32 = -1;
  auto status = context.copy_from_buffer(argmax_output, &next_token_i32,
                                         sizeof(next_token_i32));
  if (!status.is_ok()) {
    return status;
  }
  if (next_token_i32 < 0) {
    return Status::internal_error("Qwen3.5 Metal argmax returned negative token id");
  }
  return static_cast<std::int64_t>(next_token_i32);
}

Result<std::vector<float>> read_qwen35_logits_host(const mps::MpsContext& context,
                                                   const mps::MpsBuffer& logits,
                                                   std::size_t logits_values) {
  if (logits_values == 0) {
    return Status::invalid_argument("Qwen3.5 logits readback requires logits");
  }
  std::vector<float> logits_host(logits_values, 0.0F);
  auto status = context.copy_from_buffer(logits, logits_host.data(),
                                         logits_host.size() * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  return logits_host;
}

Result<std::vector<float>> read_qwen35_logits_row_host(
  const mps::MpsContext& context, const mps::MpsBuffer& logits,
  std::size_t row, std::size_t logits_values) {
  if (logits_values == 0) {
    return Status::invalid_argument("Qwen3.5 logits row readback requires logits");
  }
  if (row > std::numeric_limits<std::size_t>::max() / logits_values) {
    return Status::invalid_argument("Qwen3.5 logits row offset overflow");
  }
  std::vector<float> logits_host(logits_values, 0.0F);
  const auto byte_offset = row * logits_values * sizeof(float);
  auto status = context.copy_from_buffer_at(logits, byte_offset, logits_host.data(),
                                            logits_host.size() * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  return logits_host;
}

std::string qwen35_join_size_values(const std::vector<std::size_t>& values) {
  std::ostringstream output;
  for (std::size_t index = 0; index < values.size(); ++index) {
    if (index > 0) {
      output << ',';
    }
    output << values[index];
  }
  return output.str();
}

Result<std::vector<CpuGenerationResult::LogitTopEntry>> make_qwen35_logits_top(
  const GgufTokenizer& tokenizer, const std::vector<float>& logits_host,
  std::size_t requested_top_k) {
  std::vector<CpuGenerationResult::LogitTopEntry> result;
  if (requested_top_k == 0) {
    return result;
  }
  if (logits_host.empty()) {
    return Status::invalid_argument("Qwen3.5 logits top-k requires logits");
  }

  const auto top_k = std::min(requested_top_k, logits_host.size());
  std::vector<std::size_t> indices(logits_host.size(), 0);
  std::iota(indices.begin(), indices.end(), std::size_t{0});
  using IndexOffset = std::vector<std::size_t>::difference_type;
  std::partial_sort(indices.begin(),
                    indices.begin() + static_cast<IndexOffset>(top_k),
                    indices.end(), [&](std::size_t lhs, std::size_t rhs) {
                      return logits_host[lhs] > logits_host[rhs];
                    });
  result.reserve(top_k);
  for (std::size_t index = 0; index < top_k; ++index) {
    const auto token_id = static_cast<std::int64_t>(indices[index]);
    auto token_text = gguf_decode_token_text(tokenizer, {token_id}, false);
    if (!token_text.is_ok()) {
      return token_text.status();
    }
    result.push_back(CpuGenerationResult::LogitTopEntry{
      token_id,
      logits_host[indices[index]],
      token_text.value(),
    });
  }
  return result;
}

Result<std::vector<CpuGenerationResult::LogitTopEntry>> read_qwen35_logits_top(
  const mps::MpsContext& context, const GgufTokenizer& tokenizer,
  const mps::MpsBuffer& logits, std::size_t logits_values, std::size_t requested_top_k) {
  if (requested_top_k == 0) {
    return std::vector<CpuGenerationResult::LogitTopEntry>{};
  }
  auto logits_host = read_qwen35_logits_host(context, logits, logits_values);
  if (!logits_host.is_ok()) {
    return logits_host.status();
  }
  return make_qwen35_logits_top(tokenizer, logits_host.value(), requested_top_k);
}

Result<std::int64_t> sample_qwen35_next_token(
  const std::vector<float>& logits, std::mt19937& rng,
  const Qwen35EffectiveSamplingConfig& sampling) {
  if (logits.empty()) {
    return Status::invalid_argument("Qwen3.5 sampling requires logits");
  }
  struct Candidate {
    std::int64_t id{0};
    float logit{0.0F};
  };

  std::vector<Candidate> candidates;
  candidates.reserve(logits.size());
  for (std::size_t index = 0; index < logits.size(); ++index) {
    candidates.push_back(Candidate{static_cast<std::int64_t>(index), logits[index]});
  }
  std::sort(candidates.begin(), candidates.end(),
            [](const Candidate& lhs, const Candidate& rhs) {
              return lhs.logit > rhs.logit;
            });

  if (sampling.top_k > 0 && sampling.top_k < candidates.size()) {
    candidates.resize(sampling.top_k);
  }

  if (sampling.top_p < 1.0) {
    const auto max_logit = static_cast<double>(candidates.front().logit);
    std::vector<double> probabilities;
    probabilities.reserve(candidates.size());
    double total_probability = 0.0;
    for (const auto& candidate : candidates) {
      const auto probability =
        std::exp(static_cast<double>(candidate.logit) - max_logit);
      probabilities.push_back(probability);
      total_probability += probability;
    }
    if (!(total_probability > 0.0)) {
      return Status::internal_error("Qwen3.5 top-p produced zero probability mass");
    }

    double cumulative = 0.0;
    std::size_t kept = 0;
    for (std::size_t index = 0; index < probabilities.size(); ++index) {
      cumulative += probabilities[index] / total_probability;
      kept = index + 1U;
      if (cumulative >= sampling.top_p) {
        break;
      }
    }
    candidates.resize(kept);
  }

  if (sampling.temperature <= 0.0) {
    return candidates.front().id;
  }

  const auto max_logit = static_cast<double>(candidates.front().logit);
  std::vector<double> weights;
  weights.reserve(candidates.size());
  double total_weight = 0.0;
  for (const auto& candidate : candidates) {
    const auto weight =
      std::exp((static_cast<double>(candidate.logit) - max_logit) /
               sampling.temperature);
    weights.push_back(weight);
    total_weight += weight;
  }
  if (!(total_weight > 0.0)) {
    return Status::internal_error("Qwen3.5 sampling produced zero probability mass");
  }

  std::uniform_real_distribution<double> distribution(0.0, 1.0);
  const auto target = total_weight * distribution(rng);
  double cumulative = 0.0;
  for (std::size_t index = 0; index < weights.size(); ++index) {
    cumulative += weights[index];
    if (target <= cumulative) {
      return candidates[index].id;
    }
  }
  return candidates.back().id;
}

Result<Qwen35MainForwardOutput> run_qwen35_decode_tokens(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache, const Qwen35WeightMap& map,
  const ModelConfig& model, const Qwen35ExecutionPlan& plan,
  const std::array<std::size_t, 4>& mrope_sections,
  const std::vector<std::int64_t>& token_ids, std::size_t start_position,
  std::size_t mrope_start_position) {
  if (token_ids.empty()) {
    return Status::invalid_argument("Qwen3.5 decode requires at least one token");
  }
  if (start_position >= plan.cache.attention_capacity_tokens ||
      token_ids.size() >
        plan.cache.attention_capacity_tokens - start_position) {
    return Status::invalid_argument("Qwen3.5 decode position exceeds cache capacity");
  }
  auto token_buffer = make_qwen35_token_id_buffer(context, token_ids);
  if (!token_buffer.is_ok()) {
    return token_buffer.status();
  }
  auto hidden = run_qwen35_token_embeddings(context, weights, token_ids,
                                            token_buffer.value(), plan.hidden_size);
  if (!hidden.is_ok()) {
    return hidden.status();
  }

  auto current_hidden = std::move(hidden.value());
  std::size_t linear_layers_executed = 0;
  std::size_t full_attention_layers_executed = 0;
  std::size_t main_layers_executed = 0;
  for (std::size_t layer_index = 0; layer_index < plan.main_layers; ++layer_index) {
    const auto& layer = map.layers[layer_index];
    switch (layer.kind) {
      case Qwen35LayerKind::linear_attention: {
        auto layer_output = run_qwen35_linear_layer_chunk(
          context, weights, cache, layer, current_hidden, token_ids.size(), plan.hidden_size,
          plan.linear_conv_kernel, plan.linear_conv_channels,
          plan.linear_key_heads, plan.linear_key_head_dim,
          plan.linear_value_heads, plan.linear_inner_size,
          linear_layers_executed, plan.cache.recurrent_r_elements_per_layer,
          plan.cache.recurrent_s_elements_per_layer,
          nullptr, layer_index, start_position, static_cast<float>(model.rms_norm_eps),
          nullptr);
        if (!layer_output.is_ok()) {
          return layer_output.status();
        }
        current_hidden = std::move(layer_output.value());
        ++linear_layers_executed;
        break;
      }
      case Qwen35LayerKind::full_attention: {
        auto layer_output = run_qwen35_full_attention_layer_chunk(
          context, weights, cache, layer, current_hidden, token_ids.size(), plan.hidden_size,
          plan.attention_heads, plan.kv_heads, plan.head_dim,
          full_attention_layers_executed, start_position,
          mrope_start_position,
          plan.cache.attention_capacity_tokens, mrope_sections,
          static_cast<float>(model.rope_theta),
          static_cast<float>(model.rms_norm_eps), nullptr);
        if (!layer_output.is_ok()) {
          return layer_output.status();
        }
        current_hidden = std::move(layer_output.value());
        ++full_attention_layers_executed;
        break;
      }
      case Qwen35LayerKind::mtp:
        return Status::invalid_argument("Qwen3.5 decode layer loop reached MTP layer");
    }
    ++main_layers_executed;
  }
  if (main_layers_executed != plan.main_layers ||
      linear_layers_executed != plan.cache.linear_attention_layers ||
      full_attention_layers_executed != plan.cache.full_attention_layers) {
    return Status::internal_error("Qwen3.5 Metal decode layer execution count mismatch");
  }
  return Qwen35MainForwardOutput{std::move(current_hidden), token_ids.size()};
}

[[maybe_unused]] Result<mps::MpsBuffer> run_qwen35_decode_token(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& cache, const Qwen35WeightMap& map,
  const ModelConfig& model, const Qwen35ExecutionPlan& plan,
  const std::array<std::size_t, 4>& mrope_sections,
  std::int64_t token_id, std::size_t position) {
  const std::vector<std::int64_t> token_ids{token_id};
  auto output = run_qwen35_decode_tokens(
    context, weights, cache, map, model, plan, mrope_sections, token_ids,
    position, position);
  if (!output.is_ok()) {
    return output.status();
  }
  return std::move(output.value().hidden);
}

Result<mps::MpsBuffer> make_qwen35_mtp_shifted_h_rows(
  const mps::MpsContext& context, const mps::MpsBuffer& previous_h,
  const mps::MpsBuffer& h_nextn_rows, std::size_t rows, std::size_t hidden_size) {
  if (rows == 0 || hidden_size == 0) {
    return Status::invalid_argument("Qwen3.5 MTP shifted h rows require dimensions");
  }
  auto shifted = make_qwen35_f32_buffer(context, rows * hidden_size,
                                        "Qwen3.5 MTP shifted h rows");
  if (!shifted.is_ok()) {
    return shifted.status();
  }
  auto status = context.copy_f32_region(previous_h, shifted.value(), 0, 0,
                                        hidden_size);
  if (!status.is_ok()) {
    return status;
  }
  if (rows > 1) {
    status = context.copy_f32_rows(h_nextn_rows, shifted.value(), rows - 1U,
                                   hidden_size, hidden_size, 0,
                                   hidden_size, hidden_size);
    if (!status.is_ok()) {
      return status;
    }
  }
  return std::move(shifted.value());
}

Result<Qwen35MtpForwardOutput> run_qwen35_mtp_embeddings(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& mtp_cache, const Qwen35LayerBindings& layer,
  const Qwen35MetalWeight& fallback_output_norm,
  const Qwen35MetalWeight& fallback_output_weight,
  const std::array<std::size_t, 4>& mrope_sections,
  std::size_t token_count, const mps::MpsBuffer& tok_embd,
  const mps::MpsBuffer& h_rows, std::size_t start_position,
  std::size_t mrope_start_position, const Qwen35ExecutionPlan& plan,
  float rope_theta, float rms_eps, const mps::MpsBuffer* mrope_position_buffer,
  bool compute_logits, bool compute_argmax,
  bool compute_top1_probability = false, RequestProfiler* profiler = nullptr) {
  if (layer.kind != Qwen35LayerKind::mtp) {
    return Status::invalid_argument("Qwen3.5 MTP embedding runner requires an MTP layer");
  }
  if (token_count == 0) {
    return Status::invalid_argument("Qwen3.5 MTP embedding runner requires tokens");
  }
  if (start_position >= plan.cache.attention_capacity_tokens ||
      token_count > plan.cache.attention_capacity_tokens - start_position) {
    return Status::invalid_argument("Qwen3.5 MTP embedding position exceeds cache capacity");
  }
  std::size_t embedding_values = 0;
  if (!checked_mul_size(token_count, plan.hidden_size, embedding_values)) {
    return Status::invalid_argument("Qwen3.5 MTP embedding size overflow");
  }
  if (tok_embd.byte_size() < embedding_values * sizeof(float) ||
      h_rows.byte_size() < embedding_values * sizeof(float)) {
    return Status::invalid_argument("Qwen3.5 MTP embedding runner input is too small");
  }

  const auto* eh_proj = find_metal_weight(weights, layer.mtp.eh_proj.name);
  const auto* enorm = find_metal_weight(weights, layer.mtp.enorm.name);
  const auto* hnorm = find_metal_weight(weights, layer.mtp.hnorm.name);
  const auto* head_norm = layer.mtp.has_shared_head_norm
                            ? find_metal_weight(weights, layer.mtp.shared_head_norm.name)
                            : &fallback_output_norm;
  const auto* head_weight = layer.mtp.has_shared_head_head
                              ? find_metal_weight(weights, layer.mtp.shared_head_head.name)
                              : &fallback_output_weight;
  if (eh_proj == nullptr || enorm == nullptr || hnorm == nullptr ||
      head_norm == nullptr || head_weight == nullptr) {
    return Status::invalid_argument("Qwen3.5 MTP embedding layer is missing weights");
  }
  if (enorm->type != 0 || hnorm->type != 0 ||
      enorm->shape != std::vector<std::uint64_t>{plan.hidden_size} ||
      hnorm->shape != std::vector<std::uint64_t>{plan.hidden_size}) {
    return Status::invalid_argument("Qwen3.5 MTP embedding norm shape/type mismatch");
  }

  auto h_norm = run_qwen35_output_norm(context, h_rows, token_count,
                                       plan.hidden_size, rms_eps, *hnorm,
                                       nullptr, "mtp.hnorm");
  if (!h_norm.is_ok()) {
    return h_norm.status();
  }
  auto e_norm = run_qwen35_output_norm(context, tok_embd, token_count,
                                       plan.hidden_size, rms_eps, *enorm,
                                       nullptr, "mtp.enorm");
  if (!e_norm.is_ok()) {
    return e_norm.status();
  }

  auto concat = make_qwen35_f32_buffer(context, token_count * plan.hidden_size * 2U,
                                       "Qwen3.5 MTP concat");
  if (!concat.is_ok()) {
    return concat.status();
  }
  Status status = Status::ok();
  for (std::size_t row = 0; row < token_count; ++row) {
    const auto source = row * plan.hidden_size;
    const auto destination = row * plan.hidden_size * 2U;
    status = context.copy_f32_region(e_norm.value(), concat.value(),
                                     source, destination, plan.hidden_size);
    if (!status.is_ok()) {
      return status;
    }
    status = context.copy_f32_region(h_norm.value(), concat.value(),
                                     source, destination + plan.hidden_size,
                                     plan.hidden_size);
    if (!status.is_ok()) {
      return status;
    }
  }

  auto projected = run_qwen35_quant_matmul(context, *eh_proj, concat.value(),
                                           token_count, plan.hidden_size * 2U,
                                           profiler, "mtp.eh_proj");
  if (!projected.is_ok()) {
    return projected.status();
  }

  Qwen35LayerBindings full_layer = layer;
  full_layer.kind = Qwen35LayerKind::full_attention;
  full_layer.full_attention = layer.mtp.attention;
  auto block_out = run_qwen35_full_attention_layer_chunk(
    context, weights, mtp_cache, full_layer, projected.value(), token_count,
    plan.hidden_size, plan.attention_heads, plan.kv_heads, plan.head_dim, 0,
    start_position, mrope_start_position, plan.cache.attention_capacity_tokens,
    mrope_sections, rope_theta, rms_eps, profiler, mrope_position_buffer);
  if (!block_out.is_ok()) {
    return block_out.status();
  }

  auto h_nextn = run_qwen35_output_norm(context, block_out.value(), token_count,
                                        plan.hidden_size, rms_eps, *head_norm,
                                        nullptr, "mtp.shared_head_norm");
  if (!h_nextn.is_ok()) {
    return h_nextn.status();
  }

  Qwen35MtpForwardOutput output;
  output.rows = token_count;
  output.h_nextn = std::move(h_nextn.value());
  if (!compute_logits) {
    return output;
  }
  auto logits = run_qwen35_lm_head_logits(
    context, output.h_nextn, token_count, plan.hidden_size, *head_weight,
    nullptr, "mtp", compute_argmax, compute_top1_probability, compute_argmax);
  if (!logits.is_ok()) {
    return logits.status();
  }
  output.logits = std::move(logits.value().logits);
  output.argmax_output = std::move(logits.value().argmax_output);
  output.top1_probability = std::move(logits.value().top1_probability);
  output.logits_values = logits.value().logits_values;
  return output;
}

Result<Qwen35MtpForwardOutput> run_qwen35_mtp_tokens_from_id_buffer(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& mtp_cache, const Qwen35LayerBindings& layer,
  const Qwen35MetalWeight& fallback_token_embedding,
  const Qwen35MetalWeight& fallback_output_norm,
  const Qwen35MetalWeight& fallback_output_weight,
  const std::array<std::size_t, 4>& mrope_sections,
  std::size_t token_count, const mps::MpsBuffer& token_id_buffer,
  const mps::MpsBuffer& h_rows, std::size_t start_position,
  std::size_t mrope_start_position, const Qwen35ExecutionPlan& plan,
  float rope_theta, float rms_eps, bool compute_logits, bool compute_argmax,
  bool compute_top1_probability = false, RequestProfiler* profiler = nullptr) {
  if (layer.kind != Qwen35LayerKind::mtp) {
    return Status::invalid_argument("Qwen3.5 MTP runner requires an MTP layer");
  }
  if (token_count == 0) {
    return Status::invalid_argument("Qwen3.5 MTP runner requires tokens");
  }
  if (start_position >= plan.cache.attention_capacity_tokens ||
      token_count > plan.cache.attention_capacity_tokens - start_position) {
    return Status::invalid_argument("Qwen3.5 MTP position exceeds cache capacity");
  }

  const auto* eh_proj = find_metal_weight(weights, layer.mtp.eh_proj.name);
  const auto* enorm = find_metal_weight(weights, layer.mtp.enorm.name);
  const auto* hnorm = find_metal_weight(weights, layer.mtp.hnorm.name);
  const auto* embed_tokens = layer.mtp.has_embed_tokens
                               ? find_metal_weight(weights, layer.mtp.embed_tokens.name)
                               : &fallback_token_embedding;
  const auto* head_norm = layer.mtp.has_shared_head_norm
                            ? find_metal_weight(weights, layer.mtp.shared_head_norm.name)
                            : &fallback_output_norm;
  const auto* head_weight = layer.mtp.has_shared_head_head
                              ? find_metal_weight(weights, layer.mtp.shared_head_head.name)
                              : &fallback_output_weight;
  if (eh_proj == nullptr || enorm == nullptr || hnorm == nullptr ||
      embed_tokens == nullptr || head_norm == nullptr || head_weight == nullptr) {
    return Status::invalid_argument("Qwen3.5 MTP layer is missing weights");
  }
  if (enorm->type != 0 || hnorm->type != 0 ||
      enorm->shape != std::vector<std::uint64_t>{plan.hidden_size} ||
      hnorm->shape != std::vector<std::uint64_t>{plan.hidden_size}) {
    return Status::invalid_argument("Qwen3.5 MTP norm shape/type mismatch");
  }

  auto tok_embd = run_qwen35_token_embeddings_from_id_buffer(
    context, *embed_tokens, token_count, token_id_buffer, plan.hidden_size);
  if (!tok_embd.is_ok()) {
    return tok_embd.status();
  }

  auto h_norm = run_qwen35_output_norm(context, h_rows, token_count,
                                       plan.hidden_size, rms_eps, *hnorm,
                                       nullptr, "mtp.hnorm");
  if (!h_norm.is_ok()) {
    return h_norm.status();
  }
  auto e_norm = run_qwen35_output_norm(context, tok_embd.value(), token_count,
                                       plan.hidden_size, rms_eps, *enorm,
                                       nullptr, "mtp.enorm");
  if (!e_norm.is_ok()) {
    return e_norm.status();
  }

  auto concat = make_qwen35_f32_buffer(context, token_count * plan.hidden_size * 2U,
                                       "Qwen3.5 MTP concat");
  if (!concat.is_ok()) {
    return concat.status();
  }
  Status status = Status::ok();
  for (std::size_t row = 0; row < token_count; ++row) {
    const auto source = row * plan.hidden_size;
    const auto destination = row * plan.hidden_size * 2U;
    status = context.copy_f32_region(e_norm.value(), concat.value(),
                                     source, destination, plan.hidden_size);
    if (!status.is_ok()) {
      return status;
    }
    status = context.copy_f32_region(h_norm.value(), concat.value(),
                                     source, destination + plan.hidden_size,
                                     plan.hidden_size);
    if (!status.is_ok()) {
      return status;
    }
  }

  auto projected = run_qwen35_quant_matmul(context, *eh_proj, concat.value(),
                                           token_count, plan.hidden_size * 2U,
                                           profiler, "mtp.eh_proj");
  if (!projected.is_ok()) {
    return projected.status();
  }

  Qwen35LayerBindings full_layer = layer;
  full_layer.kind = Qwen35LayerKind::full_attention;
  full_layer.full_attention = layer.mtp.attention;
  auto block_out = run_qwen35_full_attention_layer_chunk(
    context, weights, mtp_cache, full_layer, projected.value(), token_count,
    plan.hidden_size, plan.attention_heads, plan.kv_heads, plan.head_dim, 0,
    start_position, mrope_start_position, plan.cache.attention_capacity_tokens,
    mrope_sections, rope_theta, rms_eps, profiler);
  if (!block_out.is_ok()) {
    return block_out.status();
  }

  auto h_nextn = run_qwen35_output_norm(context, block_out.value(), token_count,
                                        plan.hidden_size, rms_eps, *head_norm,
                                        nullptr, "mtp.shared_head_norm");
  if (!h_nextn.is_ok()) {
    return h_nextn.status();
  }

  Qwen35MtpForwardOutput output;
  output.rows = token_count;
  output.h_nextn = std::move(h_nextn.value());
  if (!compute_logits) {
    return output;
  }
  auto logits = run_qwen35_lm_head_logits(
    context, output.h_nextn, token_count, plan.hidden_size, *head_weight,
    nullptr, "mtp", compute_argmax, compute_top1_probability, compute_argmax);
  if (!logits.is_ok()) {
    return logits.status();
  }
  output.logits = std::move(logits.value().logits);
  output.argmax_output = std::move(logits.value().argmax_output);
  output.top1_probability = std::move(logits.value().top1_probability);
  output.logits_values = logits.value().logits_values;
  return output;
}

Result<Qwen35MtpForwardOutput> run_qwen35_mtp_tokens(
  const mps::MpsContext& context, const Qwen35MetalWeightStore& weights,
  Qwen35MetalCache& mtp_cache, const Qwen35LayerBindings& layer,
  const Qwen35MetalWeight& fallback_token_embedding,
  const Qwen35MetalWeight& fallback_output_norm,
  const Qwen35MetalWeight& fallback_output_weight,
  const std::array<std::size_t, 4>& mrope_sections,
  const std::vector<std::int64_t>& token_ids, const mps::MpsBuffer& h_rows,
  std::size_t start_position, std::size_t mrope_start_position,
  const Qwen35ExecutionPlan& plan, float rope_theta, float rms_eps,
  bool compute_logits, bool compute_argmax, bool compute_top1_probability = false,
  RequestProfiler* profiler = nullptr) {
  if (layer.kind != Qwen35LayerKind::mtp) {
    return Status::invalid_argument("Qwen3.5 MTP runner requires an MTP layer");
  }
  if (token_ids.empty()) {
    return Status::invalid_argument("Qwen3.5 MTP runner requires tokens");
  }
  if (start_position >= plan.cache.attention_capacity_tokens ||
      token_ids.size() >
        plan.cache.attention_capacity_tokens - start_position) {
    return Status::invalid_argument("Qwen3.5 MTP position exceeds cache capacity");
  }

  const auto* eh_proj = find_metal_weight(weights, layer.mtp.eh_proj.name);
  const auto* enorm = find_metal_weight(weights, layer.mtp.enorm.name);
  const auto* hnorm = find_metal_weight(weights, layer.mtp.hnorm.name);
  const auto* embed_tokens = layer.mtp.has_embed_tokens
                               ? find_metal_weight(weights, layer.mtp.embed_tokens.name)
                               : &fallback_token_embedding;
  const auto* head_norm = layer.mtp.has_shared_head_norm
                            ? find_metal_weight(weights, layer.mtp.shared_head_norm.name)
                            : &fallback_output_norm;
  const auto* head_weight = layer.mtp.has_shared_head_head
                              ? find_metal_weight(weights, layer.mtp.shared_head_head.name)
                              : &fallback_output_weight;
  if (eh_proj == nullptr || enorm == nullptr || hnorm == nullptr ||
      embed_tokens == nullptr || head_norm == nullptr || head_weight == nullptr) {
    return Status::invalid_argument("Qwen3.5 MTP layer is missing weights");
  }
  if (enorm->type != 0 || hnorm->type != 0 ||
      enorm->shape != std::vector<std::uint64_t>{plan.hidden_size} ||
      hnorm->shape != std::vector<std::uint64_t>{plan.hidden_size}) {
    return Status::invalid_argument("Qwen3.5 MTP norm shape/type mismatch");
  }

  auto token_buffer = make_qwen35_token_id_buffer(context, token_ids);
  if (!token_buffer.is_ok()) {
    return token_buffer.status();
  }
  auto tok_embd = run_qwen35_token_embeddings_from_weight(
    context, *embed_tokens, token_ids, token_buffer.value(), plan.hidden_size);
  if (!tok_embd.is_ok()) {
    return tok_embd.status();
  }

  auto h_norm = run_qwen35_output_norm(context, h_rows, token_ids.size(),
                                       plan.hidden_size, rms_eps, *hnorm,
                                       nullptr, "mtp.hnorm");
  if (!h_norm.is_ok()) {
    return h_norm.status();
  }
  auto e_norm = run_qwen35_output_norm(context, tok_embd.value(), token_ids.size(),
                                       plan.hidden_size, rms_eps, *enorm,
                                       nullptr, "mtp.enorm");
  if (!e_norm.is_ok()) {
    return e_norm.status();
  }

  auto concat = make_qwen35_f32_buffer(context, token_ids.size() * plan.hidden_size * 2U,
                                       "Qwen3.5 MTP concat");
  if (!concat.is_ok()) {
    return concat.status();
  }
  Status status = Status::ok();
  for (std::size_t row = 0; row < token_ids.size(); ++row) {
    const auto source = row * plan.hidden_size;
    const auto destination = row * plan.hidden_size * 2U;
    status = context.copy_f32_region(e_norm.value(), concat.value(),
                                     source, destination, plan.hidden_size);
    if (!status.is_ok()) {
      return status;
    }
    status = context.copy_f32_region(h_norm.value(), concat.value(),
                                     source, destination + plan.hidden_size,
                                     plan.hidden_size);
    if (!status.is_ok()) {
      return status;
    }
  }

  auto projected = run_qwen35_quant_matmul(context, *eh_proj, concat.value(),
                                           token_ids.size(), plan.hidden_size * 2U,
                                           profiler, "mtp.eh_proj");
  if (!projected.is_ok()) {
    return projected.status();
  }

  Qwen35LayerBindings full_layer = layer;
  full_layer.kind = Qwen35LayerKind::full_attention;
  full_layer.full_attention = layer.mtp.attention;
  auto block_out = run_qwen35_full_attention_layer_chunk(
    context, weights, mtp_cache, full_layer, projected.value(), token_ids.size(),
    plan.hidden_size, plan.attention_heads, plan.kv_heads, plan.head_dim, 0,
    start_position, mrope_start_position, plan.cache.attention_capacity_tokens,
    mrope_sections, rope_theta, rms_eps, profiler);
  if (!block_out.is_ok()) {
    return block_out.status();
  }

  auto h_nextn = run_qwen35_output_norm(context, block_out.value(), token_ids.size(),
                                        plan.hidden_size, rms_eps, *head_norm,
                                        nullptr, "mtp.shared_head_norm");
  if (!h_nextn.is_ok()) {
    return h_nextn.status();
  }

  Qwen35MtpForwardOutput output;
  output.rows = token_ids.size();
  output.h_nextn = std::move(h_nextn.value());
  if (!compute_logits) {
    return output;
  }
  auto logits = run_qwen35_lm_head_logits(
    context, output.h_nextn, token_ids.size(), plan.hidden_size, *head_weight,
    nullptr, "mtp", compute_argmax, compute_top1_probability, compute_argmax);
  if (!logits.is_ok()) {
    return logits.status();
  }
  output.logits = std::move(logits.value().logits);
  output.argmax_output = std::move(logits.value().argmax_output);
  output.top1_probability = std::move(logits.value().top1_probability);
  output.logits_values = logits.value().logits_values;
  return output;
}

}  // namespace

Result<Qwen35ExecutionPlan> build_qwen35_execution_plan(
  const ModelConfig& config, const Qwen35WeightMap& weights,
  std::size_t prompt_tokens, const Qwen35RuntimeOptions& options) {
  if (!config.gguf || config.architecture != "qwen35") {
    return Status::invalid_argument("Qwen3.5 execution plan requires qwen35 GGUF config");
  }
  if (prompt_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 execution plan requires at least one prompt token");
  }
  if (options.prefill_chunk_tokens == 0) {
    return Status::invalid_argument("Qwen3.5 prefill chunk size must be positive");
  }
  if (options.sequence_slots == 0) {
    return Status::invalid_argument("Qwen3.5 sequence slot count must be positive");
  }

  auto hidden = positive_size(config.hidden_size, "hidden_size");
  auto main_layers = positive_size(config.main_layer_count, "main_layer_count");
  auto total_layers = positive_size(config.total_layer_count, "total_layer_count");
  auto head_dim = positive_size(config.head_dim, "head_dim");
  auto attention_heads = positive_size(config.num_attention_heads, "attention_heads");
  auto kv_heads = positive_size(config.num_key_value_heads, "kv_heads");
  auto linear_key_heads = positive_size(config.linear_num_key_heads, "linear_key_heads");
  auto linear_value_heads = positive_size(config.linear_num_value_heads, "linear_value_heads");
  auto linear_key_head_dim = positive_size(config.linear_key_head_dim, "linear_key_head_dim");
  auto linear_inner = positive_size(config.linear_inner_size, "linear_inner_size");
  auto linear_conv_kernel =
    positive_size(config.linear_conv_kernel_dim, "linear_conv_kernel_dim");
  if (!hidden.is_ok()) {
    return hidden.status();
  }
  if (!main_layers.is_ok()) {
    return main_layers.status();
  }
  if (!total_layers.is_ok()) {
    return total_layers.status();
  }
  if (!head_dim.is_ok()) {
    return head_dim.status();
  }
  if (!attention_heads.is_ok()) {
    return attention_heads.status();
  }
  if (!kv_heads.is_ok()) {
    return kv_heads.status();
  }
  if (!linear_key_heads.is_ok()) {
    return linear_key_heads.status();
  }
  if (!linear_value_heads.is_ok()) {
    return linear_value_heads.status();
  }
  if (!linear_key_head_dim.is_ok()) {
    return linear_key_head_dim.status();
  }
  if (!linear_inner.is_ok()) {
    return linear_inner.status();
  }
  if (!linear_conv_kernel.is_ok()) {
    return linear_conv_kernel.status();
  }
  if (linear_inner.value() % linear_value_heads.value() != 0) {
    return Status::invalid_argument("linear_inner_size must be divisible by linear_value_heads");
  }
  if (weights.layers.size() < main_layers.value()) {
    return Status::invalid_argument("Qwen3.5 weight map has fewer layers than main layers");
  }

  Qwen35ExecutionPlan plan;
  plan.hidden_size = hidden.value();
  plan.main_layers = main_layers.value();
  plan.total_layers = total_layers.value();
  plan.head_dim = head_dim.value();
  plan.attention_heads = attention_heads.value();
  plan.kv_heads = kv_heads.value();
  plan.linear_key_heads = linear_key_heads.value();
  plan.linear_value_heads = linear_value_heads.value();
  plan.linear_key_head_dim = linear_key_head_dim.value();
  plan.linear_inner_size = linear_inner.value();
  plan.linear_value_head_dim = linear_inner.value() / linear_value_heads.value();
  plan.linear_conv_kernel = linear_conv_kernel.value();
  plan.linear_conv_channels =
    plan.linear_inner_size + 2U * plan.linear_key_heads * plan.linear_key_head_dim;
  plan.layer_kinds.reserve(plan.main_layers);
  for (std::size_t index = 0; index < plan.main_layers; ++index) {
    plan.layer_kinds.push_back(weights.layers[index].kind);
  }

  plan.prefill.prompt_tokens = prompt_tokens;
  plan.prefill.chunk_tokens = options.prefill_chunk_tokens;
  plan.prefill.chunk_count = ceil_div(prompt_tokens, options.prefill_chunk_tokens);
  plan.prefill.final_chunk_tokens =
    prompt_tokens - (plan.prefill.chunk_count - 1U) * options.prefill_chunk_tokens;
  plan.prefill.output_only_last_token = !options.output_all_prompt_logits;

  const auto requested_capacity = prompt_tokens + options.decode_tokens;
  plan.cache.attention_capacity_tokens =
    std::max(options.context_tokens, std::max<std::size_t>(1, requested_capacity));
  plan.cache.recurrent_slots = options.sequence_slots;
  plan.cache.recurrent_rows = options.sequence_slots * (1U + options.recurrent_snapshot_count);
  plan.cache.full_attention_layers = weights.full_attention_layers;
  plan.cache.linear_attention_layers = weights.linear_attention_layers;
  plan.cache.kv_dim = plan.kv_heads * plan.head_dim;
  plan.cache.kv_cache_f16 = options.kv_cache_f16;
  plan.cache.kv_cache_element_bytes =
    options.kv_cache_f16 ? sizeof(std::uint16_t) : sizeof(float);
  plan.cache.recurrent_r_elements_per_layer =
    (plan.linear_conv_kernel - 1U) * plan.linear_conv_channels;
  plan.cache.recurrent_s_elements_per_layer = plan.linear_key_head_dim * plan.linear_inner_size;

  auto kv_bytes = cache_bytes(plan.cache.full_attention_layers,
                              plan.cache.attention_capacity_tokens,
                              plan.cache.kv_dim * 2U,
                              plan.cache.kv_cache_element_bytes,
                              "Qwen3.5 KV cache");
  if (!kv_bytes.is_ok()) {
    return kv_bytes.status();
  }
  auto r_bytes = cache_bytes(plan.cache.linear_attention_layers, plan.cache.recurrent_rows,
                             plan.cache.recurrent_r_elements_per_layer, sizeof(float),
                             "Qwen3.5 recurrent R cache");
  if (!r_bytes.is_ok()) {
    return r_bytes.status();
  }
  auto s_bytes = cache_bytes(plan.cache.linear_attention_layers, plan.cache.recurrent_rows,
                             plan.cache.recurrent_s_elements_per_layer, sizeof(float),
                             "Qwen3.5 recurrent S cache");
  if (!s_bytes.is_ok()) {
    return s_bytes.status();
  }
  auto total_bytes = add_cache_bytes(kv_bytes.value(), r_bytes.value(),
                                     "Qwen3.5 total cache");
  if (!total_bytes.is_ok()) {
    return total_bytes.status();
  }
  total_bytes = add_cache_bytes(total_bytes.value(), s_bytes.value(),
                                "Qwen3.5 total cache");
  if (!total_bytes.is_ok()) {
    return total_bytes.status();
  }
  plan.cache.kv_cache_bytes = kv_bytes.value();
  plan.cache.recurrent_r_cache_bytes = r_bytes.value();
  plan.cache.recurrent_s_cache_bytes = s_bytes.value();
  plan.cache.total_cache_bytes = total_bytes.value();

  return plan;
}

std::string format_qwen35_execution_plan(const Qwen35ExecutionPlan& plan) {
  std::ostringstream output;
  output << "Qwen3.5 execution plan: ok\n";
  output << "Layers: " << plan.main_layers << " main / " << plan.total_layers << " total\n";
  output << "- full_attention: " << plan.cache.full_attention_layers << '\n';
  output << "- linear_attention: " << plan.cache.linear_attention_layers << '\n';
  output << "Hidden: " << plan.hidden_size << '\n';
  output << "Full attention: heads=" << plan.attention_heads
         << ", kv_heads=" << plan.kv_heads << ", head_dim=" << plan.head_dim
         << ", kv_dim=" << plan.cache.kv_dim << '\n';
  output << "Linear attention: key_heads=" << plan.linear_key_heads
         << ", value_heads=" << plan.linear_value_heads
         << ", key_head_dim=" << plan.linear_key_head_dim
         << ", value_head_dim=" << plan.linear_value_head_dim
         << ", inner=" << plan.linear_inner_size
         << ", conv_channels=" << plan.linear_conv_channels << '\n';
  output << "Prefill: prompt_tokens=" << plan.prefill.prompt_tokens
         << ", chunk_tokens=" << plan.prefill.chunk_tokens
         << ", chunks=" << plan.prefill.chunk_count
         << ", final_chunk=" << plan.prefill.final_chunk_tokens
         << ", output_only_last="
         << (plan.prefill.output_only_last_token ? "yes" : "no") << '\n';
  output << "Cache bytes: kv=" << plan.cache.kv_cache_bytes
         << " (" << (plan.cache.kv_cache_f16 ? "f16" : "f32") << ")"
         << ", recurrent_r=" << plan.cache.recurrent_r_cache_bytes
         << ", recurrent_s=" << plan.cache.recurrent_s_cache_bytes
         << ", total=" << plan.cache.total_cache_bytes << '\n';
  return output.str();
}

Result<Qwen35MatmulBenchResult> benchmark_qwen35_metal_matmul(
  const Qwen35MatmulBenchConfig& config) {
  if (config.model_dir.empty()) {
    return Status::invalid_argument("Qwen3.5 matmul bench requires --model");
  }
  if (config.tensor_name.empty()) {
    return Status::invalid_argument("Qwen3.5 matmul bench requires --tensor");
  }
  if (config.tokens == 0) {
    return Status::invalid_argument("Qwen3.5 matmul bench tokens must be positive");
  }
  if (config.timed_iterations == 0) {
    return Status::invalid_argument(
      "Qwen3.5 matmul bench timed iterations must be positive");
  }

  auto gguf_path = resolve_gguf_model_path(config.model_dir);
  if (!gguf_path.is_ok()) {
    return gguf_path.status();
  }
  auto gguf = read_gguf_file(gguf_path.value());
  if (!gguf.is_ok()) {
    return gguf.status();
  }
  const auto* tensor = find_gguf_tensor(gguf.value(), config.tensor_name);
  if (tensor == nullptr) {
    return Status::invalid_argument("Qwen3.5 matmul bench tensor not found: " +
                                    config.tensor_name);
  }
  if (tensor->shape.size() != 2U) {
    return Status::invalid_argument("Qwen3.5 matmul bench tensor must be 2D: " +
                                    tensor->name);
  }
  if (tensor->type != 12U && tensor->type != 13U && tensor->type != 14U) {
    return Status::invalid_argument(
      "Qwen3.5 matmul bench currently supports only Q4_K/Q5_K/Q6_K tensors: " +
      tensor->name);
  }

  auto cols_result = checked_size_t(tensor->shape[0], tensor->name.c_str());
  auto rows_result = checked_size_t(tensor->shape[1], tensor->name.c_str());
  if (!cols_result.is_ok()) {
    return cols_result.status();
  }
  if (!rows_result.is_ok()) {
    return rows_result.status();
  }
  const auto cols = cols_result.value();
  const auto rows = rows_result.value();
  if (cols == 0 || rows == 0) {
    return Status::invalid_argument("Qwen3.5 matmul bench tensor shape is empty: " +
                                    tensor->name);
  }
  if (cols % 256U != 0) {
    return Status::invalid_argument(
      "Qwen3.5 matmul bench K-quant cols must be divisible by 256: " +
      tensor->name);
  }

  std::size_t input_values = 0;
  std::size_t output_values = 0;
  if (!checked_mul_size(config.tokens, cols, input_values) ||
      !checked_mul_size(config.tokens, rows, output_values)) {
    return Status::invalid_argument("Qwen3.5 matmul bench value count overflow");
  }
  std::uint64_t input_bytes_u64 = static_cast<std::uint64_t>(input_values);
  auto byte_status =
    checked_mul_assign(input_bytes_u64, sizeof(float), "Qwen3.5 matmul bench input");
  if (!byte_status.is_ok()) {
    return byte_status;
  }
  std::uint64_t output_bytes_u64 = static_cast<std::uint64_t>(output_values);
  byte_status =
    checked_mul_assign(output_bytes_u64, sizeof(float), "Qwen3.5 matmul bench output");
  if (!byte_status.is_ok()) {
    return byte_status;
  }
  auto input_bytes = checked_size_t(input_bytes_u64, "Qwen3.5 matmul bench input");
  auto output_bytes = checked_size_t(output_bytes_u64, "Qwen3.5 matmul bench output");
  auto weight_bytes = checked_size_t(tensor->byte_size, tensor->name.c_str());
  if (!input_bytes.is_ok()) {
    return input_bytes.status();
  }
  if (!output_bytes.is_ok()) {
    return output_bytes.status();
  }
  if (!weight_bytes.is_ok()) {
    return weight_bytes.status();
  }

  auto mapped = GgufMappedData::open(gguf.value());
  if (!mapped.is_ok()) {
    return mapped.status();
  }
  auto tensor_bytes = mapped.value().tensor_bytes(*tensor);
  if (!tensor_bytes.is_ok()) {
    return tensor_bytes.status();
  }
  if (tensor_bytes.value().data == nullptr ||
      tensor_bytes.value().size != weight_bytes.value()) {
    return Status::internal_error("Qwen3.5 matmul bench tensor bytes mismatch: " +
                                  tensor->name);
  }

  auto context_result = mps::MpsContext::create();
  if (!context_result.is_ok()) {
    return context_result.status();
  }
  auto context = std::move(context_result.value());
  auto weight_buffer = context.make_buffer(weight_bytes.value());
  if (!weight_buffer.is_ok()) {
    return weight_buffer.status();
  }
  auto status = context.copy_to_buffer(weight_buffer.value(), tensor_bytes.value().data,
                                       tensor_bytes.value().size);
  if (!status.is_ok()) {
    return status;
  }

  std::vector<float> input(input_values, 0.0F);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const auto bucket = static_cast<int>(index % 23U) - 11;
    input[index] = static_cast<float>(bucket) * 0.03125F;
  }
  auto input_buffer = context.make_buffer(input_bytes.value());
  if (!input_buffer.is_ok()) {
    return input_buffer.status();
  }
  status = context.copy_to_buffer(input_buffer.value(), input.data(), input_bytes.value());
  if (!status.is_ok()) {
    return status;
  }
  auto output_buffer = context.make_buffer(output_bytes.value());
  if (!output_buffer.is_ok()) {
    return output_buffer.status();
  }
  status = context.zero_buffer(output_buffer.value(), output_bytes.value());
  if (!status.is_ok()) {
    return status;
  }

  auto run_once = [&]() -> Status {
    MpsGraphGuard graph(context);
    auto graph_status = graph.begin();
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    switch (tensor->type) {
      case 12:
        graph_status = context.matmul_q4_k_f32_device(
          weight_buffer.value(), rows, cols, config.tokens,
          input_buffer.value(), output_buffer.value());
        break;
      case 13:
        graph_status = context.matmul_q5_k_f32_device(
          weight_buffer.value(), rows, cols, config.tokens,
          input_buffer.value(), output_buffer.value());
        break;
      case 14:
        graph_status = context.matmul_q6_k_f32_device(
          weight_buffer.value(), rows, cols, config.tokens,
          input_buffer.value(), output_buffer.value());
        break;
      default:
        graph_status = Status::invalid_argument(
          "Qwen3.5 matmul bench unsupported tensor type");
        break;
    }
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    return graph.commit();
  };

  for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
    status = run_once();
    if (!status.is_ok()) {
      return status;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < config.timed_iterations; ++iteration) {
    status = run_once();
    if (!status.is_ok()) {
      return status;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto total_seconds =
    std::chrono::duration<double>(end - start).count();

  const auto sample_values = std::min<std::size_t>(output_values, 16U);
  std::vector<float> sample(sample_values, 0.0F);
  status = context.copy_from_buffer(output_buffer.value(), sample.data(),
                                    sample.size() * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  double checksum = 0.0;
  for (std::size_t index = 0; index < sample.size(); ++index) {
    checksum += static_cast<double>(sample[index]) *
                static_cast<double>(index + 1U);
  }

  Qwen35MatmulBenchResult result;
  result.gguf_path = gguf_path.value();
  result.tensor_name = tensor->name;
  result.type_name = ggml_type_name(tensor->type);
  result.dispatch = qwen35_quant_matmul_dispatch_name(tensor->type, config.tokens);
  result.type = tensor->type;
  result.tokens = config.tokens;
  result.rows = rows;
  result.cols = cols;
  result.warmup_iterations = config.warmup_iterations;
  result.timed_iterations = config.timed_iterations;
  result.output_values_per_iteration = static_cast<std::uint64_t>(output_values);
  result.total_seconds = total_seconds;
  result.average_milliseconds =
    total_seconds * 1000.0 / static_cast<double>(config.timed_iterations);
  const auto timed_iterations = static_cast<double>(config.timed_iterations);
  result.token_iterations_per_second =
    static_cast<double>(config.tokens) * timed_iterations / total_seconds;
  result.output_megavalues_per_second =
    static_cast<double>(output_values) * timed_iterations / (total_seconds * 1000000.0);
  result.sample_checksum = checksum;
  return result;
}

std::string format_qwen35_matmul_bench_result(
  const Qwen35MatmulBenchResult& result) {
  std::ostringstream output;
  output << "Qwen3.5 Metal matmul bench: ok\n";
  output << "GGUF: " << result.gguf_path.string() << '\n';
  output << "Tensor: " << result.tensor_name << '\n';
  output << "Type: " << result.type_name << " (" << result.type << ")\n";
  output << "Dispatch: " << result.dispatch << '\n';
  output << "Shape: rows=" << result.rows << ", cols=" << result.cols
         << ", tokens=" << result.tokens << '\n';
  output << "Iterations: warmup=" << result.warmup_iterations
         << ", timed=" << result.timed_iterations << '\n';
  output << "Output values/iter: " << result.output_values_per_iteration << '\n';
  output << "Total seconds: " << result.total_seconds << '\n';
  output << "Average ms/iter: " << result.average_milliseconds << '\n';
  output << "Token-iterations/s: " << result.token_iterations_per_second << '\n';
  output << "Output Mvalues/s: " << result.output_megavalues_per_second << '\n';
  output << "Sample checksum: " << result.sample_checksum << '\n';
  return output.str();
}

Result<Qwen35GdnBenchResult> benchmark_qwen35_metal_gdn(
  const Qwen35GdnBenchConfig& config) {
  if (config.tokens == 0 || config.key_heads == 0 || config.value_heads == 0 ||
      config.head_dim == 0) {
    return Status::invalid_argument("Qwen3.5 GDN bench dimensions must be positive");
  }
  if (config.timed_iterations == 0) {
    return Status::invalid_argument(
      "Qwen3.5 GDN bench timed iterations must be positive");
  }

  std::size_t key_dim = 0;
  std::size_t value_dim = 0;
  std::size_t qk_values = 0;
  std::size_t value_values = 0;
  std::size_t gate_values = 0;
  std::size_t state_rows = 0;
  std::size_t state_values = 0;
  if (!checked_mul_size(config.key_heads, config.head_dim, key_dim) ||
      !checked_mul_size(config.value_heads, config.head_dim, value_dim) ||
      !checked_mul_size(config.tokens, key_dim, qk_values) ||
      !checked_mul_size(config.tokens, value_dim, value_values) ||
      !checked_mul_size(config.tokens, config.value_heads, gate_values) ||
      !checked_mul_size(config.value_heads, config.head_dim, state_rows) ||
      !checked_mul_size(state_rows, config.head_dim, state_values)) {
    return Status::invalid_argument("Qwen3.5 GDN bench value count overflow");
  }

  auto context_result = mps::MpsContext::create();
  if (!context_result.is_ok()) {
    return context_result.status();
  }
  auto context = std::move(context_result.value());

  std::vector<float> query(qk_values, 0.0F);
  std::vector<float> key(qk_values, 0.0F);
  std::vector<float> value(value_values, 0.0F);
  std::vector<float> gate(gate_values, 0.0F);
  std::vector<float> beta(gate_values, 0.0F);
  std::vector<float> state(state_values, 0.0F);

  for (std::size_t index = 0; index < qk_values; ++index) {
    query[index] = 0.001F * static_cast<float>(1U + index % 17U);
    key[index] = 0.0007F * static_cast<float>(1U + index % 19U);
  }
  for (std::size_t index = 0; index < value_values; ++index) {
    value[index] = 0.002F * static_cast<float>(1U + index % 23U);
  }
  for (std::size_t index = 0; index < gate_values; ++index) {
    gate[index] = -0.002F * static_cast<float>(1U + index % 7U);
    beta[index] = 0.25F + 0.01F * static_cast<float>(index % 11U);
  }
  for (std::size_t index = 0; index < state_values; ++index) {
    state[index] = 0.0001F * static_cast<float>(1U + index % 13U);
  }

  auto query_buffer = make_qwen35_f32_buffer(context, query,
                                             "Qwen3.5 GDN bench query");
  auto key_buffer = make_qwen35_f32_buffer(context, key,
                                           "Qwen3.5 GDN bench key");
  auto value_buffer = make_qwen35_f32_buffer(context, value,
                                             "Qwen3.5 GDN bench value");
  auto gate_buffer = make_qwen35_f32_buffer(context, gate,
                                            "Qwen3.5 GDN bench gate");
  auto beta_buffer = make_qwen35_f32_buffer(context, beta,
                                            "Qwen3.5 GDN bench beta");
  auto state_buffer = make_qwen35_f32_buffer(context, state,
                                             "Qwen3.5 GDN bench state");
  auto output_buffer = make_qwen35_f32_buffer(context, value_values,
                                              "Qwen3.5 GDN bench output");
  if (!query_buffer.is_ok()) {
    return query_buffer.status();
  }
  if (!key_buffer.is_ok()) {
    return key_buffer.status();
  }
  if (!value_buffer.is_ok()) {
    return value_buffer.status();
  }
  if (!gate_buffer.is_ok()) {
    return gate_buffer.status();
  }
  if (!beta_buffer.is_ok()) {
    return beta_buffer.status();
  }
  if (!state_buffer.is_ok()) {
    return state_buffer.status();
  }
  if (!output_buffer.is_ok()) {
    return output_buffer.status();
  }
  auto status = context.zero_buffer(output_buffer.value(),
                                    output_buffer.value().byte_size());
  if (!status.is_ok()) {
    return status;
  }

  auto run_once = [&]() -> Status {
    MpsGraphGuard graph(context);
    auto graph_status = graph.begin();
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    graph_status = context.gated_delta_net_f32_batched_in_place(
      query_buffer.value(), key_buffer.value(), value_buffer.value(),
      gate_buffer.value(), beta_buffer.value(), state_buffer.value(),
      config.tokens, config.key_heads, config.value_heads, config.head_dim,
      output_buffer.value());
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    return graph.commit();
  };

  for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
    status = run_once();
    if (!status.is_ok()) {
      return status;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < config.timed_iterations; ++iteration) {
    status = run_once();
    if (!status.is_ok()) {
      return status;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto total_seconds =
    std::chrono::duration<double>(end - start).count();

  const auto sample_values = std::min<std::size_t>(value_values, 16U);
  std::vector<float> sample(sample_values, 0.0F);
  status = context.copy_from_buffer(output_buffer.value(), sample.data(),
                                    sample.size() * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  double checksum = 0.0;
  for (std::size_t index = 0; index < sample.size(); ++index) {
    checksum += static_cast<double>(sample[index]) *
                static_cast<double>(index + 1U);
  }

  const auto timed_iterations = static_cast<double>(config.timed_iterations);
  Qwen35GdnBenchResult result;
  result.dispatch = qwen35_gdn_dispatch_name(config.key_heads, config.value_heads,
                                             config.head_dim);
  result.tokens = config.tokens;
  result.key_heads = config.key_heads;
  result.value_heads = config.value_heads;
  result.head_dim = config.head_dim;
  result.warmup_iterations = config.warmup_iterations;
  result.timed_iterations = config.timed_iterations;
  result.state_values = static_cast<std::uint64_t>(state_values);
  result.output_values_per_iteration = static_cast<std::uint64_t>(value_values);
  result.total_seconds = total_seconds;
  result.average_milliseconds =
    total_seconds * 1000.0 / static_cast<double>(config.timed_iterations);
  result.token_iterations_per_second =
    static_cast<double>(config.tokens) * timed_iterations / total_seconds;
  result.output_megavalues_per_second =
    static_cast<double>(value_values) * timed_iterations /
    (total_seconds * 1000000.0);
  result.sample_checksum = checksum;
  return result;
}

std::string format_qwen35_gdn_bench_result(const Qwen35GdnBenchResult& result) {
  std::ostringstream output;
  output << "Qwen3.5 Metal GDN bench: ok\n";
  output << "Dispatch: " << result.dispatch << '\n';
  output << "Shape: tokens=" << result.tokens
         << ", key_heads=" << result.key_heads
         << ", value_heads=" << result.value_heads
         << ", head_dim=" << result.head_dim << '\n';
  output << "Iterations: warmup=" << result.warmup_iterations
         << ", timed=" << result.timed_iterations << '\n';
  output << "State values: " << result.state_values << '\n';
  output << "Output values/iter: " << result.output_values_per_iteration << '\n';
  output << "Total seconds: " << result.total_seconds << '\n';
  output << "Average ms/iter: " << result.average_milliseconds << '\n';
  output << "Token-iterations/s: " << result.token_iterations_per_second << '\n';
  output << "Output Mvalues/s: " << result.output_megavalues_per_second << '\n';
  output << "Sample checksum: " << result.sample_checksum << '\n';
  return output.str();
}

Result<Qwen35AttentionBenchResult> benchmark_qwen35_metal_attention(
  const Qwen35AttentionBenchConfig& config) {
  if (config.tokens == 0 || config.heads == 0 || config.kv_heads == 0 ||
      config.head_dim == 0) {
    return Status::invalid_argument(
      "Qwen3.5 attention bench dimensions must be positive");
  }
  if (config.heads % config.kv_heads != 0) {
    return Status::invalid_argument(
      "Qwen3.5 attention bench heads must be divisible by kv_heads");
  }
  if (config.head_dim > 256U) {
    return Status::invalid_argument(
      "Qwen3.5 attention bench head_dim must be at most 256");
  }
  if (config.timed_iterations == 0) {
    return Status::invalid_argument(
      "Qwen3.5 attention bench timed iterations must be positive");
  }
  if (config.start_position >
      std::numeric_limits<std::size_t>::max() - config.tokens) {
    return Status::invalid_argument(
      "Qwen3.5 attention bench key count overflow");
  }

  const auto key_count = config.start_position + config.tokens;
  const auto capacity_tokens =
    config.capacity_tokens == 0 ? key_count : config.capacity_tokens;
  if (key_count > capacity_tokens) {
    return Status::invalid_argument(
      "Qwen3.5 attention bench chunk exceeds capacity_tokens");
  }

  std::size_t attn_dim = 0;
  std::size_t kv_dim = 0;
  std::size_t query_values = 0;
  std::size_t cache_values = 0;
  std::size_t f16_cache_bytes = 0;
  if (!checked_mul_size(config.heads, config.head_dim, attn_dim) ||
      !checked_mul_size(config.kv_heads, config.head_dim, kv_dim) ||
      !checked_mul_size(config.tokens, attn_dim, query_values) ||
      !checked_mul_size(capacity_tokens, kv_dim, cache_values) ||
      !checked_mul_size(cache_values, sizeof(std::uint16_t),
                        f16_cache_bytes)) {
    return Status::invalid_argument("Qwen3.5 attention bench value count overflow");
  }

  auto context_result = mps::MpsContext::create();
  if (!context_result.is_ok()) {
    return context_result.status();
  }
  auto context = std::move(context_result.value());

  std::vector<float> query(query_values, 0.0F);
  std::vector<float> key_cache(cache_values, 0.0F);
  std::vector<float> value_cache(cache_values, 0.0F);
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] =
      static_cast<float>(static_cast<int>((index * 13U) % 31U) - 15) * 0.015F;
  }
  for (std::size_t index = 0; index < key_cache.size(); ++index) {
    key_cache[index] =
      static_cast<float>(static_cast<int>((index * 7U) % 29U) - 14) * 0.012F;
    value_cache[index] =
      static_cast<float>(static_cast<int>((index * 11U) % 37U) - 18) * 0.025F;
  }

  auto query_buffer = make_qwen35_f32_buffer(context, query,
                                             "Qwen3.5 attention bench query");
  if (!query_buffer.is_ok()) {
    return query_buffer.status();
  }

  mps::MpsBuffer key_cache_buffer;
  mps::MpsBuffer value_cache_buffer;
  Status status;
  if (config.f16_kv) {
    auto f32_key_cache = make_qwen35_f32_buffer(
      context, key_cache, "Qwen3.5 attention bench F32 key cache");
    auto f32_value_cache = make_qwen35_f32_buffer(
      context, value_cache, "Qwen3.5 attention bench F32 value cache");
    if (!f32_key_cache.is_ok()) {
      return f32_key_cache.status();
    }
    if (!f32_value_cache.is_ok()) {
      return f32_value_cache.status();
    }
    auto key_cache_result = context.make_buffer(f16_cache_bytes);
    auto value_cache_result = context.make_buffer(f16_cache_bytes);
    if (!key_cache_result.is_ok()) {
      return key_cache_result.status();
    }
    if (!value_cache_result.is_ok()) {
      return value_cache_result.status();
    }
    status = context.copy_f32_rows_to_f16(
      f32_key_cache.value(), key_cache_result.value(), capacity_tokens, kv_dim,
      kv_dim, 0, kv_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    status = context.copy_f32_rows_to_f16(
      f32_value_cache.value(), value_cache_result.value(), capacity_tokens,
      kv_dim, kv_dim, 0, kv_dim, 0);
    if (!status.is_ok()) {
      return status;
    }
    key_cache_buffer = std::move(key_cache_result.value());
    value_cache_buffer = std::move(value_cache_result.value());
  } else {
    auto key_cache_result = make_qwen35_f32_buffer(
      context, key_cache, "Qwen3.5 attention bench F32 key cache");
    auto value_cache_result = make_qwen35_f32_buffer(
      context, value_cache, "Qwen3.5 attention bench F32 value cache");
    if (!key_cache_result.is_ok()) {
      return key_cache_result.status();
    }
    if (!value_cache_result.is_ok()) {
      return value_cache_result.status();
    }
    key_cache_buffer = std::move(key_cache_result.value());
    value_cache_buffer = std::move(value_cache_result.value());
  }

  auto output_buffer = make_qwen35_f32_buffer(
    context, query_values, "Qwen3.5 attention bench output");
  if (!output_buffer.is_ok()) {
    return output_buffer.status();
  }
  status = context.zero_buffer(output_buffer.value(),
                               output_buffer.value().byte_size());
  if (!status.is_ok()) {
    return status;
  }

  auto run_once = [&]() -> Status {
    MpsGraphGuard graph(context);
    auto graph_status = graph.begin();
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    if (config.f16_kv) {
      graph_status = context.attention_f32_batched_f16_kv(
        query_buffer.value(), key_cache_buffer, value_cache_buffer, 0,
        config.start_position, config.tokens, capacity_tokens, config.heads,
        config.kv_heads, config.head_dim, output_buffer.value());
    } else {
      graph_status = context.attention_f32_batched(
        query_buffer.value(), key_cache_buffer, value_cache_buffer, 0,
        config.start_position, config.tokens, capacity_tokens, config.heads,
        config.kv_heads, config.head_dim, output_buffer.value());
    }
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    return graph.commit();
  };

  for (std::size_t iteration = 0; iteration < config.warmup_iterations; ++iteration) {
    status = run_once();
    if (!status.is_ok()) {
      return status;
    }
  }

  const auto start = std::chrono::steady_clock::now();
  for (std::size_t iteration = 0; iteration < config.timed_iterations; ++iteration) {
    status = run_once();
    if (!status.is_ok()) {
      return status;
    }
  }
  const auto end = std::chrono::steady_clock::now();
  const auto total_seconds =
    std::chrono::duration<double>(end - start).count();

  const auto sample_values = std::min<std::size_t>(query_values, 16U);
  std::vector<float> sample(sample_values, 0.0F);
  status = context.copy_from_buffer(output_buffer.value(), sample.data(),
                                    sample.size() * sizeof(float));
  if (!status.is_ok()) {
    return status;
  }
  double checksum = 0.0;
  for (std::size_t index = 0; index < sample.size(); ++index) {
    checksum += static_cast<double>(sample[index]) *
                static_cast<double>(index + 1U);
  }

  const auto timed_iterations = static_cast<double>(config.timed_iterations);
  Qwen35AttentionBenchResult result;
  result.dispatch = qwen35_attention_dispatch_name(
    config.tokens, key_count, config.head_dim, config.f16_kv);
  result.tokens = config.tokens;
  result.start_position = config.start_position;
  result.capacity_tokens = capacity_tokens;
  result.key_count = key_count;
  result.heads = config.heads;
  result.kv_heads = config.kv_heads;
  result.head_dim = config.head_dim;
  result.warmup_iterations = config.warmup_iterations;
  result.timed_iterations = config.timed_iterations;
  result.cache_values = static_cast<std::uint64_t>(cache_values);
  result.output_values_per_iteration = static_cast<std::uint64_t>(query_values);
  result.total_seconds = total_seconds;
  result.average_milliseconds =
    total_seconds * 1000.0 / static_cast<double>(config.timed_iterations);
  result.token_iterations_per_second =
    static_cast<double>(config.tokens) * timed_iterations / total_seconds;
  result.output_megavalues_per_second =
    static_cast<double>(query_values) * timed_iterations /
    (total_seconds * 1000000.0);
  result.sample_checksum = checksum;
  result.f16_kv = config.f16_kv;
  return result;
}

std::string format_qwen35_attention_bench_result(
  const Qwen35AttentionBenchResult& result) {
  std::ostringstream output;
  output << "Qwen3.5 Metal attention bench: ok\n";
  output << "Dispatch: " << result.dispatch << '\n';
  output << "KV cache: " << (result.f16_kv ? "f16" : "f32") << '\n';
  output << "Shape: tokens=" << result.tokens
         << ", start_position=" << result.start_position
         << ", key_count=" << result.key_count
         << ", capacity_tokens=" << result.capacity_tokens
         << ", heads=" << result.heads
         << ", kv_heads=" << result.kv_heads
         << ", head_dim=" << result.head_dim << '\n';
  output << "Iterations: warmup=" << result.warmup_iterations
         << ", timed=" << result.timed_iterations << '\n';
  output << "Cache values: " << result.cache_values << '\n';
  output << "Output values/iter: " << result.output_values_per_iteration << '\n';
  output << "Total seconds: " << result.total_seconds << '\n';
  output << "Average ms/iter: " << result.average_milliseconds << '\n';
  output << "Token-iterations/s: " << result.token_iterations_per_second << '\n';
  output << "Output Mvalues/s: " << result.output_megavalues_per_second << '\n';
  output << "Sample checksum: " << result.sample_checksum << '\n';
  return output.str();
}
// qwen3.5 0.8b 核心入口
Result<CpuGenerationResult> generate_qwen35_metal(const CpuGenerationRequest& request) {
  if (request.prompt.empty() && request.messages.empty()) {
    return Status::invalid_argument("prompt must not be empty");
  }
  if (request.max_new_tokens == 0) {
    return Status::invalid_argument("max_new_tokens must be greater than zero");
  }
  if (!(request.mtp_p_min >= 0.0 && request.mtp_p_min <= 1.0)) {
    return Status::invalid_argument("mtp_p_min must be in [0, 1]");
  }
  std::optional<Qwen35MmprojMetadata> image_mmproj_metadata;
  if (chat_messages_have_image_content(request.messages)) {
    if (request.mmproj_path.empty()) {
      return Status::invalid_argument(
        "image input requires a Qwen3.5 conditional mmproj GGUF");
    }
    const auto metadata = load_qwen35_mmproj_metadata(request.mmproj_path);
    if (!metadata.is_ok()) {
      return metadata.status();
    }
    if (!qwen35_mmproj_is_qwen3vl_merger(metadata.value())) {
      return Status::invalid_argument(
        "Qwen3.5 image input requires a qwen3vl_merger mmproj");
    }
    image_mmproj_metadata = metadata.value();
  }
  std::unique_lock<std::mutex> prompt_cache_lock(
    qwen35_host_prefix_cache_mutex(), std::defer_lock);
  if (request.cache_prompt) {
    prompt_cache_lock.lock();
  }

  RequestProfiler profiler(request.observability);
  profiler.set_metadata("model_dir", request.model_dir.string());
  profiler.set_metadata("device", request.compute_device.to_string());
  if (!request.observability.client_request_id.empty()) {
    profiler.set_metadata("client_request_id",
                          request.observability.client_request_id);
  }

  auto bundle = [&]() {
    auto span = profiler.scoped("request.load_model");
    return load_model_bundle(request.model_dir);
  }();
  if (!bundle.is_ok()) {
    return bundle.status();
  }
  if (!bundle.value().model.gguf) {
    return Status::invalid_argument("Qwen3.5 Metal runtime requires a GGUF model");
  }
  if (bundle.value().model.architecture != "qwen35") {
    return Status::unavailable("native Qwen3.5 Metal runtime currently supports dense qwen35; got " +
                               bundle.value().model.architecture);
  }
  if (image_mmproj_metadata.has_value()) {
    const auto compatibility =
      validate_qwen35_mmproj_text_embedding_compatibility(
        *image_mmproj_metadata, bundle.value().model.hidden_size);
    if (!compatibility.is_ok()) {
      return compatibility;
    }
  }
  /*
    默认 CpuSamplingConfig 与 Qwen3.5 GGUF generation config 合并后，
    sampling.value() 逻辑上等价于下面的 JSONC：

    {
      // false 表示使用 greedy decoding，每次直接选择概率最高的 token。
      "do_sample": false,
      // 温度越高分布越随机；greedy decoding 下不参与 token 选择。
      "temperature": 1.0,
      // 0 表示不限制只从概率最高的 K 个 token 中采样。
      "top_k": 0,
      // 1.0 表示不通过累计概率截断候选 token。
      "top_p": 1.0,
      // 随机数种子；greedy decoding 不使用随机采样，因此默认为 0。
      "seed": 0
    }

    request.sampling 中显式设置的 temperature/top_k/top_p 会覆盖模型默认值；
    do_sample=false 时这些采样参数不会改变最终选择的 token。
  */
  auto sampling = make_qwen35_effective_sampling(bundle.value().generation,
                                                 request.sampling);
  if (!sampling.is_ok()) {
    return sampling.status();
  }
  std::mt19937 sampling_rng(static_cast<std::uint32_t>(sampling.value().seed));

  auto gguf = [&]() {
    auto span = profiler.scoped("qwen35.gguf.read");
    return read_gguf_file(bundle.value().model_file);
  }();
  if (!gguf.is_ok()) {
    return gguf.status();
  }

  /*
    对当前 Qwen3.5-0.8B GGUF，map.value() 逻辑上类似下面的 JSONC。
    它只保存 tensor 的定位和布局信息，不复制 tensor 数据；每个 binding
    实际还包含 type、offset、absolute_offset 和 byte_size。

    {
      "token_embedding": {
        "name": "token_embd.weight",
        "shape": [1024, 248320]
      },
      "output_norm": {
        "name": "output_norm.weight",
        "shape": [1024]
      },
      // 当前模型没有独立 output.weight，LM Head 复用 token embedding。
      "output": {
        "name": "token_embd.weight",
        "shape": [1024, 248320]
      },
      "output_tied_to_token_embedding": true,
      "layers": [
        {
          "index": 0,
          "kind": "linear_attention",
          "attn_norm": {"name": "blk.0.attn_norm.weight"},
          "attn_post_norm": {"name": "blk.0.post_attention_norm.weight"},
          "ffn_gate": {"name": "blk.0.ffn_gate.weight"},
          "ffn_up": {"name": "blk.0.ffn_up.weight"},
          "ffn_down": {"name": "blk.0.ffn_down.weight"},
          "linear_attention": {
            "qkv": {"name": "blk.0.attn_qkv.weight"},
            "gate": {"name": "blk.0.attn_gate.weight"},
            "conv1d": {"name": "blk.0.ssm_conv1d.weight"},
            "dt_bias": {"name": "blk.0.ssm_dt.bias"},
            "a": {"name": "blk.0.ssm_a"},
            "beta": {"name": "blk.0.ssm_beta.weight"},
            "alpha": {"name": "blk.0.ssm_alpha.weight"},
            "norm": {"name": "blk.0.ssm_norm.weight"},
            "output": {"name": "blk.0.ssm_out.weight"}
          }
        }
        // 其余 23 层结构相同，kind 决定使用 linear_attention、
        // full_attention 或 mtp binding。
      ],
      "full_attention_layers": 6,
      "linear_attention_layers": 18,
      "mtp_layers": 0
    }
  */
  auto map = [&]() {
    auto span = profiler.scoped("qwen35.weight_map");
    return build_qwen35_weight_map(bundle.value().model, gguf.value());
  }();
  if (!map.is_ok()) {
    return map.status();
  }
  // tokenizer 错误会导致后续 prefill/decoding 失败，因此在这里提前返回错误。
  auto tokenizer = [&]() {
    auto span = profiler.scoped("request.tokenize.load");
    return load_gguf_tokenizer(gguf.value());
  }();
  if (!tokenizer.is_ok()) {
    return tokenizer.status();
  }
  // 用户输入
  std::string prompt = request.prompt;
  Result<std::vector<std::int64_t>> prompt_tokens{std::vector<std::int64_t>{}};
  std::optional<Qwen35MixedPrefillPlan> mixed_prefill;
  {
    auto span = profiler.scoped("request.tokenize");
    if (image_mmproj_metadata.has_value()) {
      auto mixed = build_qwen35_mixed_prefill_plan(
        tokenizer.value(), *image_mmproj_metadata, request.mmproj_path,
        request.messages, true, request.enable_thinking);
      if (!mixed.is_ok()) {
        return mixed.status();
      }
      mixed_prefill = std::move(mixed.value());
    } else if (!request.messages.empty()) {

      // format 过程
      auto formatted = format_qwen35_chat_prompt(tokenizer.value(),
                                                 request.messages, true,
                                                 request.enable_thinking);
      if (!formatted.is_ok()) {
        return formatted.status();
      }
      prompt = formatted.value();
      // 解析特殊 prompt 语法
      // 把 prompt 字符串转换成模型可以处理的 token ID 数组。
      prompt_tokens = gguf_encode_text(tokenizer.value(), prompt, false, true);
    } else {
      prompt_tokens = gguf_encode_text(tokenizer.value(), prompt, false,
                                       request.parse_special_prompt);
    }
  }
  if (!prompt_tokens.is_ok()) {
    return prompt_tokens.status();
  }


  //
  const auto prompt_token_count = mixed_prefill.has_value()
                                    ? mixed_prefill->total_tokens
                                    : prompt_tokens.value().size();
  const auto prompt_mrope_position_count =
    mixed_prefill.has_value() ? mixed_prefill->total_position_advance
                              : prompt_token_count;
  profiler.set_metadata("prompt_tokens", prompt_token_count);
  profiler.set_metadata("prompt_mrope_positions", prompt_mrope_position_count);
  profiler.set_metadata("qwen35_native_vl",
                        std::string{mixed_prefill.has_value() ? "1" : "0"});
  if (mixed_prefill.has_value()) {
    profiler.set_metadata("qwen35_native_vl_image_tokens",
                          mixed_prefill->image_tokens);
  }
  cpu::DebugDumper debug_dumper(request.debug_dump_dir);
  cpu::DebugDumper* debug_dump =
    debug_dumper.enabled() ? &debug_dumper : nullptr;
  if (debug_dump != nullptr && !mixed_prefill.has_value()) {
    debug_dump->write_i64(
      "prompt_tokens",
      {static_cast<std::uint64_t>(prompt_tokens.value().size())},
      prompt_tokens.value());
  }

  Qwen35RuntimeOptions options;
  options.decode_tokens = request.max_new_tokens;
  options.kv_cache_f16 = qwen35_f16_kv_cache_enabled();
  if (request.prefill_chunk_tokens != 0) {
    options.prefill_chunk_tokens = request.prefill_chunk_tokens;
  }
  const auto plan = [&]() {
    auto span = profiler.scoped("qwen35.plan");
    return build_qwen35_execution_plan(bundle.value().model, map.value(),
                                       prompt_token_count, options);
  }();
  if (!plan.is_ok()) {
    return plan.status();
  }
  Qwen35MtpRuntimeState mtp_state;
  mtp_state.available = map.value().mtp_layers > 0;
  mtp_state.draft_tokens = request.mtp_draft_tokens;
  mtp_state.p_min = request.mtp_p_min;
  if (!mtp_state.available) {
    mtp_state.disabled_reason = "model_has_no_mtp_layers";
  } else if (!request.enable_mtp) {
    mtp_state.disabled_reason = "request_disabled";
  } else if (map.value().mtp_layers != 1U) {
    mtp_state.disabled_reason = "only_single_mtp_layer_supported";
  } else if (sampling.value().do_sample) {
    mtp_state.disabled_reason = "sampling_not_supported_in_first_mtp_pass";
  } else if (request.mtp_draft_tokens == 0) {
    mtp_state.disabled_reason = "draft_tokens_zero";
  } else if (mixed_prefill.has_value()) {
    mtp_state.disabled_reason = "multimodal_prompt_not_supported_with_mtp";
  } else {
    mtp_state.enabled = true;
    // Start conservatively and let the adaptive controller ramp up only when
    // accepted drafts justify the extra MTP work on this machine/request.
    mtp_state.adaptive_budget = 1;
    mtp_state.verified_by_position.assign(mtp_state.draft_tokens, 0);
    mtp_state.accepted_by_position.assign(mtp_state.draft_tokens, 0);
  }
  profiler.set_metadata("qwen35_mtp_available",
                        std::string{mtp_state.available ? "1" : "0"});
  profiler.set_metadata("qwen35_mtp_enabled",
                        std::string{mtp_state.enabled ? "1" : "0"});
  profiler.set_metadata("qwen35_mtp_draft_tokens", mtp_state.draft_tokens);
  profiler.set_metadata("qwen35_mtp_p_min", std::to_string(mtp_state.p_min));
  if (!mtp_state.enabled && !mtp_state.disabled_reason.empty()) {
    profiler.set_metadata("qwen35_mtp_disabled_reason", mtp_state.disabled_reason);
  }
  profiler.set_metadata("prefill_chunk_tokens", plan.value().prefill.chunk_tokens);
  profiler.set_metadata("prefill_chunks", plan.value().prefill.chunk_count);
  profiler.set_metadata("qwen35_kv_cache_f16",
                        std::string{plan.value().cache.kv_cache_f16 ? "1" : "0"});
  profiler.set_metadata("qwen35_default_path", "llama_cpp_graph_parity");
  profiler.set_metadata("qwen35_fused_norm_gated",
                        std::string{qwen35_fused_norm_gated_enabled() ? "1" : "0"});
  const auto prefill_commit_mode = qwen35_prefill_commit_mode();
  switch (prefill_commit_mode) {
    case Qwen35PrefillCommitMode::off:
      profiler.set_metadata("qwen35_prefill_commit", "single");
      break;
    case Qwen35PrefillCommitMode::chunk:
      profiler.set_metadata("qwen35_prefill_commit", "chunk");
      break;
    case Qwen35PrefillCommitMode::layer:
      profiler.set_metadata("qwen35_prefill_commit", "layer");
      break;
  }
  const auto cache_block_tokens =
    request.cache_block_tokens == 0 ? plan.value().prefill.chunk_tokens
                                    : request.cache_block_tokens;
  const auto cache_capacity_blocks =
    request.cache_capacity_blocks == 0 ? std::size_t{16}
                                       : request.cache_capacity_blocks;
  bool prompt_cache_enabled =
    request.cache_prompt && cache_block_tokens > 0 &&
    cache_capacity_blocks > 0 &&
    cache_block_tokens == plan.value().prefill.chunk_tokens &&
    prefill_commit_mode == Qwen35PrefillCommitMode::chunk &&
    !mtp_state.enabled && !mixed_prefill.has_value();
  if (request.cache_prompt && prefill_commit_mode != Qwen35PrefillCommitMode::chunk) {
    profiler.set_metadata("prompt_cache_disabled_reason", "prefill_commit_mode");
  } else if (request.cache_prompt &&
             cache_block_tokens != plan.value().prefill.chunk_tokens) {
    profiler.set_metadata("prompt_cache_disabled_reason", "block_chunk_mismatch");
  } else if (request.cache_prompt && mtp_state.enabled) {
    profiler.set_metadata("prompt_cache_disabled_reason", "mtp_enabled");
  } else if (request.cache_prompt && mixed_prefill.has_value()) {
    profiler.set_metadata("prompt_cache_disabled_reason", "multimodal_prompt");
  }

  Qwen35PrefixCacheLookup prompt_cache_lookup;
  std::size_t prompt_cache_committed_tokens = 0;
  std::size_t prompt_cache_evicted_blocks = 0;
  if (prompt_cache_enabled) {
    Qwen35HostPrefixCacheLayout layout;
    layout.model_dir = request.model_dir;
    layout.model_file = bundle.value().model_file;
    layout.gguf_version = bundle.value().gguf_version;
    layout.gguf_tensor_count = bundle.value().gguf_tensor_count;
    layout.gguf_metadata_count = bundle.value().gguf_metadata_count;
    layout.gguf_file_size = bundle.value().gguf_file_size;
    layout.tokenizer_fingerprint = qwen35_tokenizer_fingerprint(tokenizer.value());
    layout.block_tokens = cache_block_tokens;
    layout.full_attention_layers = plan.value().cache.full_attention_layers;
    layout.linear_attention_layers = plan.value().cache.linear_attention_layers;
    layout.kv_dim = plan.value().cache.kv_dim;
    layout.kv_element_bytes = plan.value().cache.kv_cache_element_bytes;
    layout.hidden_size = plan.value().hidden_size;
    layout.recurrent_r_cache_bytes = plan.value().cache.recurrent_r_cache_bytes;
    layout.recurrent_s_cache_bytes = plan.value().cache.recurrent_s_cache_bytes;
    layout.kv_cache_f16 = plan.value().cache.kv_cache_f16;
    qwen35_host_prefix_cache().configure(
      layout, Qwen35PrefixCacheConfig{
                true,
                cache_block_tokens,
                cache_capacity_blocks,
                request.cache_reuse_min_tokens,
              });
    prompt_cache_lookup =
      qwen35_host_prefix_cache().lookup(prompt_tokens.value());
  }
  profiler.set_metadata("prompt_cache_enabled",
                        std::string{prompt_cache_enabled ? "1" : "0"});
  profiler.set_metadata("prompt_cache_block_tokens", cache_block_tokens);
  profiler.set_metadata("prompt_cache_hit_tokens",
                        prompt_cache_lookup.hit_tokens);
  profiler.set_metadata("prompt_cache_miss_tokens",
                        prompt_token_count - prompt_cache_lookup.hit_tokens);

  auto context = [&]() {
    auto span = profiler.scoped("qwen35.mps.create_context");
    return mps::MpsContext::create();
  }();
  if (!context.is_ok()) {
    return context.status();
  }
  auto cache = [&]() {
    auto span = profiler.scoped("qwen35.cache.allocate");
    return allocate_qwen35_metal_cache(context.value(), plan.value());
  }();
  if (!cache.is_ok()) {
    return cache.status();
  }
  if (mtp_state.enabled) {
    auto mtp_cache = allocate_qwen35_mtp_metal_cache(
      context.value(), plan.value(), map.value().mtp_layers);
    if (!mtp_cache.is_ok()) {
      return mtp_cache.status();
    }
    mtp_state.cache = std::move(mtp_cache.value());
    auto pending_h = make_qwen35_f32_buffer(context.value(), plan.value().hidden_size,
                                            "Qwen3.5 MTP pending h");
    if (!pending_h.is_ok()) {
      return pending_h.status();
    }
    auto zero_status = zero_allocated_buffer(
      context.value(), pending_h.value(), plan.value().hidden_size * sizeof(float),
      "Qwen3.5 MTP pending h");
    if (!zero_status.is_ok()) {
      return zero_status;
    }
    mtp_state.pending_h = std::move(pending_h.value());
    auto recurrent_r_snapshot_size = checked_size_t(
      cache.value().recurrent_r_cache_bytes, "Qwen3.5 MTP recurrent R snapshot");
    if (!recurrent_r_snapshot_size.is_ok()) {
      return recurrent_r_snapshot_size.status();
    }
    auto recurrent_s_snapshot_size = checked_size_t(
      cache.value().recurrent_s_cache_bytes, "Qwen3.5 MTP recurrent S snapshot");
    if (!recurrent_s_snapshot_size.is_ok()) {
      return recurrent_s_snapshot_size.status();
    }
    auto recurrent_r_snapshot =
      context.value().make_buffer(recurrent_r_snapshot_size.value());
    if (!recurrent_r_snapshot.is_ok()) {
      return recurrent_r_snapshot.status();
    }
    auto recurrent_s_snapshot =
      context.value().make_buffer(recurrent_s_snapshot_size.value());
    if (!recurrent_s_snapshot.is_ok()) {
      return recurrent_s_snapshot.status();
    }
    mtp_state.recurrent_r_snapshot = std::move(recurrent_r_snapshot.value());
    mtp_state.recurrent_s_snapshot = std::move(recurrent_s_snapshot.value());
  }
  if (prompt_cache_enabled && !prompt_cache_lookup.block_hashes.empty()) {
    auto restore_status = qwen35_host_prefix_cache().restore(
      context.value(), cache.value(), plan.value(),
      prompt_cache_lookup.block_hashes);
    if (!restore_status.is_ok()) {
      return restore_status;
    }
  }
  auto weights = [&]() {
    auto span = profiler.scoped("qwen35.weights.upload");
    return upload_qwen35_metal_weights(
      context.value(), gguf.value(),
      prompt_token_count > 1U && dense_prefill_weights_enabled());
  }();
  if (!weights.is_ok()) {
    return weights.status();
  }
  const auto* output_norm = find_metal_weight(weights.value(), map.value().output_norm.name);
  const auto* output_weight = find_metal_weight(weights.value(), map.value().output.name);
  const auto* token_embedding =
    find_metal_weight(weights.value(), map.value().token_embedding.name);
  if (output_norm == nullptr || output_weight == nullptr || token_embedding == nullptr) {
    return Status::invalid_argument("Qwen3.5 token/output weights are missing");
  }
  if (map.value().layers.empty()) {
    return Status::invalid_argument("Qwen3.5 weight map has no layers");
  }
  auto mrope_sections = qwen35_mrope_sections(bundle.value().model);
  if (!mrope_sections.is_ok()) {
    return mrope_sections.status();
  }
  Qwen35LogitsOutput logits_output;
  Status status;
  const bool prompt_cache_full_hit =
    prompt_cache_enabled && prompt_cache_lookup.hit_tokens == prompt_token_count &&
    !prompt_cache_lookup.block_hashes.empty();
  if (prompt_cache_full_hit) {
    auto final_hidden = make_qwen35_f32_buffer(context.value(),
                                               plan.value().hidden_size,
                                               "Qwen3.5 cached final hidden");
    if (!final_hidden.is_ok()) {
      return final_hidden.status();
    }
    const auto* cached_hidden =
      qwen35_host_prefix_cache().last_hidden(prompt_cache_lookup.block_hashes.back());
    if (cached_hidden == nullptr) {
      return Status::invalid_argument("Qwen3.5 prefix cache last hidden is missing");
    }
    status = context.value().copy_to_buffer(final_hidden.value(),
                                            cached_hidden->data(),
                                            cached_hidden->size());
    if (!status.is_ok()) {
      return status;
    }
    MpsGraphGuard logits_guard(context.value());
    status = logits_guard.begin();
    if (!status.is_ok()) {
      return status;
    }
    {
      auto logits_span = profiler.scoped("qwen35.prefill.logits");
      auto logits_result = run_qwen35_output_logits(
        context.value(), final_hidden.value(), plan.value().hidden_size,
        static_cast<float>(bundle.value().model.rms_norm_eps), *output_norm,
        *output_weight, debug_dump, "prefill", !sampling.value().do_sample);
      if (!logits_result.is_ok()) {
        return logits_result.status();
      }
      logits_output = std::move(logits_result.value());
    }
    {
      auto prefill_span = profiler.scoped("request.prefill");
      auto commit_span = profiler.scoped("qwen35.prefill.commit");
      status = logits_guard.commit();
    }
    if (!status.is_ok()) {
      return status;
    }
  } else {
    std::optional<mps::MpsBuffer> prompt_token_ids;
    if (!mixed_prefill.has_value()) {
      auto uploaded_prompt_token_ids = [&]() {
        auto span = profiler.scoped("qwen35.prompt_ids.upload");
        return make_qwen35_token_id_buffer(context.value(), prompt_tokens.value());
      }();
      if (!uploaded_prompt_token_ids.is_ok()) {
        return uploaded_prompt_token_ids.status();
      }
      prompt_token_ids = std::move(uploaded_prompt_token_ids.value());
    }
    MpsGraphGuard graph_guard(context.value());
    auto graph_status = [&]() {
      auto span = profiler.scoped("qwen35.prefill.graph_begin");
      return graph_guard.begin();
    }();
    if (!graph_status.is_ok()) {
      return graph_status;
    }
    auto first_hidden = [&]() {
      auto span = profiler.scoped("qwen35.prefill.embedding");
      if (mixed_prefill.has_value()) {
        return make_qwen35_mixed_prefill_hidden(
          context.value(), *token_embedding, *mixed_prefill, plan.value().hidden_size);
      }
      return run_qwen35_token_embeddings(
        context.value(), weights.value(), prompt_tokens.value(),
        prompt_token_ids.value(), plan.value().hidden_size);
    }();
    if (!first_hidden.is_ok()) {
      return first_hidden.status();
    }
    auto embed_dump_status = dump_qwen35_mps_f32(
      context.value(), debug_dump, "prompt.embedding",
      {static_cast<std::uint64_t>(prompt_token_count),
       static_cast<std::uint64_t>(plan.value().hidden_size)},
      first_hidden.value(), prompt_token_count * plan.value().hidden_size);
    if (!embed_dump_status.is_ok()) {
      return embed_dump_status;
    }
    std::optional<mps::MpsBuffer> mixed_mtp_input_embeddings;
    if (mixed_prefill.has_value() && mtp_state.enabled) {
      const auto& mtp_layer = map.value().layers[plan.value().main_layers];
      const auto* mtp_embed_tokens =
        mtp_layer.mtp.has_embed_tokens
          ? find_metal_weight(weights.value(), mtp_layer.mtp.embed_tokens.name)
          : token_embedding;
      if (mtp_embed_tokens == nullptr) {
        return Status::invalid_argument(
          "Qwen3.5 mixed MTP prefill is missing embed_tokens");
      }
      auto mtp_embeddings = make_qwen35_mixed_prefill_hidden(
        context.value(), *mtp_embed_tokens, *mixed_prefill, plan.value().hidden_size);
      if (!mtp_embeddings.is_ok()) {
        return mtp_embeddings.status();
      }
      mixed_mtp_input_embeddings = std::move(mtp_embeddings.value());
    }
    if (prompt_token_count == 1) {
      embed_dump_status = dump_qwen35_mps_f32(
        context.value(), debug_dump, qwen35_position_tensor_name(0, "embedding"),
        {static_cast<std::uint64_t>(plan.value().hidden_size)},
        first_hidden.value(), plan.value().hidden_size);
      if (!embed_dump_status.is_ok()) {
        return embed_dump_status;
      }
    }
    mps::MpsBuffer last_chunk_hidden;
    std::size_t last_chunk_tokens = 0;
    for (std::size_t chunk_start = prompt_cache_lookup.hit_tokens;
         chunk_start < prompt_token_count;
         chunk_start += plan.value().prefill.chunk_tokens) {
      const auto chunk_tokens =
        std::min(plan.value().prefill.chunk_tokens,
                 prompt_token_count - chunk_start);
      auto chunk_span = profiler.scoped(
        "qwen35.prefill.chunk",
        qwen35_chunk_profile_fields(chunk_start, chunk_tokens));
      auto chunk_hidden = make_qwen35_f32_buffer(
        context.value(), chunk_tokens * plan.value().hidden_size,
        "Qwen3.5 chunk hidden");
      if (!chunk_hidden.is_ok()) {
        return chunk_hidden.status();
      }
      status = context.value().copy_f32_region(
        first_hidden.value(), chunk_hidden.value(),
        chunk_start * plan.value().hidden_size, 0,
        chunk_tokens * plan.value().hidden_size);
      if (!status.is_ok()) {
        return status;
      }

      auto current_hidden = std::move(chunk_hidden.value());
      auto chunk_mrope_positions = mixed_prefill.has_value()
                                     ? make_qwen35_mrope_positions(
                                         context.value(),
                                         mixed_prefill->mrope_positions,
                                         mixed_prefill->total_tokens,
                                         chunk_start, chunk_tokens)
                                     : make_qwen35_mrope_positions(
                                         context.value(), chunk_start, chunk_tokens);
      if (!chunk_mrope_positions.is_ok()) {
        return chunk_mrope_positions.status();
      }
      std::size_t linear_layers_executed = 0;
      std::size_t full_attention_layers_executed = 0;
      std::size_t main_layers_executed = 0;
      for (std::size_t layer_index = 0; layer_index < plan.value().main_layers; ++layer_index) {
        const auto& layer = map.value().layers[layer_index];
        switch (layer.kind) {
          case Qwen35LayerKind::linear_attention: {
            auto layer_span = profiler.scoped(
              "qwen35.prefill.linear_layer",
              qwen35_layer_profile_fields(layer_index, chunk_start, chunk_tokens,
                                          "linear_attention"));
            auto layer_output = run_qwen35_linear_layer_chunk(
              context.value(), weights.value(), cache.value(), layer, current_hidden,
              chunk_tokens, plan.value().hidden_size,
              plan.value().linear_conv_kernel,
              plan.value().linear_conv_channels, plan.value().linear_key_heads,
              plan.value().linear_key_head_dim, plan.value().linear_value_heads,
              plan.value().linear_inner_size, linear_layers_executed,
              plan.value().cache.recurrent_r_elements_per_layer,
              plan.value().cache.recurrent_s_elements_per_layer,
              debug_dump, layer_index, chunk_start,
              static_cast<float>(bundle.value().model.rms_norm_eps), &profiler);
            if (!layer_output.is_ok()) {
              return layer_output.status();
            }
            current_hidden = std::move(layer_output.value());
            ++linear_layers_executed;
            break;
          }
          case Qwen35LayerKind::full_attention: {
            auto layer_span = profiler.scoped(
              "qwen35.prefill.full_attention_layer",
              qwen35_layer_profile_fields(layer_index, chunk_start, chunk_tokens,
                                          "full_attention"));
            auto layer_output = run_qwen35_full_attention_layer_chunk(
              context.value(), weights.value(), cache.value(), layer, current_hidden,
              chunk_tokens, plan.value().hidden_size, plan.value().attention_heads,
              plan.value().kv_heads, plan.value().head_dim,
              full_attention_layers_executed, chunk_start,
              chunk_start,
              plan.value().cache.attention_capacity_tokens, mrope_sections.value(),
              static_cast<float>(bundle.value().model.rope_theta),
              static_cast<float>(bundle.value().model.rms_norm_eps), &profiler,
              &chunk_mrope_positions.value());
            if (!layer_output.is_ok()) {
              return layer_output.status();
            }
            current_hidden = std::move(layer_output.value());
            ++full_attention_layers_executed;
            break;
          }
          case Qwen35LayerKind::mtp:
            return Status::invalid_argument("Qwen3.5 main layer loop reached MTP layer");
        }
        ++main_layers_executed;
        if (prefill_commit_mode == Qwen35PrefillCommitMode::layer) {
          auto commit_status = qwen35_prefill_commit_and_rebegin(
            graph_guard, profiler, "qwen35.prefill.commit.layer",
            qwen35_layer_profile_fields(layer_index, chunk_start, chunk_tokens,
                                        layer.kind == Qwen35LayerKind::linear_attention
                                          ? "linear_attention"
                                          : "full_attention"));
          if (!commit_status.is_ok()) {
            return commit_status;
          }
        }
      }
      if (main_layers_executed != plan.value().main_layers ||
          linear_layers_executed != plan.value().cache.linear_attention_layers ||
          full_attention_layers_executed != plan.value().cache.full_attention_layers) {
        return Status::internal_error("Qwen3.5 Metal chunk layer execution count mismatch");
      }
      if (mtp_state.enabled) {
        auto h_nextn_rows = run_qwen35_output_norm(
          context.value(), current_hidden, chunk_tokens, plan.value().hidden_size,
          static_cast<float>(bundle.value().model.rms_norm_eps), *output_norm,
          debug_dump, "prefill.mtp_process");
        if (!h_nextn_rows.is_ok()) {
          return h_nextn_rows.status();
        }
        auto shifted_h = make_qwen35_mtp_shifted_h_rows(
          context.value(), mtp_state.pending_h, h_nextn_rows.value(),
          chunk_tokens, plan.value().hidden_size);
        if (!shifted_h.is_ok()) {
          return shifted_h.status();
        }
        const auto& mtp_layer = map.value().layers[plan.value().main_layers];
        if (mixed_prefill.has_value()) {
          auto chunk_input_embeddings = make_qwen35_f32_buffer(
            context.value(), chunk_tokens * plan.value().hidden_size,
            "Qwen3.5 mixed MTP input embeddings");
          if (!chunk_input_embeddings.is_ok()) {
            return chunk_input_embeddings.status();
          }
          const auto& mtp_embedding_source =
            mixed_mtp_input_embeddings.has_value()
              ? mixed_mtp_input_embeddings.value()
              : first_hidden.value();
          status = context.value().copy_f32_region(
            mtp_embedding_source, chunk_input_embeddings.value(),
            chunk_start * plan.value().hidden_size, 0,
            chunk_tokens * plan.value().hidden_size);
          if (!status.is_ok()) {
            return status;
          }
          auto mtp_process = run_qwen35_mtp_embeddings(
            context.value(), weights.value(), mtp_state.cache, mtp_layer,
            *output_norm, *output_weight, mrope_sections.value(), chunk_tokens,
            chunk_input_embeddings.value(), shifted_h.value(), chunk_start,
            chunk_start, plan.value(),
            static_cast<float>(bundle.value().model.rope_theta),
            static_cast<float>(bundle.value().model.rms_norm_eps),
            &chunk_mrope_positions.value(), false, false, false, &profiler);
          if (!mtp_process.is_ok()) {
            return mtp_process.status();
          }
        } else {
          std::vector<std::int64_t> chunk_token_ids(
            prompt_tokens.value().begin() + static_cast<std::ptrdiff_t>(chunk_start),
            prompt_tokens.value().begin() +
              static_cast<std::ptrdiff_t>(chunk_start + chunk_tokens));
          auto mtp_process = run_qwen35_mtp_tokens(
            context.value(), weights.value(), mtp_state.cache, mtp_layer,
            *token_embedding, *output_norm, *output_weight, mrope_sections.value(),
            chunk_token_ids, shifted_h.value(), chunk_start, chunk_start, plan.value(),
            static_cast<float>(bundle.value().model.rope_theta),
            static_cast<float>(bundle.value().model.rms_norm_eps),
            false, false, false, &profiler);
          if (!mtp_process.is_ok()) {
            return mtp_process.status();
          }
        }
        status = context.value().copy_f32_region(
          h_nextn_rows.value(), mtp_state.pending_h,
          (chunk_tokens - 1U) * plan.value().hidden_size, 0,
          plan.value().hidden_size);
        if (!status.is_ok()) {
          return status;
        }
      }
      if (prefill_commit_mode == Qwen35PrefillCommitMode::chunk) {
        if (prompt_cache_enabled && chunk_tokens == cache_block_tokens &&
            chunk_start % cache_block_tokens == 0) {
          {
            auto commit_span = profiler.scoped(
              "qwen35.prefill.commit.chunk",
              qwen35_chunk_profile_fields(chunk_start, chunk_tokens));
            auto commit_status = graph_guard.commit();
            if (!commit_status.is_ok()) {
              return commit_status;
            }
          }
          auto cache_commit = qwen35_host_prefix_cache().commit_block(
            context.value(), cache.value(), plan.value(), current_hidden,
            prompt_tokens.value(), chunk_start, chunk_tokens);
          if (!cache_commit.is_ok()) {
            return cache_commit.status();
          }
          if (cache_commit.value().committed) {
            prompt_cache_committed_tokens += cache_block_tokens;
          }
          if (cache_commit.value().evicted) {
            ++prompt_cache_evicted_blocks;
          }
          {
            auto begin_span = profiler.scoped("qwen35.prefill.graph_begin");
            auto begin_status = graph_guard.begin();
            if (!begin_status.is_ok()) {
              return begin_status;
            }
          }
        } else {
          auto commit_status = qwen35_prefill_commit_and_rebegin(
            graph_guard, profiler, "qwen35.prefill.commit.chunk",
            qwen35_chunk_profile_fields(chunk_start, chunk_tokens));
          if (!commit_status.is_ok()) {
            return commit_status;
          }
        }
      }
      last_chunk_tokens = chunk_tokens;
      last_chunk_hidden = std::move(current_hidden);
    }
    if (!last_chunk_hidden.valid() || last_chunk_tokens == 0) {
      return Status::internal_error("Qwen3.5 Metal prefill produced no final hidden state");
    }
    auto final_hidden = make_qwen35_f32_buffer(context.value(), plan.value().hidden_size,
                                               "Qwen3.5 final hidden");
    if (!final_hidden.is_ok()) {
      return final_hidden.status();
    }
    status = context.value().copy_f32_region(
      last_chunk_hidden, final_hidden.value(),
      (last_chunk_tokens - 1U) * plan.value().hidden_size, 0,
      plan.value().hidden_size);
    if (!status.is_ok()) {
      return status;
    }
    status = dump_qwen35_mps_f32(
      context.value(), debug_dump, "prefill.last_hidden",
      {static_cast<std::uint64_t>(plan.value().hidden_size)},
      final_hidden.value(), plan.value().hidden_size);
    if (!status.is_ok()) {
      return status;
    }
    {
      auto logits_span = profiler.scoped("qwen35.prefill.logits");
      auto logits_result = run_qwen35_output_logits(
        context.value(), final_hidden.value(), plan.value().hidden_size,
        static_cast<float>(bundle.value().model.rms_norm_eps), *output_norm,
        *output_weight, debug_dump, "prefill", !sampling.value().do_sample);
      if (!logits_result.is_ok()) {
        return logits_result.status();
      }
      logits_output = std::move(logits_result.value());
    }
    {
      auto prefill_span = profiler.scoped("request.prefill");
      auto commit_span = profiler.scoped("qwen35.prefill.commit");
      status = graph_guard.commit();
    }
    if (!status.is_ok()) {
      return status;
    }
  }

  std::vector<std::int64_t> generated_tokens;
  generated_tokens.reserve(request.max_new_tokens);
  std::vector<CpuGenerationResult::LogitTopEntry> logits_top;
  std::string finish_reason = "length";
  std::size_t next_decode_position = prompt_token_count;
  std::size_t next_decode_mrope_position = prompt_mrope_position_count;

  auto sample_logits_row =
    [&](const Qwen35LogitsOutput& output, std::size_t row,
        bool update_top_entries) -> Result<std::int64_t> {
    if (row >= output.rows) {
      return Status::invalid_argument("Qwen3.5 logits row is out of range");
    }

    std::vector<float> logits_host;
    bool logits_host_loaded = false;
    std::int64_t sampled_token = 0;
    if (!sampling.value().do_sample && row == 0 &&
        output.argmax_output.valid()) {
      auto argmax_token = read_qwen35_argmax_token(
        context.value(), output.argmax_output);
      if (!argmax_token.is_ok()) {
        return argmax_token.status();
      }
      sampled_token = argmax_token.value();
    } else {
      Result<std::vector<float>> logits_host_result{
        std::vector<float>{}};
      if (output.rows == 1) {
        logits_host_result = read_qwen35_logits_host(
          context.value(), output.logits, output.logits_values);
      } else {
        logits_host_result = read_qwen35_logits_row_host(
          context.value(), output.logits, row, output.logits_values);
      }
      if (!logits_host_result.is_ok()) {
        return logits_host_result.status();
      }
      logits_host = std::move(logits_host_result.value());
      logits_host_loaded = true;
      auto row_sampling = sampling.value();
      if (!row_sampling.do_sample) {
        row_sampling.temperature = 0.0;
      }
      auto token = sample_qwen35_next_token(logits_host, sampling_rng,
                                            row_sampling);
      if (!token.is_ok()) {
        return token.status();
      }
      sampled_token = token.value();
    }

    if (update_top_entries && request.logits_top_k > 0) {
      Result<std::vector<CpuGenerationResult::LogitTopEntry>> top_entries{
        std::vector<CpuGenerationResult::LogitTopEntry>{}};
      if (logits_host_loaded) {
        top_entries = make_qwen35_logits_top(tokenizer.value(), logits_host,
                                             request.logits_top_k);
      } else if (output.rows == 1) {
        top_entries = read_qwen35_logits_top(
          context.value(), tokenizer.value(), output.logits,
          output.logits_values, request.logits_top_k);
      } else {
        auto row_logits = read_qwen35_logits_row_host(
          context.value(), output.logits, row, output.logits_values);
        if (!row_logits.is_ok()) {
          return row_logits.status();
        }
        top_entries = make_qwen35_logits_top(tokenizer.value(),
                                             row_logits.value(),
                                             request.logits_top_k);
      }
      if (!top_entries.is_ok()) {
        return top_entries.status();
      }
      logits_top = std::move(top_entries.value());
    }
    return sampled_token;
  };

  auto emit_generated_token = [&](std::int64_t token) -> Status {
    generated_tokens.push_back(token);
    if (request.stream_token) {
      auto token_text = gguf_decode_token_text(tokenizer.value(),
                                               {token}, true);
      if (!token_text.is_ok()) {
        return token_text.status();
      }
      request.stream_token(token_text.value());
    }
    return Status::ok();
  };

  auto update_mtp_pending_h =
    [&](const mps::MpsBuffer& h_nextn, std::size_t row) -> Status {
    if (!mtp_state.enabled) {
      return Status::ok();
    }
    if (row > std::numeric_limits<std::size_t>::max() / plan.value().hidden_size) {
      return Status::invalid_argument("Qwen3.5 MTP pending h row offset overflow");
    }
    return context.value().copy_f32_region(
      h_nextn, mtp_state.pending_h, row * plan.value().hidden_size, 0,
      plan.value().hidden_size);
  };

  auto snapshot_recurrent_state = [&]() -> Status {
    if (!mtp_state.enabled) {
      return Status::ok();
    }
    auto snapshot_status = context.value().copy_buffer_region(
      cache.value().recurrent_r_cache, mtp_state.recurrent_r_snapshot, 0, 0,
      static_cast<std::size_t>(cache.value().recurrent_r_cache_bytes));
    if (!snapshot_status.is_ok()) {
      return snapshot_status;
    }
    return context.value().copy_buffer_region(
      cache.value().recurrent_s_cache, mtp_state.recurrent_s_snapshot, 0, 0,
      static_cast<std::size_t>(cache.value().recurrent_s_cache_bytes));
  };

  auto restore_recurrent_state = [&]() -> Status {
    if (!mtp_state.enabled) {
      return Status::ok();
    }
    auto restore_status = context.value().copy_buffer_region(
      mtp_state.recurrent_r_snapshot, cache.value().recurrent_r_cache, 0, 0,
      static_cast<std::size_t>(cache.value().recurrent_r_cache_bytes));
    if (!restore_status.is_ok()) {
      return restore_status;
    }
    return context.value().copy_buffer_region(
      mtp_state.recurrent_s_snapshot, cache.value().recurrent_s_cache, 0, 0,
      static_cast<std::size_t>(cache.value().recurrent_s_cache_bytes));
  };

  auto read_mtp_draft_token =
    [&](const mps::MpsBuffer& argmax_output,
        const mps::MpsBuffer* probability) -> Result<std::optional<std::int64_t>> {
    auto draft_token = read_qwen35_argmax_token(context.value(), argmax_output);
    if (!draft_token.is_ok()) {
      return draft_token.status();
    }
    if (mtp_state.p_min <= 0.0) {
      return std::optional<std::int64_t>{draft_token.value()};
    }
    if (probability == nullptr || !probability->valid()) {
      return Status::internal_error(
        "Qwen3.5 MTP p_min requires draft top1 probability");
    }
    float top1_probability = 0.0F;
    auto probability_status =
      context.value().copy_from_buffer(*probability, &top1_probability,
                                       sizeof(top1_probability));
    if (!probability_status.is_ok()) {
      return probability_status;
    }
    if (static_cast<double>(top1_probability) < mtp_state.p_min) {
      ++mtp_state.confidence_stops;
      return std::optional<std::int64_t>{};
    }
    return std::optional<std::int64_t>{draft_token.value()};
  };

  auto decode_target_tokens_logits =
    [&](const std::vector<std::int64_t>& token_ids, std::size_t position,
        std::size_t mrope_position, std::string debug_prefix,
        std::size_t step) -> Result<Qwen35LogitsOutput> {
    if (token_ids.empty()) {
      return Status::invalid_argument("Qwen3.5 target decode requires tokens");
    }
    auto decode_span = profiler.scoped(
      "request.decode",
      {ProfileField{"step", std::to_string(step)},
       ProfileField{"tokens", std::to_string(token_ids.size())},
       ProfileField{"position", std::to_string(position)},
       ProfileField{"mrope_position", std::to_string(mrope_position)}});
    MpsGraphGuard decode_guard(context.value());
    auto decode_status = decode_guard.begin();
    if (!decode_status.is_ok()) {
      return decode_status;
    }
    auto decoded = run_qwen35_decode_tokens(
      context.value(), weights.value(), cache.value(), map.value(),
      bundle.value().model, plan.value(), mrope_sections.value(),
      token_ids, position, mrope_position);
    if (!decoded.is_ok()) {
      return decoded.status();
    }
    auto h_nextn = run_qwen35_output_norm(
      context.value(), decoded.value().hidden, decoded.value().rows,
      plan.value().hidden_size,
      static_cast<float>(bundle.value().model.rms_norm_eps), *output_norm,
      debug_dump, debug_prefix);
    if (!h_nextn.is_ok()) {
      return h_nextn.status();
    }
    auto logits_result = run_qwen35_lm_head_logits(
      context.value(), h_nextn.value(), decoded.value().rows,
      plan.value().hidden_size, *output_weight, debug_dump, debug_prefix,
      !sampling.value().do_sample && decoded.value().rows == 1U);
    if (!logits_result.is_ok()) {
      return logits_result.status();
    }
    logits_result.value().h_nextn = std::move(h_nextn.value());
    {
      auto commit_span = profiler.scoped(
        "qwen35.decode.commit",
        {ProfileField{"step", std::to_string(step)}});
      decode_status = decode_guard.commit();
    }
    if (!decode_status.is_ok()) {
      return decode_status;
    }
    return std::move(logits_result.value());
  };

  auto decode_target_token_logits =
    [&](std::int64_t token, std::size_t position, std::size_t mrope_position,
        std::string debug_prefix,
        std::size_t step) -> Result<Qwen35LogitsOutput> {
    const std::vector<std::int64_t> token_ids{token};
    return decode_target_tokens_logits(token_ids, position, mrope_position,
                                       std::move(debug_prefix), step);
  };

  auto draft_mtp_tokens =
    [&](std::int64_t first_token, std::size_t start_position,
        std::size_t start_mrope_position,
        std::size_t budget) -> Result<std::vector<std::int64_t>> {
    std::vector<std::int64_t> drafted_tokens;
    if (!mtp_state.enabled || budget == 0) {
      return drafted_tokens;
    }
    drafted_tokens.reserve(budget);
    const auto& mtp_layer = map.value().layers[plan.value().main_layers];
    const auto* mtp_embed_tokens =
      mtp_layer.mtp.has_embed_tokens
        ? find_metal_weight(weights.value(), mtp_layer.mtp.embed_tokens.name)
        : token_embedding;
    const bool dynamic_embedding =
      mtp_embed_tokens != nullptr &&
      (mtp_embed_tokens->type == 12U || mtp_embed_tokens->type == 13U ||
       mtp_embed_tokens->type == 14U);
    if (dynamic_embedding) {
      auto initial_token_buffer =
        make_qwen35_token_id_buffer(context.value(), {first_token});
      if (!initial_token_buffer.is_ok()) {
        return initial_token_buffer.status();
      }

      auto draft_span = profiler.scoped(
        "qwen35.mtp.draft_chain",
        {ProfileField{"tokens", std::to_string(budget)},
         ProfileField{"position", std::to_string(start_position)},
         ProfileField{"mrope_position", std::to_string(start_mrope_position)}});
      MpsGraphGuard draft_guard(context.value());
      auto draft_status = draft_guard.begin();
      if (!draft_status.is_ok()) {
        return draft_status;
      }
      const mps::MpsBuffer* token_input = &initial_token_buffer.value();
      const mps::MpsBuffer* h_input = &mtp_state.pending_h;
      std::vector<mps::MpsBuffer> draft_argmax_buffers;
      std::vector<mps::MpsBuffer> draft_h_buffers;
      std::vector<mps::MpsBuffer> draft_probability_buffers;
      draft_argmax_buffers.reserve(budget);
      draft_h_buffers.reserve(budget);
      if (mtp_state.p_min > 0.0) {
        draft_probability_buffers.reserve(budget);
      }
      for (std::size_t index = 0; index < budget; ++index) {
        auto mtp_out = run_qwen35_mtp_tokens_from_id_buffer(
          context.value(), weights.value(), mtp_state.cache, mtp_layer,
          *token_embedding, *output_norm, *output_weight, mrope_sections.value(),
          1, *token_input, *h_input, start_position + index,
          start_mrope_position + index, plan.value(),
          static_cast<float>(bundle.value().model.rope_theta),
          static_cast<float>(bundle.value().model.rms_norm_eps),
          true, true, mtp_state.p_min > 0.0, &profiler);
        if (!mtp_out.is_ok()) {
          return mtp_out.status();
        }
        auto output = std::move(mtp_out.value());
        draft_argmax_buffers.push_back(std::move(output.argmax_output));
        if (mtp_state.p_min > 0.0) {
          draft_probability_buffers.push_back(std::move(output.top1_probability));
        }
        draft_h_buffers.push_back(std::move(output.h_nextn));
        token_input = &draft_argmax_buffers.back();
        h_input = &draft_h_buffers.back();
      }
      {
        auto commit_span = profiler.scoped(
          "qwen35.mtp.draft_chain.commit",
          {ProfileField{"tokens", std::to_string(budget)}});
        draft_status = draft_guard.commit();
      }
      if (!draft_status.is_ok()) {
        return draft_status;
      }
      for (std::size_t index = 0; index < draft_argmax_buffers.size(); ++index) {
        const mps::MpsBuffer* probability =
          mtp_state.p_min > 0.0 ? &draft_probability_buffers[index] : nullptr;
        auto draft_token = read_mtp_draft_token(
          draft_argmax_buffers[index], probability);
        if (!draft_token.is_ok()) {
          return draft_token.status();
        }
        if (!draft_token.value().has_value()) {
          break;
        }
        drafted_tokens.push_back(*draft_token.value());
        ++mtp_state.drafted_tokens;
      }
      return drafted_tokens;
    }

    std::int64_t token = first_token;
    const mps::MpsBuffer* h_input = &mtp_state.pending_h;
    mps::MpsBuffer draft_h;
    for (std::size_t index = 0; index < budget; ++index) {
      auto draft_span = profiler.scoped(
        "qwen35.mtp.draft",
        {ProfileField{"index", std::to_string(index)},
         ProfileField{"position", std::to_string(start_position + index)},
         ProfileField{"mrope_position",
                      std::to_string(start_mrope_position + index)}});
      MpsGraphGuard decode_guard(context.value());
      auto draft_status = decode_guard.begin();
      if (!draft_status.is_ok()) {
        return draft_status;
      }
      const std::vector<std::int64_t> token_ids{token};
      auto mtp_out = run_qwen35_mtp_tokens(
        context.value(), weights.value(), mtp_state.cache, mtp_layer,
        *token_embedding, *output_norm, *output_weight, mrope_sections.value(),
        token_ids, *h_input, start_position + index, start_mrope_position + index,
        plan.value(),
        static_cast<float>(bundle.value().model.rope_theta),
        static_cast<float>(bundle.value().model.rms_norm_eps),
        true, true, mtp_state.p_min > 0.0, &profiler);
      if (!mtp_out.is_ok()) {
        return mtp_out.status();
      }
      {
        auto commit_span = profiler.scoped(
          "qwen35.mtp.draft.commit",
          {ProfileField{"index", std::to_string(index)}});
        draft_status = decode_guard.commit();
      }
      if (!draft_status.is_ok()) {
        return draft_status;
      }
      const mps::MpsBuffer* probability =
        mtp_state.p_min > 0.0 ? &mtp_out.value().top1_probability : nullptr;
      auto draft_token = read_mtp_draft_token(
        mtp_out.value().argmax_output, probability);
      if (!draft_token.is_ok()) {
        return draft_token.status();
      }
      if (!draft_token.value().has_value()) {
        break;
      }
      drafted_tokens.push_back(*draft_token.value());
      ++mtp_state.drafted_tokens;
      token = *draft_token.value();
      draft_h = std::move(mtp_out.value().h_nextn);
      h_input = &draft_h;
    }
    return drafted_tokens;
  };

  auto update_mtp_adaptive_budget =
    [&](std::size_t drafted_count, std::size_t accepted_count) {
    if (!mtp_state.enabled || mtp_state.draft_tokens <= 1U ||
        drafted_count == 0) {
      return;
    }
    mtp_state.adaptive_window_drafted += drafted_count;
    mtp_state.adaptive_window_accepted += accepted_count;
    const auto sample_threshold =
      std::max<std::size_t>(16U, mtp_state.adaptive_budget * 8U);
    if (mtp_state.adaptive_window_drafted < sample_threshold) {
      return;
    }

    const auto window_drafted = mtp_state.adaptive_window_drafted;
    const auto window_accepted = mtp_state.adaptive_window_accepted;
    bool changed = false;
    if (mtp_state.adaptive_budget > 1U &&
        window_accepted * 4U < window_drafted * 3U) {
      --mtp_state.adaptive_budget;
      changed = true;
    } else if (mtp_state.adaptive_budget < mtp_state.draft_tokens &&
               window_accepted == window_drafted) {
      ++mtp_state.adaptive_budget;
      changed = true;
    }
    if (changed) {
      ++mtp_state.adaptive_changes;
      mtp_state.adaptive_window_drafted = 0;
      mtp_state.adaptive_window_accepted = 0;
    } else if (mtp_state.adaptive_window_drafted >= sample_threshold * 2U) {
      mtp_state.adaptive_window_drafted = 0;
      mtp_state.adaptive_window_accepted = 0;
    }
  };

  std::size_t logits_output_row = 0;
  if (mtp_state.enabled) {
    std::size_t decode_step = 0;
    bool stop_generation = false;
    while (generated_tokens.size() < request.max_new_tokens && !stop_generation) {
      auto sampled = sample_logits_row(logits_output, logits_output_row, true);
      if (!sampled.is_ok()) {
        return sampled.status();
      }
      logits_output_row = 0;
      const auto next_token = sampled.value();
      if (is_qwen35_eos_token(next_token, bundle.value().model,
                              bundle.value().generation, tokenizer.value())) {
        finish_reason = "stop";
        break;
      }
      status = emit_generated_token(next_token);
      if (!status.is_ok()) {
        return status;
      }
      if (generated_tokens.size() >= request.max_new_tokens) {
        break;
      }

      const auto verified_position = next_decode_position;
      const auto verified_mrope_position = next_decode_mrope_position;
      const auto draft_budget = std::min(
        mtp_state.adaptive_budget,
        request.max_new_tokens - generated_tokens.size());
      auto drafted = draft_mtp_tokens(next_token, verified_position,
                                      verified_mrope_position, draft_budget);
      if (!drafted.is_ok()) {
        return drafted.status();
      }

      std::vector<std::int64_t> verify_tokens;
      verify_tokens.reserve(1U + drafted.value().size());
      verify_tokens.push_back(next_token);
      verify_tokens.insert(verify_tokens.end(), drafted.value().begin(),
                           drafted.value().end());

      status = snapshot_recurrent_state();
      if (!status.is_ok()) {
        return status;
      }

      auto target_logits = decode_target_tokens_logits(
        verify_tokens, verified_position, verified_mrope_position,
        std::string{"decode.mtp.verify."} + std::to_string(decode_step),
        decode_step);
      if (!target_logits.is_ok()) {
        return target_logits.status();
      }

      std::size_t accepted_count = 0;
      for (std::size_t draft_index = 0; draft_index < drafted.value().size();
           ++draft_index) {
        if (generated_tokens.size() >= request.max_new_tokens) {
          break;
        }
        auto target_next = sample_logits_row(target_logits.value(), draft_index, true);
        if (!target_next.is_ok()) {
          return target_next.status();
        }
        ++mtp_state.verify_steps;
        if (draft_index < mtp_state.verified_by_position.size()) {
          ++mtp_state.verified_by_position[draft_index];
        }
        if (is_qwen35_eos_token(target_next.value(), bundle.value().model,
                                bundle.value().generation, tokenizer.value())) {
          finish_reason = "stop";
          stop_generation = true;
          break;
        }
        if (target_next.value() != drafted.value()[draft_index]) {
          break;
        }
        status = emit_generated_token(drafted.value()[draft_index]);
        if (!status.is_ok()) {
          return status;
        }
        ++accepted_count;
        ++mtp_state.accepted_tokens;
        if (draft_index < mtp_state.accepted_by_position.size()) {
          ++mtp_state.accepted_by_position[draft_index];
        }
        if (generated_tokens.size() >= request.max_new_tokens) {
          break;
        }
      }
      update_mtp_adaptive_budget(drafted.value().size(), accepted_count);

      const auto committed_tokens = 1U + accepted_count;
      if (stop_generation) {
        next_decode_position = verified_position + committed_tokens;
        next_decode_mrope_position =
          verified_mrope_position + committed_tokens;
        finish_reason = "stop";
        break;
      }

      if (accepted_count == drafted.value().size()) {
        next_decode_position = verified_position + committed_tokens;
        next_decode_mrope_position =
          verified_mrope_position + committed_tokens;
        ++decode_step;
        status = update_mtp_pending_h(target_logits.value().h_nextn,
                                      committed_tokens - 1U);
        if (!status.is_ok()) {
          return status;
        }
        logits_output_row = committed_tokens - 1U;
        logits_output = std::move(target_logits.value());
      } else {
        status = restore_recurrent_state();
        if (!status.is_ok()) {
          return status;
        }
        std::vector<std::int64_t> commit_tokens;
        commit_tokens.reserve(committed_tokens);
        commit_tokens.push_back(next_token);
        commit_tokens.insert(commit_tokens.end(), drafted.value().begin(),
                             drafted.value().begin() +
                               static_cast<std::ptrdiff_t>(accepted_count));
        auto committed_logits = decode_target_tokens_logits(
          commit_tokens, verified_position, verified_mrope_position,
          std::string{"decode.mtp.commit."} + std::to_string(decode_step),
          decode_step);
        if (!committed_logits.is_ok()) {
          return committed_logits.status();
        }
        next_decode_position = verified_position + committed_tokens;
        next_decode_mrope_position =
          verified_mrope_position + committed_tokens;
        ++decode_step;
        status = update_mtp_pending_h(committed_logits.value().h_nextn,
                                      committed_tokens - 1U);
        if (!status.is_ok()) {
          return status;
        }
        logits_output_row = committed_tokens - 1U;
        logits_output = std::move(committed_logits.value());
      }
    }
  } else {
    for (std::size_t step = 0; step < request.max_new_tokens; ++step) {
      if (step > 0) {
        auto logits_result = decode_target_token_logits(
          generated_tokens.back(), next_decode_position,
          next_decode_mrope_position,
          std::string{"decode."} + std::to_string(step), step);
        if (!logits_result.is_ok()) {
          return logits_result.status();
        }
        ++next_decode_position;
        ++next_decode_mrope_position;
        logits_output = std::move(logits_result.value());
      }

      auto sampled = sample_logits_row(logits_output, 0, true);
      if (!sampled.is_ok()) {
        return sampled.status();
      }
      const auto next_token = sampled.value();
      if (is_qwen35_eos_token(next_token, bundle.value().model,
                              bundle.value().generation, tokenizer.value())) {
        finish_reason = "stop";
        break;
      }

      status = emit_generated_token(next_token);
      if (!status.is_ok()) {
        return status;
      }
    }
  }

  auto decoded = gguf_decode_token_text(tokenizer.value(), generated_tokens, true);
  if (!decoded.is_ok()) {
    return decoded.status();
  }

  CpuGenerationResult result{};
  result.implemented = true;
  result.text = decoded.value();
  result.request_id = profiler.request_id();
  result.finish_reason = finish_reason;
  result.prompt_tokens = prompt_token_count;
  result.generated_tokens = generated_tokens.size();
  result.logits_top = std::move(logits_top);
  result.kv_cache.available = true;
  result.kv_cache.layers = plan.value().cache.full_attention_layers;
  result.kv_cache.kv_heads = plan.value().kv_heads;
  result.kv_cache.head_dim = plan.value().head_dim;
  result.kv_cache.kv_dim = plan.value().cache.kv_dim;
  result.kv_cache.capacity_tokens = plan.value().cache.attention_capacity_tokens;
  result.kv_cache.used_tokens = next_decode_position;
  result.kv_cache.key_bytes = cache.value().key_cache_bytes;
  result.kv_cache.value_bytes = cache.value().value_cache_bytes;
  result.kv_cache.total_bytes = cache.value().key_cache_bytes + cache.value().value_cache_bytes;
  result.mtp.available = mtp_state.available;
  result.mtp.enabled = mtp_state.enabled;
  result.mtp.layers = map.value().mtp_layers;
  result.mtp.draft_tokens = mtp_state.draft_tokens;
  result.mtp.p_min = mtp_state.p_min;
  result.mtp.drafted_tokens = mtp_state.drafted_tokens;
  result.mtp.accepted_tokens = mtp_state.accepted_tokens;
  result.mtp.verify_steps = mtp_state.verify_steps;
  result.mtp.confidence_stops = mtp_state.confidence_stops;
  result.mtp.adaptive_budget = mtp_state.adaptive_budget;
  result.mtp.adaptive_changes = mtp_state.adaptive_changes;
  result.mtp.verified_by_position = mtp_state.verified_by_position;
  result.mtp.accepted_by_position = mtp_state.accepted_by_position;
  result.mtp.disabled_reason = mtp_state.disabled_reason;
  profiler.set_metadata("qwen35_mtp_drafted_tokens", mtp_state.drafted_tokens);
  profiler.set_metadata("qwen35_mtp_accepted_tokens", mtp_state.accepted_tokens);
  profiler.set_metadata("qwen35_mtp_verify_steps", mtp_state.verify_steps);
  profiler.set_metadata("qwen35_mtp_confidence_stops",
                        mtp_state.confidence_stops);
  profiler.set_metadata("qwen35_mtp_adaptive_budget",
                        mtp_state.adaptive_budget);
  profiler.set_metadata("qwen35_mtp_adaptive_changes",
                        mtp_state.adaptive_changes);
  if (!mtp_state.verified_by_position.empty()) {
    profiler.set_metadata("qwen35_mtp_verified_by_position",
                          qwen35_join_size_values(mtp_state.verified_by_position));
  }
  if (!mtp_state.accepted_by_position.empty()) {
    profiler.set_metadata("qwen35_mtp_accepted_by_position",
                          qwen35_join_size_values(mtp_state.accepted_by_position));
  }
  if (prompt_cache_enabled) {
    const auto cache_stats = qwen35_host_prefix_cache().stats();
    result.prompt_cache.enabled = true;
    result.prompt_cache.block_tokens = cache_stats.block_tokens;
    result.prompt_cache.capacity_blocks = cache_stats.capacity_blocks;
    result.prompt_cache.stored_blocks = cache_stats.stored_blocks;
    result.prompt_cache.hit_tokens = prompt_cache_lookup.hit_tokens;
    result.prompt_cache.miss_tokens = prompt_token_count - prompt_cache_lookup.hit_tokens;
    result.prompt_cache.committed_tokens = prompt_cache_committed_tokens;
    result.prompt_cache.evicted_blocks = prompt_cache_evicted_blocks;
    profiler.set_metadata("prompt_cache_committed_tokens",
                          prompt_cache_committed_tokens);
    profiler.set_metadata("prompt_cache_evicted_blocks",
                          prompt_cache_evicted_blocks);
    profiler.set_metadata("prompt_cache_stored_blocks",
                          cache_stats.stored_blocks);
  }
  profiler.set_metadata("generated_tokens", result.generated_tokens);
  profiler.set_status("ok");
  const auto artifacts = profiler.write_artifacts();
  if (!artifacts.output_dir.empty()) {
    result.profile_dir = artifacts.output_dir;
  }
  return result;
}

}  // namespace toyllm
