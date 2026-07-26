#pragma once

#include "toyllm/core/device.hpp"
#include "toyllm/core/status.hpp"
#include "toyllm/runtime/chat_message.hpp"
#include "toyllm/runtime/profiling.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

struct CpuSamplingConfig {
  bool do_sample{false};
  bool temperature_set{false};
  bool top_k_set{false};
  bool top_p_set{false};
  bool seed_set{false};
  double temperature{1.0};
  std::size_t top_k{0};
  double top_p{1.0};
  std::uint64_t seed{0};
};

struct CpuGenerationRequest {
  std::filesystem::path model_dir{
    "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf"};
  std::string prompt;
  std::size_t max_new_tokens{16};
  std::size_t prefill_chunk_tokens{0};
  std::size_t logits_top_k{0};
  bool parse_special_prompt{false};
  bool enable_thinking{false};
  bool enable_mtp{true};
  std::size_t mtp_draft_tokens{3};
  double mtp_p_min{0.30};
  bool cache_prompt{false};
  std::size_t cache_reuse_min_tokens{0};
  std::size_t cache_block_tokens{0};
  std::size_t cache_capacity_blocks{0};
  std::vector<ChatMessage> messages;
  std::filesystem::path mmproj_path;
  std::filesystem::path debug_dump_dir;
  bool verify_kv_cache{false};
  Device compute_device{Device::cpu()};
  CpuSamplingConfig sampling;
  std::function<void(std::string_view)> stream_token;
  ObservabilityConfig observability;
};

struct CpuKvCacheReport {
  bool available{false};
  std::size_t layers{0};
  std::size_t kv_heads{0};
  std::size_t head_dim{0};
  std::size_t kv_dim{0};
  std::size_t capacity_tokens{0};
  std::size_t used_tokens{0};
  std::uint64_t key_bytes{0};
  std::uint64_t value_bytes{0};
  std::uint64_t total_bytes{0};
};

struct CpuPromptCacheReport {
  bool enabled{false};
  std::size_t block_tokens{0};
  std::size_t capacity_blocks{0};
  std::size_t stored_blocks{0};
  std::size_t hit_tokens{0};
  std::size_t miss_tokens{0};
  std::size_t committed_tokens{0};
  std::size_t evicted_blocks{0};
};

struct CpuMtpReport {
  bool available{false};
  bool enabled{false};
  std::size_t layers{0};
  std::size_t draft_tokens{0};
  double p_min{0.0};
  std::size_t drafted_tokens{0};
  std::size_t accepted_tokens{0};
  std::size_t verify_steps{0};
  std::size_t confidence_stops{0};
  std::size_t adaptive_budget{0};
  std::size_t adaptive_changes{0};
  std::vector<std::size_t> verified_by_position;
  std::vector<std::size_t> accepted_by_position;
  std::string disabled_reason;
};

struct CpuGenerationResult {
  bool implemented{false};
  std::string text;
  std::string request_id;
  std::string finish_reason{"stop"};
  std::size_t prompt_tokens{0};
  std::size_t generated_tokens{0};
  std::filesystem::path profile_dir;
  std::vector<std::string> missing_dependencies;
  CpuKvCacheReport kv_cache;
  CpuPromptCacheReport prompt_cache;
  CpuMtpReport mtp;
  bool kv_cache_verified{false};
  struct LogitTopEntry {
    std::int64_t token_id{0};
    float logit{0.0F};
    std::string text;
  };
  std::vector<LogitTopEntry> logits_top;
};

[[nodiscard]] Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request);
[[nodiscard]] std::string format_cpu_generation_result(const CpuGenerationResult& result);
[[nodiscard]] Result<std::string> format_weight_summary(const std::filesystem::path& model_dir);

}  // namespace toyllm
