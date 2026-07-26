#pragma once

#include "toyllm/core/device.hpp"
#include "toyllm/core/status.hpp"
#include "toyllm/runtime/profiling.hpp"

#include <cstddef>
#include <filesystem>
#include <string>

namespace toyllm {

struct OpenAIGatewayConfig {
  std::string host{"127.0.0.1"};
  int port{8080};
  std::filesystem::path model_dir{
    "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf"};
  std::filesystem::path mmproj_path;
  std::string model_id{"qwen3.5-0.8b"};
  Device compute_device{Device::cpu()};
  std::size_t default_max_tokens{16};
  std::size_t prefill_chunk_tokens{0};
  std::size_t context_size{0};
  std::size_t parallel_slots{1};
  bool cache_prompt{true};
  std::size_t cache_reuse_min_tokens{0};
  std::size_t cache_block_tokens{0};
  std::size_t cache_capacity_blocks{16};
  bool enable_mtp{true};
  std::size_t mtp_draft_tokens{3};
  double mtp_p_min{0.30};
  ObservabilityConfig observability;
};

[[nodiscard]] Status serve_openai_gateway(const OpenAIGatewayConfig& config);

}  // namespace toyllm
