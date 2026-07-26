#include "toyllm/backends/mps/mps_backend.hpp"
#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/cpu_inference.hpp"
#include "toyllm/runtime/gguf_tokenizer.hpp"
#include "toyllm/runtime/qwen35_multimodal.hpp"
#include "toyllm/runtime/qwen35_prefix_cache.hpp"
#include "toyllm/runtime/qwen35_runtime.hpp"
#include "toyllm/runtime/reasoning_parser.hpp"

#include <cassert>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

void test_mps_backend() {
  const auto info = toyllm::mps::query_backend();
  if (info.available) {
    assert(!info.device_name.empty());
  }
  const auto status = toyllm::mps::run_operator_smoke_test();
  if (info.available && info.compute_ready) {
    assert(status.is_ok());
  } else {
    assert(!status.is_ok());
  }
}

void append_reasoning_deltas(
  const std::vector<toyllm::ReasoningDelta>& deltas,
  std::string& reasoning, std::string& content) {
  for (const auto& delta : deltas) {
    if (delta.kind == toyllm::ReasoningDelta::Kind::reasoning) {
      reasoning += delta.text;
    } else {
      content += delta.text;
    }
  }
}

void test_reasoning_parser() {
  const auto parsed = toyllm::split_reasoning_content(
    "<think>\nprivate\n</think>\n\npublic", false,
    toyllm::ReasoningFormat::deepseek);
  assert(parsed.reasoning_content == "private");
  assert(parsed.content == "public");

  toyllm::ReasoningStreamParser stream{
    true, toyllm::ReasoningFormat::deepseek};
  std::string reasoning;
  std::string content;
  append_reasoning_deltas(stream.push("pri"), reasoning, content);
  append_reasoning_deltas(
    stream.push("vate\n</think>\n\npublic"), reasoning, content);
  append_reasoning_deltas(stream.finish(), reasoning, content);
  assert(reasoning == "private");
  assert(content == "public");
}

toyllm::GgufTokenizer make_test_tokenizer() {
  toyllm::GgufTokenizer tokenizer;
  tokenizer.model = "gpt2";
  tokenizer.pre = "qwen35";
  tokenizer.tokens = {"<|im_start|>", "<|im_end|>", "h", "i"};
  tokenizer.token_types = {
    static_cast<std::int64_t>(toyllm::GgufTokenType::control),
    static_cast<std::int64_t>(toyllm::GgufTokenType::control),
    static_cast<std::int64_t>(toyllm::GgufTokenType::normal),
    static_cast<std::int64_t>(toyllm::GgufTokenType::normal),
  };
  for (std::size_t id = 0; id < tokenizer.tokens.size(); ++id) {
    tokenizer.token_to_id.emplace(
      tokenizer.tokens[id], static_cast<std::int64_t>(id));
  }
  return tokenizer;
}

void test_qwen35_tokenizer_and_chat_template() {
  const auto tokenizer = make_test_tokenizer();
  const auto encoded =
    toyllm::gguf_encode_text(tokenizer, "hi", false, true);
  assert(encoded.is_ok());
  assert((encoded.value() == std::vector<std::int64_t>{2, 3}));

  const auto special = toyllm::gguf_encode_text(
    tokenizer, "<|im_start|>hi<|im_end|>", false, true);
  assert(special.is_ok());
  assert((special.value() ==
          std::vector<std::int64_t>{0, 2, 3, 1}));

  const std::vector<toyllm::ChatMessage> messages{
    toyllm::ChatMessage{"user", "hi"},
  };
  const auto prompt = toyllm::format_qwen35_chat_prompt(
    tokenizer, messages, true, false);
  assert(prompt.is_ok());
  assert(prompt.value() ==
         "<|im_start|>user\nhi<|im_end|>\n"
         "<|im_start|>assistant\n<think>\n\n</think>\n\n");
}

void test_qwen35_image_helpers() {
  const toyllm::ChatMessage text_message{"user", "hello"};
  assert(!toyllm::chat_message_has_image_content(text_message));

  toyllm::ChatMessage image_message{"user", "describe"};
  image_message.content_parts.push_back(toyllm::ChatContentPart{
    toyllm::ChatContentPartKind::image_url,
    {},
    "data:image/png;base64,SGVsbG8=",
    "auto",
  });
  assert(toyllm::chat_message_has_image_content(image_message));
  assert(toyllm::chat_messages_have_image_content(
    {text_message, image_message}));

  const auto parsed = toyllm::parse_qwen35_image_data_url(
    "data:image/png;base64,SGVsbG8=");
  assert(parsed.is_ok());
  assert(parsed.value().mime_type == "image/png");
  assert((parsed.value().bytes ==
          std::vector<std::uint8_t>{'H', 'e', 'l', 'l', 'o'}));

  const std::vector<std::uint8_t> png_header{
    0x89U, 0x50U, 0x4EU, 0x47U, 0x0DU, 0x0AU, 0x1AU, 0x0AU,
    0x00U, 0x00U, 0x00U, 0x0DU, 0x49U, 0x48U, 0x44U, 0x52U,
    0x00U, 0x00U, 0x00U, 0x02U, 0x00U, 0x00U, 0x00U, 0x03U,
  };
  const auto dimensions =
    toyllm::infer_qwen35_image_dimensions("image/png", png_header);
  assert(dimensions.is_ok());
  assert(dimensions.value().width == 2);
  assert(dimensions.value().height == 3);
}

void test_qwen35_prefix_cache() {
  toyllm::Qwen35PrefixCacheIndex cache{
    toyllm::Qwen35PrefixCacheConfig{
      true,
      2,
      2,
      0,
    }};
  const std::vector<std::int64_t> tokens{1, 2, 3, 4, 5};
  assert(cache.lookup(tokens).hit_tokens == 0);
  assert(cache.commit_block(tokens, 0).inserted);
  assert(cache.commit_block(tokens, 2).inserted);
  const auto hit = cache.lookup(tokens);
  assert(hit.hit_tokens == 4);
  assert(hit.block_hashes.size() == 2);

  cache.clear();
  assert(cache.lookup(tokens).hit_tokens == 0);
  assert(cache.stats().stored_blocks == 0);
}

void test_qwen35_execution_plan() {
  toyllm::ModelConfig config;
  config.gguf = true;
  config.architecture = "qwen35";
  config.hidden_size = 1024;
  config.main_layer_count = 24;
  config.total_layer_count = 24;
  config.head_dim = 256;
  config.num_attention_heads = 8;
  config.num_key_value_heads = 2;
  config.linear_num_key_heads = 16;
  config.linear_num_value_heads = 16;
  config.linear_key_head_dim = 128;
  config.linear_inner_size = 2048;
  config.linear_conv_kernel_dim = 4;

  toyllm::Qwen35WeightMap weights;
  for (std::int64_t index = 0; index < 24; ++index) {
    toyllm::Qwen35LayerBindings layer;
    layer.index = index;
    layer.kind = ((index + 1) % 4 == 0)
                   ? toyllm::Qwen35LayerKind::full_attention
                   : toyllm::Qwen35LayerKind::linear_attention;
    if (layer.kind == toyllm::Qwen35LayerKind::full_attention) {
      ++weights.full_attention_layers;
    } else {
      ++weights.linear_attention_layers;
    }
    weights.layers.push_back(layer);
  }

  toyllm::Qwen35RuntimeOptions options;
  options.decode_tokens = 16;
  options.prefill_chunk_tokens = 32;
  auto plan =
    toyllm::build_qwen35_execution_plan(config, weights, 100, options);
  assert(plan.is_ok());
  assert(plan.value().prefill.chunk_count == 4);
  assert(plan.value().prefill.final_chunk_tokens == 4);
  assert(plan.value().cache.full_attention_layers == 6);
  assert(plan.value().cache.linear_attention_layers == 18);
  assert(plan.value().cache.attention_capacity_tokens == 116);
}

void test_qwen35_bench_formatters() {
  toyllm::Qwen35MatmulBenchResult matmul;
  matmul.gguf_path = "qwen35.gguf";
  matmul.tensor_name = "blk.0.attn_qkv.weight";
  matmul.type_name = "Q5_K";
  matmul.dispatch = "mul_mm_simd_64x32";
  matmul.rows = 6144;
  matmul.cols = 1024;
  const auto matmul_text =
    toyllm::format_qwen35_matmul_bench_result(matmul);
  assert(matmul_text.find("Qwen3.5 Metal matmul bench: ok") !=
         std::string::npos);

  toyllm::Qwen35GdnBenchResult gdn;
  gdn.dispatch = "qwen35_simdgroup_4rows";
  const auto gdn_text = toyllm::format_qwen35_gdn_bench_result(gdn);
  assert(gdn_text.find("Qwen3.5 Metal GDN bench: ok") !=
         std::string::npos);

  toyllm::Qwen35AttentionBenchResult attention;
  attention.dispatch = "flash256_f16_kv+tail";
  attention.f16_kv = true;
  const auto attention_text =
    toyllm::format_qwen35_attention_bench_result(attention);
  assert(attention_text.find("Qwen3.5 Metal attention bench: ok") !=
         std::string::npos);
}

void test_real_qwen35_gguf_when_present() {
  const std::filesystem::path model{
    "models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf"};
  if (!std::filesystem::exists(model)) {
    return;
  }
  const auto bundle = toyllm::load_model_bundle(model);
  assert(bundle.is_ok());
  assert(bundle.value().model.architecture == "qwen35");
  assert(bundle.value().model.hidden_size == 1024);
  assert(bundle.value().tokenizer.available);

  const auto summary = toyllm::format_weight_summary(model);
  assert(summary.is_ok());
  assert(summary.value().find("Qwen3.5 GGUF mapping: ok") !=
         std::string::npos);
}

}  // namespace

int main() {
  test_mps_backend();
  test_reasoning_parser();
  test_qwen35_tokenizer_and_chat_template();
  test_qwen35_image_helpers();
  test_qwen35_prefix_cache();
  test_qwen35_execution_plan();
  test_qwen35_bench_formatters();
  test_real_qwen35_gguf_when_present();
  std::cout << "Qwen3.5 smoke tests: ok\n";
  return 0;
}
