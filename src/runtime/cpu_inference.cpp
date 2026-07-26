#include "toyllm/runtime/cpu_inference.hpp"

#include "toyllm/model/model_config.hpp"
#include "toyllm/runtime/gguf_reader.hpp"
#include "toyllm/runtime/qwen35_multimodal.hpp"
#include "toyllm/runtime/qwen35_runtime.hpp"
#include "toyllm/runtime/qwen35_weight_map.hpp"

#include <algorithm>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace toyllm {

namespace {

std::string escape_debug_text(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        escaped += "\\\\";
        break;
      case '"':
        escaped += "\\\"";
        break;
      case '\n':
        escaped += "\\n";
        break;
      case '\r':
        escaped += "\\r";
        break;
      case '\t':
        escaped += "\\t";
        break;
      default:
        escaped.push_back(character);
        break;
    }
  }
  return escaped;
}

std::string format_mtp_position_counts(const CpuMtpReport& report) {
  if (report.verified_by_position.empty() && report.accepted_by_position.empty()) {
    return {};
  }
  const auto positions = std::max(report.verified_by_position.size(),
                                  report.accepted_by_position.size());
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < positions; ++index) {
    if (index > 0) {
      output << ", ";
    }
    const auto accepted = index < report.accepted_by_position.size()
                            ? report.accepted_by_position[index]
                            : std::size_t{0};
    const auto verified = index < report.verified_by_position.size()
                            ? report.verified_by_position[index]
                            : std::size_t{0};
    output << accepted << '/' << verified;
  }
  output << ']';
  return output.str();
}

std::string format_tensor_shape(const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0; index < shape.size(); ++index) {
    if (index > 0) {
      output << ", ";
    }
    output << shape[index];
  }
  output << ']';
  return output.str();
}

}  // namespace

Result<CpuGenerationResult> generate_cpu(const CpuGenerationRequest& request) {
  auto gguf_path = resolve_gguf_model_path(request.model_dir);
  if (!gguf_path.is_ok()) {
    return Status::invalid_argument(
      "Qwen3.5 inference requires a GGUF model: " + gguf_path.status().message());
  }
  return generate_qwen35_metal(request);
}

std::string format_cpu_generation_result(const CpuGenerationResult& result) {
  std::ostringstream output;
  output << result.text << '\n';
  if (!result.logits_top.empty()) {
    output << "logits_top:\n";
    for (std::size_t index = 0; index < result.logits_top.size(); ++index) {
      const auto& entry = result.logits_top[index];
      output << index << " token_id=" << entry.token_id
             << " logit=" << entry.logit
             << " text=\"" << escape_debug_text(entry.text) << "\"\n";
    }
  }
  if (result.mtp.available) {
    output << "mtp: " << (result.mtp.enabled ? "enabled" : "disabled")
           << ", layers=" << result.mtp.layers
           << ", draft_tokens=" << result.mtp.draft_tokens
           << ", p_min=" << result.mtp.p_min
           << ", drafted=" << result.mtp.drafted_tokens
           << ", accepted=" << result.mtp.accepted_tokens
           << ", verify_steps=" << result.mtp.verify_steps
           << ", confidence_stops=" << result.mtp.confidence_stops;
    if (result.mtp.enabled &&
        (result.mtp.adaptive_budget != result.mtp.draft_tokens ||
         result.mtp.adaptive_changes != 0U)) {
      output << ", adaptive_budget=" << result.mtp.adaptive_budget
             << ", adaptive_changes=" << result.mtp.adaptive_changes;
    }
    const auto position_counts = format_mtp_position_counts(result.mtp);
    if (!position_counts.empty()) {
      output << ", accepted_by_position=" << position_counts;
    }
    if (!result.mtp.enabled && !result.mtp.disabled_reason.empty()) {
      output << ", reason=" << result.mtp.disabled_reason;
    }
    output << '\n';
  }
  return output.str();
}

Result<std::string> format_weight_summary(const std::filesystem::path& model_dir) {
  auto bundle = load_model_bundle(model_dir);
  if (!bundle.is_ok()) {
    return bundle.status();
  }
  auto gguf = read_gguf_file(bundle.value().model_file);
  if (!gguf.is_ok()) {
    return gguf.status();
  }

  std::vector<const GgufTensorInfo*> tensors;
  tensors.reserve(gguf.value().tensors.size());
  std::uint64_t tensor_bytes = 0;
  for (const auto& tensor : gguf.value().tensors) {
    if (tensor.byte_size > std::numeric_limits<std::uint64_t>::max() - tensor_bytes) {
      return Status::invalid_argument("GGUF tensor byte total overflows uint64");
    }
    tensor_bytes += tensor.byte_size;
    tensors.push_back(&tensor);
  }
  std::sort(tensors.begin(), tensors.end(),
            [](const GgufTensorInfo* lhs, const GgufTensorInfo* rhs) {
              return lhs->name < rhs->name;
            });

  std::ostringstream output;
  output << "Weights: ok\n";
  output << "Format: GGUF v" << gguf.value().version << '\n';
  output << "File: " << bundle.value().model_file.string() << '\n';
  output << "File size: " << gguf.value().file_size << " bytes\n";
  output << "Tensor data bytes: " << tensor_bytes << '\n';
  output << "Tensor count: " << gguf.value().tensor_count << '\n';
  output << "Metadata entries: " << gguf.value().metadata_count << '\n';
  output << "Alignment: " << gguf.value().alignment << '\n';
  output << "GGML types:\n";
  for (const auto& [name, count] : gguf_tensor_type_counts(gguf.value())) {
    output << "- " << name << ": " << count << '\n';
  }
  output << "First tensors:\n";
  const auto preview_count = std::min<std::size_t>(tensors.size(), 12U);
  for (std::size_t index = 0; index < preview_count; ++index) {
    const auto& tensor = *tensors[index];
    output << "- " << tensor.name << ' ' << format_tensor_shape(tensor.shape)
           << ' ' << ggml_type_name(tensor.type) << '\n';
  }

  if (bundle.value().model.architecture == "qwen35") {
    auto map = build_qwen35_weight_map(bundle.value().model, gguf.value());
    if (!map.is_ok()) {
      return map.status();
    }
    output << format_qwen35_weight_map_summary(map.value());
    output << "Qwen3.5 GGUF mapping: ok\n";
  } else if (bundle.value().model.architecture == "clip") {
    auto plan = plan_qwen35_vision_graph(bundle.value().model_file);
    if (!plan.is_ok()) {
      return plan.status();
    }
    output << format_qwen35_vision_graph_plan(plan.value());
    output << "Qwen3.5 VL native vision graph plan: ok\n";
  } else {
    return Status::invalid_argument(
      "unsupported Qwen3.5 GGUF architecture: " + bundle.value().model.architecture);
  }
  output << "Validation: ok\n";
  return output.str();
}

}  // namespace toyllm
