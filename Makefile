CXX ?= clang++
BUILD_DIR ?= build/manual
MODEL ?= models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf
PROMPT ?= hello
CHAT_TOKENS ?= 16
QWEN35_MODEL ?= models/qwen3.5-0.8b
QWEN35_MTP_MODEL ?= models/qwen3.5-0.8b-mtp/Qwen3.5-0.8B-Q4_K_M.gguf
QWEN35_MMPROJ ?= models/qwen3.5-0.8b/mmproj-Qwen3.5-0.8B-BF16.gguf
QWEN35_IMAGE ?=
BINARY := kraken-infer
SMOKE_TEST := kraken-infer-smoke-test

COMMON_FLAGS := -std=c++20 -Wall -Wextra -Wpedantic -Wconversion -Wshadow -Iinclude
DEBUG_FLAGS := -O0 -g
RELEASE_FLAGS := -O3 -DNDEBUG
MPS_FLAGS := -DKRAKEN_INFER_ENABLE_MPS=1
IMAGEIO_FLAGS := -DKRAKEN_INFER_ENABLE_APPLE_IMAGEIO=1
APPLE_FRAMEWORKS := -framework Foundation -framework CoreGraphics -framework ImageIO -framework Metal -framework MetalPerformanceShaders

CORE_SRCS := \
	src/core/status.cpp \
	src/model/model_config.cpp \
	src/runtime/cpu/debug_dump.cpp \
	src/runtime/cpu_inference.cpp \
	src/runtime/gguf_reader.cpp \
	src/runtime/gguf_tokenizer.cpp \
	src/runtime/openai_gateway.cpp \
	src/runtime/profiling.cpp \
	src/runtime/qwen35_image_decode.mm \
	src/runtime/qwen35_prefix_cache.cpp \
	src/runtime/qwen35_runtime.cpp \
	src/runtime/qwen35_multimodal.cpp \
	src/runtime/qwen35_vl_adapter.cpp \
	src/runtime/qwen35_weight_map.cpp \
	src/runtime/reasoning_parser.cpp \
	src/backends/mps/mps_backend.mm

.PHONY: all debug release test qwen35-vl-test qwen35-vl-mtp-test cli inspect weights doctor infer run chat serve compare-transformers mps-info clean

all: debug

debug: $(BUILD_DIR)/$(BINARY) $(BUILD_DIR)/$(SMOKE_TEST)

release:
	$(MAKE) BUILD_DIR=build/release EXTRA_FLAGS="$(RELEASE_FLAGS)" debug

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/$(BINARY): $(CORE_SRCS) apps/kraken_infer_main.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MPS_FLAGS) $(IMAGEIO_FLAGS) $^ $(APPLE_FRAMEWORKS) -o $@

$(BUILD_DIR)/$(SMOKE_TEST): $(CORE_SRCS) tests/smoke_test.cpp | $(BUILD_DIR)
	$(CXX) $(COMMON_FLAGS) $(DEBUG_FLAGS) $(EXTRA_FLAGS) $(MPS_FLAGS) $(IMAGEIO_FLAGS) $^ $(APPLE_FRAMEWORKS) -o $@

test: $(BUILD_DIR)/$(SMOKE_TEST)
	./$(BUILD_DIR)/$(SMOKE_TEST)

qwen35-vl-test: $(BUILD_DIR)/$(BINARY)
	python3 scripts/test_qwen35_vl_gateway.py \
		--binary ./$(BUILD_DIR)/$(BINARY) \
		--model $(QWEN35_MODEL) \
		--mmproj $(QWEN35_MMPROJ) $(if $(QWEN35_IMAGE),--image $(QWEN35_IMAGE),) \
		--timeout 180

qwen35-vl-mtp-test: $(BUILD_DIR)/$(BINARY)
	python3 scripts/test_qwen35_vl_gateway.py \
		--binary ./$(BUILD_DIR)/$(BINARY) \
		--model $(QWEN35_MTP_MODEL) \
		--mmproj $(QWEN35_MMPROJ) $(if $(QWEN35_IMAGE),--image $(QWEN35_IMAGE),) \
		--timeout 180 \
		--expect-mtp-disabled-reason multimodal_prompt_not_supported_with_mtp

cli: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) help

inspect: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) inspect $(MODEL)

weights: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) weights $(MODEL)

doctor: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) doctor $(MODEL)

infer: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) infer --model $(MODEL) --prompt "$(PROMPT)"

run: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) run --model $(MODEL) --prompt "$(PROMPT)"

chat: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) chat --model $(MODEL) --max-new-tokens $(CHAT_TOKENS)

serve: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) serve --model $(MODEL)

mps-info: $(BUILD_DIR)/$(BINARY)
	./$(BUILD_DIR)/$(BINARY) mps

clean:
	rm -rf build
