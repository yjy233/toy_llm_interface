cmake_minimum_required(VERSION 3.24)

project(
  kraken-infer
  VERSION 0.1.0
  DESCRIPTION "A small from-scratch LLM inference runtime for macOS MPS experiments"
  LANGUAGES CXX
)

if(APPLE)
  enable_language(OBJCXX)
endif()

if(DEFINED TOYLLM_ENABLE_MPS AND NOT DEFINED KRAKEN_INFER_ENABLE_MPS)
  set(KRAKEN_INFER_ENABLE_MPS "${TOYLLM_ENABLE_MPS}" CACHE BOOL
      "Enable Metal Performance Shaders backend on macOS" FORCE)
endif()
if(DEFINED TOYLLM_BUILD_TESTS AND NOT DEFINED KRAKEN_INFER_BUILD_TESTS)
  set(KRAKEN_INFER_BUILD_TESTS "${TOYLLM_BUILD_TESTS}" CACHE BOOL "Build smoke tests" FORCE)
endif()

option(KRAKEN_INFER_ENABLE_MPS "Enable Metal Performance Shaders backend on macOS" ON)
option(KRAKEN_INFER_BUILD_TESTS "Build smoke tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(ProjectWarnings)

add_library(kraken_infer_core)
add_library(kraken_infer::core ALIAS kraken_infer_core)

target_sources(
  kraken_infer_core
  PRIVATE
    src/core/status.cpp
    src/model/model_config.cpp
    src/runtime/cpu/debug_dump.cpp
    src/runtime/gguf_reader.cpp
    src/runtime/cpu_inference.cpp
    src/runtime/gguf_tokenizer.cpp
    src/runtime/openai_gateway.cpp
    src/runtime/profiling.cpp
    src/runtime/qwen35_prefix_cache.cpp
    src/runtime/qwen35_runtime.cpp
    src/runtime/qwen35_multimodal.cpp
    src/runtime/qwen35_vl_adapter.cpp
    src/runtime/qwen35_weight_map.cpp
    src/runtime/reasoning_parser.cpp
)

target_include_directories(
  kraken_infer_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(kraken_infer_core PUBLIC cxx_std_20)
kraken_infer_set_project_warnings(kraken_infer_core)

if(APPLE)
  target_sources(kraken_infer_core PRIVATE src/runtime/qwen35_image_decode.mm)
  target_compile_definitions(kraken_infer_core PUBLIC KRAKEN_INFER_ENABLE_APPLE_IMAGEIO=1)
  target_link_libraries(
    kraken_infer_core
    PUBLIC
      "-framework Foundation"
      "-framework CoreGraphics"
      "-framework ImageIO"
  )
else()
  target_sources(kraken_infer_core PRIVATE src/runtime/qwen35_image_decode_stub.cpp)
  target_compile_definitions(kraken_infer_core PUBLIC KRAKEN_INFER_ENABLE_APPLE_IMAGEIO=0)
endif()

if(APPLE AND KRAKEN_INFER_ENABLE_MPS)
  target_sources(kraken_infer_core PRIVATE src/backends/mps/mps_backend.mm)
  target_compile_definitions(kraken_infer_core PUBLIC KRAKEN_INFER_ENABLE_MPS=1)
  target_link_libraries(
    kraken_infer_core
    PUBLIC
      "-framework Foundation"
      "-framework Metal"
      "-framework MetalPerformanceShaders"
  )
else()
  target_sources(kraken_infer_core PRIVATE src/backends/mps/mps_backend_stub.cpp)
  target_compile_definitions(kraken_infer_core PUBLIC KRAKEN_INFER_ENABLE_MPS=0)
endif()

add_executable(kraken-infer apps/kraken_infer_main.cpp)
target_link_libraries(kraken-infer PRIVATE kraken_infer::core)
kraken_infer_set_project_warnings(kraken-infer)

if(KRAKEN_INFER_BUILD_TESTS)
  enable_testing()
  add_executable(kraken-infer-smoke-test tests/smoke_test.cpp)
  target_link_libraries(kraken-infer-smoke-test PRIVATE kraken_infer::core)
  kraken_infer_set_project_warnings(kraken-infer-smoke-test)
  add_test(NAME kraken-infer-smoke-test COMMAND kraken-infer-smoke-test)
  set_tests_properties(kraken-infer-smoke-test PROPERTIES
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()
