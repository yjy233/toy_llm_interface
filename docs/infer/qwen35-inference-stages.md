# Qwen3.5 推理阶段：从 ChatMessage 到输出文本

本文说明本项目中 Qwen3.5 请求从结构化聊天消息进入模型，再逐 token
生成输出的完整流程，并区分四个容易混淆的概念：

- Chat templating（聊天模板格式化）
- Tokenization（分词和编码）
- Prefill（提示词预填充）
- Decode（自回归解码）

## 总体流程

```text
ChatMessage / prompt
        │
        │ 1. Chat templating / Prompt serialization
        ▼
Qwen3.5 prompt 字符串
        │
        │ 2. Tokenization
        ▼
Prompt token IDs
        │
        │ 3. Prefill
        ▼
模型状态 + 首个 logits
        │
        │ 4. Sampling
        ▼
第一个生成 token
        │
        │ 5. Decode + Sampling（循环）
        ▼
后续生成 token IDs
        │
        │ 6. Detokenization
        ▼
输出文本
```

前两个阶段属于输入预处理，Prefill 和 Decode 才属于神经网络推理。

## 1. Chat templating

对应函数：

```cpp
format_qwen35_chat_prompt(
  tokenizer,
  messages,
  add_generation_prompt,
  enable_thinking);
```

专业术语可以称为：

- Chat templating
- Prompt formatting
- Prompt serialization

它将结构化消息：

```json
[
  {
    "role": "user",
    "content": "你好"
  }
]
```

转换为 Qwen3.5 能识别的 prompt：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>

</think>

```

这个阶段只拼接字符串，不执行 Transformer，也不建立 KV Cache。

### 消息边界 token

每条消息由下面的格式生成：

```cpp
prompt << "<|im_start|>"
       << message.role
       << '\n'
       << message.content
       << "<|im_end|>\n";
```

`<|im_start|>` 表示一条消息开始，`<|im_end|>` 表示消息结束。格式化前会检查
两个 token 是否存在于 GGUF 内嵌 tokenizer 的词表：

```cpp
if (!gguf_token_id(tokenizer, "<|im_start|>").has_value() ||
    !gguf_token_id(tokenizer, "<|im_end|>").has_value()) {
  return Status::invalid_argument(
    "Qwen3.5 tokenizer is missing chat control tokens");
}
```

这项检查可以避免控制 token 被误当作普通字符进行 BPE 编码。

### add_generation_prompt

`add_generation_prompt` 决定是否在历史消息后追加 assistant 回复头：

```text
<|im_start|>assistant
```

推理请求通常将它设置为 `true`，含义是“历史对话已经结束，现在轮到 assistant
继续生成”。它与是否启用 thinking 是两个不同的配置。

### enable_thinking

当 `add_generation_prompt=true` 时：

```cpp
if (enable_thinking) {
  prompt << "<think>\n";
} else {
  prompt << "<think>\n\n</think>\n\n";
}
```

- `enable_thinking=true`：只写入 `<think>` 开头，让模型继续生成思考内容。
- `enable_thinking=false`：预先写入空的 thinking 块，让模型直接生成正式回答。

当前推理调用逻辑相当于：

```cpp
format_qwen35_chat_prompt(
  tokenizer.value(),
  request.messages,
  true,                     // add_generation_prompt
  request.enable_thinking); // enable_thinking
```

## 2. Tokenization

格式化完成后，prompt 字符串进入：

```cpp
gguf_encode_text(
  tokenizer.value(),
  prompt,
  false, // add_special
  true); // parse_special
```

该阶段将字符串转换为 token ID 数组：

```text
"<|im_start|>user\n你好..."
                │
                ▼
[special_id, text_id, text_id, ...]
```

`parse_special=true` 表示识别 `<|im_start|>`、`<|im_end|>` 等控制 token。
编码器先调用 `partition_special_tokens()` 将输入划分为特殊片段和普通文本片段：

```jsonc
[
  {"special": true,  "text": "<|im_start|>"},
  {"special": false, "text": "user\n你好"},
  {"special": true,  "text": "<|im_end|>"},
  {"special": false, "text": "\n"}
]
```

处理规则为：

- 特殊 token：直接从 `token_to_id` 查出 ID，不执行普通 BPE。
- 普通文本：先按 Qwen3.5 规则切分，再执行 byte-level BPE。

## 3. prompt_token_count

纯文本请求使用 token ID 数组长度：

```cpp
prompt_token_count = prompt_tokens.value().size();
```

因此 Chat Template 添加的消息边界、role、换行、thinking 标记和 assistant
generation prompt 都计入 prompt token 数。

图文请求使用 mixed prefill 计划中的总数：

```cpp
prompt_token_count = mixed_prefill->total_tokens;
```

其逻辑关系是：

```text
total_tokens = text_tokens + image_tokens
```

图片 token 来自 mmproj 产生的视觉 embedding，它们在 Prefill 中占据与文本 token
相同的 hidden-state 行，但不经过文本 tokenizer。

## 4. Prefill

Prefill 也称 prompt processing。它将完整 prompt token 序列一次或分块送入
Qwen3.5 网络：

```text
Prompt token IDs / image embeddings
        │
        ▼
Token Embedding
        │
        ▼
18 层 Linear Attention + 6 层 Full Attention
        │
        ├── 更新 Full Attention KV Cache
        ├── 更新 Linear Attention recurrent state
        └── 产生每个位置的 hidden state
        ▼
Output Norm + LM Head
        ▼
最后一个 prompt 位置的 logits
```

Prefill 的主要作用是：

1. 让模型处理和理解完整输入上下文。
2. 建立 Decode 阶段复用的 KV Cache 和 recurrent state。
3. 产生用于选择第一个输出 token 的 logits。

当 prompt 很长时，`prefill_chunk_tokens` 可以让 runtime 分块处理 prompt。分块只
改变计算和内存调度方式，不改变逻辑上的 prompt token 数。

Profiler 中 Prefill 使用：

```cpp
profiler.scoped("request.prefill");
```

所以性能分析里的 `request.prefill` 不包含 Chat templating 和 Tokenization；
它记录的是模型计算阶段。

## 5. Logits 与 Sampling

LM Head 为词表中的每个 token 产生一个 logit。Sampling 根据有效采样配置选择
下一个 token：

```jsonc
{
  "do_sample": false,
  "temperature": 1.0,
  "top_k": 0,
  "top_p": 1.0,
  "seed": 0
}
```

默认 `do_sample=false`，使用 greedy decoding，即选择 logit 最大的 token。
第一个生成 token 使用 Prefill 产生的 logits；后续 token 使用每次 Decode
产生的 logits。

## 6. Decode

Decode 是自回归生成循环。每轮把新 token 输入模型，并生成下一轮 logits：

```text
当前 token
    │
    ▼
run_qwen35_decode_tokens()
    │
    ├── 读取并更新 KV Cache
    ├── 读取并更新 recurrent state
    └── 计算当前 token 的 hidden state
    ▼
Output Norm + LM Head
    │
    ▼
logits
    │
    ▼
Sampling
    │
    ▼
下一个 token
```

与 Prefill 的区别：

| 阶段 | 输入 | 主要目的 |
| --- | --- | --- |
| Prefill | 整个 prompt 或 prompt chunk | 理解上下文并建立模型缓存 |
| Decode | 通常是一个新 token | 复用缓存并继续生成 |

Profiler 中 Decode 使用：

```cpp
profiler.scoped("request.decode");
```

循环会在达到 `max_new_tokens`、产生 EOS 或遇到其他停止条件时结束。

## 7. Detokenization

生成的 token IDs 最后通过以下函数转换为 UTF-8 文本：

```cpp
gguf_decode_token_text(
  tokenizer.value(),
  generated_tokens,
  true); // skip_control
```

`skip_control=true` 表示输出时跳过 control 和 unused 类型的 token。

## 8. 代码入口

主要实现位置：

```text
src/runtime/qwen35_runtime.cpp
├── generate_qwen35_metal()       推理总入口
├── prompt 格式化与 tokenization
├── Prefill
├── run_qwen35_decode_tokens()    Decode forward
├── LM Head 与 Sampling
└── 输出解码

src/runtime/gguf_tokenizer.cpp
├── load_gguf_tokenizer()
├── format_qwen35_chat_prompt()
├── gguf_encode_text()
├── partition_special_tokens()
└── gguf_decode_token_text()
```

最简洁的术语边界是：

```text
format_qwen35_chat_prompt() = Chat templating
gguf_encode_text()          = Tokenization
Prompt 的模型 Forward       = Prefill
生成 token 的循环 Forward   = Decode
gguf_decode_token_text()    = Detokenization
```
