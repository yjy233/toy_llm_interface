# Qwen3.5 Byte-Level BPE 编码逻辑

本文只说明本项目 `gguf_encode_text()` 使用的 Qwen3.5 tokenizer 编码过程。
实现位于 `src/runtime/gguf_tokenizer.cpp`。

## 1. BPE 的作用

BPE（Byte Pair Encoding）将 prompt 文本转换为模型词表中的 token ID：

```text
Prompt 字符串
    │
    │ 特殊 token 识别、预切分、byte-level 映射、BPE merge
    ▼
Token IDs
    │
    ▼
Qwen3.5 Prefill
```

入口函数：

```cpp
Result<std::vector<std::int64_t>> gguf_encode_text(
  const GgufTokenizer& tokenizer,
  std::string_view text,
  bool add_special,
  bool parse_special);
```

例如 Chat Template 先生成：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
```

`gguf_encode_text()` 再将它编码为：

```text
[im_start_id, user_text_ids..., im_end_id, im_start_id, assistant_ids...]
```

这里的 ID 是 token 在 GGUF 词表中的索引。

## 2. 运行时数据来自 GGUF

本项目不会在推理时训练 BPE。完整词表和已经训练好的 merge 规则都保存在
Qwen3.5 GGUF metadata：

```jsonc
{
  "tokenizer.ggml.model": "gpt2",
  "tokenizer.ggml.pre": "qwen35",
  "tokenizer.ggml.tokens": ["..."],
  "tokenizer.ggml.token_type": [1, 1, 3],
  "tokenizer.ggml.merges": [
    "token_a token_b",
    "token_c token_d"
  ]
}
```

当前 encoder 只支持：

```text
tokenizer.ggml.model = gpt2
tokenizer.ggml.pre   = qwen35
```

其他组合会返回 `invalid_argument`。

## 3. Loader 构建的索引

`load_gguf_tokenizer()` 读取 GGUF 后，在内存中创建两张索引表。

### token_to_id

词表数组的下标就是 token ID：

```jsonc
{
  "<|im_start|>": 248044,
  "<|im_end|>": 248045,
  "hello": 14990
}
```

上面的数值仅用于解释结构；实际 ID 以当前 GGUF 为准。

构建逻辑相当于：

```cpp
for (std::size_t id = 0; id < tokens.size(); ++id) {
  token_to_id[tokens[id]] = id;
}
```

### merge_ranks

`tokenizer.ggml.merges` 的数组顺序就是 merge 优先级：

```jsonc
{
  "a + b": 0,
  "ab + c": 1,
  "x + y": 2
}
```

rank 越小，合并优先级越高。实现使用不可见分隔符 `0x1f` 将左右 token
拼成哈希表 key，避免普通字符串拼接产生歧义。

## 4. 完整编码流程

```text
输入 UTF-8 文本
    │
    ├─ ① 可选添加 BOS
    │
    ├─ ② 分离特殊 token
    │      ├─ 特殊片段：直接写入 token ID
    │      └─ 普通片段：继续处理
    │
    ├─ ③ Qwen3.5 预切分
    │
    ├─ ④ GPT-2 byte-level 映射
    │
    ├─ ⑤ BPE 反复合并最高优先级相邻 pair
    │
    ├─ ⑥ 最终 token 查 token_to_id
    │
    └─ ⑦ 可选添加 EOS
    ▼
std::vector<int64_t>
```

## 5. 第一步：可选添加 BOS/EOS

当 `add_special=true` 时，encoder 会根据 tokenizer 配置决定是否添加 BOS 和
EOS：

```cpp
if (add_special) {
  append_special_bos(tokenizer, ids);
}

// 编码正文

if (add_special) {
  append_special_eos(tokenizer, ids);
}
```

`add_special=true` 不代表一定添加；还需要 GGUF 中的 `add_bos_token` 或
`add_eos_token` 为 `true`，并且相应 ID 有效。

Qwen3.5 Chat 推理当前使用：

```cpp
gguf_encode_text(tokenizer, prompt, false, true);
```

因此 Chat 路径不会由 encoder 额外添加 BOS/EOS。

## 6. 第二步：分离特殊 token

Chat prompt 中可能包含：

```text
<|im_start|>
<|im_end|>
```

这些字符串应该各自成为一个完整 token，不能按照普通字符执行 BPE。

`partition_special_tokens()` 先将输入拆分：

```jsonc
[
  {
    "special": true,
    "text": "<|im_start|>",
    "token_id": "<GGUF 中的 ID>"
  },
  {
    "special": false,
    "text": "user\n你好"
  },
  {
    "special": true,
    "text": "<|im_end|>",
    "token_id": "<GGUF 中的 ID>"
  }
]
```

对特殊片段，主循环直接写入 ID：

```cpp
if (fragment.token) {
  ids.push_back(fragment.token_id);
  continue;
}
```

### parse_special

Token 类型决定它是否参与特殊 token 匹配：

| Token 类型 | `parse_special=false` | `parse_special=true` |
| --- | --- | --- |
| `user_defined` | 识别 | 识别 |
| `control` | 不识别 | 识别 |
| `unknown` | 不识别 | 识别 |
| 其他类型 | 不识别 | 不识别 |

Chat 编码传入 `parse_special=true`，因此可以识别控制 token。

匹配前，候选特殊 token 按字符串长度从长到短排序。当前位置同时匹配多个 token
时优先选择较长的 token，避免短 token 提前截断长 token。

## 7. 第三步：Qwen3.5 预切分

普通文本进入：

```cpp
split_qwen35_words(fragment_text)
```

这一步称为 pre-tokenization。它先按 Unicode 字符类型划分较小片段，再对每个
片段独立执行 BPE。

当前实现识别：

- 字母和组合音标
- 数字
- 空白、换行和回车
- 标点及其他符号
- 英文缩写后缀：`'s`、`'t`、`'m`、`'d`、`'re`、`'ve`、`'ll`

数字当前按单个 Unicode 数字切分。连续字母和组合音标通常留在同一个片段中。
空格、连续空白和换行使用专门规则处理，使切分行为与 Qwen3.5 tokenizer
预期一致。

预切分不是最终 tokenization。一个预切分片段仍可能通过 BPE 生成一个或多个
token。

## 8. 第四步：Byte-level 映射

预切分片段在进入 BPE 前会调用：

```cpp
byte_encoded(text, byte_encoder())
```

GPT-2 byte-level BPE 为 0 到 255 的每个原始字节建立一个可打印 Unicode
表示。这样任意 UTF-8 输入都可以先表示成基础符号序列，而不必依赖传统的
unknown token。

例如一个汉字在 UTF-8 中由多个字节组成：

```text
Unicode 文本
    ↓ UTF-8 bytes
多个 byte-level 符号
    ↓ BPE merge
一个或多个词表 token
```

这里处理的是 UTF-8 字节，而不是简单地把每个汉字固定映射成一个 token。
最终是否合并为更大的 token 取决于 GGUF 中的 merge rank 和词表。

## 9. 第五步：BPE merge

每个 byte-encoded 片段进入：

```cpp
bpe_encode_word(tokenizer, encoded_word)
```

初始状态下，片段被拆成 byte encoder 产生的 Unicode codepoint 字符串：

```text
[a, b, c]
```

每轮执行：

1. 枚举所有相邻 pair。
2. 在 `merge_ranks` 中查找每个 pair。
3. 选择 rank 最小的 pair。
4. 合并该 pair。
5. 重复，直到没有 pair 可以合并。

核心逻辑：

```cpp
while (word.size() > 1) {
  // 查找 rank 最小的相邻 pair
  // 将该 pair 合并
  // 没有可合并 pair 时退出
}
```

### 简化示例

假设 merge rank 为：

```jsonc
{
  "a + b": 1,
  "b + c": 5,
  "ab + c": 2
}
```

输入：

```text
[a, b, c]
```

第一轮相邻 pair：

```text
(a, b) rank=1
(b, c) rank=5
```

选择 rank 更小的 `(a, b)`：

```text
[ab, c]
```

下一轮：

```text
(ab, c) rank=2
```

继续合并：

```text
[abc]
```

最后查询：

```cpp
tokenizer.token_to_id["abc"]
```

注意：这个例子只解释算法，merge 和 ID 不是当前 Qwen3.5 GGUF 的真实值。

## 10. 第六步：转换为 token ID

BPE 停止后，每个最终 token 通过 `token_to_id` 查找：

```cpp
const auto it = tokenizer.token_to_id.find(token);
if (it != tokenizer.token_to_id.end()) {
  ids.push_back(it->second);
}
```

实现还包含防御性的 byte fallback：如果合并后的 token 没有直接出现在词表中，
它会逐字节尝试查找基础 byte token。正常的 Qwen3.5 GGUF 词表和 merge 规则
应当能够覆盖编码结果。

## 11. 解码过程

生成阶段得到 token IDs 后，通过：

```cpp
gguf_decode_token_text(tokenizer, ids, true)
```

进行解码：

```text
Token IDs
    │
    ├─ 根据 ID 读取 tokens[id]
    ├─ 可选跳过 control / unused token
    ├─ byte decoder 将可打印 Unicode 映射还原为原始字节
    ▼
UTF-8 文本
```

Byte encoder 和 byte decoder 是互逆关系：

```text
原始 UTF-8 bytes
   → byte encoder
   → BPE token
   → token ID
   → tokens[id]
   → byte decoder
   → 原始 UTF-8 bytes
```

## 12. 与 Prefill 的边界

`gguf_encode_text()` 属于 Tokenization，不属于 Prefill：

```text
format_qwen35_chat_prompt()  Chat templating
             ↓
gguf_encode_text()           Tokenization / BPE
             ↓
prompt token IDs
             ↓
Qwen3.5 模型 forward          Prefill
```

`prompt_token_count` 对纯文本请求就是编码结果的长度：

```cpp
prompt_token_count = prompt_tokens.value().size();
```

因此 BPE 如何切分文本会直接影响：

- Prompt token 数
- Prefill 计算量
- 上下文窗口占用
- KV Cache 和 recurrent state 的位置推进

## 13. 主要代码位置

```text
include/toyllm/runtime/gguf_tokenizer.hpp
└── GgufTokenizer 和公开 API

src/runtime/gguf_tokenizer.cpp
├── load_gguf_tokenizer()        加载 GGUF tokenizer metadata
├── partition_special_tokens()   分离特殊 token
├── split_qwen35_words()         Qwen3.5 pre-tokenization
├── byte_encoded()               GPT-2 byte-level 映射
├── bpe_encode_word()            BPE merge
├── gguf_encode_text()           编码总入口
└── gguf_decode_token_text()     token ID 解码
```

本地查看 prompt 的实际编码结果：

```bash
./build/debug/kraken-infer tokenize \
  --model models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf \
  --prompt "你好" \
  --parse-special
```

当前本地 Qwen3.5-0.8B GGUF 的实际输出为：

```text
[109266]
```

这表示“你好”在当前词表和 merge 规则下被编码为一个 token。该结果属于当前
GGUF；替换词表或模型版本后，token ID 和 token 数都可能变化。
