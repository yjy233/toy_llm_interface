# Qwen3.5 MRoPE：文本与图像的位置编码

本文只说明本项目中 Qwen3.5 的 MRoPE（Multimodal Rotary Position
Embedding）实现，包括 GGUF 参数、位置坐标生成、Metal 旋转公式，以及
Prefill/Decode 阶段如何消费这些位置。

MRoPE 不是 Prefill 的别名。它是 Full Attention 内部对 Query 和 Key
执行的位置编码操作；Prefill 和 Decode 都会调用它。

## 1. 为什么需要 MRoPE

普通文本 RoPE 只需要一个一维位置：

```text
token:     A  B  C  D
position:  0  1  2  3
```

图片经过视觉编码器后是二维网格 token。仅使用一维序号会丢失“同行、同列”
关系，所以 Qwen3.5 为每个 token 准备四路位置坐标：

```text
section 0: t    时间/序列坐标
section 1: y    图像行坐标
section 2: x    图像列坐标
section 3: aux  第四路坐标
```

当前实现中，文本 token 的四路坐标相同；图像 token 使用 `t/y/x` 三路坐标，
第四路固定为 `0`。

## 2. 参数从哪里来

模型加载阶段从主模型 GGUF metadata 读取：

```text
{architecture}.rope.freq_base
{architecture}.rope.dimension_sections
```

加载后的字段是：

```cpp
model.rope_theta
model.rope_dimension_sections
```

仓库中真实模型
`models/qwen3.5-0.8b/Qwen3.5-0.8B-Q4_K_M.gguf` 的相关信息为：

```jsonc
{
  "general.architecture": "qwen35",
  "qwen35.attention.head_count": 8,
  "qwen35.attention.head_count_kv": 2,
  "qwen35.attention.key_length": 256,
  "qwen35.rope.freq_base": 10000000,
  "qwen35.rope.dimension_sections": [11, 11, 10, 0],
  "qwen35.full_attention_interval": 4
}
```

上面的 JSON 只是把 GGUF 的 typed key-value metadata 写成便于阅读的形式，
不是磁盘中的真实 JSON 文件。

每个 section 数值表示使用该坐标的“旋转维度对”数量：

```text
t:   11 对
y:   11 对
x:   10 对
aux:  0 对
合计: 32 对 = 64 个标量维度
```

代码据此计算：

```cpp
mrope_pairs = 11 + 11 + 10 + 0;  // 32
mrope_dims = 2 * mrope_pairs;     // 64
```

该模型每个 attention head 有 `256` 维，因此 MRoPE 旋转每个 head 的前
`64` 维，剩余 `192` 维保持不变。

如果 GGUF 没有 `rope.dimension_sections`，当前 loader 使用
`[11, 11, 10, 0]` 作为兼容默认值。运行时仍会验证必须至少有四段、每段
非负且总和非零。

## 3. Position buffer 的内存布局

MRoPE position buffer 是 `int32`，逻辑 shape 为：

```text
[4, token_count]
```

实际使用 section-major 的一维布局：

```cpp
positions[section * total_tokens + token]
```

也就是：

```text
[所有 token 的 t]
[所有 token 的 y]
[所有 token 的 x]
[所有 token 的 aux]
```

不能把它误解为逐 token 交错的
`[t0, y0, x0, aux0, t1, y1, ...]`。

## 4. 文本位置如何生成

纯文本或混合 Prompt 中的文本 chunk 都使用普通递增位置。设 chunk 从
`position_start` 开始，则第 `i` 个 token 为：

```text
t = y = x = aux = position_start + i
```

例如三个文本 token 从位置 `7` 开始：

```text
t:    [7, 8, 9]
y:    [7, 8, 9]
x:    [7, 8, 9]
aux:  [7, 8, 9]
```

对应实现是 `append_mixed_text_mrope_positions()`。非多模态批次由
`make_qwen35_mrope_positions(context, start_position, tokens)` 生成同样的
四路序列。

## 5. 图片位置如何生成

视觉编码器输出的图片 token 按合并后的二维网格排列。设网格大小为：

```text
merge_grid_x = W
merge_grid_y = H
image_tokens = W * H
```

代码按 row-major 顺序遍历：

```cpp
token = row * W + col;
t = position_start;
y = position_start + row;
x = position_start + col;
aux = 0;
```

例如一张 `3 x 2` 网格图片从 MRoPE 位置 `2` 开始，共产生六个图片 token：

```text
图片 token:  p0 p1 p2 p3 p4 p5
网格坐标:   0,0 0,1 0,2 1,0 1,1 1,2

t:           2  2  2  2  2  2
y:           2  2  2  3  3  3
x:           2  3  4  2  3  4
aux:         0  0  0  0  0  0
```

这段逻辑位于 `append_mixed_image_mrope_positions()`。

### 图片为什么不按 token 数推进位置

图片虽然有 `W * H` 个 token，但下一个 chunk 的 MRoPE 起点只推进：

```cpp
position_advance = max(W, H);
```

因为二维网格在坐标轴上占用的是最大边长，而不是展平后的 token 数。KV Cache
仍然必须为全部 `W * H` 个图片 token 保留物理槽位。

所以多模态请求中必须区分：

```text
prompt_token_count
  = 实际进入模型、占用 KV Cache 的 token 行数

prompt_mrope_position_count
  = Prompt 结束后，下一个生成 token 使用的 MRoPE 位置
```

## 6. 一个完整的混合位置例子

下面故意省略 Chat Template 的视觉边界 token，只观察三个连续 chunk：

```text
2 个文本 token
3 x 2 图片网格，共 6 个图片 token
2 个文本 token
```

得到：

```text
物理 token index:  0  1 | 2  3  4  5  6  7 | 8  9
chunk:             T  T | I  I  I  I  I  I | T  T

t:                 0  1 | 2  2  2  2  2  2 | 5  6
y:                 0  1 | 2  2  2  3  3  3 | 5  6
x:                 0  1 | 2  3  4  2  3  4 | 5  6
aux:               0  1 | 0  0  0  0  0  0 | 5  6
```

计数结果：

```text
prompt_token_count          = 2 + 6 + 2 = 10
prompt_mrope_position_count = 2 + 3 + 2 = 7
```

因此第一个 Decode token：

```text
写入 KV Cache 的物理位置 = 10
使用的四路 MRoPE 位置    = [7, 7, 7, 7]
```

纯文本请求中每个 token 恰好推进一个位置，所以这两个计数相等；只有图片等
多模态输入会让它们产生差异。

## 7. Metal kernel 实际做了什么

Full Attention 得到 Query 和 Key 后，对二者分别原地执行：

```cpp
context.mrope_f32_in_place(...);
```

对旋转区域内的第 `i` 个维度对：

```text
freq(i)  = 1 / theta^(2i / n_dims)
angle(i) = position(section(i), token) * freq(i)

x0' = x0 * cos(angle) - x1 * sin(angle)
x1' = x1 * cos(angle) + x0 * sin(angle)
```

本项目的维度对是 half-split 配对，不是相邻配对：

```text
x0 = values[i]
x1 = values[n_dims / 2 + i]
```

以当前 0.8B 模型的 `n_dims = 64` 为例：

```text
pair 0  : dim 0  与 dim 32
pair 1  : dim 1  与 dim 33
...
pair 31 : dim 31 与 dim 63
```

section 决定该维度对读取哪一路 position：

```text
pair  0..10 -> t
pair 11..21 -> y
pair 22..31 -> x
aux         -> 当前配置没有分配维度对
```

`theta` 来自 GGUF 的 `rope.freq_base`。不同 pair 的频率不同：低索引维度变化
更快，高索引维度变化更慢，从而同时表示局部和长距离相对位置。

注意，MRoPE 只改变 Q/K，不改变 V。之后 attention score 中的
`Q · K` 就携带了文本顺序或图片二维相对位置信息。

## 8. 在推理流程中的位置

当前调用链如下：

```text
主模型 GGUF
  └─ load_model_bundle()
      └─ 读取 rope.freq_base / rope.dimension_sections

ChatMessage + 图片
  └─ plan_qwen35_multimodal_prompt()
      └─ 规划文本 chunk、图片网格和视觉边界 token
          └─ tokenize_qwen35_multimodal_prompt()
              └─ 计算 token_count / position_advance
                  └─ build_qwen35_mixed_prefill_plan()
                      ├─ 拼接文本 embedding 与图片 embedding
                      └─ 生成 [4, total_tokens] MRoPE positions

Prefill / Decode
  └─ run_qwen35_full_attention_layer_chunk()
      ├─ Q/K projection
      ├─ Q/K norm
      ├─ mrope_f32_in_place(Query)
      ├─ mrope_f32_in_place(Key)
      ├─ Key/Value 写入 KV Cache
      └─ causal attention
```

Qwen3.5-0.8B 有 `24` 个主层，`full_attention_interval = 4`，因此当前层表
包含 `18` 个 Linear Attention 层和 `6` 个 Full Attention 层。MRoPE 只在
Full Attention 路径使用；Linear Attention 路径不读取该 position buffer。

## 9. Prefill 与 Decode 如何衔接

Prefill 时：

- 纯文本请求按 `[0, 1, 2, ...]` 生成四路相同的位置。
- 多模态请求先构建完整位置表，再按 prefill chunk 切出
  `[4, chunk_tokens]` 的连续 section-major buffer。
- 同一批位置会被每个 Full Attention 层重复用于旋转该层的 Q/K。

Prefill 完成后，运行时分别初始化：

```cpp
next_decode_position = prompt_token_count;
next_decode_mrope_position = prompt_mrope_position_count;
```

Decode 生成的是普通文本 token，因此每提交一个 token，这两个计数都加一。
MTP speculative decode 一次可能提交多个 token，计数则同时增加
`committed_tokens`。

## 10. 关键约束与常见误解

- `rope.dimension_sections` 的值是旋转维度对数量，不是 token 数。
- position buffer 固定有四个 section，即使第四段长度是 `0`。
- 图像 token 数决定 embedding/KV Cache 行数；图像网格边长决定 MRoPE
  position advance。
- `prompt_token_count` 不是最后一个 MRoPE position。
- MRoPE 不负责分词、Chat Template 或图片编码，它只给 attention 的 Q/K
  注入位置信息。
- MRoPE position 使用 `int32`，构建 buffer 时会检查位置是否越界。
- `mrope_dims` 必须为正数、是偶数且不能超过 `head_dim`。

## 11. 代码索引

| 作用 | 位置 |
|---|---|
| 从 GGUF 读取 RoPE 参数 | `src/model/model_config.cpp` |
| 校验四个 MRoPE sections | `qwen35_mrope_sections()` |
| 规划多模态 chunk | `plan_qwen35_multimodal_prompt()` |
| 计算 token/position advance | `tokenize_qwen35_multimodal_prompt()` |
| 生成文本四路位置 | `append_mixed_text_mrope_positions()` |
| 生成图片 `t/y/x/aux` 位置 | `append_mixed_image_mrope_positions()` |
| 构建并切分 position buffer | `make_qwen35_mrope_positions()` |
| Full Attention 消费位置 | `run_qwen35_full_attention_layer_chunk()` |
| MPS host 参数和校验 | `MpsContext::mrope_f32_in_place()` |
| Metal 旋转公式 | `mrope_f32_in_place` kernel |

这些函数主要分布在：

```text
src/model/model_config.cpp
src/runtime/qwen35_multimodal.cpp
src/runtime/qwen35_runtime.cpp
src/backends/mps/mps_backend.mm
include/toyllm/backends/mps/mps_backend.hpp
```
