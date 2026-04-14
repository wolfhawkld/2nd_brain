# KV Cache

## 定义

KV Cache（Key-Value Cache）是 LLM 推理时的缓存机制，存储已计算过的 Key 和 Value 向量，避免在生成每个新 token 时重复计算历史序列的 KV 值。

## 核心原理

### Transformer 推理过程

自回归生成时，每生成一个新 token：
1. 需要计算该 token 的 Query（$Q_{new}$）
2. 需要与所有历史 token 的 Key、Value 计算注意力

**问题**：如果每次都重新计算所有历史 KV，计算量随序列长度线性增长，非常低效。

### KV Cache 的作用

| 无 KV Cache | 有 KV Cache |
|------------|-------------|
| 每步重新计算所有历史 KV | 只计算新 token 的 KV |
| 计算量 $O(n^2)$ | 计算量 $O(n)$（累积） |
| 内存占用低但慢 | 内存占用高但快 |

KV Cache 存储的是：
- 已生成 token 对应的 Key 向量：$K_1, K_2, ..., K_{n-1}$
- 已生成 token 对应的 Value 向量：$V_1, V_2, ..., V_{n-1}$

生成新 token 时：
- 只计算 $Q_n$, $K_n$, $V_n$
- 将 $K_n$, $V_n$ append 到 cache
- 用 $Q_n$ 与完整的 cached KV 计算注意力

## 内存占用计算

KV Cache 内存 = $2 \times \text{layers} \times \text{seq\_len} \times \text{hidden\_dim} \times \text{bytes\_per\_element}$

示例（LLaMA-7B，FP16）：
- Layers: 32
- Hidden dim: 4096
- Bytes: 2 (FP16)
- 2048 tokens: ~512 MB
- 128K tokens: ~32 GB

**结论**：长上下文场景下，KV Cache 是内存瓶颈的主要来源。

## 优化技术

### PagedAttention (vLLM)

- 将 KV Cache 分页管理，类似操作系统虚拟内存
- 非连续内存存储，提高内存利用率
- 支持内存共享（beam search、parallel sampling）

### KV Cache 压缩

- ** eviction**：丢弃不重要的 KV（如 H2O、StreamingLLM）
- **quantization**：KV Cache 量化（FP16 → INT8/INT4）
- **sparsity**：只保留关键位置的 KV

### Multi-Query Attention (MQA) / Grouped-Query Attention (GQA)

- 多个 Query 共享一组 KV
- 显著减少 KV Cache 大小
- LLaMA-2 使用 GQA

## 与 Flash Attention 的关系

Flash Attention 优化**计算效率**（减少内存读写）
KV Cache 优化**推理效率**（避免重复计算）

两者互补：
- Flash Attention 让训练/长序列推理更快
- KV Cache 让自回归生成更高效

## 前置知识

- [[注意力机制]] - KV Cache 的本质是缓存注意力计算结果
- [[Transformer架构]] - 理解 decoder-only 架构的自回归生成
- [[flash-attention]] - 计算层面的优化

## 应用场景

- 长对话聊天机器人
- 长文档问答
- 批量推理（batch inference）
- 多轮对话系统

## 参考资料

- vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (Kwon et al., 2023)
- Efficient Memory Management for Large Language Model Serving with PagedAttention
- H2O: Heavy-Hitter Oracle for Efficient LLM Decoding