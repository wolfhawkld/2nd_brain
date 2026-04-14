# Flash Attention

## 定义

Flash Attention 是一种高效的注意力计算算法，通过优化内存访问模式显著减少 Transformer 模型在长序列上的计算开销和内存占用。

## 核心原理

### 传统注意力的问题

标准注意力机制的计算过程：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

传统实现需要：
1. 计算 $QK^T$ 得到完整的注意力矩阵（$N \times N$）
2. 应用 softmax
3. 与 $V$ 相乘

**问题**：$N \times N$ 的注意力矩阵需要大量内存，尤其是长序列时。内存读写成为瓶颈。

### Flash Attention 的优化

核心思想：**分块计算（tiling）+ 在线 softmax**

1. **分块处理**：将 Q、K、V 分成小块，逐块计算
2. **在线 softmax**：不需要存储完整注意力矩阵，逐块累积 softmax 结果
3. **内存重用**：利用 GPU SRAM（快速内存）避免频繁读写 HBM（慢速内存）

数学技巧：
- 使用增量式 softmax 计算：$\text{softmax}(x_1, ..., x_n)$ 可以通过分块逐步累积
- 保持数值稳定性：减去最大值技巧

## 性能提升

| 指标 | 传统注意力 | Flash Attention |
|------|-----------|-----------------|
| 内存占用 | $O(N^2)$ | $O(N)$ |
| 计算速度 | 慢（内存瓶颈） | 快 2-4x |
| 最大序列长度 | 受内存限制 | 可处理更长序列 |

## 版本演进

- **Flash Attention v1** (2022)：首次提出分块计算
- **Flash Attention v2** (2023)：优化并行化，进一步提速
- **Flash Attention v3** (2024)：支持 FP8，H100 GPU 优化

## 应用场景

- 长文档处理（书籍、论文）
- 长视频理解
- 大规模代码分析
- 扩展 LLM 上下文窗口（从 2K → 128K+）

## 前置知识

- [[注意力机制]] - 基础注意力计算
- [[Transformer架构]] - Flash Attention 的应用场景
- GPU 内存层次结构（SRAM vs HBM）

## 相关概念

- [[kv-cache]] - KV Cache 与 Flash Attention 配合优化推理
- PagedAttention - 另一种内存优化技术（vLLM 使用）

## 参考资料

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Tri Dao et al., 2022)
- FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023)