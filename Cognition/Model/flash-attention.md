# Flash Attention

## 定义

Flash Attention 是一种高效的注意力计算算法，通过 **Online Softmax + 分块计算（Tiling）** 显著减少 Transformer 模型在长序列上的计算开销和内存占用。核心是将原本 memory-bound 的注意力运算变为 compute-bound。

## 核心原理：Online Softmax

### 传统注意力的问题

标准注意力机制：
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

传统实现需要：
1. 计算完整的 $N \times N$ 注意力矩阵
2. 应用 softmax（需要全局 max 和 sum）
3. 与 $V$ 相乘

**瓶颈**：$N \times N$ 矩阵必须写入 HBM（高带宽内存），内存读写成为瓶颈。

### Online Softmax 的数学推导

Softmax 本质是一个"全局依赖"算子——需要知道所有值才能归一化。但 Online Softmax 揭示了一个反直觉的事实：**softmax 可以边算边更新，不需要一次性看到全部数据**。

#### 公式推导

假设已处理集合 A，新来 block B。维护两个状态：

**最大值 m**：
$$m_A = \max_{i \in A} x_i$$

**归一化分母 l**：
$$l_A = \sum_{i \in A} e^{x_i - m_A}$$

新 block B 到来后，新的最大值：
$$m' = m_{A \cup B} = \max(m_A, m_B)$$

关键：新的归一化分母如何更新？
$$l' = l_{A \cup B} = \sum_{i \in A \cup B} e^{x_i - m'}$$

展开计算：
$$l' = \sum_{i \in A} e^{x_i - m'} + \sum_{i \in B} e^{x_i - m'}$$

$$= \sum_{i \in A} e^{x_i - m_A} \cdot e^{m_A - m'} + \sum_{i \in B} e^{x_i - m'}$$

$$= l_A \cdot e^{m_A - m'} + \sum_{i \in B} e^{x_i - m'}$$

**核心公式**：
$$l' = l_A \cdot e^{m_A - m'} + l_B$$

#### 数值例子

一行 score：`[1, -2, 4, 0]`

直接算 softmax 分母 Z：
$$Z = e^{1-4} + e^{-2-4} + e^{4-4} + e^{0-4} = e^{-3} + e^{-6} + 1 + e^{-4}$$

分成两个 block：`[1, -2]` 和 `[4, 0]`

Block A：$m_A = 1$, $l_A = e^{1-1} + e^{-2-1} = 1 + e^{-3}$

Block B 来了，新最大值 $m' = 4$

**坐标系补偿**：把 Block A 的贡献对齐到新坐标系：
$$l_A \cdot e^{m_A - m'} = (1 + e^{-3}) \cdot e^{1-4} = e^{-3} + e^{-6}$$

Block B 的贡献：
$$e^{4-4} + e^{0-4} = 1 + e^{-4}$$

合并后与直接计算完全一致！

### 为什么可行？

Softmax 具有**平移不变性**：
$$\text{softmax}(x) = \text{softmax}(x + c)$$

减 max 是选一个**参考坐标系**。当最大值变化时，对历史贡献做精确的**代数补偿**——这不是近似，而是严格等价的数学变换。

**关键条件**：
- 平移/缩放不变性
- 可结合的累积结构（$\sum e^{x_i}$ 可以分块求和再合并）

## Flash Attention 的工程实现

### 分块计算流程

1. **分块**：将 Q、K、V 分成小块（每个块可放入 SRAM）
2. **逐块计算**：
   - 对每个 Q block，遍历所有 K、V blocks
   - 维护状态 $(m, l, o)$ — 最大值、分母、输出累积
   - 用 Online Softmax 公式更新
3. **避免 HBM 读写**：计算全部在 SRAM 完成

### 内存层次利用

| 内存类型 | 容量 | 速度 | Flash Attention 利用 |
|---------|------|------|---------------------|
| HBM | ~40 GB | ~1.5 TB/s | 只存 Q、K、V（各 O(N)） |
| SRAM | ~192 KB | ~19 TB/s | 存分块计算结果 |

传统注意力需要 HBM 存 $O(N^2)$ 的注意力矩阵，Flash Attention 只需 $O(N)$。

## 性能对比

| 指标 | 传统注意力 | Flash Attention |
|------|-----------|-----------------|
| 内存占用 | $O(N^2)$ | $O(N)$ |
| HBM 访问次数 | $O(N^2)$ | $O(N)$ |
| 计算速度 | Memory-bound | Compute-bound |
| 最大序列长度 | 受 HBM 限制 | 受 SRAM 分块数限制 |

## 版本演进

- **Flash Attention v1** (2022)：Online Softmax + Tiling
- **Flash Attention v2** (2023)：优化并行化，减少非矩阵乘 FLOPs
- **Flash Attention v3** (2024)：FP8 支持，H100 异步优化

## 更广泛的 Streaming Reduction 模式

Online Softmax 并非特例，它揭示了一个通用模式：**很多"全局依赖"算子都可以改写为 streaming/tiling 版本**。

| 算子 | Streaming 维度 | 维护状态 | 参考系变化 | 补偿方式 |
|------|---------------|---------|-----------|---------|
| Online Softmax | token 维 | $(m, l)$ | 最大值更新 | $e^{m_A - m'}$ |
| LayerNorm | feature 维 | $(n, \mu, M_2)$ | 均值漂移 | $(\mu_B - \mu_A)^2$ |
| Adam/RMSProp | time 维 | $(m, v)$ | 时间衰减 | $\beta^{\Delta}$ |
| 协方差矩阵 | sample 维 | $(n, \mu, \Sigma)$ | 均值移动 | 外积补偿 |

**判断标准**：如果算子满足以下条件，就可以分块化：
1. 可重写为 Reduction 形式：$\text{state} = \bigoplus_i \phi(x_i)$
2. 状态 $O(1)$（与数据规模无关）
3. 存在 Merge 函数：$\text{state}_{A \cup B} = \text{merge}(\text{state}_A, \text{state}_B)$

## 前置知识

- [[注意力机制]] - 基础注意力计算
- [[Transformer架构]] - Flash Attention 的应用场景
- GPU 内存层次（SRAM vs HBM）

## 相关概念

- [[kv-cache]] - KV Cache 与 Flash Attention 配合优化推理
- PagedAttention - 另一种内存优化技术（vLLM 使用）

## 参考资料

- FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (Tri Dao et al., 2022)
- Online normalizer calculation for softmax (2018)
- 李宏毅 ML 2026 Spring：加快語言模型生成速度(1/2)：Flash Attention
- 从 FlashAttention 到 Streaming Reduction：如何把"全局算子"改写成可分块计算

---
*参考李宏毅 ML 2026 Spring 课程 · Nemo 整理*