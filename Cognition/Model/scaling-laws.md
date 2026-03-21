# Scaling Laws

## 核心认知

**Scaling Law：模型性能与规模呈可预测的幂律关系。**

OpenAI 2020 年发现：模型性能主要受三个因素影响，且呈幂律关系：
- **N** — 模型参数量
- **D** — 训练数据量
- **C** — 计算量

公式简化表示：
```
Loss ∝ N^(-α) × D^(-β)
```

其中 α、β 为正数，意味着参数和数据增加都会降低损失。

## 关键结论

### 1. 幂律关系
在跨越多个数量级的范围内，性能与规模的关系高度可预测。

### 2. 计算最优
给定计算预算 C，最优分配：
- 模型大小 N* ∝ C^0.74
- 数据量 D* ∝ C^0.26

即：计算增加时，应更多投入于模型规模，而非数据量。

### 3. Chinchilla 修正
DeepMind 2022 年指出：之前的模型普遍**欠训练**。
- 实际最优：N 和 D 应同步 scaling
- 很多大模型数据量不足

## 与架构的关系

Scaling Law 是**现象描述**，非架构专属。但不同架构实现的简洁程度不同：

| 架构 | Scaling 难度 | 原因 |
|-----|-------------|------|
| **Decoder-only** | ⭐ 低 | 架构简洁，单向注意力，易扩展 |
| Encoder-only | 中 | 双向注意力，MLM 目标不适合生成 scaling |
| Encoder-Decoder | 高 | 架构复杂，工程难度大 |

[[decoder-dominance|Decoder 主导时代]] 的一个关键原因：**它是实现 scaling law 最简洁的架构**。

## 实践意义

- **可预测性**：给定预算，可预测所需模型规模和性能
- **投资决策**：指导算力、数据、参数的分配
- **架构选择**：简洁架构更利于 scaling

## 反直觉点

- 更大的模型 ≠ 更好的性能（需要配套的数据和计算）
- 很多"大模型"实际上是欠训练的（Chinchilla 启示）
- Scaling law 在极大规模下可能失效（尚未观察到）

## 相关认知

- [[decoder-dominance|Decoder 主导时代的成因]]
- [[emergent-abilities|涌现能力]]
- [[compute-optimal-training|计算最优训练]]

## 参考

- Scaling Laws for Neural Language Models (Kaplan et al., 2020)
- Training Compute-Optimal Large Language Models (Chinchilla, Hoffmann et al., 2022)

---
*创建: 2026-03-21*