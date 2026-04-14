# KV Cache

## 定义

KV Cache（Key-Value Cache）是 LLM 推理时的缓存机制，存储已计算过的 Key 和 Value 向量，避免在自回归生成每个新 token 时重复计算历史序列的 KV 值。

**本质**：KV Cache 是自回归生成的"时间状态机"——保存历史状态，让每步只需增量计算。

## 核心原理

### Transformer 推理的两阶段

| 阶段 | 特点 | 计算模式 |
|------|------|---------|
| **Prefill** | 处理输入 prompt | 并行计算所有 token 的 KV |
| **Decoding** | 逐 token 生成 | 自回归，每步只算新 token |

Decoding 阶段的计算不对称性：
- 每个新 token 只需计算自己的 $Q_n$, $K_n$, $V_n$
- 但注意力需要与**所有历史** KV 计算

### 无 KV Cache vs 有 KV Cache

| 无 KV Cache | 有 KV Cache |
|------------|-------------|
| 每步重新计算所有历史 KV | 只计算新 token 的 KV |
| 计算复杂度累积 $O(n^2)$ | 每步 $O(1)$（总累积 $O(n)$） |
| 内存低但极慢 | 内存高但快 |

KV Cache 的作用：
1. Prefill 阶段计算并存储 prompt 所有 token 的 K、V
2. Decoding 阶段：
   - 只计算新 token 的 $K_n$, $V_n$
   - Append 到 cache
   - 用 $Q_n$ 与完整 cached KV 计算注意力

## 显存占用精确量化

### 计算公式

每个 token 的 KV Cache 内存：
$$\text{Memory}_{\text{per\_token}} = 2 \times \text{layers} \times \text{heads}_{KV} \times \text{dim}_{\text{head}} \times \text{bytes}$$

**注意**：使用 GQA/MQA 的模型，$\text{heads}_{KV} < \text{heads}_Q$

### 实例计算

**LLaMA-3-8B (FP16)**：
- Layers: 32
- KV Heads: 8 (GQA，Query 头是 32)
- Head dim: 128
- Bytes: 2 (FP16)

$$\text{Memory} = 2 \times 32 \times 8 \times 128 \times 2 = 131,072 \text{ Bytes} \approx 128 \text{ KB/token}$$

**序列长度影响**：
| 序列长度 | KV Cache 占用 |
|---------|--------------|
| 2048 tokens | ~256 MB |
| 16K tokens | ~2 GB |
| 128K tokens | ~16 GB |

**结论**：长上下文场景下，KV Cache 是显存瓶颈的主要来源。

## 传统管理的失效：内存碎片

原生 PyTorch/FasterTransformer 采用**预分配连续显存**：

### 问题

| 碎片类型 | 描述 |
|---------|------|
| **Internal Fragmentation** | 为最大长度预留空间，实际生成往往远短，空间空置 |
| **Reservation Gap** | 无法预测输出长度，必须锁定剩余空间，无法被其他 batch 共享 |

显存利用率仅 ~60%，大量预留空间浪费。

## PagedAttention：内存分页映射

vLLM 提出的 **PagedAttention** 将操作系统虚拟内存思想引入 GPU 显存管理：

### 核心数据结构

```
┌─────────────────┐     ┌─────────────────┐
│ Logical Blocks  │     │ Physical Blocks │
│ (模型感知的KV)   │ ←── │ Block Table     │
│ 逻辑连续        │     │ (显存离散存储)   │
└─────────────────┘     └─────────────────┘
```

- **Block Size**：通常 16 tokens
- **Logical Blocks**：模型视角的连续 KV 序列
- **Physical Blocks**：显存中离散分布的存储块
- **Block Table**：逻辑→物理映射（类似 OS 页表）

### 动态按需分配

```
Token 生成流程：
┌──────────┐
│ Block 1  │ ← 填满后申请下一个
│ (tokens  │
│  1-16)   │
└──────────┘
      ↓
从 Free Block Pool 申请 Block 2
      ↓
┌──────────┐
│ Block 2  │ ← 继续填充...
│ (tokens  │
│ 17-32)   │
└──────────┘
```

**效果**：显存利用率从 ~60% 提升至 **96%+**

### Copy-on-Write (CoW) 与并行采样

PagedAttention 天然支持高效并行采样：

| 场景 | 传统方案 | PagedAttention |
|------|---------|----------------|
| Beam Search | 每个 beam 复制完整 KV | 共享 prompt 的物理块 |
| Parallel Sampling | 每个样本独立 KV | 只在分歧点 CoW |

**机制**：多个请求共享同一 Prompt 的物理块，仅在需要写入不同新 token 时触发 CoW，极大降低内存冗余。

## Continuous Batching

配合 PagedAttention，vLLM 实现 **Continuous Batching**：

- 同一 batch 可处理不同阶段（Prefill/Decoding）的请求
- 消除传统 batch 必须等待最长请求的"木桶效应"
- 显著提高 Effective Batch Size 和吞吐量

## 真实硬件约束

**RTX 4090 (24GB) 部署 LLaMA-3-8B**：

| 内存用途 | 占用 |
|---------|------|
| 模型权重 | ~15 GB |
| KV Cache 可用 | ~9 GB |

**容量估算**：
- FP16：~72K tokens
- FP8 KV Cache 量化：~144K tokens

**注意**：单卡 24GB 无法实现"百万上下文"，需要多卡张量并行或 CPU Offloading。

## 技术演进

| 优化方向 | 方法 |
|---------|------|
| **KV Cache 量化** | INT8/FP8，压缩单元空间 |
| **Block Size 调优** | 16 是碎片与调度开销的平衡点 |
| **计算下沉** | 结合 FlashAttention-3，减少 HBM↔SRAM 搬运 |
| **KV Eviction** | 丢弃不重要位置的 KV（StreamingLLM, H2O） |
| **Prefix Caching** | 跨请求复用共享 prefix 的 KV |

## 总结

| 技术 | 解决的问题 |
|------|-----------|
| **KV Cache** | $O(n^2)$ 计算复杂度 → $O(n)$ |
| **PagedAttention** | $O(n)$ 空间管理失控 → 动态按需分配 |

## 前置知识

- [[注意力机制]] - KV Cache 缓存的是注意力计算的中间结果
- [[Transformer架构]] - 理解 decoder-only 自回归生成
- [[flash-attention]] - 计算层面的优化（两者互补）

## 应用场景

- 长对话聊天机器人
- 长文档问答
- 批量推理服务（vLLM, TensorRT-LLM）
- 多轮对话系统

## 参考资料

- vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention (Kwon et al., 2023)
- 李宏毅 ML 2026 Spring：加快語言模型生成速度(2/2)：KV Cache
- 大模型推理-Page attention 内存分页术（腾讯云开发者社区）
- H2O: Heavy-Hitter Oracle for Efficient LLM Decoding

---
*参考李宏毅 ML 2026 Spring 课程 · Nemo 整理*