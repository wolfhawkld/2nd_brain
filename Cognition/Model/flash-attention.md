# Flash Attention

## 定义

Flash Attention 是一种高效的注意力计算算法，通过 **代理值技巧 + 分块计算（Tiling）** 将 softmax 从 3-pass 算法压缩到 1-pass，显著减少 Transformer 模型在长序列上的内存读写次数。

## 核心原理：从 3-pass 到 1-pass

### 标准 Softmax：3-pass 算法

传统 softmax 需要遍历三次所有元素：

| 趟数 | 操作 | 说明 |
|-----|------|------|
| 第1趟 | 求 $d = \max(a_i)$ | 需遍历全部元素 |
| 第2趟 | 求 $s = \sum e^{a_i - d}$ | 需遍历全部元素 |
| 第3趟 | 计算 $y_i = \frac{e^{a_i - d}}{s}$ | 得到最终结果 |

公式：
$$y_i = \frac{e^{a_i - d}}{\sum_{j=1}^{L} e^{a_j - d}}$$

其中减去 $d$（最大值）是为了数值稳定，避免指数溢出。

### 引入代理值：2-pass 算法

关键优化：引入**代理值** $s_k$，不再需要先遍历得到最大值。

定义：
- $d_k$：第 k 个 chunk 的最大值
- $s_k$：第 k 个 chunk 的指数和代理值

**代理值的递推关系**：
$$s_k = s_{k-1} \cdot e^{d_{k-1} - d_k} + \sum_{i=kN+1}^{(k+1)N} e^{a_i - d_k}$$

这个公式的关键：当遍历完所有 chunks 后，$s_{\text{last}}$ 等于真实的指数和。

**2-pass 算法流程**：
| 趟数 | 操作 |
|-----|------|
| 第1趟 | 同时求 $d_k$ 和 $s_k$（边遍历边更新） |
| 第2趟 | 计算最终值 $y_i = \frac{e^{a_i - d}}{s_{\text{last}}}$ |

### Flash Attention：1-pass 算法

对于 attention 计算，还可以再引入一个代理值 $o_k$，将 2-pass 压缩为 1-pass。

定义：
- $o_k$：第 k 个 chunk 的输出代理值

Attention 公式（第 k 个 chunk）：
$$o_k = \sum_{i \in \text{chunk}_k} \frac{e^{a_i - d_k}}{s_k} \cdot v_i$$

**输出代理值的递推关系**：
$$o_k = o_{k-1} \cdot \frac{s_{k-1}}{s_k} \cdot e^{d_{k-1} - d_k} + \sum_{i=kN+1}^{(k+1)N} \frac{e^{a_i - d_k}}{s_k} \cdot v_i$$

**关键性质**：
- 代理值当前 chunk 和前一 chunk 存在递推关系
- 当遍历完所有 chunks 后，$o_{\text{last}}$ 就是最终输出

**1-pass 算法流程**：
一次遍历中同时更新：
- $d_k$（当前 chunk 最大值）
- $s_k$（指数和代理值）
- $o_k$（输出代理值）

遍历结束后，$o_{\text{last}}$ 就是最终输出，**一步到位**。

## 分块实现

### 为什么需要分块？

序列长度 $L$ 很大时，单次无法载入全部数据。利用矩阵分块：

- Q、K、V、O 都进行分块（chunk size = $N$）
- 每次载入一个 chunk，在 SRAM 中完成计算
- 关键：**整个过程只和 chunk size $N$ 有关，和序列长度 $L$ 完全解耦**

### 分块后的算法

对于每个 Q 的 chunk：
1. 遍历所有 K、V 的 chunks
2. 每个 chunk 内：
   - 先求局部最大值 $d_{\text{local}}$
   - 再与历史最大值比较更新 $d_{\text{new}} = \max(d_{\text{old}}, d_{\text{local}})$
   - 用递推公式更新 $s_k$ 和 $o_k$

**内存交互**：
- 输入：从 HBM 读取 Q、K、V 分块
- 输出：直接写入 HBM 的 O 分块
- 中间结果：不缓存，在 SRAM 中完成

## 性能对比

| 指标 | 传统 Attention | Flash Attention |
|------|---------------|-----------------|
| 算法趟数 | 3-pass | 1-pass |
| 内存占用 | $O(L^2)$（注意力矩阵） | $O(L)$ |
| HBM 读写次数 | 多次中间读写 | 一次读入、一次写出 |
| 计算类型 | Memory-bound | Compute-bound |

## 版本演进

| 版本 | 优化点 |
|------|-------|
| **Flash Attention v1** | 代理值 + 分块计算 |
| **Flash Attention v2** | 减少 non-matmul 计算；增加 seqlen 维度并行；Warp Partitioning 优化 |
| **Flash Attention v3** | FP8 支持；H100 异步优化 |

## 更广泛的 Streaming Reduction 模式

代理值技巧不仅适用于 softmax，很多"全局依赖"算子都可以用类似方法改写：

| 算子 | 代理值 | 递推关系 |
|------|-------|---------|
| Online Softmax | $s_k$（指数和） | $s_k = s_{k-1} \cdot e^{d_{k-1} - d_k} + \sum e^{a_i - d_k}$ |
| LayerNorm | $\tilde{\mu}_k$, $\tilde{\sigma}_k$ | Welford 算法 |
| Attention Output | $o_k$ | $o_k = o_{k-1} \cdot \frac{s_{k-1}}{s_k} \cdot e^{d_{k-1} - d_k} + ...$ |

**判断标准**：如果算子满足：
- 可分解为局部贡献
- 存在有限维状态（与数据规模无关）
- 状态可合并（有递推关系）

就可以压缩为 streaming/tiling 版本。

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
- 从 Online Softmax 到 FlashAttention（腾讯云开发者社区）

---
*参考李宏毅 ML 2026 Spring 课程 · Nemo 整理*