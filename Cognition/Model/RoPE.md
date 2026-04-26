# RoPE（Rotary Position Embedding，旋转位置编码）

> RoPE 是一种把位置信息编码为“旋转”的 Transformer 位置编码方法。它不直接把位置向量加到 token embedding 上，而是在注意力计算前旋转 Query 和 Key，使 QK 内积天然包含相对位置信息。

---

## 直觉理解

Transformer 的注意力机制本身不知道 token 顺序：

```text
我 / 爱 / 你
你 / 爱 / 我
```

如果没有位置信息，Self-Attention 只看到一堆 token，无法知道谁在前、谁在后。

RoPE 的做法是：

> token 在第 m 个位置，就把它的 Q/K 向量按位置 m 旋转一个角度。

位置越靠后，旋转角度越大；不同维度对使用不同频率旋转。

这样两个 token 做注意力内积时，结果不只取决于内容相似度，也取决于二者的相对距离 m - n。

---

## 核心思想

传统位置编码通常是：

```text
x_m = token_embedding + position_embedding_m
```

RoPE 则是：

```text
q_m = R_m q
k_n = R_n k
attention_score = q_m^T k_n
```

其中 R_m、R_n 是由位置决定的旋转矩阵。

关键性质：

```text
(R_m q)^T (R_n k) = q^T R_{n-m} k
```

也就是说，注意力分数可以转化为只依赖相对位置 n-m 的形式。

这就是 RoPE 的核心价值：

> 用绝对位置的旋转实现相对位置的注意力依赖。

---

## 二维旋转形式

对一个二维向量 (x₁, x₂)，旋转角度 θ：

```text
[cos θ  -sin θ] [x₁]
[sin θ   cos θ] [x₂]
```

得到：

```text
x₁' = x₁ cos θ - x₂ sin θ
x₂' = x₁ sin θ + x₂ cos θ
```

RoPE 会把高维向量按两两一组拆开：

```text
(x₁, x₂), (x₃, x₄), ..., (x_{d-1}, x_d)
```

每一组用不同频率旋转：

```text
θ_i = base^{-2i/d}
位置 m 的旋转角 = m · θ_i
```

低维频率高，适合表达短距离变化；高维频率低，适合表达长距离变化。

---

## 为什么作用在 Q 和 K 上？

Self-Attention 的核心分数是：

```text
score(i, j) = q_i · k_j
```

RoPE 直接改造 q_i 与 k_j，使位置关系进入注意力分数。

它通常不作用在 V 上，因为：

- Q/K 决定“看谁”；
- V 决定“拿什么信息”；
- 位置信息主要影响注意力权重，而不是 value 内容本身。

所以 RoPE 常见实现是：

```text
Q_rot = apply_rope(Q, position)
K_rot = apply_rope(K, position)
Attention = softmax(Q_rot K_rot^T / sqrt(d)) V
```

---

## 与其他位置编码对比

| 方法 | 注入方式 | 相对位置能力 | 长度外推 | 参数量 |
|------|----------|--------------|----------|--------|
| Sinusoidal PE | 加到 embedding | 间接 | 一般 | 0 |
| Learned PE | 学习位置向量 | 弱 | 弱 | O(Ld) |
| Relative Position Bias | 加到 attention score | 强 | 取决于设计 | 有参数 |
| ALiBi | attention score 加线性偏置 | 强 | 很强 | 极少 |
| RoPE | 旋转 Q/K | 强 | 较强 | 0 |

RoPE 的优势在于：它既保留绝对位置编码的实现简洁性，又让 attention score 显式具有相对位置结构。

---

## 关键性质/特点

| 特点 | 说明 |
|------|------|
| 无额外可训练参数 | 旋转角由位置和频率规则生成 |
| 相对位置自然出现 | Q/K 内积依赖相对距离 m-n |
| 长度外推较好 | 比 learned position embedding 更适合外推 |
| 与 Attention 强耦合 | 直接改造 Q/K，而不是 embedding 本身 |
| LLM 标配 | LLaMA、GLM、PaLM、Qwen 等大量模型采用或改造 RoPE |

---

## 深度学习中的应用

### 1. 大语言模型位置编码

RoPE 已成为 decoder-only LLM 的常见默认选择。原因：

- 自回归生成只需要处理 Q/K 的位置旋转；
- 无需学习固定长度的位置表；
- 对长上下文扩展比 learned PE 更友好；
- 与 KV Cache 兼容。

### 2. 长上下文扩展

RoPE 虽然具备外推能力，但直接外推到远超训练长度时也会退化。因此现代长上下文模型常配合：

- RoPE scaling
- NTK-aware scaling
- YaRN
- LongRoPE
- partial RoPE

例如 DeepSeek-V4 技术报告中提到，在 CSA/HCA 中对 query、KV entry 以及 attention output 的部分维度应用 RoPE，以在压缩注意力中保留相对位置信息。

### 3. 多模态与二维位置

RoPE 的思想也可扩展到图像、音频等场景：

- 1D RoPE：文本序列
- 2D RoPE：图像 patch 的行列位置
- 多轴 RoPE：视频、空间-时间序列

---

## Python 实现（简化版）

下面是最小化 RoPE 实现，展示“按偶/奇维度成对旋转”的核心逻辑：

```python
import torch

def rotate_half(x):
    # x: [..., d]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

def apply_rope(x, positions, base=10000):
    # x: [batch, seq_len, dim]
    dim = x.shape[-1]
    device = x.device

    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=device).float() / dim))
    angles = positions[:, None].float() * inv_freq[None, :]  # [seq_len, dim/2]

    # 复制到偶/奇维度
    angles = torch.repeat_interleave(angles, repeats=2, dim=-1)  # [seq_len, dim]
    cos = angles.cos()[None, :, :]
    sin = angles.sin()[None, :, :]

    return x * cos + rotate_half(x) * sin

batch, seq_len, dim = 2, 8, 64
q = torch.randn(batch, seq_len, dim)
k = torch.randn(batch, seq_len, dim)
pos = torch.arange(seq_len)

q_rot = apply_rope(q, pos)
k_rot = apply_rope(k, pos)
```

实际工程中通常会缓存 cos/sin，以减少重复计算。

---

## 常见误区

| 误区 | 纠正 |
|------|------|
| RoPE 是加法位置编码 | 不是加法，而是旋转 Q/K |
| RoPE 只表达绝对位置 | 它用绝对位置旋转，但 QK 内积产生相对位置依赖 |
| RoPE 可以无限外推 | 直接外推会退化，长上下文常需 scaling 技术 |
| RoPE 会增加参数量 | 标准 RoPE 无可训练参数 |
| RoPE 作用在所有 QKV 上 | 常规做法只作用在 Q 和 K 上 |

---

## 记忆要点

| 要点 | 内容 |
|------|------|
| 一句话 | RoPE = 用旋转把位置写进 Q/K |
| 核心公式 | (R_m q)^T(R_n k)=q^T R_{n-m} k |
| 位置来源 | 位置 m 决定旋转角度 mθ |
| 相对位置 | 通过 QK 内积自然出现 m-n |
| 工程优势 | 无参数、适配 KV Cache、LLM 常用 |
| 长上下文 | 通常需要 RoPE scaling / partial RoPE 等改造 |

---

## 记忆口诀

> **RoPE 不加位置，RoPE 旋转位置；QK 一相乘，相对距离自然现。**

---

## 相关概念

- [[位置编码]] — RoPE 是现代位置编码的重要类型
- [[Transformer架构]] — RoPE 服务于 Transformer 注意力层
- [[注意力机制]] — RoPE 直接改变 Q/K 内积
- [[kv-cache|KV Cache]] — RoPE 需与自回归缓存位置一致
- [[flash-attention|Flash Attention]] — 高效注意力实现中常集成 RoPE kernel
- [[奇异值]] — 二者都涉及线性变换的几何直觉；RoPE 是保长度旋转，奇异值描述拉伸

---

## 参考资料

- Su et al., RoFormer: Enhanced Transformer with Rotary Position Embedding, arXiv:2104.09864
- EleutherAI Blog: Rotary Embeddings, A Relative Revolution
- HuggingFace RoFormer documentation
- DeepSeek-V4 技术报告中 partial RoPE 与 CSA/HCA 相关说明

---

*创建时间: 2026-04-26*
*来源: RoFormer 论文 + Transformer 位置编码资料 + DeepSeek-V4 技术报告上下文*
*分类: Cognition/Model/关键机制*
