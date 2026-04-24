# Muon Optimizer（Muon 优化器）

> 一种专门为神经网络隐藏层**2D参数（矩阵）**设计的优化器，通过 **Newton-Schulz 迭代**对动量矩阵进行**正交化**，实现更快的收敛速度。

---

## 背景与诞生

| 信息 | 内容 |
|------|------|
| **作者** | Keller Jordan |
| **发布时间** | 2024年12月（博客形式） |
| **后续** | 作者入职 OpenAI |
| **关键改进** | 月之暗面（2025年2月）开源改进版，算力需求减少48% |

**有趣的故事**：Keller 仅用一篇博客文章就获得 OpenAI offer，他直言"几乎所有优化器的论文都是假的"，坚持实际效果大于学术装饰。

---

## 核心原理

### 传统优化器的局限

Adam、AdamW 等优化器将所有参数**展平为向量**，使用向量范数约束更新：

```
问题：忽略了参数的矩阵结构
结果：更新方向不是目标函数下降最快的方向
```

### Muon 的突破

Muon 保留参数的**矩阵结构**，使用**谱范数**（矩阵2范数）约束更新：

```
优势：考虑了线性映射对输出的影响
约束：||ΔW||₂₂ ≤ ε → 控制输出变化量
```

---

## 算法步骤

给定权重矩阵 W ∈ R^{n×m}：

**Step 1: 计算动量**
```
M_t = β M_{t-1} + G_t
```
其中 G_t 是当前梯度，β 是动量系数。

**Step 2: 正交化（Newton-Schulz迭代）**
```
O_t = Newton-Schulz(M_t, T)
```
T 通常取 5 步，近似计算 msign(M_t)。

**Step 3: 更新权重**
```
W_{t+1} = W_t - η O_t
```

---

## msign 函数（矩阵符号函数）

Muon 的核心是 **msign**，即 sign 函数的矩阵推广：

**标量版本**：
```
sign(x) = x / |x| = x(x²)^{-1/2}
```

**矩阵版本**：
```
msign(M) = M(M^T M)^{-1/2}
```

这是通过 SVD 极分解得到的正交因子：
```
M = U Σ V^T  →  msign(M) = U V^T
```

---

## Newton-Schulz 迭代

为什么不直接用 SVD？

| 方法 | 精度 | 速度 |
|------|------|------|
| SVD | 精确 | 慢，计算成本高 |
| Newton-Schulz | 近似 | 快，仅涉及矩阵乘法 |

**Newton-Schulz 多项式**：

```
f(x) = ax + bx³ + cx⁵ + ...
```

使 x=1 成为吸引不动点：
- f(1) = 1
- |f'(1)| < 1

**迭代收敛**：对于 x ∈ (0,1]，多次迭代后收敛到 1。

---

## 实际使用

### 混合策略

Muon **只作用于 2D 参数**（矩阵），1D 参数仍用 AdamW：

```
网络参数分布：
├─ 2D 参数（隐藏层权重） → Muon
├─ 1D 参数（bias、LayerNorm） → AdamW
└─ 输入/输出层 → AdamW
```

### FLOP 开销

```
开销 = T × m / B
```

- T：NS 迭代步数（通常=5）
- m：模型维度
- B：批次 token 数

**示例**：
- NanoGPT：m=768, B=524288 → 开销 0.7%
- CIFAR-10：m=16384, B=16M → 开销 0.5%

---

## 性能表现

### 与 AdamW 对比

| 任务 | AdamW | Muon | 提升 |
|------|-------|------|------|
| NanoGPT 验证损失达 3.28 | 基准 | 1.35倍速度 | 35% |
| 1.5B 参数模型达 GPT-2 XL | 13.3小时 | 10小时 | 33% |
| 月之暗面 8B 模型 | 基准 | FLOP 52% | 48%节省 |

### Moonlight 模型

月之暗面基于 DeepSeek-V3-Small 架构，用改进版 Muon 训练：

```
Moonlight（MoE）
├─ 总参数：15.29B
├─ 激活参数：2.24B
├─ 训练 token：5.7T
└─ 性能：推进帕累托前沿
```

---

## 月之暗面改进

### 问题 1：更新幅度不一致

Muon 对形状 [A,B] 的矩阵，更新幅度为 `sqrt(1/max(A,B))`：

- MLP（宽矩阵）→ 更新过小
- Attention Head → 更新过大

**解决方案**：基于形状调整每个参数的学习率

### 问题 2：分布式训练不兼容

Muon 需要完整梯度矩阵，但现有框架（ZeRO-1、Megatron-LM）按元素切分。

**解决方案**：重新设计梯度聚合策略

---

## 数学洞察

### 范数视角

```
约束 ||ΔW||₂₂ ≤ ε
```

谱范数（矩阵2范数）约束 → 控制线性映射输出变化

### 与其他方法的关系

| 优化器 | 约束方式 |
|--------|----------|
| SGD | 向量 L2 范数 |
| Adam | 元素级自适应 |
| Shampoo | 行/列级 Fisher |
| Muon | 矩阵谱范数 |

**Muon ≈ 无累积的 Shampoo**（正交化梯度）

---

## Python 实现（简化版）

```python
import torch

def newton_schulz_orthogonalize(M, steps=5):
    """Newton-Schulz迭代近似正交化"""
    # 系数 (a, b) = (3.4445, 4.7750) 或 (1.5, -0.5)
    a, b = 3.4445, 4.7750
    
    for _ in range(steps):
        M = a * M + b * M @ M.T @ M
    return M

def muon_update(W, G, momentum_buffer, lr, beta=0.9):
    """Muon优化器单步更新"""
    # 1. 动量
    momentum_buffer = beta * momentum_buffer + G
    
    # 2. 正交化
    orthogonal_update = newton_schulz_orthogonalize(momentum_buffer)
    
    # 3. 更新
    W = W - lr * orthogonal_update
    return W, momentum_buffer
```

---

## 应用场景

| 场景 | 推荐 |
|------|------|
| 大规模 LLM 预训练 | ✅ Muon 高效 |
| Transformer 语言模型 | ✅ 已验证有效 |
| CNN（CIFAR-10） | ✅ 创造训练速度记录 |
| 小模型、快速实验 | ⚠️ AdamW 更简单 |
| 分布式训练（ZeRO/Megatron） | ⚠️ 需要额外适配 |

---

## 记忆要点

| 要点 | 内容 |
|------|------|
| **核心思想** | 保留矩阵结构，用谱范数约束 |
| **关键技术** | Newton-Schulz 迭代正交化 |
| **msign** | 矩阵版 sign 函数，提取正交因子 |
| **使用策略** | 2D参数用Muon，1D参数用AdamW |
| **开销** | FLOP 仅 0.5-0.7% |

---

## 记忆口诀

> **Muon = 矩阵谱范数 + 正交化更新 = 比AdamW快 = 月暗改进算力减半**

---

## 相关概念

- [[AdamW]] — 当前主流优化器，Muon 的对比基准
- [[SGD-Momentum]] — Muon 的基础动量计算
- [[Newton-Schulz迭代]] — 正交化的高效近似方法
- [[谱范数]] — 矩阵2范数，Muon 的约束方式
- [[正交矩阵]] — msign 提取的正交因子
- [[Shampoo]] — 类似思想的二阶优化器
- [[Moonlight]] — 月之暗面用 Muon 训练的 MoE 模型

---

## 参考资料

- Keller Jordan 博客：https://kellerjordan.github.io/posts/muon/
- GitHub：https://github.com/KellerJordan/Muon
- 月之暗面论文：Moonlight (https://github.com/MoonshotAI/Moonlight)
- 苏剑林分析：https://spaces.ac.cn/archives/10592
- 收敛性证明：Convergence of Muon with Newton-Schulz

---

*创建时间: 2026-04-21*
*来源: 深度学习优化器前沿研究*
*分类: Cognition/Math/优化理论*