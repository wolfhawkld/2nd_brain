# Transformer架构

## 概念

Transformer是一种基于自注意力机制的架构，摆脱了RNN的序列依赖，实现完全并行计算。是现代大语言模型的基础架构。

## 核心架构

### Encoder-Decoder结构

```
Encoder: 输入 → N层编码器 → 编码表示
Decoder: 编码表示 + 已生成 → N层解码器 → 输出
```

### 编码器层 (Encoder Layer)

```
输入 → Multi-Head Attention → Add & Norm → Feed Forward → Add & Norm → 输出
```

1. **多头自注意力**: 各位置互相注意
2. **前馈网络 (FFN)**: 逐位置非线性变换
3. **残差连接 + LayerNorm**: 稳定训练

### 解码器层 (Decoder Layer)

```
输入 → Masked Self-Attention → Add & Norm
    → Cross-Attention → Add & Norm
    → Feed Forward → Add & Norm → 输出
```

1. **Masked自注意力**: 不能看未来
2. **Cross-Attention**: Query来自解码器，K/V来自编码器
3. **前馈网络**

## 关键组件

### 多头注意力 (Multi-Head Attention)

```
MultiHead(Q,K,V) = Concat(head_1,...,head_h) * W^O
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

- 多个子空间并行学习
- 典型配置: 8头或16头

### 前馈网络 (FFN)

```
FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
```

- 两个线性层 + ReLU/GELU
- 中间维度通常是隐藏层的4倍

### 层归一化 (LayerNorm)

```
LN(x) = γ * (x - μ) / σ + β
```

- 稳定训练，适应变长序列
- **Pre-Norm vs Post-Norm**: Pre-Norm更稳定

### 位置编码 (Positional Encoding)

```
PE(pos, 2i) = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

- 为序列添加位置信息
- 现代变体: 可学习位置编码、RoPE、ALiBi

## 主要变体

### Encoder-only (BERT系列)
- 双向上下文理解
- **应用**: 文本分类、命名实体识别、语义匹配

### Decoder-only (GPT系列)
- 单向自回归生成
- **应用**: 文本生成、对话、代码生成

### Encoder-Decoder (T5, BART)
- 编码器理解，解码器生成
- **应用**: 翻译、摘要、问答

## 计算复杂度

| 操作 | 复杂度 | 说明 |
|-----|--------|------|
| 自注意力 | O(n²d) | 序列长度二次方 |
| FFN | O(nd²) | 隐藏层维度 |
| 总体 | O(n²d + nd²) | 序列长时注意力主导 |

## 与RNN对比

| 特性 | Transformer | RNN |
|-----|------------|-----|
| 并行性 | 完全并行 | 序列依赖 |
| 长距离依赖 | O(1)直接连接 | O(n)逐步传递 |
| 计算复杂度 | O(n²) | O(n) |
| 内存占用 | 高 | 低 |

## 扩展阅读

- **Vision Transformer (ViT)**: 图像patch化 + Transformer
- **Swin Transformer**: 层次化、滑动窗口注意力
- **Longformer/BigBird**: 稀疏注意力，支持长序列

## 相关概念

- [[注意力机制]] - Transformer的核心
- [[位置编码]] - 为无位置偏置的注意力添加位置信息
- [[残差连接]] - 深层网络训练的关键
- [[归一化层]] - LayerNorm稳定训练
- [[混合精度训练]] - 大模型训练必备
- [[梯度消失与梯度爆炸]] - Pre-Norm缓解的问题