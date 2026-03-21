# Transformer 架构演化

## 核心认知

**Transformer 从 Encoder-Decoder 起步，分化出三条路径，最终 Decoder-only 成为 LLM 主流。**

---

## 一、起源：Attention Is All You Need (2017)

### 论文信息

| 项目 | 内容 |
|-----|------|
| 标题 | Attention Is All You Need |
| 机构 | Google Brain (现 DeepMind) |
| 发表 | NeurIPS 2017 |
| 引用 | 10万+ 次 |

### 核心作者

| 作者 | 后续发展 |
|-----|---------|
| **Ashish Vaswani** | 第一作者，后加入 Adept AI |
| **Noam Shazeer** | 创立 Character.AI |
| **Niki Parmar** | Google Research |
| **Jakob Uszkoreit** | 创立 Inceptive |
| **Llion Jones** | 后加入 Sakana AI |
| **Aidan N. Gomez** | 创立 Cohere |
| **Łukasz Kaiser** | 后加入 OpenAI |
| **Illia Polosukhin** | 创立 NEAR Protocol |

### 原版架构

```
Transformer (2017)
├── Encoder (6层)
│   └── Multi-Head Self-Attention (双向) + Feed-Forward
│
└── Decoder (6层)
    ├── Masked Self-Attention (单向)
    ├── Cross-Attention (关注 Encoder 输出)
    └── Feed-Forward
```

**核心创新**：自注意力机制（Self-Attention）取代 RNN，实现完全并行化。

---

## 二、架构分化

### 时间线

| 年份 | 架构 | 模型 | 用途 |
|-----|------|------|------|
| **2017** | Encoder-Decoder | Transformer 原版 | 翻译 |
| **2018** | Encoder-only | BERT | 理解任务 |
| **2018** | Decoder-only | GPT-1 | 生成任务 |
| **2020** | Decoder-only | GPT-3 | LLM 时代开启 |
| **2022+** | Decoder-only | LLaMA、Claude、Gemini | LLM 主流 |

### 三种架构对比

| 架构 | 注意力方向 | 典型模型 | 擅长任务 |
|-----|----------|---------|---------|
| **Encoder-only** | 双向 | BERT、RoBERTa | 分类、NER、语义匹配 |
| **Decoder-only** | 单向（因果） | GPT、LLaMA、Claude | 生成、对话、推理 |
| **Encoder-Decoder** | 编码双向 + 解码单向 | T5、BART | 翻译、摘要 |

### 演化路径

```
Transformer (2017)
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 ▼
Encoder-only      Encoder-Decoder   Decoder-only
(BERT 2018)       (T5 2019)         (GPT 2018)
    │                 │                 │
    ▼                 ▼                 ▼
理解任务           翻译/摘要         生成任务
(分类/NER)                           │
                                      ▼
                              GPT-3 (2020)
                                      │
                                      ▼
                              LLM 时代主流
                           (LLaMA/Claude/Gemini)
```

---

## 三、为何 Decoder-only 胜出

详见 [[decoder-dominance|Decoder 主导时代的成因]]：

1. **生成是通用接口** — 自然语言可作为任何任务的输入输出
2. **训练效率** — 单向注意力计算更简单，易于并行
3. **扩展性** — 架构简洁，更容易 scale up
4. **涌现能力** — 大规模 decoder 展现出 encoder 难以具备的推理能力

---

## 四、与深度学习基础的关系

### 概念分层

| 概念 | 发明者 | 时间 | 层级 |
|-----|-------|------|------|
| **反向传播** | Hinton + Rumelhart + Williams | 1986 | 训练算法 |
| **深度学习基础** | Hinton 团队 | 1980s-2010s | 基础范式 |
| **Transformer 架构** | Google (Vaswani 等) | 2017 | 模型架构 |
| **自注意力机制** | 同上 | 2017 | 核心组件 |

### 贡献关系

```
Hinton 奠基
├── 反向传播 (1986) → 让深度网络训练成为可能
├── 深度信念网络 (2006) → 深度学习复兴
├── Dropout (2012) → 正则化技术
└── Capsule Network (2017)
        ↓
Google 团队
├── Transformer (2017) → 自注意力架构
└── 彻底改变 NLP
        ↓
现代 LLM
├── GPT 系列 (OpenAI)
├── LLaMA 系列 (Meta)
├── Claude (Anthropic)
└── Gemini (Google)
```

**比喻**：Hinton 是"修路的人"（训练算法），Transformer 团队是"造车的人"（模型架构）。

---

## 五、关键里程碑

| 时间 | 事件 | 意义 |
|-----|------|------|
| 2017 | Attention Is All You Need | Transformer 诞生 |
| 2018 | BERT | Encoder-only 成为主流理解模型 |
| 2018 | GPT-1 | Decoder-only 路线开启（被低估） |
| 2020 | GPT-3 | 涌现能力震惊业界，Decoder 成为 LLM 标准 |
| 2022 | ChatGPT | 对话式 AI 爆发 |
| 2023+ | LLaMA、Claude、Gemini | Decoder-only 统治 LLM |

---

## 六、反直觉点

1. **原版 Transformer 不是为 LLM 设计的** — 目标是翻译
2. **BERT 曾是主流** — 2018-2020 年 Encoder-only 更受关注
3. **GPT-1 被低估** — Decoder-only 早期被认为是"简单生成"
4. **Hinton 不是 Transformer 作者** — 他奠基了深度学习，但 Transformer 是 Google 团队的作品

---

## 相关认知

- [[decoder-dominance|Decoder 主导时代的成因]]
- [[scaling-laws|Scaling Laws]]
- [[emergent-abilities|涌现能力]]

## 参考

- Attention Is All You Need (Vaswani et al., 2017)
- BERT: Pre-training of Deep Bidirectional Transformers (2018)
- Improving Language Understanding by Generative Pre-Training (GPT-1, 2018)

---
*创建: 2026-03-21*