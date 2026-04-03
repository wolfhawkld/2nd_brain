# MOC: 模型架构与深度学习

深度学习模型架构、核心概念与演进历程。

---

## 核心概念

### 基础组件
- [[激活函数]] — ReLU/Sigmoid/GELU，决定网络非线性能力
- [[损失函数]] — MSE/Cross-Entropy，训练的优化目标
- [[反向传播]] — 链式法则，梯度逐层传递
- [[梯度消失与梯度爆炸]] — 深层网络的训练挑战

### 学习范式
- [[在线学习]] — 流式数据，持续增量更新，省内存
- [[联邦学习]] — 数据不动模型动，隐私保护的分布式协作

### 网络架构
- [[卷积神经网络]] — 局部感受野，图像处理标配
- [[循环神经网络]] — 序列建模，LSTM/GRU
- [[Transformer架构]] — 自注意力，现代大模型基石
- [[MoE架构]] — 稀疏激活，参数效率与规模平衡

### 关键机制
- [[注意力机制]] — Self-Attention/Cross-Attention，动态关注
- [[位置编码]] — RoPE/ALiBi，为序列注入位置信息
- [[残差连接]] — 梯度高速公路，深层网络训练关键
- [[归一化层]] — BatchNorm/LayerNorm，稳定训练
- [[正则化]] — Dropout/Weight Decay，防止过拟合
- [[感受野]] — 卷积网络的视野范围

### 表示与生成
- [[词嵌入]] — Word2Vec/BERT，词的向量表示
- [[扩散模型]] — 去噪生成，图像生成新范式

---

## 模型演进

- [[transformer-evolution|Transformer 架构演化]] — 从 Encoder-Decoder 到 Decoder-only 主流
- [[decoder-dominance|Decoder 主导时代的成因]]
- [[scaling-laws|Scaling Laws]] — 规模与性能的幂律关系
- [[emergent-abilities|涌现能力]] — 规模超过阈值后突然出现的能力

---

## 架构类型

### Decoder-only
- GPT 系列、LLaMA、Claude

### Encoder-only
- BERT、RoBERTa

### Encoder-Decoder
- T5、BART、原版 Transformer

---

*整合: 2026-03-22 · Nemo + Outis*