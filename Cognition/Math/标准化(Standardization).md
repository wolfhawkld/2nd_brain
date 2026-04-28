# 标准化（Standardization）

> 把**不同量纲的数据放到统一的框架里**看待——通过 Z-score 变换使特征具有均值为 0、标准差为 1 的分布，消除量纲差异的同时保留原始分布形态。

---

## 直觉理解

你的理解很精准：**标准化就是把不同量纲的数据放到统一的一个框架里**。

想象你要比较一个班级的成绩和身高：
- 成绩：0-100 分
- 身高：140-190 cm

这两个特征的量纲完全不同，如果直接用原始数值训练模型，身高（数值大）会在计算中占据主导地位。标准化的做法是：**不看绝对值，看"你比平均高多少、低多少"**。成绩 85 分如果比平均高 1.2 个标准差，身高 175 cm 如果也比平均高 1.2 个标准差，那么在这两个维度上，这个学生的"相对位置"是一样的。

> **核心思想**：标准化不是"拍扁到 [0,1]"，而是把每个数据点变成"偏离均值几个标准差"的分数——保留了"谁比谁大多少"的相对信息，但丢掉了绝对量纲。

---

## 核心定义/公式

### Z-score 标准化（最常用）

$$x' = \frac{x - \mu}{\sigma}$$

- $\mu$：特征列的均值
- $\sigma$：特征列的标准差
- 结果：均值为 0，标准差为 1（但不限制范围）

### 其他标准化方法

| 方法 | 公式 | 特点 |
|------|------|------|
| **Z-score** | $(x-\mu)/\sigma$ | 保留分布形态，对异常值相对鲁棒 |
| **Robust Scaler** | $(x - \text{median}) / \text{IQR}$ | 对异常值完全鲁棒 |
| **MaxAbs Scaler** | $x / \max(|x|)$ | 保持稀疏性，适合稀疏数据 |

---

## 关键性质/特点

| 性质 | 说明 |
|------|------|
| **均值为 0，标准差为 1** | 变换后的数据分布以原点为中心 |
| **保留分布形态** | 不改变原始分布的形状（偏度、峰度不变） |
| **线性变换** | 不改变数据之间的相对顺序和距离比例 |
| **对异常值相对鲁棒** | 相比归一化（用 max/min），标准化（用均值和标准差）受极端值影响较小 |
| **无界输出** | 输出范围不固定在 [0,1]，理论上可以取任意实数 |
| **依赖整体分布** | $\mu$ 和 $\sigma$ 会随样本变化，但通常比 max/min 更稳定 |

---

## 标准化 vs 归一化

> 你的理解"标准化是把不同量纲放到统一框架"正好点出了标准化与归一化的核心差异。

| 维度 | 标准化（Standardization） | 归一化（Normalization） |
|------|--------------------------|------------------------|
| **核心操作** | 以"偏离均值多少标准差"衡量 | 线性压缩到固定区间 |
| **公式** | $x' = (x-\mu)/\sigma$ | $x' = (x-x_{\min})/(x_{\max}-x_{\min})$ |
| **输出范围** | 无界（通常 -∞ 到 +∞） | 固定 [0, 1]（或 [-1,1]） |
| **分布保持** | ✅ 保留原始分布形态 | ❌ 可能改变分布形态 |
| **异常值敏感度** | 较低（受整体均值和标准差缓冲） | 很高（极端值会"挤压"正常值） |
| **适用条件** | 无严格范围要求，数据可能有异常值 | 已知数据边界，分布较均匀 |
| **稳定性** | 相对稳定（$\mu,\sigma$ 比 max/min 稳定） | 不稳定（新数据超出 [min,max] 需重新计算） |
| **比喻** | "第几名"——不看绝对成绩，看相对排名 | "百分比"——统一压缩到 0%~100% |

### 直观例子

数据：`[1, 2, 3, 4, 5, 100]`（100 是异常值）

- **归一化后**：`[0, 0.01, 0.02, 0.03, 0.04, 1]` → 正常值全部被"挤"到 0~0.04
- **标准化后**：大约 `[-0.8, -0.75, -0.7, -0.65, -0.6, 2.9]` → 正常值保持间距，异常值独显

---

## 机器学习（ML）中的应用

### 为什么很多 ML 算法需要标准化？

许多 ML 算法基于**距离**或**梯度**，量纲不同会导致：

1. **距离失真**：KNN、K-means、SVM（RBF核）中，大数值特征主导距离计算
2. **梯度不均**：线性/逻辑回归中，不同量纲导致梯度下降收敛慢（椭圆等高线 → 圆形等高线）
3. **正则化不公平**：L1/L2 正则化对不同量纲的特征施加相同惩罚，小数值特征被"过度惩罚"

### 必须标准化的算法

| 算法 | 为什么 |
|------|--------|
| **SVM（RBF核）** | 距离计算依赖特征尺度，量纲不同导致核函数偏向大数值特征 |
| **KNN / K-means** | 欧氏距离受量纲影响极大 |
| **PCA** | 方差解释受量纲影响——大数值特征会"劫持"主成分方向 |
| **逻辑回归 / 线性回归** | 梯度下降收敛速度受特征尺度影响（椭圆形误差面 → 圆形） |
| **Lasso / Ridge** | 正则化惩罚对不同尺度特征不公平 |
| **神经网络** | 不同量纲导致梯度爆炸/消失，收敛困难 |

### 不需要标准化的算法

| 算法 | 为什么 |
|------|--------|
| **决策树 / 随机森林** | 基于阈值分裂，只关心相对顺序，不关心绝对数值 |
| **XGBoost / LightGBM** | 同上，树模型不受量纲影响 |
| **朴素贝叶斯** | 基于条件概率独立性假设，理论上需要标准化，但实际影响有限 |

---

## 深度学习（DL）中的应用

### 数据标准化（输入层之前）

深度学习对输入尺度极其敏感：

- **梯度稳定性**：标准化输入使各维度的梯度在同一量级，避免某些参数更新过快/过慢
- **收敛加速**：输入以原点为中心、方差为 1 → 激活函数（如 tanh、sigmoid）工作在敏感区
- **权重初始化**：He/Xavier 初始化假设输入已标准化，否则初始化假设失效

### 内部标准化（网络层之间）

深度学习中"标准化"的另一层含义是网络内部的归一化层：

| 方法 | 作用维度 | 场景 |
|------|---------|------|
| **Batch Normalization** | 对 batch 内每个特征标准化 | CNN、前馈网络 |
| **Layer Normalization** | 对每个样本的特征维度标准化 | Transformer、RNN |
| **Instance Normalization** | 对每个样本的每个通道标准化 | 风格迁移 |
| **Group Normalization** | 对每组通道标准化 | 小 batch 场景 |

> **注意**：这些"归一化层"实际做的是 Z-score 标准化（减去均值除以标准差 + 可学习的缩放和平移），名字叫 Normalization 但数学上是 Standardization。

---

## 强化学习（RL）中的应用

RL 中标准化比监督学习更重要，因为数据分布会随策略变化而漂移（non-stationary）。

### 1. 状态标准化（State Normalization）

$$
s' = \frac{s - \mu_{\text{running}}}{\sigma_{\text{running}}}
$$

- 使用**运行时均值和标准差**（running mean/std），因为 RL 数据是流式的
- 消除不同状态维度间的量纲差异（如位置 vs 速度 vs 角度）
- 在复杂环境（如 MuJoCo、IsaacGym）中几乎是**标配**

### 2. 奖励标准化（Reward Normalization / Scaling）

$$
r' = \frac{r}{\sigma_{\text{running}}}
$$

- 仅除以标准差（不减均值），因为保留奖励的正负号很重要
- 防止 reward scale 过大导致梯度爆炸
- 也常用固定裁剪：$r = \text{clip}(r, -1, 1)$（如 DQN 的 Atari 实验）

### 3. 优势函数标准化（Advantage Normalization）

PPO 中关键的一步：

$$
\hat{A}' = \frac{\hat{A} - \mu_{\text{batch}}}{\sigma_{\text{batch}}}
$$

- 在一个 mini-batch 内对 advantage 做 Z-score 标准化
- 使优势值均值为 0，防止策略更新偏向某个方向
- PPO 原始论文中的默认做法

### 4. 观测标准化（Observation Normalization）

在连续控制任务中（如机器人），传感器读数（关节角度、力矩、IMU 数据）量纲差异大，标准化的观测是训练稳定的前提。

---

## Python 实现

```python
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ===== 手动实现 Z-score 标准化 =====
def z_score_normalize(X):
    """X: (n_samples, n_features)"""
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mu) / sigma


# ===== 使用 sklearn =====
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # 计算 μ,σ 并变换
X_test_scaled  = scaler.transform(X_test)         # 用训练集的 μ,σ 变换测试集
# 注意：测试集不能用 fit_transform！必须用训练集的 scaler


# ===== RL 中运行时标准化 =====
class RunningMeanStd:
    """用于 RL 的运行时均值和标准差追踪"""
    def __init__(self, shape=(), epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


# ===== 完整对比：标准化 vs 归一化 =====
data = np.array([[1], [2], [3], [4], [5], [100]])  # 含异常值 100

# 标准化
std_scaler = StandardScaler()
print("Z-score 标准化:", std_scaler.fit_transform(data).ravel())
# ≈ [-0.81, -0.77, -0.73, -0.69, -0.65, 3.65]

# 归一化 [0,1]
minmax = MinMaxScaler()
print("Min-Max 归一化:", minmax.fit_transform(data).ravel())
# ≈ [0.0, 0.01, 0.02, 0.03, 0.04, 1.0]
# → 正常值 1~5 被 "挤" 到了 0~0.04，几乎无法区分！
```

---

## 记忆要点

| 要点 | 内容 |
|------|------|
| **本质** | 把不同量纲的数据放到统一框架——以"偏离均值多少标准差"统一衡量 |
| **公式** | $x' = (x-\mu)/\sigma$ |
| **结果** | 均值 0，标准差 1 |
| **vs 归一化** | 标准化是"第几名"（保留相对间距）；归一化是"百分比"（压缩到 [0,1]） |
| **ML 中谁需要** | 基于距离/梯度的算法（SVM、KNN、K-means、PCA、LR、NN） |
| **ML 中谁不需要** | 树模型（决策树、XGBoost） |
| **DL 中** | 输入标准化加速收敛；BN/LN 内部标准化稳定训练 |
| **RL 中** | State/Reward/Advantage 标准化都是标配，且用 running mean/std |
| **关键陷阱** | 测试集必须用训练集的 μ 和 σ，不能自己 fit！ |

---

## 记忆口诀

> **标准就是「比均高几 σ」——减均值除标准差，零心一宽保留形。**

---

## 相关概念

- [[归一化(Normalization)]] — 同一家族的另一分支：压缩到 [0,1] 区间
- [[Batch Normalization]] — DL 内部标准化层，数学上就是 Z-score + 可学习参数
- [[梯度下降(Gradient Descent)]] — 标准化后梯度等高线从椭圆变圆形，收敛加速
- [[特征工程(Feature Engineering)]] — 标准化属于特征缩放（Feature Scaling）
- [[主成分分析(PCA)]] — PCA 前必须标准化，否则大数值特征劫持主成分
- [[协方差与相关系数]] — 标准化后协方差 = 相关系数

---

*创建时间: 2026-04-29*
*来源: 综合阿里云开发者社区、知乎、论文实践 + Damon 的直觉理解*
*分类: Cognition/Math*
