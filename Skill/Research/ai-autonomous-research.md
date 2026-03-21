# AI Agent 自主研究

## 核心技能

**将研究过程抽象为可执行的工作流，让 AI Agent 自主完成研究任务。**

人类角色从"执行者"转变为"策略设计者"——定义工作流、设定目标、监督结果。

## 代表性项目

### 1. Karpathy AutoResearch

**目标**：ML 训练自动优化

**架构**：极简（3 文件）
```
autoresearch/
├── train.py      # 固定的训练代码
├── program.md    # 策略文档（人类编写）
└── agent.py      # 自主实验循环
```

**工作流**：
```
改代码 → 训练 5min → 评估 → 改进/回滚 → 循环
```

**核心创新**：
- 人类写 `program.md`（策略），而非 `train.py`（代码）
- 固定时间窗口（5min），快速迭代
- 自动保留改进、回滚失败

**成果**：一夜 50 次实验，无需人类干预

### 2. AI超元域 ClawTeam

**目标**：通用任务的多 Agent 协作

**架构**：Manager/Worker
```
ClawTeam/
├── Manager Agent  # 任务拆解、分配、协调
└── Worker Agents  # 并行执行、独立上下文
```

**工作流**：
```
需求 → Manager 拆解 → 分配 Workers → 并行执行 → 协调 → 汇总
```

**核心创新**：
- 一句自然语言自动组建团队
- Worker 隔离运行，避免"记忆爆炸"
- 实时协调，像真正的团队

### 3. Research Navigator

**目标**：研究方向评估与分析

**架构**：Agent + Skills
```
research_navigator/
├── agents/
│   └── research-agent.md    # 协调 Agent
├── skills/
│   ├── research-search/     # 检索
│   ├── research-analyzer/   # 分析
│   ├── research-literature/ # 文献
│   ├── research-validator/  # 验证
│   └── research-reporter/   # 报告
└── shared-memory/           # 数据共享
```

**工作流模式**：

| 模式 | 流程 | 适用场景 |
|------|------|----------|
| 快速评估 | search → analyzer | "这个想法值得研究吗？" |
| 标准研究 | search → analyzer → literature → reporter | "研究一下 X" |
| 深度研究 | search → analyzer → literature → validator → reporter | "验证我的假设" |

**核心创新**：
- Agent 自主选择工作流模式
- Skills 模块化，可复用
- Shared-memory 实现数据共享

## 共同范式

### 核心模式
```
人类定义策略 → Agent 自主执行 → 迭代优化
```

### 关键设计原则

**1. 工作流抽象**
- 把研究过程拆解为可执行的步骤
- 每个步骤有明确的输入/输出
- 步骤之间通过数据传递连接

**2. Agent 自主决策**
- Agent 理解目标后自主选择行动
- 根据中间结果调整策略
- 在关键决策点可请求人类确认

**3. 模块化 Skills**
- 每个能力封装为独立 Skill
- Skill 可被多个 Agent 复用
- 便于扩展和维护

**4. 数据共享机制**
- Shared-memory / 文件系统
- 避免重复检索
- 保持上下文一致性

**5. 人类角色转变**
- 从"执行者"到"策略设计者"
- 从"写代码"到"写策略文档"
- 从"每步干预"到"关键点监督"

## 实践要点

### 何时使用

| 场景 | 适用性 |
|-----|--------|
| 重复性研究任务 | ✅ 高度适合 |
| 需要大量检索分析 | ✅ 高度适合 |
| 探索性研究 | ⚠️ 部分适合（需人类引导） |
| 创新性研究 | ⚠️ 部分适合（策略需迭代） |
| 高风险决策 | ❌ 需人类主导 |

### 设计工作流

1. **识别步骤**：研究过程有哪些环节？
2. **定义接口**：每个环节的输入/输出是什么？
3. **选择 Agent**：哪些步骤需要自主决策？
4. **设计 Skills**：每个能力如何封装？
5. **设定决策点**：哪些地方需要人类确认？

### 常见问题

**Q: Agent 能完全替代研究者吗？**

不能。Agent 擅长：
- 执行定义清晰的步骤
- 大规模检索和整理
- 重复性工作

人类仍需：
- 定义研究问题
- 设计创新方向
- 做高风险决策
- 验证结果的合理性

**Q: 如何保证研究质量？**

- 设计验证环节（如 research-validator）
- 设置评分机制和阈值
- 关键结论人类复核
- 保留完整过程记录

## 相关认知

- [[agent-workflow-design|Agent 工作流设计]]
- [[skill-modularization|Skill 模块化]]
- [[human-agent-collaboration|人机协作模式]]

## 参考

- [karpathy/autoresearch](https://github.com/karpathy/autoresearch)
- AI超元域 ClawTeam 视频教程
- Research Navigator 项目文档

---
*创建: 2026-03-21*