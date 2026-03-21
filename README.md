# 2nd Brain - Damon's External Knowledge Base

体系化外脑，基于脑科学记忆系统设计。

## 顶级分类

| 分类 | 脑科学对应 | 内容 |
|-----|----------|------|
| **Cognition** | 语义记忆 | 概念、知识、思维模型 |
| **Skill** | 程序性记忆 | 技能、方法、流程 |
| **Memo** | 情景+工作记忆 | 日记、闪念、备查 |
| **Meta** | 元认知 | 方法论、复盘、体系本身 |
| **Horizon** | — | 探索、实验、待研究 |

## 目录结构

```
2nd_brain/
├── Cognition/           # 认知 - 知识与理解
│   ├── ai-research/     # AI 研究
│   ├── work/            # 工作相关
│   └── life/            # 生活常识
├── Skill/               # 技能 - 如何做事
│   ├── coding/          # 编程
│   ├── writing/         # 写作
│   └── management/      # 管理
├── Memo/                # 记录 - 保存与备忘
│   ├── daily/           # 日记
│   ├── fleeting/        # 闪念捕捉
│   └── reference/       # 备查资料
├── Meta/                # 元认知 - 关于认知的认知
│   ├── methodology/     # 方法论
│   └── reviews/         # 定期复盘
└── Horizon/             # 探索 - 未来与可能
    ├── questions/       # 待探索问题
    └── experiments/     # 实验记录
```

## 双链语法

使用 `[[note-name]]` 创建双向链接：

```markdown
今天研究了 [[RAG]]，参考了 [[Nemo]] 的建议...
```

Obsidian 会自动生成关系图谱。

## AI 协作

| AI | 机器 | 定位 |
|---|---|---|
| **Outis** | Sugarbox | 通用研究、知识管理 |
| **Nemo** | cube | 飞书、日常协作 |

日常通过 Git 异步同步，紧急事项 A2A 实时通知。

---
*由 Damon + Outis + Nemo 共同构建*