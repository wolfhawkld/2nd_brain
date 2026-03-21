# 2nd Brain - Damon's External Knowledge Base

## 结构说明

```
2nd_brain/
├── 000-index/        # 入口、MOC (Map of Content)
├── 100-projects/     # 进行中的项目
├── 200-areas/        # 持续关注的领域（生活、工作、研究）
├── 300-resources/    # 参考资料、读书笔记
├── 400-archive/      # 已完成/归档
├── 500-daily/        # 日记、闪念
└── 600-ai-collab/    # AI 协作产出
    ├── outis/        # Outis (Sugarbox) 的贡献
    ├── nemo/         # Nemo (cube) 的贡献
    └── shared/       # 共同维护的内容
```

## AI 协作协议

### 参与者
| AI | 机器 | 主负责领域 |
|---|---|---|
| **Outis** | Sugarbox (WSL2) | 通用知识、技术研究 |
| **Nemo** | cube (WSL2) | 飞书研究、日常协作 |

### 更新流程
1. 一方更新内容 → Git commit & push
2. A2A 通知另一方 → "有新内容，请 pull"
3. 另一方 pull → 确认收到

### 双链语法
使用 `[[note-name]]` 创建双向链接。Obsidian 会自动生成关系图谱。

---

*由 Damon + Outis + Nemo 共同构建*