# A2A 协议

Agent-to-Agent 通信协议，让 Outis 和 Nemo 无需人类中介即可对话。

## 技术实现

### 插件位置
`~/.openclaw/extensions/a2a/`

### 通信方式
- HTTP API (`/v1/chat/completions`)
- 文件共享 (`/a2a/files/*`)

### 配置
```json5
a2a: {
  enabled: true,
  peers: [{ id: "nemo", host: "192.168.0.110", port: 18789 }],
  fileShare: { enabled: true, basePath: "~/.openclaw/a2a-share" }
}
```

## 协作规则

### 触发通知
当一方发现以下情况时，通知另一方：
1. **知识库更新** - 生成了新文档
2. **重要发现** - Damon 研究相关关键信息
3. **系统变更** - 配置更新、新 skills

### 分工
| AI | 主负责 | 次负责 |
|---|---|---|
| Outis | 通用、技术研究 | 知识库维护 |
| Nemo | 飞书、日常协作 | 知识库同步 |

## 历史里程碑
- **2026-03-09**: A2A 首次成功通信
- **2026-03-13**: 双向对话测试成功

---
*创建: 2026-03-21*