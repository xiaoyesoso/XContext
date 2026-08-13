# XContext — 统一上下文窗口管理服务

[English](./README.en.md) | 简体中文

> **Context Window = f(Context)**

XContext 是一个 Python 后端服务，为 LLM Agent 系统提供统一的上下文管理抽象。它将记忆管理、用户画像、对话历史、工具结果、任务状态等异构上下文统一为标准模型，并通过可配置的管道动态编排上下文窗口。

## 核心理念

```
Window_t = Inject(
             Order(
               Compress(
                 Select(AllContext_t, TaskState_t),
                 TokenBudget_t
               )
             )
           )
```

服务将上下文窗口的构建分解为五个阶段：**Retrieve → Select → Compress → Order → Inject**，并引入 **Context Compiler** 概念——窗口不仅取决于全部上下文，还取决于当前任务状态（`TaskState_t`）和 Token 预算（`TokenBudget_t`）。

## 功能特性

### 窗口策略

| 策略 | 说明 |
|------|------|
| `sliding` | 滑动窗口：保留最近 N 条原始上下文，超出预算的旧条目被截断 |
| `summary` | 摘要窗口：每 N 轮生成一次摘要，旧条目替换为摘要项 |
| `hybrid` | 混合窗口：用户输入保持原始，模型输出按块摘要，保留最近几轮原始输出 |
| `dynamic` | 动态编排：基于任务状态和 Token 预算，按重要性×预算模式选择压缩级别，缓存感知排序 |

### 动态编排（Phase 7）

- **预算模式切换**：根据剩余 Token 比例自动切换 Full（>50%）/ Balanced（20-50%）/ Compact（5-20%）/ Minimal（<5%）四种模式
- **五级压缩（L0–L4）**：L0 丢弃 → L1 关键词 → L2 一句话 → L3 结构化摘要 → L4 原始保留，按重要性×预算模式查表决策
- **场景化压缩变体**：同一内容可预生成不同场景的压缩版本，运行时按场景选择
- **级联规则**：同一关联组的互补项保持同一压缩级别；重叠项可丢弃冗余
- **缓存感知排序**：稳定内容（约束、硬规则、已确认事实）前置构建 Prompt Prefix Cache 友好前缀，易变内容（用户输入、工具结果）后置
- **负向上下文**：用户拒绝的内容（`authority=denied`）转换为一句话"勿重复"提醒，防止 Agent 回退到已否定方案
- **失败历史反馈**：记录因缺失上下文导致的任务失败，后续同类任务自动提升相关上下文类型的优先级和最低压缩级别
- **多模式选择器**：规则选择器（默认）、关键词检索选择器、LLM 模型选择器（默认关闭）

### 分层上下文管理

| 层级 | 说明 |
|------|------|
| `working` | 当前步骤的活跃上下文 |
| `session` | 当前会话上下文 |
| `long_term` | 长期记忆（用户确认的事实、画像等） |
| `archive` | 冷归档（文件系统存储） |

支持可配置的晋升/降级规则，例如用户确认事实后自动从 `session` 晋升到 `long_term`。

## 技术栈

- **语言**：Python 3.11+
- **Web 框架**：FastAPI
- **数据校验**：Pydantic v2
- **关系存储**：SQLAlchemy 2.0 + Alembic
- **热缓存**：Redis
- **Token 估算**：tiktoken
- **LLM 集成**：OpenAI SDK 兼容（SiliconFlow 等）
- **测试**：pytest + httpx

## 快速开始

### 环境准备

```bash
cd backend
pip install -r requirements.txt
```

### 配置

在项目根目录创建 `.env` 文件：

```env
API_KEY=your_api_key_here
BASE_URL=https://api.siliconflow.cn/v1
FLASH_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
PRO_LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro
DATABASE_URL=sqlite+aiosqlite:///:memory:
REDIS_URL=redis://localhost:6379/0
```

> 测试环境下设置 `SUMMARIZER_MODE=mock` 可跳过真实 LLM 调用。

### 启动服务

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker 部署

```bash
# 默认使用 mock 摘要模式，无需真实 API Key
docker compose up -d

# 使用真实 LLM 调用（需在 .env 中配置 API_KEY）
SUMMARIZER_MODE=real docker compose up -d

# 查看服务日志
docker compose logs -f api

# 停止服务
docker compose down
```

Docker Compose 会自动启动 API 服务和 Redis，并将 SQLite 数据库持久化到 `db-data` 卷中。服务默认监听 `http://localhost:8000`。

### 运行测试

```bash
cd backend
python -m pytest tests/ -v
```

## API 接口

### 健康检查

```
GET /health
```

### 上下文项管理

```
POST   /context/items?session_id={session_id}     创建上下文项
GET    /context/items?session_id={session_id}     列出上下文项
GET    /context/items/{item_id}?session_id={id}   获取单个上下文项
DELETE /context/items/{item_id}?session_id={id}   删除上下文项
```

**创建上下文项示例：**

```json
{
  "type": "constraint",
  "content": "必须严格遵守退款政策",
  "source": "user",
  "scope": "current_task",
  "authority": "hard_rule",
  "priority": 10
}
```

### 窗口编排

```
POST /context/windows/compose
```

**基础请求（滑动窗口）：**

```json
{
  "session_id": "my-session",
  "strategy": "sliding",
  "max_tokens": 4096
}
```

**动态编排请求（含任务状态和 Token 预算）：**

```json
{
  "session_id": "my-session",
  "strategy": "dynamic",
  "max_tokens": 4096,
  "task_state": {
    "current_step": "process_refund",
    "goal": "处理退款请求",
    "progress": "正在检查退款政策",
    "missing_context": []
  },
  "token_budget": {
    "total": 32000,
    "reserved": 8000
  },
  "scenario": "refund"
}
```

**响应：**

```json
{
  "session_id": "my-session",
  "strategy": "dynamic",
  "items": [...],
  "prompt_fragment": "...",
  "total_tokens": 3200,
  "item_count": 12,
  "budget_mode": "full"
}
```

### 分层管理

```
GET  /context/layers                            列出配置的层级
POST /context/layers/{layer}/promote            晋升上下文项到下一层
     ?session_id={id}&item_id={item_id}
```

### 归档管理

```
POST /context/archive/{item_id}?session_id={id}    归档上下文项
GET  /context/archive/{item_id}?session_id={id}    获取归档项
GET  /context/archive?session_id={id}              列出归档项
```

### 指标查询

```
GET /metrics/{session_id}
```

返回检索数、选择数、压缩数、排序数、上下文总 Token、窗口 Token、预算模式等指标。

## 上下文项模型

| 字段 | 类型 | 说明 |
|------|------|------|
| `id` | str | 唯一标识（自动生成） |
| `type` | enum | `constraint` / `fact` / `tool_result` / `user_input` / `model_output` / `summary` / `profile` |
| `content` | any | 上下文内容 |
| `source` | enum | `user` / `agent` / `tool` / `internal` / `external` |
| `scope` | enum | `current_step` / `current_task` / `current_session` / `current_user` / `global` |
| `authority` | enum | `hard_rule` / `confirmed` / `inferred` / `assumed` / `denied` |
| `confidence` | float | 置信度 (0.0–1.0) |
| `priority` | int | 优先级（越高越重要） |
| `token_cost` | int | Token 成本（自动估算） |
| `layer` | str | 所在层级 |
| `version` | int | 版本号 |
| `compression_level` | enum | 压缩级别 `l0`–`l4`（压缩后填充） |
| `correlation_group` | str | 关联组标识（用于级联规则） |
| `expires_at` | datetime | 过期时间 |

## 压缩决策表

按重要性 × 预算模式选择压缩级别：

| 重要性 | Full (>50%) | Balanced (20-50%) | Compact (5-20%) | Minimal (<5%) |
|--------|-------------|-------------------|------------------|----------------|
| Critical | L4 | L4 | L4 | L4 |
| High | L4 | L4 | L3 | L2 |
| Medium | L4 | L3 | L2 | L1 |
| Low | L3 | L2 | L1 | L0 |

**重要性分类规则：**

- **Critical**：硬规则（`hard_rule`）、当前用户输入
- **High**：已确认事实（`confirmed`）、高优先级（`priority >= 10`）
- **Medium**：推断内容（`inferred`）、工具结果、用户画像
- **Low**：假设内容（`assumed`）、历史摘要

## 项目结构

```
XContext/
├── README.md                   # 中文 README（主文件）
├── README.en.md                # 英文 README
├── .env                        # 环境变量配置（不入库）
├── .gitignore
├── docker-compose.yml          # Docker Compose: API + Redis
├── AGENTS.md                   # AI 助手项目指南
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI 应用入口
│   │   ├── models.py           # Pydantic 数据模型
│   │   ├── db_models.py        # SQLAlchemy ORM 模型
│   │   ├── config.py           # 环境变量配置
│   │   ├── dependencies.py     # 依赖注入
│   │   ├── api/                # RESTful 路由
│   │   │   ├── health.py
│   │   │   ├── items.py
│   │   │   ├── windows.py
│   │   │   ├── layers.py
│   │   │   ├── metrics.py
│   │   │   └── archive.py
│   │   ├── core/               # 核心引擎
│   │   │   ├── engine.py       # ContextEngine 管道编排
│   │   │   ├── pipeline.py     # 管道阶段抽象与基础实现
│   │   │   ├── budget.py       # BudgetModeResolver 预算模式解析
│   │   │   ├── compression.py  # DynamicCompressor L0-L4 动态压缩
│   │   │   ├── ordering.py     # CacheAwareOrderer 缓存感知排序
│   │   │   ├── selectors.py    # 动态/检索/模型选择器
│   │   │   ├── failure_history.py  # 失败历史追踪
│   │   │   ├── layers.py       # LayerManager 分层管理
│   │   │   ├── summarizer.py   # 摘要生成器（Mock）
│   │   │   ├── llm.py          # OpenAI 兼容 LLM 摘要器
│   │   │   ├── tokenizer.py    # tiktoken Token 估算
│   │   │   ├── metrics.py      # 指标收集器
│   │   │   ├── database.py     # SQLAlchemy 数据库配置
│   │   │   └── logging_config.py
│   │   ├── services/
│   │   │   └── context_service.py  # 应用服务层
│   │   └── repositories/       # 存储仓库
│   │       ├── base.py         # 仓库抽象基类
│   │       ├── memory.py       # 内存仓库
│   │       ├── redis_repo.py   # Redis 仓库
│   │       ├── sql.py          # SQLAlchemy 仓库
│   │       ├── composite.py    # Redis + SQL 复合仓库
│   │       └── archive.py      # 文件系统归档仓库
│   ├── alembic/                # 数据库迁移脚本
│   │   ├── env.py
│   │   └── versions/
│   ├── alembic.ini
│   ├── tests/
│   │   ├── conftest.py         # 测试夹具
│   │   ├── test_api.py         # API 集成测试
│   │   ├── test_pipeline.py    # 管道单元测试
│   │   ├── test_dynamic.py     # 动态编排测试（Phase 7）
│   │   ├── test_layers.py      # 分层管理测试
│   │   └── test_archive.py     # 归档测试
│   ├── data/                   # SQLite 数据库目录（Docker 卷）
│   ├── Dockerfile
│   ├── .dockerignore
│   └── requirements.txt
├── extra_doc/                  # 外部参考文档（不入库）
└── openspec/                   # OpenSpec 变更/规格工作区（不入库）
    └── changes/
        └── context-window-service/
            ├── proposal.md
            ├── design.md       # 系统设计文档（SDD）
            ├── tasks.md
            └── specs/
                └── context-api/
                    └── spec.md
```

## 测试

```bash
cd backend

# 运行全部测试
python -m pytest tests/ -v

# 仅运行动态编排测试
python -m pytest tests/test_dynamic.py -v

# 运行并显示覆盖率
python -m pytest tests/ --cov=app --cov-report=term-missing
```

当前测试覆盖：53 项测试全部通过，涵盖 API 集成、管道各阶段、动态压缩、缓存排序、负向上下文、失败历史、17K 预算分配实战案例等。

## License

MIT
