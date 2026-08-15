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

### 摘要与细节召回（Phase 8-10）

- **多类型摘要体系**：对话整体摘要、章节摘要（~20 轮）、关键事实（6 类：目标 / 硬约束 / 软偏好 / 确认-否认事实 / 决策 / 实体）、模型输出摘要、中间结果摘要，按粒度和生命周期分层存储
- **模型可读摘要**：高密度压缩，删除口语填充词，保留关键语义（目标、实体、事实、约束、决策），由轻量模型生成，显著降低 Token 消耗
- **异步摘要调度**：支持 end-of-turn、start-of-next-turn、watchdog 三种触发时机，通过 `asyncio.Event` 实现等待完成机制，避免摘要未就绪导致的竞态
- **K 轮原文窗口**：保留最近 K 轮（默认 K=10）原始上下文，解决摘要与检索库之间的同步延迟问题
- **混合检索召回**：基于关键词 + 向量的混合检索 + 重排序，从 ES / 向量库召回被压缩掉的细节
- **冲突裁决**：区分"强化"（互补，保留全部）与"冲突"（矛盾，择优），支持 last-write-wins（最新优先）和 authority precedence（权威优先）两种策略
- **迭代式细节召回**：LLM 评估上下文充分性 → 主动请求缺失细节 → 召回并合并 → 重试循环，含最大迭代次数和窗口溢出保护

### 用户画像（Phase 11）

- **五维画像**：目标（想达成什么）/ 能力（能理解什么）/ 偏好（喜欢讨厌什么）/ 决策（通常怎么选择）/ 关系（谁重要、关系如何），画像作为一等上下文项进入同一管道
- **显式厌恶转硬约束**：用户明确拒绝的内容（如"不要小米"）自动提升为 `hard_rule` 高权威约束项，注入时按约束处理
- **关系画像建模**：人物表 + 事件表双结构；事实与观点分离（`objective_fact` vs `user_interpretation`）；态度方向性独立存储；事件驱动信任更新（`trust↑/↓`）并保留证据链
- **电商品类偏好**：全局 / 品类 / 当前购物三层架构；订单价格百分位计算（"该用户电子品类通常买 top 30% 价格带"）；相似类目 fallback（裤子缺历史时参考上衣，置信度 ×0.6）
- **场景化加载**：按当前场景（退款 / 推荐 / 教育 / 社交）只加载相关画像子集，关系数据按提及加载，防止画像膨胀撑爆窗口
- **规格说明与可接受广告**：画像 + 当前请求推导结构化规格（价格区间、必需特性、排除品牌）；广告可小幅偏离规格但须落在可接受边界内（如 300-500 → 240-600，1200 拒绝）

### 分层上下文管理

| 层级 | 说明 |
|------|------|
| `working` | 当前步骤的活跃上下文 |
| `session` | 当前会话上下文 |
| `long_term` | 长期记忆（用户确认的事实、画像等） |
| `archive` | 冷归档（文件系统存储） |

支持可配置的晋升/降级规则，例如用户确认事实后自动从 `session` 晋升到 `long_term`。

### Agent 对话 Demo

内置一个基于 React 18 的前端 Demo（`frontend/index.html`），将后端 API 完整串联。右侧面板是**对话驱动的观察面板**：摘要提取与画像构造在每轮对话结束时后台自动进行（end_of_turn），摘要、画像事实、召回细节在下一轮开始时同步注入上下文——全程无需手动触发。

- **流式回复**：Agent 回复通过 SSE（Server-Sent Events）逐 token 推送，实现打字机效果
- **对话剧本引导**：内置客服退款 / 手机推荐两套剧本，覆盖 Full → Balanced → Compact → Minimal 四种预算模式，支持一键运行完整剧本
- **上下文窗口面板**：实时展示预算模式、检索/选择/压缩/排序计数、Token 用量条、分层架构与上下文项列表（类型/权威级别/压缩级别 L0–L4/Token 成本）
- **摘要与召回面板**：本轮同步注入报告（召回关键词、注入摘要类型、画像项数、召回数）、多类型摘要（每轮自动生成）、后台任务列表（end_of_turn 自动调度）、K 轮原始窗口统计、本轮自动召回结果（含命中关键词）
- **用户画像面板**：五维画像事实（对话中自动提取，显式厌恶自动转硬约束）、关系画像（人物 + 事件）、品类偏好（三层架构 + 相似类目回退）、推荐规格与可接受广告边界（推荐场景自动推导）
- **中英文切换**：默认中文，一键切换英文

#### 前端使用指引

1. 启动服务后，浏览器打开 `http://localhost:8765/`（自动跳转到 Demo 页面）
2. 在左侧输入框直接对话；或点击"显示剧本"选择剧本后，点击步骤填充或"运行完整剧本"自动演示
3. 对话过程中观察右侧三个面板自动更新：真实 LLM 模式下，对话结束后约 10–60 秒摘要/画像后台任务完成，面板自动刷新，全程无需点击任何触发按钮

![主界面与上下文窗口面板](docs/images/demo-chat-window.png)

![摘要与召回面板：本轮同步注入、自动摘要与后台任务](docs/images/demo-summaries.png)

![用户画像面板：五维画像与推荐规格/广告边界](docs/images/demo-profile.png)

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

### Agent 对话

```
POST /chat          非流式对话（等待完整回复后返回）
POST /chat/stream   流式对话（SSE 逐 token 推送）
```

**请求体：**

```json
{
  "session_id": "my-session",
  "message": "你好，我想退款",
  "strategy": "dynamic",
  "max_tokens": 4096,
  "token_budget": {
    "total": 32000,
    "reserved": 8000,
    "remaining": 5000
  },
  "scenario": "refund"
}
```

> `token_budget.remaining` 可选，用于模拟预算收紧场景。未提供时后端按 `total - reserved` 计算。

**非流式响应（`POST /chat`）：**

```json
{
  "session_id": "my-session",
  "reply": "您好！请问您的订单号是多少？",
  "items": [...],
  "prompt_fragment": "...",
  "total_tokens": 3200,
  "item_count": 5,
  "budget_mode": "balanced",
  "metrics": {...}
}
```

**流式响应（`POST /chat/stream`）：**

SSE 事件序列：

| 事件 | 说明 |
|------|------|
| `meta` | 流式开始前推送 prompt_fragment 和 budget_mode |
| `delta` | 逐块推送回复文本（多次） |
| `done` | 流式结束后推送 items 和 metrics |

```
data: {"type": "meta", "prompt_fragment": "...", "budget_mode": "full"}

data: {"type": "delta", "content": "你好"}

data: {"type": "delta", "content": "！"}

data: {"type": "done", "items": [...], "metrics": {...}}
```

### 用户画像

```
GET  /profiles/{user_id}                                画像摘要（人物 + 品类偏好）
POST /profiles/{user_id}/extract                        从会话上下文提取画像事实（LLM）
GET  /profiles/{user_id}/dimension/{dimension}          按维度查询画像项（需 session_id）
GET  /profiles/{user_id}/relationships/persons          人物列表
POST /profiles/{user_id}/relationships/persons          创建/更新人物
GET  /profiles/{user_id}/relationships/events           事件列表（可按 person_id 过滤）
POST /profiles/{user_id}/relationships/events           记录关系事件（自动更新态度）
POST /profiles/{user_id}/preferences/percentiles        订单历史价格百分位计算（离线）
GET  /profiles/{user_id}/preferences/{category_id}      品类偏好列表
POST /profiles/{user_id}/preferences/{category_id}      创建/更新品类偏好
GET  /profiles/{user_id}/preferences/{category_id}/price 价格偏好（含相似类目 fallback）
POST /profiles/{user_id}/recommendation-spec            推导推荐规格说明
POST /profiles/{user_id}/acceptable-ads                 计算可接受广告边界
POST /profiles/{user_id}/acceptable-ads/check           校验广告候选是否在边界内
```

**画像提取响应示例（`POST /profiles/{user_id}/extract`）：**

```json
[
  {
    "fact": {
      "dimension": "preference",
      "content": "不喜欢小米品牌",
      "is_dislike": true,
      "is_hard_requirement": true,
      "confidence": 0.95
    },
    "context_item_id": "76f0540c-..."
  }
]
```

> `is_dislike=true` 的画像事实会以 `authority=hard_rule`、`priority=10` 存为约束项，注入窗口时按硬约束处理。

**可接受广告边界示例（`POST /profiles/{user_id}/acceptable-ads`）：**

```json
{
  "spec": {"price_range": [300.0, 500.0], "excluded_brands": [], "...": "..."},
  "min_price": 240.0,
  "max_price": 600.0,
  "slack_ratio": 0.2
}
```

> 规格区间 300-500、松弛系数 0.2 时，广告价格落在 240-600 内可接受，1200 会被拒绝。

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
| `profile_dimension` | enum? | 画像维度：`goal` / `capability` / `preference` / `decision` / `relationship`（仅 profile 类型项） |
| `profile_tier` | enum? | 画像层级：`global` / `category` / `current_shopping`（仅 profile 类型项） |
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
├── frontend/
│   └── index.html              # 前端 Demo（React 18 单文件，Agent 对话界面）
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
│   │   │   ├── archive.py
│   │   │   ├── profiles.py      # 用户画像接口（人物/事件/品类偏好/广告）
│   │   │   └── chat.py          # Agent 对话接口（含 SSE 流式）
│   │   ├── core/               # 核心引擎
│   │   │   ├── engine.py       # ContextEngine 管道编排
│   │   │   ├── pipeline.py     # 管道阶段抽象与基础实现
│   │   │   ├── budget.py       # BudgetModeResolver 预算模式解析
│   │   │   ├── compression.py  # DynamicCompressor L0-L4 动态压缩
│   │   │   ├── ordering.py     # CacheAwareOrderer 缓存感知排序
│   │   │   ├── selectors.py    # 动态/检索/模型选择器
│   │   │   ├── failure_history.py  # 失败历史追踪
│   │   │   ├── key_facts.py    # 关键事实提取（6 类分类）
│   │   │   ├── model_readable.py   # 模型可读高密度压缩
│   │   │   ├── async_summary.py    # 异步摘要调度（3 种触发时机）
│   │   │   ├── detail_recall.py    # 细节召回 + K 轮原文窗口
│   │   │   ├── conflict_resolution.py  # 冲突裁决（最新优先/权威优先）
│   │   │   ├── iterative_recall.py # LLM 驱动的迭代式召回循环
│   │   │   ├── user_profile.py     # 五维画像提取（LLM + Mock）
│   │   │   ├── relationship_profile.py  # 关系画像（人物表 + 事件表）
│   │   │   ├── category_preference.py   # 品类偏好 + 价格百分位 + 相似类目
│   │   │   ├── profile_selector.py      # 场景化画像加载
│   │   │   ├── recommendation_spec.py   # 推荐规格 + 可接受广告边界
│   │   │   ├── layers.py       # LayerManager 分层管理
│   │   │   ├── summarizer.py   # 摘要生成器（Mock）
│   │   │   ├── llm.py          # OpenAI 兼容 LLM 摘要器
│   │   │   ├── tokenizer.py    # tiktoken Token 估算
│   │   │   ├── metrics.py      # 指标收集器
│   │   │   ├── database.py     # SQLAlchemy 数据库配置
│   │   │   └── logging_config.py
│   │   ├── services/
│   │   │   ├── context_service.py  # 应用服务层
│   │   │   └── profile_service.py  # 用户画像服务层
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
│   │   ├── test_summary_recall.py  # 摘要与细节召回测试（Phase 8-10）
│   │   ├── test_profile.py     # 用户画像测试（Phase 11）
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

# 仅运行摘要与细节召回测试
python -m pytest tests/test_summary_recall.py -v

# 仅运行用户画像测试
python -m pytest tests/test_profile.py -v

# 运行并显示覆盖率
python -m pytest tests/ --cov=app --cov-report=term-missing
```

当前测试覆盖：133 项测试全部通过，涵盖 API 集成、管道各阶段、动态压缩、缓存排序、负向上下文、失败历史、17K 预算分配实战案例，多类型摘要提取、模型可读压缩、异步摘要调度、K 轮原文窗口驱逐/召回、冲突裁决、迭代式召回循环，以及五维画像提取、关系画像事实/观点分离与方向性态度、品类价格百分位与相似类目 fallback、场景化加载、推荐规格推导、可接受广告过滤等。

## License

MIT
