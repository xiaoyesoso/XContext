# Plan: Agent Chat Frontend Demo

## Context

XContext 后端已完整实现上下文管理 API，但缺少可视化前端。用户希望有一个 Agent 对话 demo，把后端 API 真正用起来——展示用户消息如何成为上下文项、窗口如何被动态编排、压缩级别如何变化、Token 预算如何分配。

## 方案概述

单 HTML 文件 + Vue 3 (CDN)，由 FastAPI 直接作为静态文件提供。后端新增 `/chat` 端点串联完整 Agent 流程：用户输入 → 上下文编排 → LLM 调用 → 响应存储。

## 后端改动

### 1. `backend/app/main.py` — 添加 CORS + 静态文件 + chat 路由

- 添加 `CORSMiddleware`，允许所有源（demo 用途）
- 挂载 `StaticFiles` 到 `/static`，提供前端 HTML
- 添加根路由 `/` 重定向到 `index.html`
- 注册 `chat.router`

### 2. `backend/app/api/chat.py` — 新增 `/chat` 端点

**请求模型 `ChatRequest`：**
```python
session_id: str
message: str                    # 用户消息
strategy: WindowStrategy = DYNAMIC
max_tokens: int = 4096
task_state: Optional[TaskState] = None
token_budget: Optional[TokenBudget] = None
scenario: Optional[str] = None
```

**响应模型 `ChatResponse`：**
```python
session_id: str
reply: str                      # LLM 回复文本
items: list[ContextItem]        # 编排后的窗口项
prompt_fragment: str            # 组合的 prompt
total_tokens: int
item_count: int
budget_mode: Optional[BudgetMode]
metrics: Optional[dict]         # 从 metrics_collector 获取
```

**流程：**
1. 创建 `user_input` context item（`POST` 等价的 `context_service.create_item`）
2. 调用 `context_service.compose_window()` 编排上下文窗口
3. 用 `prompt_fragment` 作为 system context，调用 LLM（复用 `OpenAISummarizer` 的 `AsyncOpenAI` client，使用 `pro_llm_model`）
4. 创建 `model_output` context item 存储 LLM 回复
5. 从 `metrics_collector` 获取最新指标
6. 返回 `ChatResponse`

**LLM 调用细节：**
- 复用 `app.config.settings` 中的 `api_key`、`base_url`、`pro_llm_model`
- system prompt: `"You are a helpful assistant. Use the following context to answer the user's question.\n\n{prompt_fragment}"`
- user message: 原始用户消息
- 当 `SUMMARIZER_MODE=mock` 时，返回 mock 回复（不调用 LLM）

### 3. `backend/app/dependencies.py` — 添加 LLM client 依赖

- 新增 `get_llm_client()` 函数，返回 `AsyncOpenAI` 实例
- 或直接在 `chat.py` 中内联创建

## 前端 `backend/static/index.html`

单文件 Vue 3 应用，通过 CDN 加载 Vue 3 + Marked.js（Markdown 渲染）。

### 布局（三栏）

```
┌─────────────────────────────────────────────────┐
│  Header: XContext Agent Demo                    │
├──────────────────────┬──────────────────────────┤
│                      │  Config Panel            │
│  Chat Panel          │  - Strategy selector     │
│                      │  - Max tokens slider     │
│  [user] Hello        │  - Budget (total/reserved)│
│  [agent] Hi there!   │  - Scenario input        │
│                      │                          │
│  [user] Refund plz   │  Context Items           │
│  [agent] Sure! ...   │  - item cards with       │
│                      │    type/authority/       │
│                      │    compression/token_cost│
│  ┌──────────────┐    │                          │
│  │ Input...  [▶]│    │  Metrics                 │
│  └──────────────┘    │  - retrieved/selected/   │
│                      │    compressed counts     │
│                      │  - budget_mode badge     │
│                      │  - total/window tokens   │
│                      │                          │
│                      │  Prompt Preview          │
│                      │  (collapsible)           │
└──────────────────────┴──────────────────────────┘
```

### 功能

1. **Chat 面板**：
   - 消息列表（用户右侧蓝色，Agent 左侧灰色）
   - Markdown 渲染 Agent 回复
   - 输入框 + 发送按钮（Enter 发送）
   - 自动滚动到最新消息
   - 加载中状态指示

2. **配置面板**：
   - Strategy 下拉选择（sliding/summary/hybrid/dynamic）
   - Max tokens 滑块（512–16384）
   - 当 strategy=dynamic 时显示 Token Budget 输入（total, reserved）
   - Scenario 文本输入
   - Session ID 自动生成 + 显示

3. **上下文项列表**：
   - 每项显示：type badge（颜色区分）、content 截断、authority、compression_level（L0-L4 彩色标签）、token_cost
   - 按时间排序

4. **指标面板**：
   - retrieved_count / selected_count / compressed_count / ordered_count
   - budget_mode 彩色 badge（Full=绿, Balanced=蓝, Compact=橙, Minimal=红）
   - total_context_tokens / window_tokens 进度条

5. **Prompt 预览**：
   - 可折叠的 `<pre>` 块，显示 `prompt_fragment`

## 关键文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `backend/app/api/chat.py` | 新建 | `/chat` 端点 |
| `backend/app/models.py` | 修改 | 添加 `ChatRequest` / `ChatResponse` 模型 |
| `backend/app/main.py` | 修改 | CORS + 静态文件 + chat router |
| `backend/static/index.html` | 新建 | Vue 3 前端 demo |

## 验证方式

1. 启动服务：`cd backend && SUMMARIZER_MODE=real python -m uvicorn app.main:app --port 8765`
2. 浏览器打开 `http://localhost:8765/`
3. 发送消息，验证：
   - Chat 面板显示用户和 Agent 消息
   - Context 面板显示上下文项及压缩级别
   - Metrics 面板显示预算模式和 token 统计
   - 切换 strategy 后再次发消息，观察压缩级别变化
4. Mock 模式测试：`SUMMARIZER_MODE=mock` 启动，验证 UI 流程完整（Agent 返回 mock 文本）
5. Docker 部署：`docker compose up -d` 后访问 `http://localhost:8000/`
