# XContext — Unified Context Window Management Service

English | [简体中文](./README.md)

> **Context Window = f(Context)**

XContext is a Python backend service that provides a unified context management abstraction for LLM Agent systems. It normalizes heterogeneous context sources — memory, user profiles, conversation history, tool results, and task state — into a standard model, and dynamically orchestrates the context window through a configurable pipeline.

## Core Concept

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

The service decomposes context window construction into five stages: **Retrieve → Select → Compress → Order → Inject**, and introduces the **Context Compiler** concept — the window depends not only on all available context but also on the current task state (`TaskState_t`) and token budget (`TokenBudget_t`).

## Features

### Window Strategies

| Strategy | Description |
|----------|-------------|
| `sliding` | Sliding window: keeps the most recent N raw items, truncating older ones beyond budget |
| `summary` | Summary window: generates summaries every N turns, replacing older items with summary items |
| `hybrid` | Hybrid window: keeps user inputs raw, summarizes model outputs in blocks, preserves recent raw outputs |
| `dynamic` | Dynamic orchestration: selects compression levels by importance × budget mode based on task state and token budget, with cache-aware ordering |

### Dynamic Orchestration (Phase 7)

- **Budget mode switching**: automatically switches between Full (>50%), Balanced (20-50%), Compact (5-20%), and Minimal (<5%) based on remaining token ratio
- **Five-level compression (L0–L4)**: L0 drop → L1 keywords → L2 one-sentence → L3 structured summary → L4 raw, decided by importance × budget mode lookup table
- **Scenario-dependent compression variants**: pre-build multiple compressed versions per scenario, select at runtime
- **Cascading rules**: complementary items in the same correlation group share a compression level; redundant items can be dropped
- **Cache-aware ordering**: stable content (constraints, hard rules, confirmed facts) placed first for Prompt Prefix Cache friendliness; volatile content (user input, tool results) placed last
- **Negative context**: user-rejected content (`authority=denied`) converted to one-sentence "do not repeat" reminders, preventing the Agent from reverting to discredited approaches
- **Failure history feedback**: records task failures caused by missing context, automatically elevates priority and minimum compression level for implicated context types in subsequent similar tasks
- **Multi-mode selectors**: rule-based (default), keyword retrieval, LLM model-based (off by default)

### Layered Context Management

| Layer | Description |
|-------|-------------|
| `working` | Active context for the current step |
| `session` | Current session context |
| `long_term` | Long-term memory (confirmed facts, user profile, etc.) |
| `archive` | Cold archive (filesystem storage) |

Supports configurable promotion/demotion rules, e.g., automatically promoting a confirmed fact from `session` to `long_term`.

## Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Data Validation**: Pydantic v2
- **Relational Storage**: SQLAlchemy 2.0 + Alembic
- **Hot Cache**: Redis
- **Token Estimation**: tiktoken
- **LLM Integration**: OpenAI SDK compatible (SiliconFlow, etc.)
- **Testing**: pytest + httpx

## Quick Start

### Prerequisites

```bash
cd backend
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
API_KEY=your_api_key_here
BASE_URL=https://api.siliconflow.cn/v1
FLASH_LLM_MODEL=deepseek-ai/DeepSeek-V4-Flash
PRO_LLM_MODEL=deepseek-ai/DeepSeek-V4-Pro
DATABASE_URL=sqlite+aiosqlite:///:memory:
REDIS_URL=redis://localhost:6379/0
```

> Set `SUMMARIZER_MODE=mock` in test environments to skip real LLM calls.

### Start the Service

```bash
cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker Deployment

```bash
docker-compose up -d
```

### Run Tests

```bash
cd backend
python -m pytest tests/ -v
```

## API Endpoints

### Health Check

```
GET /health
```

### Context Item Management

```
POST   /context/items?session_id={session_id}     Create a context item
GET    /context/items?session_id={session_id}     List context items
GET    /context/items/{item_id}?session_id={id}   Get a single context item
DELETE /context/items/{item_id}?session_id={id}   Delete a context item
```

**Create context item example:**

```json
{
  "type": "constraint",
  "content": "Must strictly follow the refund policy",
  "source": "user",
  "scope": "current_task",
  "authority": "hard_rule",
  "priority": 10
}
```

### Window Composition

```
POST /context/windows/compose
```

**Basic request (sliding window):**

```json
{
  "session_id": "my-session",
  "strategy": "sliding",
  "max_tokens": 4096
}
```

**Dynamic orchestration request (with task state and token budget):**

```json
{
  "session_id": "my-session",
  "strategy": "dynamic",
  "max_tokens": 4096,
  "task_state": {
    "current_step": "process_refund",
    "goal": "Handle refund request",
    "progress": "Checking refund policy",
    "missing_context": []
  },
  "token_budget": {
    "total": 32000,
    "reserved": 8000
  },
  "scenario": "refund"
}
```

**Response:**

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

### Layer Management

```
GET  /context/layers                            List configured layers
POST /context/layers/{layer}/promote            Promote a context item to the next layer
     ?session_id={id}&item_id={item_id}
```

### Archive Management

```
POST /context/archive/{item_id}?session_id={id}    Archive a context item
GET  /context/archive/{item_id}?session_id={id}    Retrieve an archived item
GET  /context/archive?session_id={id}              List archived items
```

### Metrics

```
GET /metrics/{session_id}
```

Returns retrieved count, selected count, compressed count, ordered count, total context tokens, window tokens, budget mode, and other metrics.

## Context Item Model

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique identifier (auto-generated) |
| `type` | enum | `constraint` / `fact` / `tool_result` / `user_input` / `model_output` / `summary` / `profile` |
| `content` | any | Context content |
| `source` | enum | `user` / `agent` / `tool` / `internal` / `external` |
| `scope` | enum | `current_step` / `current_task` / `current_session` / `current_user` / `global` |
| `authority` | enum | `hard_rule` / `confirmed` / `inferred` / `assumed` / `denied` |
| `confidence` | float | Confidence score (0.0–1.0) |
| `priority` | int | Priority (higher = more important) |
| `token_cost` | int | Token cost (auto-estimated) |
| `layer` | str | Layer name |
| `version` | int | Version number |
| `compression_level` | enum | Compression level `l0`–`l4` (populated after compression) |
| `correlation_group` | str | Correlation group ID (for cascading rules) |
| `expires_at` | datetime | Expiration time |

## Compression Decision Table

Compression level selected by importance × budget mode:

| Importance | Full (>50%) | Balanced (20-50%) | Compact (5-20%) | Minimal (<5%) |
|------------|-------------|-------------------|------------------|----------------|
| Critical | L4 | L4 | L4 | L4 |
| High | L4 | L4 | L3 | L2 |
| Medium | L4 | L3 | L2 | L1 |
| Low | L3 | L2 | L1 | L0 |

**Importance classification rules:**

- **Critical**: hard rules (`hard_rule`), current user input
- **High**: confirmed facts (`confirmed`), high priority (`priority >= 10`)
- **Medium**: inferred content (`inferred`), tool results, user profiles
- **Low**: assumed content (`assumed`), historical summaries

## Project Structure

```
XContext/
├── README.md                   # Chinese README (main)
├── README.en.md                # English README
├── .env                        # Environment config (not tracked)
├── .gitignore
├── docker-compose.yml
├── AGENTS.md                   # AI assistant project guide
├── backend/
│   ├── app/
│   │   ├── main.py             # FastAPI application entry
│   │   ├── models.py           # Pydantic data models
│   │   ├── config.py           # Environment configuration
│   │   ├── dependencies.py     # Dependency injection
│   │   ├── api/                # RESTful routes
│   │   │   ├── health.py
│   │   │   ├── items.py
│   │   │   ├── windows.py
│   │   │   ├── layers.py
│   │   │   ├── metrics.py
│   │   │   └── archive.py
│   │   ├── core/               # Core engine
│   │   │   ├── engine.py       # ContextEngine pipeline orchestration
│   │   │   ├── pipeline.py     # Pipeline stage ABCs and base implementations
│   │   │   ├── budget.py       # BudgetModeResolver
│   │   │   ├── compression.py  # DynamicCompressor L0-L4
│   │   │   ├── ordering.py     # CacheAwareOrderer
│   │   │   ├── selectors.py    # Dynamic/Retrieval/Model selectors
│   │   │   ├── failure_history.py  # Failure history tracker
│   │   │   ├── layers.py       # LayerManager
│   │   │   ├── summarizer.py   # Summarizer (Mock)
│   │   │   ├── llm.py          # OpenAI-compatible LLM summarizer
│   │   │   ├── tokenizer.py    # tiktoken token estimation
│   │   │   ├── metrics.py      # Metrics collector
│   │   │   ├── database.py     # SQLAlchemy database config
│   │   │   └── logging_config.py
│   │   ├── services/
│   │   │   └── context_service.py  # Application service layer
│   │   └── repositories/       # Storage repositories
│   │       ├── base.py         # Repository ABC
│   │       ├── memory.py       # In-memory repository
│   │       ├── composite.py    # Redis + SQL composite repository
│   │       └── archive.py      # Filesystem archive repository
│   ├── tests/
│   │   ├── test_api.py         # API integration tests
│   │   ├── test_pipeline.py    # Pipeline unit tests
│   │   ├── test_dynamic.py     # Dynamic orchestration tests (Phase 7)
│   │   ├── test_layers.py      # Layer management tests
│   │   └── test_archive.py     # Archive tests
│   ├── Dockerfile
│   └── requirements.txt
├── extra_doc/                  # External reference docs (not tracked)
└── openspec/                   # OpenSpec change/spec workspace (not tracked)
    └── changes/
        └── context-window-service/
            ├── proposal.md
            ├── design.md       # System Design Document (SDD)
            ├── tasks.md
            └── specs/
                └── context-api/
                    └── spec.md
```

## Testing

```bash
cd backend

# Run all tests
python -m pytest tests/ -v

# Run only dynamic orchestration tests
python -m pytest tests/test_dynamic.py -v

# Run with coverage
python -m pytest tests/ --cov=app --cov-report=term-missing
```

Current test coverage: 53 tests passing, covering API integration, pipeline stages, dynamic compression, cache-aware ordering, negative context, failure history, and the 17K budget allocation worked example.

## License

MIT
