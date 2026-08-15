# AGENTS.md — Project Guide for AI Assistants

## 1. Project Overview

XContext is a Python backend service that implements the unified context-management abstraction for LLM Agent systems:

> Context Window = f(Context)

The service exposes RESTful APIs for context ingestion, layered context management, and configurable context-window composition strategies (sliding window, summary-based window, hybrid raw/summary, and dynamic orchestration with budget-mode-aware compression and cache-aware ordering). It also includes a multi-type summary subsystem (conversation/chapter/key-facts/model-readable summaries), async summary scheduling, detail recall with a K-turn raw window, conflict resolution, an iterative recall loop driven by LLM sufficiency evaluation, and a user-profile subsystem (five dimensions, relationship person/event tables, three-tier e-commerce preferences, scenario-aware loading, recommendation specs with acceptable-ad boundaries). An Agent chat endpoint with SSE streaming and a Vue 3 frontend demo is included.

## 2. Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Data Validation**: Pydantic v2
- **Relational Persistence**: SQLAlchemy 2.0 + Alembic
- **Hot Cache**: Redis
- **Token Estimation**: tiktoken
- **LLM Integration**: OpenAI SDK-compatible models (SiliconFlow, etc.)
- **Testing**: pytest + httpx
- **Containerization**: Docker + Docker Compose
- **Spec / Design Workflow**: OpenSpec (`openspec` CLI)

## 3. Development Conventions

### 3.1 Code Comments

All code comments and docstrings MUST be written in **English**.

```python
# Good: English comment
# Bad: 中文注释
```

### 3.2 Language for User-facing Artifacts

User-facing documents (OpenSpec artifacts, READMEs, design docs) should be written in **Chinese** unless the user explicitly asks for English.

### 3.3 Code Style

- Follow PEP 8.
- Use type hints where practical.
- Prefer `pathlib` over raw string paths.
- Keep functions focused and small.
- Avoid premature abstraction.

## 4. Workflow

### 4.1 OpenSpec-driven Changes

All non-trivial changes MUST go through the OpenSpec workflow:

1. Create or reuse a change directory under `openspec/changes/<change-name>/`.
2. Required artifacts (in order):
   - `proposal.md` — motivation and scope
   - `specs/<capability>/spec.md` — requirements with `## ADDED Requirements` and `#### Scenario:` blocks
   - `design.md` — technical design document (SDD)
   - `tasks.md` — implementation tasks
3. Validate the change before implementation:
   ```bash
   openspec validate --changes <change-name> --json
   ```
4. Implement the tasks in `tasks.md`.
5. Run tests locally before considering the change complete.

### 4.2 Directory Layout

```text
XContext/
├── AGENTS.md                   # This file
├── README.md                   # Chinese README (main)
├── README.en.md                # English README
├── .gitignore                  # Ignores extra_doc/ and openspec/
├── .env                        # Environment config (NOT tracked)
├── docker-compose.yml          # Docker Compose: API + Redis
├── backend/                    # Python backend service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py             # FastAPI application entry
│   │   ├── models.py           # Pydantic schemas (ContextItem, TaskState, etc.)
│   │   ├── db_models.py        # SQLAlchemy ORM models
│   │   ├── config.py           # Environment configuration (pydantic-settings)
│   │   ├── dependencies.py     # Dependency injection
│   │   ├── api/                # RESTful routes
│   │   │   ├── health.py
│   │   │   ├── items.py
│   │   │   ├── windows.py
│   │   │   ├── layers.py
│   │   │   ├── metrics.py
│   │   │   ├── archive.py
│   │   │   ├── profiles.py     # User profile endpoints (persons/events/preferences/ads)
│   │   │   └── chat.py         # Agent chat endpoint (SSE streaming)
│   │   ├── core/               # Core engine
│   │   │   ├── engine.py       # ContextEngine pipeline orchestration
│   │   │   ├── pipeline.py     # Pipeline stage ABCs and base implementations
│   │   │   ├── budget.py       # BudgetModeResolver (Full/Balanced/Compact/Minimal)
│   │   │   ├── compression.py  # DynamicCompressor L0-L4
│   │   │   ├── ordering.py     # CacheAwareOrderer
│   │   │   ├── selectors.py    # Dynamic/Retrieval/Model selectors
│   │   │   ├── failure_history.py  # Failure history tracker
│   │   │   ├── key_facts.py    # KeyFact extractor (6-category classification)
│   │   │   ├── model_readable.py   # Model-readable high-density compressor
│   │   │   ├── async_summary.py    # Async summary scheduler (3 trigger modes)
│   │   │   ├── detail_recall.py    # Detail retriever + K-turn raw window
│   │   │   ├── conflict_resolution.py  # Conflict resolver (last-write-wins / authority)
│   │   │   ├── iterative_recall.py # LLM-driven iterative recall loop
│   │   │   ├── user_profile.py     # Five-dimension profile extractor (LLM + Mock)
│   │   │   ├── relationship_profile.py  # Relationship profiles (person + event tables)
│   │   │   ├── category_preference.py   # Category prefs + price percentile + sibling fallback
│   │   │   ├── profile_selector.py      # Scenario-aware profile loading
│   │   │   ├── recommendation_spec.py   # Recommendation spec + acceptable-ad boundary
│   │   │   ├── layers.py       # LayerManager
│   │   │   ├── summarizer.py   # Summarizer (Mock)
│   │   │   ├── llm.py          # OpenAI-compatible LLM summarizer
│   │   │   ├── tokenizer.py    # tiktoken token estimation
│   │   │   ├── metrics.py      # Metrics collector
│   │   │   ├── database.py     # SQLAlchemy database config
│   │   │   └── logging_config.py
│   │   ├── services/
│   │   │   ├── context_service.py  # Application service layer
│   │   │   └── profile_service.py  # User profile service layer
│   │   └── repositories/       # Storage repositories
│   │       ├── base.py         # Repository ABC
│   │       ├── memory.py       # In-memory repository
│   │       ├── redis_repo.py   # Redis repository
│   │       ├── sql.py          # SQLAlchemy repository
│   │       ├── composite.py    # Redis + SQL composite repository
│   │       └── archive.py      # Filesystem archive repository
│   ├── static/                 # Frontend demo (Vue 3 single-file)
│   │   └── index.html          # Agent chat UI
│   ├── alembic/                # Database migration scripts
│   │   ├── env.py
│   │   └── versions/
│   ├── alembic.ini
│   ├── tests/
│   │   ├── conftest.py         # Test fixtures
│   │   ├── test_api.py         # API integration tests
│   │   ├── test_pipeline.py    # Pipeline unit tests
│   │   ├── test_dynamic.py     # Dynamic orchestration tests (Phase 7)
│   │   ├── test_summary_recall.py  # Summary & detail recall tests (Phase 8-10)
│   │   ├── test_profile.py     # User profile tests (Phase 11)
│   │   ├── test_layers.py      # Layer management tests
│   │   └── test_archive.py     # Archive tests
│   ├── data/                   # SQLite database directory (Docker volume)
│   ├── Dockerfile
│   ├── .dockerignore
│   └── requirements.txt
├── extra_doc/                  # External reference documents (NOT tracked by git)
└── openspec/                   # OpenSpec change/spec workspace (NOT tracked by git)
    ├── config.yaml
    └── changes/
        └── context-window-service/
            ├── proposal.md
            ├── design.md       # System Design Document (SDD)
            ├── tasks.md
            ├── specs/context-api/spec.md
            └── assets/
```

## 5. Testing and Validation

- Run Python syntax checks:
  ```bash
  python3 -m py_compile backend/app/*.py backend/app/core/*.py
  ```
- Run the test suite:
  ```bash
  cd backend && python -m pytest tests/ -v
  ```
- Validate OpenSpec changes:
  ```bash
  openspec validate --changes <change-name> --json
  ```

Current test coverage: 133 tests passing, covering API integration, pipeline stages, dynamic compression, cache-aware ordering, negative context, failure history, the 17K budget allocation worked example, multi-type summary extraction, model-readable compression, async summary scheduling, K-turn raw window eviction/recall, conflict resolution, iterative recall loops, five-dimension profile extraction, relationship fact/opinion separation with directional attitudes, category price percentile with sibling fallback, scenario-aware loading, spec derivation, and acceptable-ad filtering.

## 6. Docker Deployment

- Build and start all services (API + Redis):
  ```bash
  docker compose up -d
  ```
- Use real LLM calls (requires `API_KEY` in `.env`):
  ```bash
  SUMMARIZER_MODE=real docker compose up -d
  ```
- The Dockerfile copies `app/`, `static/`, `alembic/`, and `alembic.ini`, runs `alembic upgrade head` on startup, and persists SQLite data to a Docker volume.

## 7. Git and Submission

- Do NOT commit `extra_doc/` or `openspec/` to GitHub. They are already listed in `.gitignore`.
- Do NOT commit `.env` — it contains API keys and model names.
- Prefer conservative, known-working solutions backed by metrics or local validation.
- Maintain versioned submission files and a submission history log when the user asks for submissions.

## 8. External APIs and Tools

- LLM integration uses OpenAI SDK-compatible models (configured via `.env`).
- Document processing may use FinixDoc-VL API or OCR tools when required.

## 9. Contact and Context

- User prefers data-driven optimization based on metrics and experimental results.
- User is proficient in Python and document-processing pipelines.
- When in doubt, ask clarifying questions rather than making assumptions.
