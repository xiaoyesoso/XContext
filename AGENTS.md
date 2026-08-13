# AGENTS.md — Project Guide for AI Assistants

## 1. Project Overview

XContext is a Python backend service that implements the unified context-management abstraction for LLM Agent systems:

> Context Window = f(Context)

The service exposes RESTful APIs for context ingestion, layered context management, and configurable context-window composition strategies (sliding window, summary-based window, hybrid raw/summary).

## 2. Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: FastAPI
- **Data Validation**: Pydantic v2
- **Relational Persistence**: SQLAlchemy 2.0 + Alembic (planned)
- **Hot Cache**: Redis
- **Token Estimation**: tiktoken
- **Async Tasks** (planned): Celery
- **Testing**: pytest + httpx
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
├── AGENTS.md               # This file
├── .gitignore              # Ignores extra_doc/ and openspec/
├── backend/                # Python backend service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py         # FastAPI application entry
│   │   ├── models.py       # Pydantic schemas
│   │   └── engine.py       # Context pipeline engine
│   └── requirements.txt
├── extra_doc/              # External reference documents (NOT tracked by git)
└── openspec/               # OpenSpec change/spec workspace (NOT tracked by git)
    ├── config.yaml
    └── changes/
        └── context-window-service/
            ├── proposal.md
            ├── design.md
            ├── tasks.md
            ├── specs/context-api/spec.md
            └── assets/
```

## 5. Testing and Validation

- Run Python syntax checks:
  ```bash
  python3 -m py_compile backend/app/*.py
  ```
- Run the test suite (when available):
  ```bash
  cd backend && pytest
  ```
- Validate OpenSpec changes:
  ```bash
  openspec validate --changes <change-name> --json
  ```

## 6. Git and Submission

- Do NOT commit `extra_doc/` or `openspec/` to GitHub. They are already listed in `.gitignore`.
- Prefer conservative, known-working solutions backed by metrics or local validation.
- Maintain versioned submission files and a submission history log when the user asks for submissions.

## 7. External APIs and Tools

- LLM integration uses OpenAI SDK-compatible models.
- Document processing may use FinixDoc-VL API or OCR tools when required.

## 8. Contact and Context

- User prefers data-driven optimization based on metrics and experimental results.
- User is proficient in Python and document-processing pipelines.
- When in doubt, ask clarifying questions rather than making assumptions.
