import asyncio
import json
import os

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from openai import AsyncOpenAI

from app.config import settings
from app.dependencies import get_context_service, get_metrics_collector
from app.models import (
    ChatRequest,
    ChatResponse,
    ComposeRequest,
    ContextAuthority,
    ContextItemCreateRequest,
    ContextScope,
    ContextSource,
    ContextType,
)

router = APIRouter(prefix="/chat", tags=["chat"])

_is_mock = os.getenv("SUMMARIZER_MODE", "").lower() == "mock"

# Lazily initialised OpenAI client for chat completions.
_chat_client: AsyncOpenAI | None = None


def _get_chat_client() -> AsyncOpenAI:
    global _chat_client
    if _chat_client is None:
        _chat_client = AsyncOpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
    return _chat_client


def _system_prompt(system_context: str) -> str:
    """Build the system prompt with language-consistency instruction."""
    return (
        "You are a helpful assistant. "
        "Use the following context to answer the user's question. "
        "Always respond in the same language as the user's message.\n\n"
        f"{system_context}"
    )


async def _call_llm(system_context: str, user_message: str) -> str:
    """Call the LLM and return the full reply text."""
    client = _get_chat_client()
    response = await client.chat.completions.create(
        model=settings.pro_llm_model,
        messages=[
            {"role": "system", "content": _system_prompt(system_context)},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


def _sse(data: dict) -> str:
    """Format a dict as a Server-Sent Event data frame."""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _budget_mode_value(mode) -> str | None:
    """Safely convert an optional BudgetMode enum to its string value."""
    return mode.value if mode else None


async def _ingest_user_message(service, request: ChatRequest) -> None:
    """Store the user message as a context item."""
    await service.create_item(
        request.session_id,
        ContextItemCreateRequest(
            type=ContextType.USER_INPUT,
            content=request.message,
            source=ContextSource.USER,
            scope=ContextScope.CURRENT_STEP,
            authority=ContextAuthority.CONFIRMED,
        ),
    )


async def _store_agent_reply(service, session_id: str, reply: str) -> None:
    """Store the agent reply as a context item."""
    await service.create_item(
        session_id,
        ContextItemCreateRequest(
            type=ContextType.MODEL_OUTPUT,
            content=reply,
            source=ContextSource.AGENT,
            scope=ContextScope.CURRENT_SESSION,
            authority=ContextAuthority.INFERRED,
        ),
    )


def _collect_metrics(session_id: str) -> dict | None:
    """Read the latest window metrics for a session."""
    collector = get_metrics_collector()
    latest = collector.get_latest(session_id)
    if latest is None:
        return None
    return {
        "retrieved_count": latest.retrieved_count,
        "selected_count": latest.selected_count,
        "compressed_count": latest.compressed_count,
        "ordered_count": latest.ordered_count,
        "total_context_tokens": latest.total_context_tokens,
        "window_tokens": latest.window_tokens,
        "budget_mode": _budget_mode_value(latest.budget_mode),
    }


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent chat endpoint: ingest user message, compose window, call LLM, store reply."""
    service = get_context_service()

    # 1. Store the user message as a context item.
    await _ingest_user_message(service, request)

    # 2. Compose the context window.
    compose_response = await service.compose_window(
        ComposeRequest(
            session_id=request.session_id,
            strategy=request.strategy,
            max_tokens=request.max_tokens,
            task_state=request.task_state,
            token_budget=request.token_budget,
            scenario=request.scenario,
        )
    )

    # 3. Call the LLM (or return a mock reply).
    if _is_mock:
        reply = f"[Mock Agent] Received: {request.message}"
    else:
        reply = await _call_llm(compose_response.prompt_fragment, request.message)

    # 4. Store the agent reply as a context item.
    await _store_agent_reply(service, request.session_id, reply)

    # 5. Collect metrics.
    metrics = _collect_metrics(request.session_id)

    return ChatResponse(
        session_id=request.session_id,
        reply=reply,
        items=compose_response.items,
        prompt_fragment=compose_response.prompt_fragment,
        total_tokens=compose_response.total_tokens,
        item_count=compose_response.item_count,
        budget_mode=compose_response.budget_mode,
        metrics=metrics,
    )


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Streaming chat endpoint using Server-Sent Events.

    Event sequence:
      1. meta   — prompt fragment + budget mode (before LLM call)
      2. delta  — incremental reply text (repeated)
      3. done   — context items + metrics (after reply is stored)
    """
    service = get_context_service()

    # 1. Store the user message as a context item.
    await _ingest_user_message(service, request)

    # 2. Compose the context window (synchronous, before streaming).
    compose_response = await service.compose_window(
        ComposeRequest(
            session_id=request.session_id,
            strategy=request.strategy,
            max_tokens=request.max_tokens,
            task_state=request.task_state,
            token_budget=request.token_budget,
            scenario=request.scenario,
        )
    )

    async def event_generator():
        # Emit metadata so the UI can render the prompt preview early.
        yield _sse({
            "type": "meta",
            "prompt_fragment": compose_response.prompt_fragment,
            "budget_mode": _budget_mode_value(compose_response.budget_mode),
        })

        # 3. Stream the LLM reply token-by-token.
        full_reply = ""
        if _is_mock:
            # Simulate streaming for mock mode.
            mock_reply = f"[Mock Agent] Received: {request.message}"
            for i in range(0, len(mock_reply), 3):
                delta = mock_reply[i : i + 3]
                full_reply += delta
                yield _sse({"type": "delta", "content": delta})
                await asyncio.sleep(0.03)
        else:
            client = _get_chat_client()
            stream = await client.chat.completions.create(
                model=settings.pro_llm_model,
                messages=[
                    {"role": "system", "content": _system_prompt(compose_response.prompt_fragment)},
                    {"role": "user", "content": request.message},
                ],
                temperature=0.7,
                max_tokens=1024,
                stream=True,
            )
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta = chunk.choices[0].delta.content
                    full_reply += delta
                    yield _sse({"type": "delta", "content": delta})

        # 4. Store the complete agent reply.
        await _store_agent_reply(service, request.session_id, full_reply)

        # 5. Emit the final event with items and metrics.
        items_data = [
            item.model_dump(mode="json") for item in compose_response.items
        ]
        yield _sse({
            "type": "done",
            "items": items_data,
            "metrics": _collect_metrics(request.session_id),
            "budget_mode": _budget_mode_value(compose_response.budget_mode),
        })

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
