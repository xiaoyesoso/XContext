import os

from fastapi import APIRouter
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


async def _call_llm(system_context: str, user_message: str) -> str:
    """Call the LLM and return the reply text."""
    client = _get_chat_client()
    response = await client.chat.completions.create(
        model=settings.pro_llm_model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant. "
                    "Use the following context to answer the user's question.\n\n"
                    f"{system_context}"
                ),
            },
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content or ""


@router.post("", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Agent chat endpoint: ingest user message, compose window, call LLM, store reply."""
    service = get_context_service()

    # 1. Store the user message as a context item.
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
        reply = await _call_llm(
            compose_response.prompt_fragment, request.message
        )

    # 4. Store the agent reply as a context item.
    await service.create_item(
        request.session_id,
        ContextItemCreateRequest(
            type=ContextType.MODEL_OUTPUT,
            content=reply,
            source=ContextSource.AGENT,
            scope=ContextScope.CURRENT_SESSION,
            authority=ContextAuthority.INFERRED,
        ),
    )

    # 5. Collect metrics.
    metrics = None
    collector = get_metrics_collector()
    latest = collector.get_latest(request.session_id)
    if latest is not None:
        metrics = {
            "retrieved_count": latest.retrieved_count,
            "selected_count": latest.selected_count,
            "compressed_count": latest.compressed_count,
            "ordered_count": latest.ordered_count,
            "total_context_tokens": latest.total_context_tokens,
            "window_tokens": latest.window_tokens,
            "budget_mode": latest.budget_mode,
        }

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
