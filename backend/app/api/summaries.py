"""Summary & detail-recall API endpoints.

Exposes the multi-type summary subsystem (conversation / chapter /
key facts / model-readable), the async summary scheduler, the K-turn
raw window, keyword detail recall, the iterative recall loop, and
conflict resolution to the frontend demo.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_context_service, get_summary_service

router = APIRouter(prefix="/context/summaries", tags=["context-summaries"])


class ExtractSummariesRequest(BaseModel):
    session_id: str
    types: Optional[list[str]] = None  # Defaults to all four types.


class AsyncScheduleRequest(BaseModel):
    session_id: str
    trigger: str = "end_of_turn"  # end_of_turn | start_of_next_turn | watchdog


class AsyncTaskWaitRequest(BaseModel):
    session_id: str
    task_id: str
    timeout: float = 30.0


class KTurnSyncRequest(BaseModel):
    session_id: str
    k: int = Field(default=10, ge=1, le=100)


class RecallRequest(BaseModel):
    query: str
    session_id: str
    top_k: int = Field(default=5, ge=1, le=20)


class IterativeRecallRequest(BaseModel):
    session_id: str
    catalog: Optional[list[str]] = None
    window_budget_tokens: Optional[int] = None


class ConflictEntry(BaseModel):
    content: str
    authority: Optional[str] = None


class ConflictResolveRequest(BaseModel):
    entries: list[ConflictEntry]
    strategy: str = "last_write_wins"  # last_write_wins | authority_precedence


@router.post("/extract")
async def extract_summaries(request: ExtractSummariesRequest) -> dict:
    """Generate multi-type summaries for all items of a session."""
    service = get_context_service()
    items = await service.list_items(request.session_id)
    if not items:
        raise HTTPException(status_code=400, detail="Session has no context items")
    try:
        return await get_summary_service().extract_summaries(
            request.session_id, items, request.types
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/async")
async def schedule_async_summary(request: AsyncScheduleRequest) -> dict:
    """Schedule an async summary task for the session items."""
    service = get_context_service()
    items = await service.list_items(request.session_id)
    if not items:
        raise HTTPException(status_code=400, detail="Session has no context items")
    if request.trigger not in ("end_of_turn", "start_of_next_turn", "watchdog"):
        raise HTTPException(status_code=400, detail="Invalid trigger")
    return get_summary_service().schedule_async_summary(
        request.session_id, items, request.trigger
    )


@router.get("/async/tasks")
async def list_async_tasks(session_id: str) -> list[dict]:
    """List all async summary tasks of a session."""
    return get_summary_service().list_async_tasks(session_id)


@router.post("/async/wait")
async def wait_for_async_task(request: AsyncTaskWaitRequest) -> dict:
    """Wait for an async summary task to complete."""
    result = await get_summary_service().wait_for_task(
        request.session_id, request.task_id, request.timeout
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.post("/k-turn")
async def sync_k_turn_window(request: KTurnSyncRequest) -> dict:
    """Replay the session turns into a K-turn raw window and report state."""
    service = get_context_service()
    items = await service.list_items(request.session_id)
    return get_summary_service().sync_k_turn_window(
        request.session_id, items, request.k
    )


@router.post("/recall")
async def recall_details(request: RecallRequest) -> list[dict]:
    """Keyword recall over indexed session items."""
    service = get_context_service()
    items = await service.list_items(request.session_id)
    return await get_summary_service().recall_details(
        request.query, items, session_id=request.session_id, top_k=request.top_k
    )


@router.post("/iterative")
async def run_iterative_recall(request: IterativeRecallRequest) -> dict:
    """Run the LLM-driven iterative recall loop for the session."""
    service = get_context_service()
    items = await service.list_items(request.session_id)
    if not items:
        raise HTTPException(status_code=400, detail="Session has no context items")
    return await get_summary_service().run_iterative_recall(
        request.session_id,
        items,
        catalog=request.catalog,
        window_budget_tokens=request.window_budget_tokens,
    )


@router.post("/conflicts/resolve")
async def resolve_conflicts(request: ConflictResolveRequest) -> dict:
    """Resolve conflicts among the provided entries."""
    if len(request.entries) < 2:
        raise HTTPException(status_code=400, detail="At least two entries required")
    if request.strategy not in ("last_write_wins", "authority_precedence"):
        raise HTTPException(status_code=400, detail="Invalid strategy")
    return get_summary_service().resolve_conflicts(
        [entry.model_dump() for entry in request.entries], request.strategy
    )
