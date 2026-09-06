"""REST endpoints for long-task drift control.

Provides event sourcing, snapshots, checkpoints, verifier runs,
restore, and phase handoff operations.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.dependencies import get_chat_orchestrator, get_context_service
from app.core.checkpoint import CheckpointManager
from app.core.drift import ContextVerifier
from app.core.handoff import HandoffBuilder
from app.core.task_events import EventStore

router = APIRouter(prefix="/tasks", tags=["tasks"])


class VerifyRequest(BaseModel):
    """Optional parameters for a manual verifier run."""

    turn: int | None = None


class SnapshotRequest(BaseModel):
    """Optional parameters for a manual snapshot."""

    goal: str | None = None


class RestoreRequest(BaseModel):
    """Restore to a trusted snapshot."""

    snapshot_id: str = Field(min_length=1)


class HandoffRequest(BaseModel):
    """Create a phase handoff."""

    phase_name: str = Field(min_length=1)


class RecallRequest(BaseModel):
    """Recall items from a handoff by keywords."""

    keywords: list[str] = Field(min_length=1)
    top_k: int = 5


def _get_orchestrator():
    return get_chat_orchestrator()


@router.get("/{session_id}/events")
async def list_events(session_id: str, include_tainted: bool = False) -> dict:
    """Return the recorded event stream for a session."""
    orchestrator = _get_orchestrator()
    events = orchestrator._event_store.list_events(session_id, include_tainted)
    return {
        "session_id": session_id,
        "count": orchestrator._event_store.count(session_id),
        "events": [e.model_dump(mode="json") for e in events],
    }


@router.get("/{session_id}/snapshots")
async def list_snapshots(session_id: str) -> dict:
    """Return all trusted snapshots for a session."""
    snapshots = _get_orchestrator()._event_store.list_snapshots(session_id)
    return {
        "session_id": session_id,
        "snapshots": [s.model_dump(mode="json") for s in snapshots],
    }


@router.post("/{session_id}/snapshots")
async def create_snapshot(session_id: str, request: SnapshotRequest | None = None) -> dict:
    """Manually create a trusted snapshot for the current session state."""
    orchestrator = _get_orchestrator()
    context = get_context_service()
    items = await context.list_items(session_id)
    goal = (request and request.goal) or orchestrator._extract_goal(session_id)
    constraints = [
        i.content_as_string()
        for i in items
        if i.authority.value in ("hard_rule", "denied")
    ]
    facts = [
        i.content_as_string()
        for i in items
        if i.type.value == "fact" and i.authority.value == "confirmed"
    ]
    turn = orchestrator._turns.get(session_id, 0)
    snapshot = orchestrator._event_store.create_snapshot(
        session_id,
        goal=goal,
        hard_constraints=constraints,
        confirmed_facts=facts,
        task_state={"turn": turn, "item_count": len(items)},
        turn=turn,
    )
    return snapshot.model_dump(mode="json")


@router.post("/{session_id}/verify")
async def verify(session_id: str, request: VerifyRequest | None = None) -> dict:
    """Run the context verifier over the current session state."""
    orchestrator = _get_orchestrator()
    context = get_context_service()
    items = await context.list_items(session_id)
    events = orchestrator._event_store.list_events(session_id, include_tainted=True)
    turn = (request and request.turn) or orchestrator._turns.get(session_id, 0)
    verifier = ContextVerifier()
    report = verifier.verify(
        session_id, items, events=events, goal=orchestrator._extract_goal(session_id), turn=turn
    )
    from app.core.task_events import EventType

    orchestrator._event_store.append(
        session_id,
        EventType.DRIFT_DETECTED,
        payload=report.model_dump(mode="json"),
        turn=turn,
    )
    return report.model_dump(mode="json")


@router.get("/{session_id}/checkpoints")
async def list_checkpoints(session_id: str) -> dict:
    """Return checkpoint events for a session."""
    events = _get_orchestrator()._event_store.list_events(session_id)
    checkpoints = [
        e for e in events
        if e.type.value in ("checkpoint_passed", "checkpoint_failed")
    ]
    return {
        "session_id": session_id,
        "count": len(checkpoints),
        "checkpoints": [e.model_dump(mode="json") for e in checkpoints],
    }


@router.post("/{session_id}/checkpoints")
async def run_checkpoint(session_id: str) -> dict:
    """Manually trigger a checkpoint for a session."""
    orchestrator = _get_orchestrator()
    context = get_context_service()
    items = await context.list_items(session_id)
    turn = orchestrator._turns.get(session_id, 0)
    report = orchestrator._checkpoint_manager.run_checkpoint(
        session_id,
        items,
        goal=orchestrator._extract_goal(session_id),
        task_state=None,
        turn=turn,
    )
    return report.model_dump(mode="json")


@router.post("/{session_id}/restore")
async def restore(session_id: str, request: RestoreRequest) -> dict:
    """Restore session state to a trusted snapshot."""
    orchestrator = _get_orchestrator()
    state = orchestrator._event_store.restore(session_id, request.snapshot_id)
    if state is None:
        raise HTTPException(status_code=404, detail="Snapshot not found")
    return {"session_id": session_id, "snapshot_id": request.snapshot_id, "state": state}


@router.post("/{session_id}/handoff")
async def create_handoff(session_id: str, request: HandoffRequest) -> dict:
    """Build and store a phase handoff for the current session."""
    orchestrator = _get_orchestrator()
    context = get_context_service()
    items = await context.list_items(session_id)
    events = orchestrator._event_store.list_events(session_id, include_tainted=True)
    handoff = HandoffBuilder.build(
        session_id,
        request.phase_name,
        items,
        events=events,
        goal=orchestrator._extract_goal(session_id),
    )
    orchestrator._handoff_store.save(handoff, items)
    return handoff.model_dump(mode="json")


@router.get("/{session_id}/handoffs")
async def list_handoffs(session_id: str) -> dict:
    """Return all handoffs for a session."""
    handoffs = _get_orchestrator()._handoff_store.list_handoffs(session_id)
    return {
        "session_id": session_id,
        "handoffs": [h.model_dump(mode="json") for h in handoffs],
    }


@router.post("/handoff/{handoff_id}/recall")
async def recall_from_handoff(handoff_id: str, request: RecallRequest) -> dict:
    """Recall original phase items from a handoff by keywords."""
    results = _get_orchestrator()._handoff_store.recall(
        handoff_id, request.keywords, top_k=request.top_k
    )
    return {"handoff_id": handoff_id, "keywords": request.keywords, "items": results}
