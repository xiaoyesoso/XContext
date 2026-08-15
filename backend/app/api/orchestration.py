"""Observation endpoint for the conversation-driven orchestration flow.

The frontend polls this after each turn instead of triggering summary or
profile construction itself: construction happens automatically at
end_of_turn (async) and injection at the start of the next turn (sync).
"""

from fastapi import APIRouter

from app.dependencies import get_chat_orchestrator

router = APIRouter(prefix="/orchestration", tags=["orchestration"])


@router.get("/state")
async def orchestration_state(session_id: str) -> dict:
    """Return summaries, tasks, K-turn state, recalls, and profile state."""
    return await get_chat_orchestrator().get_state(session_id)
