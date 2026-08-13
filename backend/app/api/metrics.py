from fastapi import APIRouter, HTTPException

from app.dependencies import get_metrics_collector

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.get("/{session_id}")
async def get_metrics(session_id: str) -> dict:
    """Return the latest metrics for a session."""
    collector = get_metrics_collector()
    metrics = collector.get_latest(session_id)
    if metrics is None:
        raise HTTPException(
            status_code=404, detail="No metrics found for this session"
        )
    return {
        "session_id": metrics.session_id,
        "retrieved_count": metrics.retrieved_count,
        "selected_count": metrics.selected_count,
        "compressed_count": metrics.compressed_count,
        "ordered_count": metrics.ordered_count,
        "total_context_tokens": metrics.total_context_tokens,
        "window_tokens": metrics.window_tokens,
        "timestamp": metrics.timestamp.isoformat(),
    }
