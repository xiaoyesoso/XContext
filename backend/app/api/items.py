from fastapi import APIRouter, HTTPException

from app.dependencies import get_context_service
from app.models import ContextItem, ContextItemCreateRequest
from app.services.context_service import ContextService

router = APIRouter(prefix="/context/items", tags=["context-items"])


def _service() -> ContextService:
    return get_context_service()


@router.post("", response_model=ContextItem)
async def create_item(session_id: str, request: ContextItemCreateRequest) -> ContextItem:
    """Create a context item for a session."""
    return await _service().create_item(session_id, request)


@router.get("", response_model=list[ContextItem])
async def list_items(session_id: str) -> list[ContextItem]:
    """List all context items for a session."""
    return await _service().list_items(session_id)


@router.get("/{item_id}", response_model=ContextItem)
async def get_item(session_id: str, item_id: str) -> ContextItem:
    """Get a single context item by id."""
    item = await _service().get_item(session_id, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Context item not found")
    return item


@router.delete("/{item_id}")
async def delete_item(session_id: str, item_id: str) -> dict[str, bool]:
    """Delete a context item by id."""
    deleted = await _service().delete_item(session_id, item_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Context item not found")
    return {"deleted": True}
