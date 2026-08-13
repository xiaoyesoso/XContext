from fastapi import APIRouter, HTTPException

from app.dependencies import get_archive_repository, get_context_service
from app.models import ContextItem

router = APIRouter(prefix="/context/archive", tags=["context-archive"])


@router.post("/{item_id}")
async def archive_item(session_id: str, item_id: str) -> ContextItem:
    """Move a context item from active storage to cold archive."""
    service = get_context_service()
    archive_repo = get_archive_repository()

    item = await service.get_item(session_id, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Context item not found")

    archived = await archive_repo.add(session_id, item)
    await service.delete_item(session_id, item_id)
    return archived


@router.get("/{item_id}")
async def get_archived_item(session_id: str, item_id: str) -> ContextItem:
    """Retrieve a single archived item by id."""
    archive_repo = get_archive_repository()
    item = await archive_repo.get(session_id, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Archived item not found")
    return item


@router.get("")
async def list_archived_items(session_id: str) -> list[ContextItem]:
    """List all archived items for a session."""
    archive_repo = get_archive_repository()
    return await archive_repo.list(session_id)
