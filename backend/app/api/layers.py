from fastapi import APIRouter, HTTPException

from app.dependencies import get_context_service, get_layer_manager
from app.models import ContextItem

router = APIRouter(prefix="/context/layers", tags=["context-layers"])


@router.get("")
async def list_layers() -> list[dict]:
    """List configured context layers."""
    manager = get_layer_manager()
    return [layer.model_dump(exclude={"predicate"}) for layer in manager.list_layers()]


@router.post("/{layer}/promote")
async def promote_item(layer: str, session_id: str, item_id: str) -> ContextItem:
    """Promote a context item to the next layer if it satisfies the rules."""
    service = get_context_service()
    manager = get_layer_manager()

    item = await service.get_item(session_id, item_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Context item not found")
    if item.layer != layer:
        raise HTTPException(
            status_code=400,
            detail=f"Item layer {item.layer} does not match requested layer {layer}",
        )

    promoted = manager.promote(item)
    if promoted is None:
        raise HTTPException(
            status_code=400, detail="Item does not satisfy promotion rules"
        )

    # Persist the promoted item; in a MOVE scenario the original would be archived.
    await service.create_item_direct(session_id, promoted)
    return promoted
