from fastapi import APIRouter

from app.dependencies import get_context_service
from app.models import ComposeRequest, ComposeResponse
from app.services.context_service import ContextService

router = APIRouter(prefix="/context/windows", tags=["context-windows"])


def _service() -> ContextService:
    return get_context_service()


@router.post("/compose", response_model=ComposeResponse)
async def compose_window(request: ComposeRequest) -> ComposeResponse:
    """Compose a context window using the requested strategy."""
    return await _service().compose_window(request)
