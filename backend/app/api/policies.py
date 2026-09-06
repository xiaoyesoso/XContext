"""REST endpoints for context policy management and execution inspection."""

from fastapi import APIRouter, HTTPException, status

from app.dependencies import get_policy_service
from app.models import ContextPolicy

router = APIRouter(prefix="/policies", tags=["policies"])


@router.get("")
async def list_policies() -> list[dict]:
    """List registered context policies."""
    return [policy.model_dump(mode="json") for policy in get_policy_service().list_policies()]


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_policy(policy: ContextPolicy) -> dict:
    """Create a validated context policy."""
    service = get_policy_service()
    created, errors = service.create_policy(policy)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    return created.model_dump(mode="json")


@router.get("/execution/{session_id}")
async def get_policy_execution(session_id: str) -> dict:
    """Return the latest policy execution for a session."""
    execution = get_policy_service().get_execution(session_id)
    if execution is None:
        raise HTTPException(status_code=404, detail="Policy execution not found")
    return execution.model_dump(mode="json")


@router.get("/{policy_id}")
async def get_policy(policy_id: str) -> dict:
    """Get a policy by identifier."""
    service = get_policy_service()
    policy = service.get_policy(policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy.model_dump(mode="json")


@router.put("/{policy_id}")
async def update_policy(policy_id: str, policy: ContextPolicy) -> dict:
    """Replace a policy by identifier."""
    service = get_policy_service()
    updated, errors = service.update_policy(policy_id, policy)
    if errors:
        raise HTTPException(status_code=422, detail=errors)
    if updated is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return updated.model_dump(mode="json")


@router.delete("/{policy_id}")
async def delete_policy(policy_id: str) -> dict:
    """Delete a non-default policy."""
    if not get_policy_service().delete_policy(policy_id):
        raise HTTPException(status_code=404, detail="Policy not found or protected")
    return {"deleted": True, "policy_id": policy_id}


@router.post("/{policy_id}/default")
async def set_default_policy(policy_id: str) -> dict:
    """Set a policy as the default policy."""
    policy = get_policy_service().set_default(policy_id)
    if policy is None:
        raise HTTPException(status_code=404, detail="Policy not found")
    return policy.model_dump(mode="json")
