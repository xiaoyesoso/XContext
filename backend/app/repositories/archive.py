import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.models import ContextItem
from app.repositories.base import ContextRepository


def _item_to_dict(item: ContextItem) -> dict[str, Any]:
    """Serialize a context item to a JSON-serializable dictionary."""
    return {
        "id": item.id,
        "type": item.type.value,
        "content": item.content_as_string(),
        "source": item.source.value,
        "scope": item.scope.value,
        "authority": item.authority.value,
        "confidence": item.confidence,
        "priority": item.priority,
        "token_cost": item.token_cost,
        "layer": item.layer,
        "version": item.version,
        "created_at": item.created_at.isoformat(),
        "expires_at": item.expires_at.isoformat() if item.expires_at else None,
        "archived_at": datetime.now(timezone.utc).isoformat(),
    }


def _dict_to_item(data: dict[str, Any]) -> ContextItem:
    """Deserialize a dictionary to a context item."""
    from app.models import (
        ContextAuthority,
        ContextScope,
        ContextSource,
        ContextType,
    )

    return ContextItem(
        id=data["id"],
        type=ContextType(data["type"]),
        content=data["content"],
        source=ContextSource(data["source"]),
        scope=ContextScope(data["scope"]),
        authority=ContextAuthority(data["authority"]),
        confidence=data["confidence"],
        priority=data["priority"],
        token_cost=data["token_cost"],
        layer=data["layer"],
        version=data["version"],
        created_at=datetime.fromisoformat(data["created_at"]),
        expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
    )


class LocalArchiveRepository(ContextRepository):
    """File-system based archive repository for cold context storage."""

    def __init__(self, base_path: str | Path = "archive"):
        self._base = Path(base_path)
        self._base.mkdir(parents=True, exist_ok=True)

    def _session_path(self, session_id: str) -> Path:
        """Return the archive directory for a session."""
        path = self._base / session_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _item_path(self, session_id: str, item_id: str) -> Path:
        """Return the archive file path for a context item."""
        return self._session_path(session_id) / f"{item_id}.json"

    async def add(self, session_id: str, item: ContextItem) -> ContextItem:
        """Archive a context item to cold storage."""
        item_path = self._item_path(session_id, item.id)
        with item_path.open("w", encoding="utf-8") as f:
            json.dump(_item_to_dict(item), f, ensure_ascii=False, indent=2)
        return item

    async def get(self, session_id: str, item_id: str) -> ContextItem | None:
        """Retrieve an archived item by id."""
        item_path = self._item_path(session_id, item_id)
        if not item_path.exists():
            return None
        with item_path.open("r", encoding="utf-8") as f:
            return _dict_to_item(json.load(f))

    async def list(self, session_id: str) -> list[ContextItem]:
        """List all archived items for a session."""
        session_path = self._session_path(session_id)
        items = []
        for item_path in sorted(session_path.glob("*.json")):
            with item_path.open("r", encoding="utf-8") as f:
                items.append(_dict_to_item(json.load(f)))
        return items

    async def delete(self, session_id: str, item_id: str) -> bool:
        """Delete an archived item by id."""
        item_path = self._item_path(session_id, item_id)
        if item_path.exists():
            item_path.unlink()
            return True
        return False

    async def clear(self, session_id: str) -> None:
        """Remove all archived items for a session."""
        session_path = self._session_path(session_id)
        for item_path in session_path.glob("*.json"):
            item_path.unlink()
