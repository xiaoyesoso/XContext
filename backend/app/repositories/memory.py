from app.models import ContextItem
from app.repositories.base import ContextRepository


class InMemoryContextRepository(ContextRepository):
    """In-memory repository for quick iteration and testing."""

    def __init__(self):
        self._store: dict[str, dict[str, ContextItem]] = {}

    async def add(self, session_id: str, item: ContextItem) -> ContextItem:
        self._store.setdefault(session_id, {})[item.id] = item
        return item

    async def get(self, session_id: str, item_id: str) -> ContextItem | None:
        return self._store.get(session_id, {}).get(item_id)

    async def list(self, session_id: str) -> list[ContextItem]:
        return list(self._store.get(session_id, {}).values())

    async def delete(self, session_id: str, item_id: str) -> bool:
        session_store = self._store.get(session_id, {})
        if item_id in session_store:
            del session_store[item_id]
            return True
        return False

    async def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
