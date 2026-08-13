from app.models import ContextItem
from app.repositories.base import ContextRepository


class CompositeContextRepository(ContextRepository):
    """Composite repository that reads from hot cache and falls back to warm storage.

    Writes go to both layers. Reads try hot first and backfill hot when missing.
    """

    def __init__(
        self,
        hot_repository: ContextRepository,
        warm_repository: ContextRepository,
    ):
        self._hot = hot_repository
        self._warm = warm_repository

    async def add(self, session_id: str, item: ContextItem) -> ContextItem:
        await self._hot.add(session_id, item)
        await self._warm.add(session_id, item)
        return item

    async def get(self, session_id: str, item_id: str) -> ContextItem | None:
        item = await self._hot.get(session_id, item_id)
        if item is not None:
            return item
        item = await self._warm.get(session_id, item_id)
        if item is not None:
            await self._hot.add(session_id, item)
        return item

    async def list(self, session_id: str) -> list[ContextItem]:
        hot_items = await self._hot.list(session_id)
        if hot_items:
            return hot_items
        warm_items = await self._warm.list(session_id)
        for item in warm_items:
            await self._hot.add(session_id, item)
        return warm_items

    async def delete(self, session_id: str, item_id: str) -> bool:
        hot_deleted = await self._hot.delete(session_id, item_id)
        warm_deleted = await self._warm.delete(session_id, item_id)
        return hot_deleted or warm_deleted

    async def clear(self, session_id: str) -> None:
        await self._hot.clear(session_id)
        await self._warm.clear(session_id)
