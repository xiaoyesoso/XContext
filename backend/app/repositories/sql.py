from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_models import ContextItemRecord
from app.models import ContextItem
from app.repositories.base import ContextRepository


def _record_to_item(record: ContextItemRecord) -> ContextItem:
    """Convert a database record to a domain model."""
    return ContextItem(
        id=record.id,
        type=record.type,
        content=record.content,
        source=record.source,
        scope=record.scope,
        authority=record.authority,
        confidence=record.confidence,
        priority=record.priority,
        token_cost=record.token_cost,
        layer=record.layer,
        version=record.version,
        created_at=record.created_at,
        expires_at=record.expires_at,
        profile_dimension=record.profile_dimension,
        profile_tier=record.profile_tier,
    )


def _item_to_record(session_id: str, item: ContextItem) -> ContextItemRecord:
    """Convert a domain model to a database record."""
    return ContextItemRecord(
        id=item.id,
        session_id=session_id,
        type=item.type.value,
        content=item.content_as_string(),
        source=item.source.value,
        scope=item.scope.value,
        authority=item.authority.value,
        confidence=item.confidence,
        priority=item.priority,
        token_cost=item.token_cost,
        layer=item.layer,
        version=item.version,
        created_at=item.created_at,
        expires_at=item.expires_at,
        profile_dimension=item.profile_dimension.value if item.profile_dimension else None,
        profile_tier=item.profile_tier.value if item.profile_tier else None,
    )


class SQLContextRepository(ContextRepository):
    """SQLAlchemy-based repository for context items."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def add(self, session_id: str, item: ContextItem) -> ContextItem:
        record = _item_to_record(session_id, item)
        self._session.add(record)
        await self._session.commit()
        return item

    async def get(self, session_id: str, item_id: str) -> ContextItem | None:
        stmt = select(ContextItemRecord).where(
            ContextItemRecord.id == item_id,
            ContextItemRecord.session_id == session_id,
        )
        result = await self._session.execute(stmt)
        record = result.scalar_one_or_none()
        return _record_to_item(record) if record else None

    async def list(self, session_id: str) -> list[ContextItem]:
        stmt = (
            select(ContextItemRecord)
            .where(ContextItemRecord.session_id == session_id)
            .order_by(ContextItemRecord.created_at)
        )
        result = await self._session.execute(stmt)
        return [_record_to_item(record) for record in result.scalars().all()]

    async def delete(self, session_id: str, item_id: str) -> bool:
        stmt = delete(ContextItemRecord).where(
            ContextItemRecord.id == item_id,
            ContextItemRecord.session_id == session_id,
        )
        result = await self._session.execute(stmt)
        await self._session.commit()
        return result.rowcount > 0

    async def clear(self, session_id: str) -> None:
        stmt = delete(ContextItemRecord).where(
            ContextItemRecord.session_id == session_id
        )
        await self._session.execute(stmt)
        await self._session.commit()
