import json
from datetime import datetime, timezone

from redis.asyncio import Redis

from app.models import ContextItem
from app.repositories.base import ContextRepository


def _item_to_dict(session_id: str, item: ContextItem) -> dict:
    """Serialize a context item to a dictionary for Redis storage."""
    return {
        "id": item.id,
        "session_id": session_id,
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
    }


def _dict_to_item(data: dict) -> ContextItem:
    """Deserialize a dictionary from Redis storage to a context item."""
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


class RedisContextRepository(ContextRepository):
    """Redis-based repository for hot context items."""

    def __init__(self, redis: Redis, key_prefix: str = "ctx"):
        self._redis = redis
        self._key_prefix = key_prefix

    def _session_key(self, session_id: str) -> str:
        return f"{self._key_prefix}:session:{session_id}"

    def _item_key(self, session_id: str, item_id: str) -> str:
        return f"{self._key_prefix}:session:{session_id}:item:{item_id}"

    async def add(self, session_id: str, item: ContextItem) -> ContextItem:
        data = _item_to_dict(session_id, item)
        pipe = self._redis.pipeline()
        item_key = self._item_key(session_id, item.id)
        pipe.set(item_key, json.dumps(data))
        pipe.zadd(self._session_key(session_id), {item.id: item.created_at.timestamp()})
        await pipe.execute()
        return item

    async def get(self, session_id: str, item_id: str) -> ContextItem | None:
        raw = await self._redis.get(self._item_key(session_id, item_id))
        if raw is None:
            return None
        return _dict_to_item(json.loads(raw))

    async def list(self, session_id: str) -> list[ContextItem]:
        item_ids = await self._redis.zrange(self._session_key(session_id), 0, -1)
        if not item_ids:
            return []
        keys = [self._item_key(session_id, item_id) for item_id in item_ids]
        raw_items = await self._redis.mget(keys)
        return [_dict_to_item(json.loads(raw)) for raw in raw_items if raw is not None]

    async def delete(self, session_id: str, item_id: str) -> bool:
        pipe = self._redis.pipeline()
        item_key = self._item_key(session_id, item_id)
        pipe.delete(item_key)
        pipe.zrem(self._session_key(session_id), item_id)
        results = await pipe.execute()
        return bool(results[0])

    async def clear(self, session_id: str) -> None:
        item_ids = await self._redis.zrange(self._session_key(session_id), 0, -1)
        if item_ids:
            keys = [self._item_key(session_id, item_id) for item_id in item_ids]
            keys.append(self._session_key(session_id))
            await self._redis.delete(*keys)
        else:
            await self._redis.delete(self._session_key(session_id))
