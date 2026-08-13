"""Cache-aware ordering for Prompt Prefix Cache optimization."""

from app.models import ContextAuthority, ContextItem, ContextType
from app.core.pipeline import Orderer, _item_cost


# Item types that are inherently volatile and must sit in the tail,
# regardless of their authority.
_VOLATILE_TYPES = {
    ContextType.USER_INPUT,
    ContextType.MODEL_OUTPUT,
    ContextType.TOOL_RESULT,
}

# Item types that are stable across turns and should sit in the cache prefix.
_STABLE_TYPES = {
    ContextType.CONSTRAINT,
    ContextType.PROFILE,
    ContextType.FACT,
}

_STABLE_AUTHORITIES = {
    ContextAuthority.HARD_RULE,
    ContextAuthority.CONFIRMED,
}


def _is_stable(item: ContextItem) -> bool:
    """Return True if the item should be placed in the stable cache prefix."""
    if item.type in _VOLATILE_TYPES:
        return False
    if item.type in _STABLE_TYPES:
        return True
    if item.authority in _STABLE_AUTHORITIES:
        return True
    return False


class CacheAwareOrderer(Orderer):
    """Orders items to maximize Prompt Prefix Cache hits.

    Stable items (constraints, hard rules, confirmed facts, profile/templates)
    are placed first in a deterministic order. Volatile items (user input,
    tool results, model output, summaries) follow, sorted by priority then
    recency. Current user input is always placed last.

    The token budget cap is applied greedily from the front (stable prefix
    first), preserving the cache-aware order — unlike SlidingWindowOrderer
    which re-sorts by recency.
    """

    def __init__(self, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    async def order(self, items: list[ContextItem]) -> list[ContextItem]:
        stable = [item for item in items if _is_stable(item)]
        volatile = [item for item in items if not _is_stable(item)]

        # Stable: deterministic order by authority rank, then priority, then type
        authority_rank = {
            ContextAuthority.HARD_RULE: 0,
            ContextAuthority.CONFIRMED: 1,
            ContextAuthority.INFERRED: 2,
            ContextAuthority.ASSUMED: 3,
            ContextAuthority.DENIED: 4,
        }
        stable.sort(
            key=lambda item: (
                authority_rank.get(item.authority, 99),
                -item.priority,
                item.type.value,
            )
        )

        # Volatile: priority desc, then user_input last, then by recency
        volatile.sort(
            key=lambda item: (
                -item.priority,
                1 if item.type == ContextType.USER_INPUT else 0,
                item.created_at,
            )
        )

        combined = stable + volatile

        # Apply token budget cap greedily from the front, preserving order
        result: list[ContextItem] = []
        total = 0
        for item in combined:
            cost = _item_cost(item)
            if total + cost > self.max_tokens and result:
                break
            result.append(item)
            total += cost

        return result
