import asyncio

from app.core.layers import LayerManager, default_promotion_rules
from app.models import ContextAuthority, ContextItem, ContextScope, ContextSource, ContextType


def _make_confirmed_fact(content: str = "fact") -> ContextItem:
    return ContextItem(
        type=ContextType.FACT,
        content=content,
        source=ContextSource.USER,
        scope=ContextScope.CURRENT_SESSION,
        authority=ContextAuthority.CONFIRMED,
        layer="session",
    )


def test_layer_manager_lists_default_layers():
    manager = LayerManager()
    layers = manager.list_layers()
    names = {layer.name for layer in layers}
    assert names >= {"working", "session", "long_term", "archive"}


def test_promote_confirmed_fact_from_session_to_long_term():
    manager = LayerManager(rules=default_promotion_rules())
    item = _make_confirmed_fact()
    promoted = manager.promote(item)

    assert promoted is not None
    assert promoted.layer == "long_term"
    assert promoted.version == item.version + 1


def test_no_promotion_for_assumed_item():
    manager = LayerManager(rules=default_promotion_rules())
    item = ContextItem(
        type=ContextType.FACT,
        content="guess",
        source=ContextSource.AGENT,
        scope=ContextScope.CURRENT_SESSION,
        authority=ContextAuthority.ASSUMED,
        layer="session",
    )
    promoted = manager.promote(item)
    assert promoted is None


def test_promote_top_layer_returns_none():
    manager = LayerManager(rules=default_promotion_rules())
    item = _make_confirmed_fact()
    item.layer = "archive"
    promoted = manager.promote(item)
    assert promoted is None
