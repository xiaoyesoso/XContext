import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from app.core.pipeline import DefaultSelector, HybridOrderer, PlainTextInjector, SlidingWindowOrderer, SummaryOrderer
from app.core.summarizer import MockSummarizer
from app.models import ContextAuthority, ContextItem, ContextScope, ContextSource, ContextType


def _make_item(
    content: str,
    created_at: datetime | None = None,
    type: ContextType = ContextType.USER_INPUT,
    authority: ContextAuthority = ContextAuthority.ASSUMED,
    priority: int = 0,
    expires_at: datetime | None = None,
    token_cost: int = 0,
) -> ContextItem:
    source = ContextSource.USER if type == ContextType.USER_INPUT else ContextSource.AGENT
    return ContextItem(
        type=type,
        content=content,
        source=source,
        scope=ContextScope.CURRENT_SESSION,
        authority=authority,
        priority=priority,
        token_cost=token_cost,
        created_at=created_at or datetime.now(timezone.utc),
        expires_at=expires_at,
    )


def test_default_selector_drops_expired_and_denied():
    now = datetime.now(timezone.utc)
    expired = _make_item("expired", created_at=now - timedelta(hours=2), expires_at=now - timedelta(hours=1))
    denied = _make_item("denied", authority=ContextAuthority.DENIED)
    active = _make_item("active")

    selector = DefaultSelector()
    result = selector.select([expired, denied, active])

    assert len(result) == 1
    assert result[0].content == "active"


def test_default_selector_sorts_by_priority_then_authority():
    low = _make_item("low", priority=0, authority=ContextAuthority.ASSUMED)
    high = _make_item("high", priority=10, authority=ContextAuthority.ASSUMED)
    confirmed = _make_item("confirmed", priority=10, authority=ContextAuthority.CONFIRMED)
    rule = _make_item("rule", priority=10, authority=ContextAuthority.HARD_RULE)

    selector = DefaultSelector()
    result = selector.select([low, high, confirmed, rule])

    assert [item.content for item in result] == ["rule", "confirmed", "high", "low"]


def test_sliding_window_keeps_most_recent_within_budget():
    now = datetime.now(timezone.utc)
    items = [
        _make_item("a", created_at=now - timedelta(seconds=3), token_cost=5),
        _make_item("b", created_at=now - timedelta(seconds=2), token_cost=5),
        _make_item("c", created_at=now - timedelta(seconds=1), token_cost=5),
    ]

    orderer = SlidingWindowOrderer(max_tokens=10)
    result = asyncio.run(orderer.order(items))

    assert [item.content for item in result] == ["b", "c"]


def test_plain_text_injector_serializes_items():
    items = [_make_item("hello")]
    injector = PlainTextInjector()
    prompt = injector.inject(items)

    assert "[user_input]" in prompt
    assert "hello" in prompt


def test_summary_orderer_summarizes_old_blocks():
    now = datetime.now(timezone.utc)
    summarizer = MockSummarizer()
    items = [
        _make_item("msg0", created_at=now - timedelta(seconds=6), token_cost=5),
        _make_item("msg1", created_at=now - timedelta(seconds=5), token_cost=5),
        _make_item("msg2", created_at=now - timedelta(seconds=4), token_cost=5),
        _make_item("msg3", created_at=now - timedelta(seconds=3), token_cost=5),
        _make_item("msg4", created_at=now - timedelta(seconds=2), token_cost=5),
        _make_item("msg5", created_at=now - timedelta(seconds=1), token_cost=5),
    ]

    orderer = SummaryOrderer(max_tokens=100, window_size=3, summarizer=summarizer)
    result = asyncio.run(orderer.order(items))

    types = [item.type for item in result]
    assert ContextType.SUMMARY in types
    # The most recent window_size items remain raw.
    assert types[-3:] == [ContextType.USER_INPUT, ContextType.USER_INPUT, ContextType.USER_INPUT]


def test_hybrid_orderer_keeps_user_inputs_raw():
    now = datetime.now(timezone.utc)
    summarizer = MockSummarizer()
    items = [
        _make_item("user0", type=ContextType.USER_INPUT, created_at=now - timedelta(seconds=5), token_cost=2),
        _make_item("model0", type=ContextType.MODEL_OUTPUT, created_at=now - timedelta(seconds=4), token_cost=10),
        _make_item("user1", type=ContextType.USER_INPUT, created_at=now - timedelta(seconds=3), token_cost=2),
        _make_item("model1", type=ContextType.MODEL_OUTPUT, created_at=now - timedelta(seconds=2), token_cost=10),
        _make_item("model2", type=ContextType.MODEL_OUTPUT, created_at=now - timedelta(seconds=1), token_cost=10),
    ]

    orderer = HybridOrderer(max_tokens=100, window_size=3, summarizer=summarizer)
    result = asyncio.run(orderer.order(items))

    contents = [item.content for item in result]
    # User inputs are always kept raw.
    assert "user0" in contents
    assert "user1" in contents
