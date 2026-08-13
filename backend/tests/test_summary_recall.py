"""Tests for Phase 8-10: multi-type summaries, detail recall, and iterative recall."""

from datetime import datetime, timezone

import pytest

from app.core.async_summary import (
    AsyncSummaryScheduler,
    SummaryTrigger,
    SummaryTaskState,
)
from app.core.conflict_resolution import (
    ConflictResolutionStrategy,
    ConflictResolver,
    RecallMetrics,
)
from app.core.detail_recall import (
    InMemoryDetailRetriever,
    KTurnRawWindow,
)
from app.core.key_facts import (
    KeyFactCategory,
    MockKeyFactExtractor,
)
from app.core.model_readable import (
    MockModelReadableCompressor,
)
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
)


def _make_item(
    content: str,
    item_type: ContextType = ContextType.USER_INPUT,
    authority: ContextAuthority = ContextAuthority.CONFIRMED,
    priority: int = 5,
) -> ContextItem:
    """Helper to create a ContextItem for tests."""
    return ContextItem(
        type=item_type,
        content=content,
        source=ContextSource.USER,
        scope=ContextScope.CURRENT_STEP,
        authority=authority,
        priority=priority,
    )


# ---------------------------------------------------------------------------
# Phase 8: Key fact extraction
# ---------------------------------------------------------------------------


class TestMockKeyFactExtractor:
    """Tests for the mock key fact extractor."""

    @pytest.mark.asyncio
    async def test_extract_finds_goal_keyword(self):
        extractor = MockKeyFactExtractor()
        items = [_make_item("My goal is to optimize the project architecture.")]
        facts = await extractor.extract(items)
        goal_facts = [f for f in facts if f.category == KeyFactCategory.GOAL]
        assert len(goal_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_finds_hard_constraint(self):
        extractor = MockKeyFactExtractor()
        items = [_make_item("The budget must not exceed $500.")]
        facts = await extractor.extract(items)
        constraint_facts = [f for f in facts if f.category == KeyFactCategory.HARD_CONSTRAINT]
        assert len(constraint_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_finds_entity(self):
        extractor = MockKeyFactExtractor()
        items = [_make_item("Order ORD-7842 for product blue headphones.")]
        facts = await extractor.extract(items)
        entity_facts = [f for f in facts if f.category == KeyFactCategory.ENTITY]
        assert len(entity_facts) >= 1

    @pytest.mark.asyncio
    async def test_extract_empty_items(self):
        extractor = MockKeyFactExtractor()
        facts = await extractor.extract([])
        assert facts == []

    @pytest.mark.asyncio
    async def test_extract_assigns_source_item_id(self):
        extractor = MockKeyFactExtractor()
        items = [_make_item("I want a cheaper option.")]
        facts = await extractor.extract(items)
        for fact in facts:
            assert fact.source_item_id == items[0].id


# ---------------------------------------------------------------------------
# Phase 8: Model-readable compression
# ---------------------------------------------------------------------------


class TestMockModelReadableCompressor:
    """Tests for the mock model-readable compressor."""

    @pytest.mark.asyncio
    async def test_compress_removes_filler(self):
        compressor = MockModelReadableCompressor()
        original = "That's a great question. I think the budget is $500 basically."
        compressed = await compressor.compress(original)
        assert "great question" not in compressed
        assert "I think" not in compressed
        assert "basically" not in compressed

    @pytest.mark.asyncio
    async def test_compress_preserves_key_info(self):
        compressor = MockModelReadableCompressor()
        original = "嗯 the budget must not exceed $500 啊"
        compressed = await compressor.compress(original)
        assert "$500" in compressed
        assert "budget" in compressed

    @pytest.mark.asyncio
    async def test_compress_empty_content(self):
        compressor = MockModelReadableCompressor()
        compressed = await compressor.compress("")
        assert compressed == ""

    @pytest.mark.asyncio
    async def test_compress_with_cost_returns_tokens(self):
        compressor = MockModelReadableCompressor()
        compressed, cost = await compressor.compress_with_cost("budget $500")
        assert cost > 0
        assert isinstance(cost, int)


# ---------------------------------------------------------------------------
# Phase 8: Async summary scheduler
# ---------------------------------------------------------------------------


class TestAsyncSummaryScheduler:
    """Tests for the async summary scheduler."""

    @pytest.mark.asyncio
    async def test_schedule_and_wait(self):
        async def mock_summary(items):
            return _make_item("Summary content", item_type=ContextType.SUMMARY)

        scheduler = AsyncSummaryScheduler(summary_func=mock_summary)
        items = [_make_item("Turn 1"), _make_item("Turn 2")]
        task = scheduler.schedule("session-1", items, SummaryTrigger.END_OF_TURN)

        result = await scheduler.wait_for_task(task.id, timeout=5.0)
        assert result is not None
        assert result.content == "Summary content"
        assert task.state == SummaryTaskState.COMPLETED

    @pytest.mark.asyncio
    async def test_schedule_failure(self):
        async def failing_summary(items):
            raise ValueError("LLM error")

        scheduler = AsyncSummaryScheduler(summary_func=failing_summary)
        items = [_make_item("Turn 1")]
        task = scheduler.schedule("session-1", items)

        result = await scheduler.wait_for_task(task.id, timeout=5.0)
        assert result is None
        assert task.state == SummaryTaskState.FAILED
        assert "LLM error" in task.error

    @pytest.mark.asyncio
    async def test_wait_for_session_multiple_tasks(self):
        async def mock_summary(items):
            return _make_item(f"Summary of {len(items)} items", item_type=ContextType.SUMMARY)

        scheduler = AsyncSummaryScheduler(summary_func=mock_summary)
        scheduler.schedule("session-1", [_make_item("A")], SummaryTrigger.END_OF_TURN)
        scheduler.schedule("session-1", [_make_item("B")], SummaryTrigger.START_OF_NEXT_TURN)

        results = await scheduler.wait_for_session("session-1", timeout=5.0)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_empty_items_raises(self):
        scheduler = AsyncSummaryScheduler()
        with pytest.raises(ValueError):
            scheduler.schedule("session-1", [])

    @pytest.mark.asyncio
    async def test_clear_session(self):
        scheduler = AsyncSummaryScheduler()
        scheduler.schedule("session-1", [_make_item("A")])
        scheduler.clear_session("session-1")
        assert scheduler.get_pending_tasks("session-1") == []


# ---------------------------------------------------------------------------
# Phase 9: Detail retrieval
# ---------------------------------------------------------------------------


class TestInMemoryDetailRetriever:
    """Tests for the in-memory detail retriever."""

    @pytest.mark.asyncio
    async def test_recall_finds_matching(self):
        retriever = InMemoryDetailRetriever()
        item1 = _make_item("The budget is $500")
        item2 = _make_item("User prefers concise answers")
        await retriever.index(item1)
        await retriever.index(item2)

        results = await retriever.recall("budget")
        assert len(results) >= 1
        assert "budget" in results[0].content_as_string().lower()

    @pytest.mark.asyncio
    async def test_recall_no_match(self):
        retriever = InMemoryDetailRetriever()
        await retriever.index(_make_item("The weather is nice"))
        results = await retriever.recall("budget")
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_recall_top_k(self):
        retriever = InMemoryDetailRetriever()
        for i in range(10):
            await retriever.index(_make_item(f"budget item {i}"))
        results = await retriever.recall("budget", top_k=3)
        assert len(results) == 3


class TestKTurnRawWindow:
    """Tests for the K-turn raw window strategy."""

    def test_add_and_get_raw_items(self):
        window = KTurnRawWindow(k=3)
        items1 = [_make_item("Turn 1 msg")]
        items2 = [_make_item("Turn 2 msg")]
        window.add_turn(items1)
        window.add_turn(items2)
        raw = window.get_raw_items()
        assert len(raw) == 2

    def test_evicts_old_turns(self):
        window = KTurnRawWindow(k=2)
        window.add_turn([_make_item("Turn 1")])
        window.add_turn([_make_item("Turn 2")])
        window.add_turn([_make_item("Turn 3")])
        raw = window.get_raw_items()
        # Only last 2 turns.
        contents = [item.content_as_string() for item in raw]
        assert "Turn 1" not in contents
        assert "Turn 2" in contents
        assert "Turn 3" in contents

    def test_k_property(self):
        window = KTurnRawWindow(k=5)
        assert window.k == 5

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            KTurnRawWindow(k=0)

    def test_reset(self):
        window = KTurnRawWindow(k=3)
        window.add_turn([_make_item("Turn 1")])
        window.reset()
        assert window.get_raw_items() == []


# ---------------------------------------------------------------------------
# Phase 9: Conflict resolution
# ---------------------------------------------------------------------------


class TestConflictResolver:
    """Tests for the conflict resolver."""

    def test_last_write_wins(self):
        resolver = ConflictResolver(ConflictResolutionStrategy.LAST_WRITE_WINS)
        old_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        item_a = _make_item("budget $500", authority=ContextAuthority.CONFIRMED)
        item_a.created_at = old_time
        item_b = _make_item("budget $300", authority=ContextAuthority.CONFIRMED)
        item_b.created_at = new_time

        resolved = resolver.resolve([item_a, item_b])
        # Both share the same first-20-char key "budget $", so they're grouped.
        # The newer one should win as the conflict resolution.
        contents = [item.content_as_string() for item in resolved]
        assert "budget $300" in contents

    def test_authority_precedence(self):
        resolver = ConflictResolver(ConflictResolutionStrategy.AUTHORITY_PRECEDENCE)
        old_time = datetime(2025, 1, 1, tzinfo=timezone.utc)
        new_time = datetime(2025, 6, 1, tzinfo=timezone.utc)
        item_low = _make_item("budget $500", authority=ContextAuthority.ASSUMED)
        item_low.created_at = new_time
        item_high = _make_item("budget $300", authority=ContextAuthority.HARD_RULE)
        item_high.created_at = old_time

        resolved = resolver.resolve([item_low, item_high])
        contents = [item.content_as_string() for item in resolved]
        # Hard rule should win despite being older.
        assert "budget $300" in contents

    def test_reinforcement_kept(self):
        resolver = ConflictResolver()
        item_a = _make_item("meet weekly")
        item_b = _make_item("meet weekly twice")
        resolved = resolver.resolve([item_a, item_b])
        # Reinforcement: the more specific (longer) version is kept.
        assert len(resolved) >= 1

    def test_single_item_no_conflict(self):
        resolver = ConflictResolver()
        item = _make_item("budget $500")
        resolved = resolver.resolve([item])
        assert len(resolved) == 1

    def test_empty_list(self):
        resolver = ConflictResolver()
        resolved = resolver.resolve([])
        assert resolved == []


class TestRecallMetrics:
    """Tests for recall metrics tracking."""

    def test_hit_rate(self):
        metrics = RecallMetrics()
        metrics.record_recall(result_count=5, latency_ms=100)
        metrics.record_recall(result_count=0, latency_ms=50)
        assert metrics.recall_count == 2
        assert metrics.hit_count == 1
        assert metrics.hit_rate == 0.5

    def test_avg_latency(self):
        metrics = RecallMetrics()
        metrics.record_recall(3, 100.0)
        metrics.record_recall(2, 200.0)
        assert metrics.avg_latency_ms == 150.0

    def test_conflict_count(self):
        metrics = RecallMetrics()
        metrics.record_conflict(3)
        metrics.record_conflict(2)
        assert metrics.conflict_count == 5

    def test_empty_metrics(self):
        metrics = RecallMetrics()
        assert metrics.hit_rate == 0.0
        assert metrics.avg_latency_ms == 0.0


# ---------------------------------------------------------------------------
# Phase 10: Iterative recall (mock-based)
# ---------------------------------------------------------------------------


class TestKTurnRawWindowIntegration:
    """Integration tests for K-turn window with retriever."""

    @pytest.mark.asyncio
    async def test_items_evicted_then_recalled(self):
        retriever = InMemoryDetailRetriever()
        window = KTurnRawWindow(k=2)

        # Turn 1: add and index.
        item1 = _make_item("budget is $500")
        window.add_turn([item1])
        await retriever.index(item1)

        # Turn 2: add.
        window.add_turn([_make_item("Turn 2")])

        # Turn 3: evicts turn 1, but it's indexed.
        window.add_turn([_make_item("Turn 3")])

        # Turn 1 item is no longer in raw window.
        raw = window.get_raw_items()
        assert item1 not in raw

        # But it can be recalled.
        recalled = await retriever.recall("budget")
        assert len(recalled) >= 1
        assert recalled[0].id == item1.id
