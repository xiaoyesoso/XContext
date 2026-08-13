"""Tests for Phase 7 — Context Compiler & Dynamic Orchestration."""

import asyncio
from datetime import datetime, timedelta, timezone

from app.core.budget import BudgetModeResolver
from app.core.compression import (
    DynamicCompressor,
    ItemImportanceClassifier,
    apply_compression,
)
from app.core.failure_history import FailureHistoryTracker
from app.core.ordering import CacheAwareOrderer
from app.core.selectors import DynamicSelector, NegativeContextHandler, RetrievalSelector
from app.models import (
    BudgetMode,
    CompressionLevel,
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
    Importance,
    TaskState,
    TokenBudget,
)


def _make_item(
    content: str,
    type: ContextType = ContextType.USER_INPUT,
    authority: ContextAuthority = ContextAuthority.ASSUMED,
    priority: int = 0,
    token_cost: int = 0,
    correlation_group: str | None = None,
    expires_at: datetime | None = None,
) -> ContextItem:
    source = (
        ContextSource.USER if type == ContextType.USER_INPUT else ContextSource.AGENT
    )
    return ContextItem(
        type=type,
        content=content,
        source=source,
        scope=ContextScope.CURRENT_SESSION,
        authority=authority,
        priority=priority,
        token_cost=token_cost,
        created_at=datetime.now(timezone.utc),
        correlation_group=correlation_group,
        expires_at=expires_at,
    )


# ---------------------------------------------------------------------------
# BudgetModeResolver
# ---------------------------------------------------------------------------


class TestBudgetModeResolver:
    def test_full_mode(self):
        resolver = BudgetModeResolver()
        budget = TokenBudget(total=10000, reserved=1000)
        assert resolver.resolve(budget) == BudgetMode.FULL

    def test_balanced_mode(self):
        resolver = BudgetModeResolver()
        budget = TokenBudget(total=10000, remaining=3000)
        assert resolver.resolve(budget) == BudgetMode.BALANCED

    def test_compact_mode(self):
        resolver = BudgetModeResolver()
        budget = TokenBudget(total=10000, remaining=1000)
        assert resolver.resolve(budget) == BudgetMode.COMPACT

    def test_minimal_mode(self):
        resolver = BudgetModeResolver()
        budget = TokenBudget(total=10000, remaining=300)
        assert resolver.resolve(budget) == BudgetMode.MINIMAL

    def test_custom_thresholds(self):
        resolver = BudgetModeResolver(
            full_threshold=0.8, balanced_threshold=0.5, compact_threshold=0.1
        )
        budget = TokenBudget(total=10000, remaining=6000)
        assert resolver.resolve(budget) == BudgetMode.BALANCED


# ---------------------------------------------------------------------------
# ItemImportanceClassifier
# ---------------------------------------------------------------------------


class TestItemImportanceClassifier:
    def setup_method(self):
        self.clf = ItemImportanceClassifier()

    def test_hard_rule_is_critical(self):
        item = _make_item("rule", authority=ContextAuthority.HARD_RULE)
        assert self.clf.classify(item) == Importance.CRITICAL

    def test_user_input_is_critical(self):
        item = _make_item("hello", type=ContextType.USER_INPUT)
        assert self.clf.classify(item) == Importance.CRITICAL

    def test_confirmed_is_high(self):
        item = _make_item("fact", type=ContextType.FACT, authority=ContextAuthority.CONFIRMED)
        assert self.clf.classify(item) == Importance.HIGH

    def test_high_priority_is_high(self):
        item = _make_item("important", type=ContextType.FACT, priority=10)
        assert self.clf.classify(item) == Importance.HIGH

    def test_tool_result_is_medium(self):
        item = _make_item("result", type=ContextType.TOOL_RESULT)
        assert self.clf.classify(item) == Importance.MEDIUM

    def test_assumed_summary_is_low(self):
        item = _make_item(
            "summary text here", type=ContextType.SUMMARY, authority=ContextAuthority.ASSUMED
        )
        assert self.clf.classify(item) == Importance.LOW


# ---------------------------------------------------------------------------
# apply_compression
# ---------------------------------------------------------------------------


class TestApplyCompression:
    def test_l0_returns_none(self):
        item = _make_item("content")
        assert apply_compression(item, CompressionLevel.L0) is None

    def test_l4_returns_item_with_marker(self):
        item = _make_item("content")
        result = apply_compression(item, CompressionLevel.L4)
        assert result is not None
        assert result.compression_level == CompressionLevel.L4
        assert result.content == "content"

    def test_l1_truncates_to_keywords(self):
        item = _make_item("the quick brown fox jumps over the lazy dog again")
        result = apply_compression(item, CompressionLevel.L1)
        assert result is not None
        assert result.compression_level == CompressionLevel.L1
        assert len(result.content_as_string().split()) <= 11  # 10 + "..."

    def test_l2_returns_first_sentence(self):
        item = _make_item("First sentence here. Second sentence follows.")
        result = apply_compression(item, CompressionLevel.L2)
        assert result is not None
        assert result.compression_level == CompressionLevel.L2
        assert "First sentence" in result.content_as_string()
        assert "Second" not in result.content_as_string()

    def test_l3_truncates_to_100_tokens(self):
        long_text = " ".join(["word"] * 200)
        item = _make_item(long_text)
        result = apply_compression(item, CompressionLevel.L3)
        assert result is not None
        assert result.compression_level == CompressionLevel.L3
        assert len(result.content_as_string().split()) <= 101


# ---------------------------------------------------------------------------
# DynamicCompressor
# ---------------------------------------------------------------------------


class TestDynamicCompressor:
    def test_critical_item_always_l4(self):
        compressor = DynamicCompressor()
        item = _make_item("rule", authority=ContextAuthority.HARD_RULE)
        result = compressor.compress_batch([item], budget_mode=BudgetMode.MINIMAL)
        assert len(result) == 1
        assert result[0].compression_level == CompressionLevel.L4

    def test_low_importance_dropped_in_minimal(self):
        compressor = DynamicCompressor()
        item = _make_item(
            "background info", type=ContextType.SUMMARY, authority=ContextAuthority.ASSUMED
        )
        result = compressor.compress_batch([item], budget_mode=BudgetMode.MINIMAL)
        assert len(result) == 0  # L0 → dropped

    def test_low_importance_l3_in_full_mode(self):
        compressor = DynamicCompressor()
        item = _make_item(
            "background info here with enough words",
            type=ContextType.SUMMARY,
            authority=ContextAuthority.ASSUMED,
        )
        result = compressor.compress_batch([item], budget_mode=BudgetMode.FULL)
        assert len(result) == 1
        assert result[0].compression_level == CompressionLevel.L3

    def test_cascading_rules_complementary_items(self):
        compressor = DynamicCompressor()
        item_a = _make_item("part a", type=ContextType.FACT, authority=ContextAuthority.INFERRED, correlation_group="g1")
        item_b = _make_item("part b", type=ContextType.SUMMARY, authority=ContextAuthority.ASSUMED, correlation_group="g1")
        # In COMPACT mode, item_a (medium) → L2, item_b (low) → L1
        # But cascading should bring both to L2 (the higher)
        result = compressor.compress_batch([item_a, item_b], budget_mode=BudgetMode.COMPACT)
        assert len(result) == 2
        levels = {r.compression_level for r in result}
        assert levels == {CompressionLevel.L2}

    def test_scenario_variant_override(self):
        item = _make_item("content", type=ContextType.SUMMARY, authority=ContextAuthority.ASSUMED, token_cost=5)
        variants = {"refund": {item.id: CompressionLevel.L4}}
        compressor = DynamicCompressor(scenario_variants=variants)
        # Without scenario: low importance in COMPACT → L1
        result_no_scenario = compressor.compress_batch([item], budget_mode=BudgetMode.COMPACT)
        assert result_no_scenario[0].compression_level == CompressionLevel.L1
        # With scenario: overridden to L4
        result_scenario = compressor.compress_batch(
            [item], budget_mode=BudgetMode.COMPACT, scenario="refund"
        )
        assert result_scenario[0].compression_level == CompressionLevel.L4


# ---------------------------------------------------------------------------
# CacheAwareOrderer
# ---------------------------------------------------------------------------


class TestCacheAwareOrderer:
    def test_stable_before_volatile(self):
        now = datetime.now(timezone.utc)
        constraint = _make_item(
            "hard rule", type=ContextType.CONSTRAINT, authority=ContextAuthority.HARD_RULE
        )
        constraint.created_at = now - timedelta(seconds=10)
        user_input = _make_item("current input", type=ContextType.USER_INPUT)
        user_input.created_at = now

        orderer = CacheAwareOrderer(max_tokens=1000)
        result = asyncio.run(orderer.order([user_input, constraint]))

        # Constraint (stable) must come before user_input (volatile)
        assert result[0].type == ContextType.CONSTRAINT
        assert result[1].type == ContextType.USER_INPUT

    def test_user_input_placed_last(self):
        constraint = _make_item(
            "rule", type=ContextType.CONSTRAINT, authority=ContextAuthority.HARD_RULE
        )
        fact = _make_item("fact", type=ContextType.FACT, authority=ContextAuthority.CONFIRMED)
        user_input = _make_item("hello", type=ContextType.USER_INPUT)
        tool_result = _make_item("result", type=ContextType.TOOL_RESULT)

        orderer = CacheAwareOrderer(max_tokens=1000)
        result = asyncio.run(orderer.order([user_input, tool_result, fact, constraint]))

        # Last item should be user_input
        assert result[-1].type == ContextType.USER_INPUT
        # First items should be stable (constraint, fact)
        assert result[0].type == ContextType.CONSTRAINT
        assert result[1].authority == ContextAuthority.CONFIRMED


# ---------------------------------------------------------------------------
# NegativeContextHandler & DynamicSelector
# ---------------------------------------------------------------------------


class TestNegativeContext:
    def test_handler_converts_denied_to_reminder(self):
        item = _make_item(
            "This plan is too fabricated. Don't use it.",
            authority=ContextAuthority.DENIED,
        )
        handler = NegativeContextHandler()
        result = handler.convert(item)
        assert result is not None
        assert "REJECTED" in result.content_as_string()
        assert result.authority == ContextAuthority.DENIED

    def test_handler_returns_none_for_non_denied(self):
        item = _make_item("normal", authority=ContextAuthority.CONFIRMED)
        handler = NegativeContextHandler()
        assert handler.convert(item) is None

    def test_dynamic_selector_retains_denied_as_reminder(self):
        active = _make_item("active", authority=ContextAuthority.CONFIRMED)
        denied = _make_item(
            "Bad plan that was rejected. Do not repeat.",
            authority=ContextAuthority.DENIED,
        )
        selector = DynamicSelector()
        result = selector.select([active, denied])
        # Active item + 1 reminder
        assert len(result) == 2
        assert result[0].content == "active"
        assert "REJECTED" in result[1].content_as_string()

    def test_dynamic_selector_drops_expired(self):
        now = datetime.now(timezone.utc)
        expired = _make_item("old", expires_at=now - timedelta(hours=1))
        active = _make_item("new")
        selector = DynamicSelector()
        result = selector.select([expired, active])
        assert len(result) == 1
        assert result[0].content == "new"


# ---------------------------------------------------------------------------
# RetrievalSelector
# ---------------------------------------------------------------------------


class TestRetrievalSelector:
    def test_keyword_ranking(self):
        task_state = TaskState(goal="find refund policy", current_step="check policy")
        item_relevant = _make_item("The refund policy allows 30 days", type=ContextType.FACT)
        item_irrelevant = _make_item("Weather is nice today", type=ContextType.FACT)
        selector = RetrievalSelector()
        result = selector.select([item_irrelevant, item_relevant], task_state)
        # Relevant item should be ranked first
        assert "refund" in result[0].content_as_string().lower()

    def test_no_task_state_falls_back(self):
        item = _make_item("content")
        selector = RetrievalSelector()
        result = selector.select([item], task_state=None)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# FailureHistoryTracker
# ---------------------------------------------------------------------------


class TestFailureHistoryTracker:
    def test_record_and_query(self):
        tracker = FailureHistoryTracker()
        tracker.record("refund_task", ContextType.FACT)
        missing = tracker.get_missing_types("refund_task")
        assert ContextType.FACT in missing

    def test_minimum_level_elevation(self):
        tracker = FailureHistoryTracker()
        tracker.record("refund_task", ContextType.PROFILE)
        level = tracker.minimum_level_for(ContextType.PROFILE, "refund_task")
        assert level == CompressionLevel.L3

    def test_no_elevation_for_clean_type(self):
        tracker = FailureHistoryTracker()
        tracker.record("refund_task", ContextType.PROFILE)
        level = tracker.minimum_level_for(ContextType.FACT, "refund_task")
        assert level is None


# ---------------------------------------------------------------------------
# Integration: Dynamic pipeline via API
# ---------------------------------------------------------------------------


class TestDynamicPipelineAPI:
    def test_dynamic_strategy_with_task_state_and_budget(self, client):
        session_id = "test-dynamic-session"

        # Ingest items
        client.post(
            f"/context/items?session_id={session_id}",
            json={
                "type": "constraint",
                "content": "Must follow the refund policy exactly.",
                "source": "user",
                "scope": "current_task",
                "authority": "hard_rule",
                "priority": 10,
            },
        )
        client.post(
            f"/context/items?session_id={session_id}",
            json={
                "type": "user_input",
                "content": "I want a refund for order 12345.",
                "source": "user",
                "scope": "current_step",
                "authority": "confirmed",
            },
        )
        client.post(
            f"/context/items?session_id={session_id}",
            json={
                "type": "summary",
                "content": "Previous conversation about shipping delays and other background topics.",
                "source": "agent",
                "scope": "current_session",
                "authority": "assumed",
            },
        )

        # Compose with dynamic strategy
        response = client.post(
            "/context/windows/compose",
            json={
                "session_id": session_id,
                "strategy": "dynamic",
                "max_tokens": 4096,
                "task_state": {
                    "current_step": "process_refund",
                    "goal": "Handle refund request",
                    "progress": "checking policy",
                    "missing_context": [],
                },
                "token_budget": {
                    "total": 32000,
                    "reserved": 8000,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["strategy"] == "dynamic"
        assert data["budget_mode"] is not None
        # Constraint should appear first (stable prefix)
        types = [item["type"] for item in data["items"]]
        if "constraint" in types:
            assert types.index("constraint") < types.index("user_input")

    def test_dynamic_strategy_minimal_budget_drops_low_importance(self, client):
        session_id = "test-minimal-session"

        client.post(
            f"/context/items?session_id={session_id}",
            json={
                "type": "constraint",
                "content": "Hard rule.",
                "source": "user",
                "scope": "current_task",
                "authority": "hard_rule",
            },
        )
        client.post(
            f"/context/items?session_id={session_id}",
            json={
                "type": "summary",
                "content": "Low importance background summary.",
                "source": "agent",
                "scope": "current_session",
                "authority": "assumed",
            },
        )

        response = client.post(
            "/context/windows/compose",
            json={
                "session_id": session_id,
                "strategy": "dynamic",
                "max_tokens": 4096,
                "token_budget": {
                    "total": 10000,
                    "remaining": 300,
                },
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["budget_mode"] == "minimal"
        types = [item["type"] for item in data["items"]]
        # Summary (low importance) should be dropped in minimal mode
        assert "summary" not in types
        # Constraint (critical) should be retained
        assert "constraint" in types


# ---------------------------------------------------------------------------
# Worked Example: 17K Budget Allocation (reference fixture)
# ---------------------------------------------------------------------------


class TestWorkedExample17K:
    """Reference fixture based on the design.md worked example.

    Demonstrates that the dynamic pipeline allocates a constrained budget
    correctly: critical items stay raw, low-importance items get compressed
    or dropped.
    """

    def test_17k_budget_allocation(self):
        # Build the 15 candidate items from the worked example
        c1 = _make_item("Continue the previous plan", type=ContextType.USER_INPUT)
        c3 = _make_item("Hard constraints: must be truthful and concise", type=ContextType.CONSTRAINT, authority=ContextAuthority.HARD_RULE)
        c5 = _make_item("User confirmed facts: 5 years experience, senior level", type=ContextType.FACT, authority=ContextAuthority.CONFIRMED)
        c7 = _make_item("History summary of 30 turns of conversation about interview prep", type=ContextType.SUMMARY, authority=ContextAuthority.ASSUMED)
        c9 = _make_item("User profile: backend engineer, Python expert", type=ContextType.PROFILE, authority=ContextAuthority.INFERRED)
        c13 = _make_item("Rejected old plan: too fabricated and generic", type=ContextType.FACT, authority=ContextAuthority.DENIED)

        items = [c1, c3, c5, c7, c9, c13]
        compressor = DynamicCompressor()

        # 17K remaining out of 32K total → ~53% → FULL mode
        # But let's test with a tighter budget to see compression
        result = compressor.compress_batch(
            items,
            budget_mode=BudgetMode.COMPACT,  # 5-20% remaining
        )

        # Critical items (c1, c3) should be L4
        c1_result = next(r for r in result if r.id == c1.id)
        c3_result = next(r for r in result if r.id == c3.id)
        assert c1_result.compression_level == CompressionLevel.L4
        assert c3_result.compression_level == CompressionLevel.L4

        # High importance (c5 confirmed) → L3 in COMPACT
        c5_result = next(r for r in result if r.id == c5.id)
        assert c5_result.compression_level == CompressionLevel.L3

        # Low importance (c7 summary) → L1 in COMPACT
        c7_result = next(r for r in result if r.id == c7.id)
        assert c7_result.compression_level == CompressionLevel.L1

        # Denied item (c13) — compressor classifies it as LOW → L1 in COMPACT.
        # In the full pipeline, DynamicSelector converts it to a reminder first.
        c13_result = next(r for r in result if r.id == c13.id)
        assert c13_result.compression_level == CompressionLevel.L1
