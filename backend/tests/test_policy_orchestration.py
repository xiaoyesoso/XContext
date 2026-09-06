"""Tests for action-based context policy orchestration."""

from datetime import datetime, timezone

import pytest

from app.core.context_actions import ActionContext, ActionRegistry, ActionResult, ContextAction
from app.core.policy_orchestrator import OrchestratorConfig, PolicyOrchestrator
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextPolicy,
    ContextScope,
    ContextSource,
    ContextType,
    PolicySegment,
)


def item(item_type: ContextType, content: str, priority: int = 0) -> ContextItem:
    return ContextItem(
        type=item_type,
        content=content,
        source=ContextSource.USER,
        scope=ContextScope.CURRENT_TASK,
        authority=ContextAuthority.CONFIRMED,
        priority=priority,
        created_at=datetime.now(timezone.utc),
    )


class FixedAction(ContextAction):
    name = "fixed"

    async def execute(self, ctx: ActionContext) -> ActionResult:
        return ActionResult(
            action_name=self.name,
            items=[item(ContextType.FACT, "a fixed fact")],
            raw_tokens=3,
            compressed_tokens=3,
        )


@pytest.mark.asyncio
async def test_registry_and_policy_execution():
    registry = ActionRegistry()
    registry.register(FixedAction())
    orchestrator = PolicyOrchestrator(
        registry=registry,
        config=OrchestratorConfig(budget_tokens=4096, reserved_tokens=0),
    )
    policy = ContextPolicy(
        policy_id="tech",
        name="Tech policy",
        scenario="tech_support",
        segments=[PolicySegment(action_name="fixed", priority=10)],
    )
    assert orchestrator.upsert_policy(policy) == []
    execution = await orchestrator.execute(
        "session-1", "user-1", "tech_support", "help", []
    )
    assert execution.policy_id == "tech"
    assert execution.segments[0].item_count == 1
    assert execution.metrics.critical_retention == 1.0
    assert len(orchestrator.get_selected_items("session-1")) == 1


def test_unknown_action_is_rejected():
    orchestrator = PolicyOrchestrator()
    policy = ContextPolicy(
        policy_id="invalid",
        name="Invalid",
        scenario="test",
        segments=[PolicySegment(action_name="not_registered")],
    )
    assert orchestrator.upsert_policy(policy)


@pytest.mark.asyncio
async def test_budget_drops_low_priority_segment():
    registry = ActionRegistry()
    registry.register(FixedAction())
    orchestrator = PolicyOrchestrator(
        registry=registry,
        config=OrchestratorConfig(budget_tokens=1, reserved_tokens=0),
    )
    policy = ContextPolicy(
        policy_id="tight",
        name="Tight",
        scenario="tight",
        segments=[PolicySegment(action_name="fixed", priority=2)],
    )
    assert orchestrator.upsert_policy(policy) == []
    execution = await orchestrator.execute("s", "u", "tight", "x", [])
    assert execution.segments[0].dropped is True
    assert execution.metrics.dropped_segments == 1
