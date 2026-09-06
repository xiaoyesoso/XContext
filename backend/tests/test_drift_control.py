"""Tests for long-task drift control: drift detection, event sourcing,
checkpoints, snapshots, restore, and phase handoff.
"""

import pytest

from app.core.checkpoint import CheckpointManager
from app.core.drift import ContextVerifier, DriftType
from app.core.handoff import HandoffBuilder, HandoffStore
from app.core.task_events import EventStore, EventType
from app.models import ContextAuthority, ContextItem, ContextSource, ContextType


def item(
    content: str,
    type: ContextType = ContextType.MODEL_OUTPUT,
    authority: ContextAuthority = ContextAuthority.INFERRED,
    item_id: str = "i1",
) -> ContextItem:
    return ContextItem(
        id=item_id,
        type=type,
        content=content,
        source=ContextSource.USER,
        scope="current_session",
        authority=authority,
    )


class TestContextVerifier:
    """Rule-based drift detection."""

    def test_no_items_is_healthy(self):
        verifier = ContextVerifier()
        report = verifier.verify("s1", [], turn=1)
        assert report.healthy is True

    def test_goal_drift_detected(self):
        verifier = ContextVerifier(goal_overlap_threshold=0.05)
        items = [
            item("帮我修复空指针，尽量少改代码", ContextType.USER_INPUT, item_id="u1"),
            item("我已经完成了数据库迁移", ContextType.MODEL_OUTPUT, item_id="a1"),
            item("接下来重构前端路由", ContextType.MODEL_OUTPUT, item_id="a2"),
        ]
        report = verifier.verify("s1", items, turn=2)
        assert any(f.drift_type == DriftType.GOAL_DRIFT for f in report.findings)

    def test_detail_loss_hard_constraint(self):
        verifier = ContextVerifier()
        items = [
            item("A接口绝对不能修改", ContextType.USER_INPUT, item_id="u1"),
            item("尽量少修改A接口", ContextType.SUMMARY, item_id="s1"),
        ]
        report = verifier.verify("s1", items, turn=1)
        assert any(f.drift_type == DriftType.DETAIL_LOSS for f in report.findings)

    def test_state_drift_failure_unrecorded(self):
        verifier = ContextVerifier()
        events = [EventStore().append("s1", EventType.STEP_COMPLETED, payload={"summary": "step1"})]
        items = [
            item("迁移失败：主键冲突", ContextType.TOOL_RESULT, item_id="t1"),
        ]
        report = verifier.verify("s1", items, events=events, turn=1)
        assert any(f.drift_type == DriftType.STATE_DRIFT for f in report.findings)

    def test_error_accumulation(self):
        verifier = ContextVerifier()
        items = [
            item("日活用户数 error", ContextType.TOOL_RESULT, item_id="t1"),
            item("日活用户数 导致配置错误", ContextType.MODEL_OUTPUT, item_id="m1"),
            item("日活用户数 error 继续扩散", ContextType.MODEL_OUTPUT, item_id="m2"),
        ]
        report = verifier.verify("s1", items, turn=2)
        assert any(f.drift_type == DriftType.ERROR_ACCUMULATION for f in report.findings)


class TestEventStore:
    """Event sourcing and snapshot mechanics."""

    def test_append_increments_seq(self):
        store = EventStore()
        e1 = store.append("s1", EventType.USER_INPUT, payload={"text": "hello"})
        e2 = store.append("s1", EventType.AGENT_REPLY, payload={})
        assert e1.seq == 1
        assert e2.seq == 2

    def test_create_snapshot_then_restore(self):
        store = EventStore()
        store.append("s1", EventType.USER_INPUT, payload={"text": "goal"})
        store.append("s1", EventType.FACT_CONFIRMED, payload={"fact": "f1"})
        snap = store.create_snapshot(
            "s1",
            goal="goal",
            hard_constraints=["c1"],
            confirmed_facts=["f1"],
            turn=2,
        )
        store.append("s1", EventType.FACT_REFUTED, payload={"fact": "f1"})
        state = store.restore("s1", snap.snapshot_id)
        assert state is not None
        assert state["confirmed_facts"] == ["f1"]
        assert state["hard_constraints"] == ["c1"]

    def test_tainted_events_hidden(self):
        store = EventStore()
        store.append("s1", EventType.USER_INPUT, payload={"text": "goal"})
        snap = store.create_snapshot("s1", turn=1)
        store.append("s1", EventType.AGENT_REPLY, payload={})
        store.restore("s1", snap.snapshot_id)
        visible = store.list_events("s1")
        assert all(e.type == EventType.USER_INPUT for e in visible)


class TestCheckpointManager:
    """Checkpoint trigger strategy and five-dim report."""

    def test_turn_interval_trigger(self):
        mgr = CheckpointManager(EventStore(), turn_interval=5)
        assert mgr.should_checkpoint(5, 0.0) is True
        assert mgr.should_checkpoint(4, 0.0) is False

    def test_usage_threshold_trigger(self):
        mgr = CheckpointManager(EventStore(), usage_threshold=0.8)
        assert mgr.should_checkpoint(1, 0.85) is True
        assert mgr.should_checkpoint(1, 0.5) is False

    def test_checkpoint_passes_when_healthy(self):
        store = EventStore()
        mgr = CheckpointManager(store)
        items = [item("goal", ContextType.USER_INPUT, item_id="u1")]
        report = mgr.run_checkpoint("s1", items, turn=1)
        assert report.passed is True
        assert any(e.type == EventType.CHECKPOINT_PASSED for e in store.list_events("s1"))

    def test_checkpoint_fails_on_drift(self):
        store = EventStore()
        mgr = CheckpointManager(store)
        items = [
            item("A接口绝对不能修改", ContextType.USER_INPUT, item_id="u1"),
            item("尽量少修改A接口", ContextType.SUMMARY, item_id="s1"),
        ]
        report = mgr.run_checkpoint("s1", items, turn=1)
        assert report.passed is False
        assert any(e.type == EventType.CHECKPOINT_FAILED for e in store.list_events("s1"))


class TestHandoff:
    """Phase handoff construction and recall."""

    def test_build_handoff(self):
        items = [
            item("帮我修空指针", ContextType.USER_INPUT, item_id="u1"),
            item("5000预算", ContextType.CONSTRAINT, ContextAuthority.HARD_RULE, "c1"),
            item("已确认是NPE", ContextType.FACT, ContextAuthority.CONFIRMED, "f1"),
        ]
        handoff = HandoffBuilder.build("s1", "分析阶段", items)
        assert handoff.phase_name == "分析阶段"
        assert "5000预算" in handoff.active_constraints
        assert "已确认是NPE" in handoff.confirmed_facts

    def test_recall_from_handoff(self):
        store = HandoffStore()
        items = [
            item("价格3000", ContextType.TOOL_RESULT, item_id="t1"),
            item("性能很强", ContextType.TOOL_RESULT, item_id="t2"),
        ]
        handoff = HandoffBuilder.build("s1", "收集", items)
        store.save(handoff, items)
        results = store.recall(handoff.handoff_id, ["价格"])
        assert len(results) == 1
        assert results[0]["id"] == "t1"


class TestDriftModels:
    def test_drift_report_refresh_health(self):
        report = ContextVerifier().verify("s1", [], turn=0)
        assert report.healthy is True
        report.findings.append(
            ContextVerifier().verify("s1", [item("x")], turn=1).findings[0]
            if ContextVerifier().verify("s1", [item("x")], turn=1).findings
            else None
        )
        # Test refresh method directly.
        report.findings = []
        assert report.refresh_health().healthy is True
