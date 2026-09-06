"""Checkpoint calibration for long-running tasks.

A checkpoint pauses execution at a strategic node and checks five
dimensions: goal alignment, constraints intact, state accuracy,
evidence sufficiency, and the gap to completion. It is the concrete
implementation of the Plan/ReAct correction point shown in the source
images.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from app.core.drift import ContextVerifier, DriftReport
from app.core.task_events import EventStore, EventType
from app.models import ContextItem


class CheckpointReport(BaseModel):
    """Result of a five-dimensional checkpoint check."""

    session_id: str
    turn: int = 0
    goal_aligned: bool = True
    constraints_intact: bool = True
    state_accurate: bool = True
    evidence_sufficient: bool = True
    gap: str = ""
    passed: bool = True
    notes: list[str] = Field(default_factory=list)
    drift_report: Optional[dict] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CheckpointManager:
    """Decides when and how to run checkpoint checks."""

    def __init__(
        self,
        event_store: EventStore,
        verifier: Optional[ContextVerifier] = None,
        turn_interval: int = 5,
        usage_threshold: float = 0.8,
        on_denied_violation: bool = True,
    ) -> None:
        self._store = event_store
        self._verifier = verifier or ContextVerifier()
        self.turn_interval = turn_interval
        self.usage_threshold = usage_threshold
        self.on_denied_violation = on_denied_violation

    def should_checkpoint(
        self,
        turn: int,
        usage_ratio: float,
        denied_violation: bool = False,
    ) -> bool:
        """Return True when a checkpoint should fire."""
        if turn > 0 and turn % self.turn_interval == 0:
            return True
        if usage_ratio >= self.usage_threshold:
            return True
        if self.on_denied_violation and denied_violation:
            return True
        return False

    def run_checkpoint(
        self,
        session_id: str,
        items: list[ContextItem],
        goal: Optional[str] = None,
        task_state: Optional[dict] = None,
        turn: int = 0,
    ) -> CheckpointReport:
        """Execute the five-dimensional check and emit an event."""
        report = CheckpointReport(session_id=session_id, turn=turn)

        # 1. Goal alignment (re-use verifier overlap logic).
        drift = self._verifier.verify(session_id, items, goal=goal, turn=turn)
        report.goal_aligned = not any(
            f.drift_type.value == "goal_drift" for f in drift.findings
        )

        # 2. Constraints intact.
        constraint_drift = any(
            f.drift_type.value == "constraint_drift" for f in drift.findings
        ) or any(f.drift_type.value == "detail_loss" for f in drift.findings)
        report.constraints_intact = not constraint_drift

        # 3. State accuracy.
        report.state_accurate = not any(
            f.drift_type.value == "state_drift" for f in drift.findings
        )

        # 4. Evidence sufficiency.
        report.evidence_sufficient = not any(
            f.drift_type.value == "evidence_drift" for f in drift.findings
        )

        # 5. Gap analysis: inspect task state and recent model output.
        report.gap = self._gap_summary(items, task_state)

        report.passed = (
            report.goal_aligned
            and report.constraints_intact
            and report.state_accurate
            and report.evidence_sufficient
        )
        report.drift_report = drift.model_dump(mode="json")
        report.notes = [f.description for f in drift.findings]

        event_type = (
            EventType.CHECKPOINT_PASSED
            if report.passed
            else EventType.CHECKPOINT_FAILED
        )
        self._store.append(
            session_id,
            event_type,
            payload=report.model_dump(mode="json"),
            turn=turn,
        )
        return report

    @staticmethod
    def _gap_summary(
        items: list[ContextItem],
        task_state: Optional[dict],
    ) -> str:
        """Heuristic answer to the question of what is still missing."""
        if task_state:
            missing = task_state.get("missing_context") or []
            if missing:
                return "Missing: " + "; ".join(missing)
        recent_agent = [
            i.content_as_string() if hasattr(i, "content_as_string") else str(i.content)
            for i in items[-3:]
            if i.type.value == "model_output"
        ]
        if recent_agent:
            text = "\n".join(recent_agent)
            if "还需要" in text or "仍需" in text or "缺少" in text or "missing" in text.lower():
                return "Agent has acknowledged missing information; waiting on evidence"
        return "No obvious gap detected"
