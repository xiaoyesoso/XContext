"""Event sourcing and trusted snapshots for long-running tasks.

Key events are appended in time order; snapshots capture a trusted state
at key nodes. Current state can always be rebuilt as
"latest snapshot + subsequent untainted events", and a restore operation
marks everything after a snapshot as tainted so execution can resume
from that trusted point.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field
from uuid import uuid4


class EventType(str, Enum):
    """Kinds of key events recorded during a long task."""

    USER_INPUT = "user_input"
    AGENT_REPLY = "agent_reply"
    CONSTRAINT_CHANGED = "constraint_changed"
    STEP_COMPLETED = "step_completed"
    STEP_FAILED = "step_failed"
    OBSERVATION = "observation"
    FACT_CONFIRMED = "fact_confirmed"
    FACT_REFUTED = "fact_refuted"
    DRIFT_DETECTED = "drift_detected"
    CHECKPOINT_PASSED = "checkpoint_passed"
    CHECKPOINT_FAILED = "checkpoint_failed"
    REPLAN = "replan"
    SNAPSHOT_CREATED = "snapshot_created"
    RESTORE = "restore"


class TaskEvent(BaseModel):
    """A single key event in the session event stream."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    seq: int = 0
    type: EventType
    turn: int = 0
    payload: dict = Field(default_factory=dict)
    item_ids: list[str] = Field(default_factory=list)
    tainted: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class TaskSnapshot(BaseModel):
    """A trusted point-in-time capture of task state."""

    snapshot_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    at_seq: int = 0
    turn: int = 0
    goal: Optional[str] = None
    hard_constraints: list[str] = Field(default_factory=list)
    confirmed_facts: list[str] = Field(default_factory=list)
    task_state: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EventStore:
    """In-memory event-sourcing store with snapshot + restore support."""

    def __init__(self) -> None:
        self._events: dict[str, list[TaskEvent]] = {}
        self._snapshots: dict[str, list[TaskSnapshot]] = {}

    # ------------------------------------------------------------------ events

    def append(
        self,
        session_id: str,
        type: EventType,
        payload: Optional[dict] = None,
        item_ids: Optional[list[str]] = None,
        turn: int = 0,
    ) -> TaskEvent:
        """Append a key event with a per-session monotonic sequence number."""
        stream = self._events.setdefault(session_id, [])
        event = TaskEvent(
            session_id=session_id,
            seq=len(stream) + 1,
            type=type,
            turn=turn,
            payload=payload or {},
            item_ids=item_ids or [],
        )
        stream.append(event)
        return event

    def list_events(
        self, session_id: str, include_tainted: bool = False
    ) -> list[TaskEvent]:
        """Return the event stream; tainted events hidden by default."""
        stream = self._events.get(session_id, [])
        if include_tainted:
            return list(stream)
        return [e for e in stream if not e.tainted]

    def count(self, session_id: str) -> int:
        """Return total recorded events (including tainted)."""
        return len(self._events.get(session_id, []))

    # --------------------------------------------------------------- snapshots

    def create_snapshot(
        self,
        session_id: str,
        goal: Optional[str] = None,
        hard_constraints: Optional[list[str]] = None,
        confirmed_facts: Optional[list[str]] = None,
        task_state: Optional[dict] = None,
        turn: int = 0,
    ) -> TaskSnapshot:
        """Capture a trusted snapshot covering all events so far."""
        stream = self._events.get(session_id, [])
        snapshot = TaskSnapshot(
            session_id=session_id,
            at_seq=stream[-1].seq if stream else 0,
            turn=turn,
            goal=goal,
            hard_constraints=hard_constraints or [],
            confirmed_facts=confirmed_facts or [],
            task_state=task_state or {},
        )
        self._snapshots.setdefault(session_id, []).append(snapshot)
        self.append(
            session_id,
            EventType.SNAPSHOT_CREATED,
            payload={"snapshot_id": snapshot.snapshot_id},
            turn=turn,
        )
        return snapshot

    def list_snapshots(self, session_id: str) -> list[TaskSnapshot]:
        """Return snapshots oldest-first."""
        return list(self._snapshots.get(session_id, []))

    def get_snapshot(self, session_id: str, snapshot_id: str) -> Optional[TaskSnapshot]:
        """Find a snapshot by id."""
        for snap in self._snapshots.get(session_id, []):
            if snap.snapshot_id == snapshot_id:
                return snap
        return None

    # ----------------------------------------------------------------- restore

    def restore(self, session_id: str, snapshot_id: str) -> Optional[dict]:
        """Roll back to a trusted snapshot.

        Events after the snapshot are marked tainted (kept for audit, hidden
        from rebuild). Returns the rebuilt state, or None if not found.
        """
        snapshot = self.get_snapshot(session_id, snapshot_id)
        if snapshot is None:
            return None
        stream = self._events.get(session_id, [])
        for event in stream:
            if event.seq > snapshot.at_seq:
                event.tainted = True
        restore_event = self.append(
            session_id,
            EventType.RESTORE,
            payload={"snapshot_id": snapshot_id, "restored_to_seq": snapshot.at_seq},
        )
        restore_event.tainted = True
        return self.rebuild_state(session_id)

    def rebuild_state(self, session_id: str) -> dict:
        """Rebuild current state as latest snapshot + subsequent valid events."""
        snapshots = self._snapshots.get(session_id, [])
        base: dict = {
            "goal": None,
            "hard_constraints": [],
            "confirmed_facts": [],
            "task_state": {},
            "from_snapshot": None,
            "applied_events": 0,
        }
        last_seq = 0
        if snapshots:
            latest = snapshots[-1]
            base.update(
                {
                    "goal": latest.goal,
                    "hard_constraints": list(latest.hard_constraints),
                    "confirmed_facts": list(latest.confirmed_facts),
                    "task_state": dict(latest.task_state),
                    "from_snapshot": latest.snapshot_id,
                }
            )
            last_seq = latest.at_seq

        applied = 0
        for event in self.list_events(session_id):
            if event.seq <= last_seq:
                continue
            self._apply_event(base, event)
            applied += 1
        base["applied_events"] = applied
        return base

    @staticmethod
    def _apply_event(state: dict, event: TaskEvent) -> None:
        """Fold one untainted event into the rebuilt state."""
        if event.type == EventType.CONSTRAINT_CHANGED:
            new_value = event.payload.get("new")
            if new_value and new_value not in state["hard_constraints"]:
                old_value = event.payload.get("old")
                if old_value in state["hard_constraints"]:
                    state["hard_constraints"].remove(old_value)
                state["hard_constraints"].append(new_value)
        elif event.type == EventType.FACT_CONFIRMED:
            fact = event.payload.get("fact")
            if fact and fact not in state["confirmed_facts"]:
                state["confirmed_facts"].append(fact)
        elif event.type == EventType.FACT_REFUTED:
            fact = event.payload.get("fact")
            if fact in state["confirmed_facts"]:
                state["confirmed_facts"].remove(fact)
        elif event.type == EventType.USER_INPUT and not state.get("goal"):
            state["goal"] = event.payload.get("text")
