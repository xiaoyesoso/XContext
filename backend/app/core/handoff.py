"""Phase handoff for long-running tasks.

A handoff is a structured boundary between two phases: instead of
passing the entire bloated context to the next phase, only the
essential continuation state is forwarded. The original phase details
remain available for targeted recall when needed.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field
from uuid import uuid4

from app.core.drift import extract_keywords
from app.models import ContextItem


class PhaseHandoff(BaseModel):
    """Structured context handoff between long-task phases."""

    handoff_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    phase_name: str
    goal: Optional[str] = None
    completed_work: list[str] = Field(default_factory=list)
    confirmed_facts: list[str] = Field(default_factory=list)
    active_constraints: list[str] = Field(default_factory=list)
    evidence_index: list[dict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    next_inputs: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HandoffBuilder:
    """Build a PhaseHandoff from context items and an event stream."""

    @staticmethod
    def build(
        session_id: str,
        phase_name: str,
        items: list[ContextItem],
        events: Optional[list] = None,
        goal: Optional[str] = None,
    ) -> PhaseHandoff:
        """Construct a handoff summarising the completed phase."""
        events = events or []
        work: list[str] = []
        facts: list[str] = []
        constraints: list[str] = []
        evidence_index: list[dict] = []
        questions: list[str] = []

        for item in items:
            content = (
                item.content_as_string()
                if hasattr(item, "content_as_string")
                else str(item.content)
            )
            if item.type.value == "tool_result":
                work.append(content[:120])
            elif item.type.value == "fact" and item.authority.value == "confirmed":
                facts.append(content)
            elif item.type.value == "constraint" or item.authority.value == "hard_rule":
                constraints.append(content)
            elif item.type.value in ("tool_result", "observation"):
                keywords = extract_keywords(content, limit=3)
                evidence_index.append(
                    {"item_id": item.id, "keywords": keywords, "summary": content[:100]}
                )

        for event in events:
            if event.type.value == "step_completed":
                work.append(event.payload.get("summary", "completed step"))
            elif event.type.value == "fact_confirmed":
                fact = event.payload.get("fact")
                if fact and fact not in facts:
                    facts.append(fact)
            elif event.type.value == "constraint_changed":
                new_constraint = event.payload.get("new")
                if new_constraint:
                    constraints.append(new_constraint)

        if not goal and items:
            user_inputs = [i for i in items if i.type.value == "user_input"]
            if user_inputs:
                goal = (
                    user_inputs[0].content_as_string()
                    if hasattr(user_inputs[0], "content_as_string")
                    else str(user_inputs[0].content)
                )

        return PhaseHandoff(
            session_id=session_id,
            phase_name=phase_name,
            goal=goal,
            completed_work=work[:20],
            confirmed_facts=facts[:20],
            active_constraints=constraints[:20],
            evidence_index=evidence_index[:20],
            open_questions=questions,
            next_inputs=HandoffBuilder._infer_next_inputs(items, events),
        )

    @staticmethod
    def _infer_next_inputs(
        items: list[ContextItem],
        events: Optional[list],
    ) -> list[str]:
        """Heuristic list of what the next phase likely needs."""
        suggestions: list[str] = []
        if any(i.type.value == "tool_result" for i in items):
            suggestions.append("analysis of collected tool results")
        if any("?" in (
            i.content_as_string() if hasattr(i, "content_as_string") else str(i.content)
        ) for i in items[-5:]):
            suggestions.append("answers to outstanding questions")
        return suggestions or ["continue toward the stated goal"]


class HandoffStore:
    """In-memory store for phase handoffs and cross-phase recall."""

    def __init__(self) -> None:
        self._handoffs: dict[str, PhaseHandoff] = {}
        self._phase_items: dict[str, list[ContextItem]] = {}

    def save(self, handoff: PhaseHandoff, items: list[ContextItem]) -> None:
        """Persist the handoff and keep the original phase items for recall."""
        self._handoffs[handoff.handoff_id] = handoff
        self._phase_items[handoff.handoff_id] = list(items)

    def list_handoffs(self, session_id: str) -> list[PhaseHandoff]:
        """Return handoffs for a session, newest first."""
        return sorted(
            [h for h in self._handoffs.values() if h.session_id == session_id],
            key=lambda h: h.created_at,
            reverse=True,
        )

    def get(self, handoff_id: str) -> Optional[PhaseHandoff]:
        """Return a handoff by id."""
        return self._handoffs.get(handoff_id)

    def recall(
        self,
        handoff_id: str,
        keywords: list[str],
        top_k: int = 5,
    ) -> list[dict]:
        """Recall original phase items matching the provided keywords."""
        items = self._phase_items.get(handoff_id, [])
        if not items or not keywords:
            return []

        def score(item: ContextItem) -> int:
            content = (
                item.content_as_string()
                if hasattr(item, "content_as_string")
                else str(item.content)
            )
            lower = content.lower()
            return sum(1 for kw in keywords if kw.lower() in lower)

        scored = [
            (score(item), item)
            for item in items
            if score(item) > 0
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [
            {
                "id": item.id,
                "type": item.type.value,
                "content": (
                    item.content_as_string()
                    if hasattr(item, "content_as_string")
                    else str(item.content)
                ),
                "score": s,
            }
            for s, item in scored[:top_k]
        ]
