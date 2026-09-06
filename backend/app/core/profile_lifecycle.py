"""Profile lifecycle management: status transitions, promotion/demotion,
and conflict resolution for user-profile facts.

Facts move through time levels (current -> candidate -> long-term) and
statuses (active -> weakened -> expired) based on 4C stability scores and
evidence freshness.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.core.stability_scorer import (
    days_between,
    score_4c,
    status_from_score,
    utc_now,
)
from app.core.user_profile import ProfileFact


class LifecycleEvent:
    """Audit record for a profile-lifecycle state change."""

    def __init__(
        self,
        user_id: str,
        fact_id: str,
        action: str,
        reason: str,
        previous_status: Optional[str] = None,
        new_status: Optional[str] = None,
        previous_time_level: Optional[str] = None,
        new_time_level: Optional[str] = None,
        conflict_with: Optional[str] = None,
    ):
        self.event_id = str(uuid4())
        self.user_id = user_id
        self.fact_id = fact_id
        self.action = action
        self.reason = reason
        self.previous_status = previous_status
        self.new_status = new_status
        self.previous_time_level = previous_time_level
        self.new_time_level = new_time_level
        self.conflict_with = conflict_with
        self.created_at = utc_now()

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "action": self.action,
            "reason": self.reason,
            "previous_status": self.previous_status,
            "new_status": self.new_status,
            "previous_time_level": self.previous_time_level,
            "new_time_level": self.new_time_level,
            "conflict_with": self.conflict_with,
            "created_at": self.created_at.isoformat(),
        }


class ProfileLifecycleManager:
    """Manages profile-fact lifecycle and conflict resolution."""

    # Default TTL rules: active weakens after 90 days without new evidence;
    # weakened expires after another 90 days.
    ACTIVE_TO_WEAKENED_DAYS = 90
    WEAKENED_TO_EXPIRED_DAYS = 90

    def __init__(
        self,
        active_to_weakened_days: int = ACTIVE_TO_WEAKENED_DAYS,
        weakened_to_expired_days: int = WEAKENED_TO_EXPIRED_DAYS,
        stability_horizon_days: int = 180,
    ):
        self.active_to_weakened_days = active_to_weakened_days
        self.weakened_to_expired_days = weakened_to_expired_days
        self.stability_horizon_days = stability_horizon_days
        self._events: dict[str, list[LifecycleEvent]] = {}

    # ------------------------------------------------------------- transitions

    def transition_status(self, fact: ProfileFact, now: datetime | None = None) -> str:
        """Move a fact through active -> weakened -> expired based on TTL."""
        if now is None:
            now = utc_now()

        if fact.status == "expired":
            return fact.status

        days_since_evidence = days_between(now, fact.latest_evidence_at)

        if fact.status == "active":
            if days_since_evidence >= self.active_to_weakened_days:
                return "weakened"
            return "active"

        if fact.status == "weakened":
            if days_since_evidence >= (
                self.active_to_weakened_days + self.weakened_to_expired_days
            ):
                return "expired"
            return "weakened"

        return fact.status or "active"

    def score_fact(self, fact: ProfileFact) -> dict:
        """Recompute 4C stability score for a fact."""
        return score_4c(
            evidence_count=fact.evidence_count or 1,
            first_evidence_at=fact.first_evidence_at,
            latest_evidence_at=fact.latest_evidence_at,
            independent_sessions=len(fact.session_ids or []),
            confirmation_kind=fact.confirmation_type or "none",
            horizon_days=self.stability_horizon_days,
        )

    def promote_or_demote(self, fact: ProfileFact) -> tuple[str, str]:
        """Return the recommended (time_level, status) for a fact.

        Uses the 4C stability score and the current evidence state. Writes
        the computed sub-scores back into the fact.
        """
        scores = self.score_fact(fact)
        fact.stability_score = scores["stability_score"]
        fact.score_count = scores["score_count"]
        fact.score_continuity = scores["score_continuity"]
        fact.score_cross_context = scores["score_cross_context"]
        fact.score_confirmation = scores["score_confirmation"]
        score_status = status_from_score(fact.stability_score)

        # current-level facts are always transient; only candidate/long-term
        # are decided by stability.
        if fact.time_level == "current":
            return "current", fact.status or "active"

        if fact.stability_score >= 85:
            return "long-term", "active"
        if fact.stability_score >= 70:
            return "long-term", score_status
        if fact.stability_score >= 30:
            return "candidate", score_status
        return "candidate", "candidate"

    # ------------------------------------------------------------ conflicts

    def resolve_conflicts(
        self,
        new_fact: ProfileFact,
        existing_facts: list[ProfileFact],
    ) -> list[tuple[ProfileFact, str, str]]:
        """Detect conflicts between a new fact and existing facts.

        Returns a list of (existing_fact, action, reason) tuples.
        Actions: keep, downgrade, replace.
        """
        results: list[tuple[ProfileFact, str, str]] = []
        for existing in existing_facts:
            if existing.id == new_fact.id:
                continue
            if not self._same_domain(new_fact, existing):
                results.append((existing, "keep", "different domain"))
                continue

            action, reason = self._compare_facts(new_fact, existing)
            results.append((existing, action, reason))
        return results

    @staticmethod
    def _same_domain(a: ProfileFact, b: ProfileFact) -> bool:
        """Two facts are in the same domain if dimension matches."""
        return a.dimension == b.dimension

    def _compare_facts(
        self, new_fact: ProfileFact, existing: ProfileFact
    ) -> tuple[str, str]:
        """Apply ordered conflict-resolution rules."""
        # Rule 1: current explicit > historical inference.
        if new_fact.time_level == "current" and new_fact.source in (
            "explicit_statement",
            "external_system",
        ):
            if existing.time_level in ("candidate", "long-term"):
                if existing.source in ("behavior", "conversation_inference"):
                    return "downgrade", "current explicit overrides historical inference"

        # Rule 2: explicit > behavior.
        if new_fact.source == "explicit_statement" and existing.source == "behavior":
            return "downgrade", "explicit statement overrides behavior inference"

        # Rule 3: user correction wins.
        if getattr(new_fact, "is_correction", False):
            return "downgrade", "user correction overrides older fact"

        # Rule 4: domain-specific > global.
        if new_fact.domain and existing.domain:
            if new_fact.domain != existing.domain and existing.domain == "global":
                return "keep", "domain-specific fact does not override global"

        # Rule 5: more recent evidence wins for same source/domain.
        if new_fact.source == existing.source and new_fact.dimension == existing.dimension:
            if (
                new_fact.latest_evidence_at
                and existing.latest_evidence_at
                and new_fact.latest_evidence_at > existing.latest_evidence_at
            ):
                return "replace", "newer evidence replaces older same-source fact"

        return "keep", "no conflict rule matched"

    # -------------------------------------------------------- offline analyze

    def offline_analyze(
        self,
        user_id: str,
        facts: list[ProfileFact],
    ) -> dict:
        """Run the full offline lifecycle pipeline.

        Applies status transitions, 4C scoring, promotion/demotion, and
        conflict resolution. Mutates facts in place and returns a report.
        """
        now = utc_now()
        events: list[LifecycleEvent] = []

        # Step 1: TTL transitions and score refresh.
        for fact in facts:
            old_status = fact.status
            fact.status = self.transition_status(fact, now)
            if fact.status != old_status:
                events.append(
                    LifecycleEvent(
                        user_id=user_id,
                        fact_id=fact.id,
                        action="status_transition",
                        reason="TTL: no new evidence",
                        previous_status=old_status,
                        new_status=fact.status,
                    )
                )

            scores = self.score_fact(fact)
            fact.stability_score = scores["stability_score"]
            fact.score_count = scores["score_count"]
            fact.score_continuity = scores["score_continuity"]
            fact.score_cross_context = scores["score_cross_context"]
            fact.score_confirmation = scores["score_confirmation"]

            old_time_level = fact.time_level
            new_time_level, new_status = self.promote_or_demote(fact)
            if new_time_level != old_time_level:
                events.append(
                    LifecycleEvent(
                        user_id=user_id,
                        fact_id=fact.id,
                        action="promotion",
                        reason=f"stability_score={fact.stability_score}",
                        previous_time_level=old_time_level,
                        new_time_level=new_time_level,
                        previous_status=fact.status,
                        new_status=new_status,
                    )
                )
            fact.time_level = new_time_level
            if new_status != "candidate":  # candidate is not a real status
                fact.status = new_status

        # Step 2: conflict resolution across facts.
        active_facts = [f for f in facts if f.status != "expired"]
        for fact in active_facts:
            others = [f for f in active_facts if f.id != fact.id]
            decisions = self.resolve_conflicts(fact, others)
            for other, action, reason in decisions:
                if action == "downgrade":
                    other.status = "weakened"
                    events.append(
                        LifecycleEvent(
                            user_id=user_id,
                            fact_id=other.id,
                            action="conflict_downgrade",
                            reason=reason,
                            previous_status="active",
                            new_status="weakened",
                            conflict_with=fact.id,
                        )
                    )
                elif action == "replace":
                    other.status = "expired"
                    events.append(
                        LifecycleEvent(
                            user_id=user_id,
                            fact_id=other.id,
                            action="conflict_replace",
                            reason=reason,
                            previous_status=other.status,
                            new_status="expired",
                            conflict_with=fact.id,
                        )
                    )

        self._events.setdefault(user_id, []).extend(events)
        return {
            "analyzed_count": len(facts),
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }

    def finalize_conversation(
        self,
        user_id: str,
        session_facts: list[ProfileFact],
    ) -> dict:
        """End-of-conversation consolidation.

        Merges duplicate facts within the same session and runs a lightweight
        conflict check. Facts are kept at current/candidate level; long-term
        promotion is left to offline analysis.
        """
        now = utc_now()
        merged: dict[str, ProfileFact] = {}
        for fact in session_facts:
            key = f"{fact.dimension.value}:{fact.content.strip().lower()}"
            if key in merged:
                existing = merged[key]
                existing.evidence_count = max(
                    existing.evidence_count or 1, fact.evidence_count or 1
                )
                if fact.latest_evidence_at and (
                    existing.latest_evidence_at is None
                    or fact.latest_evidence_at > existing.latest_evidence_at
                ):
                    existing.latest_evidence_at = fact.latest_evidence_at
                if fact.session_id and fact.session_id not in (
                    existing.session_ids or []
                ):
                    existing.session_ids = list(existing.session_ids or []) + [
                        fact.session_id
                    ]
            else:
                merged[key] = fact

        # Ensure current facts have fresh timestamps.
        for fact in merged.values():
            if fact.time_level == "current":
                fact.latest_evidence_at = now
                fact.last_evidence_at = now

        # Run a lightweight conflict pass, only downgrading contradicted
        # candidate facts within the same conversation.
        events: list[LifecycleEvent] = []
        for fact in merged.values():
            others = [f for f in merged.values() if f.id != fact.id]
            for other, action, reason in self.resolve_conflicts(fact, others):
                if action in ("downgrade", "replace") and other.status != "expired":
                    other.status = "weakened"
                    events.append(
                        LifecycleEvent(
                            user_id=user_id,
                            fact_id=other.id,
                            action="conversation_conflict",
                            reason=reason,
                            previous_status="active",
                            new_status="weakened",
                            conflict_with=fact.id,
                        )
                    )

        self._events.setdefault(user_id, []).extend(events)
        return {
            "merged_count": len(merged),
            "event_count": len(events),
            "events": [e.to_dict() for e in events],
        }

    def list_events(self, user_id: str) -> list[LifecycleEvent]:
        """Return lifecycle events for a user, newest first."""
        return list(reversed(self._events.get(user_id, [])))
