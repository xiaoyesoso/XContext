"""Tests for profile lifecycle, 4C stability scoring, and conflict resolution."""

from datetime import datetime, timedelta, timezone

import pytest

from app.core.profile_lifecycle import ProfileLifecycleManager
from app.core.stability_scorer import (
    confirmation_score,
    count_score,
    cross_context_score,
    status_from_score,
)
from app.core.user_profile import ProfileFact
from app.models import ProfileDimension


def make_fact(
    content: str = "likes small phones",
    dimension: ProfileDimension = ProfileDimension.PREFERENCE,
    time_level: str = "candidate",
    source: str = "conversation_inference",
    sessions: list[str] | None = None,
    evidence_count: int = 1,
    days_span: int = 0,
    confirmation: str = "none",
) -> ProfileFact:
    """Build a ProfileFact with lifecycle fields for testing."""
    now = datetime.now(timezone.utc)
    first = now - timedelta(days=days_span)
    return ProfileFact(
        dimension=dimension,
        content=content,
        time_level=time_level,  # type: ignore[arg-type]
        source=source,  # type: ignore[arg-type]
        session_ids=sessions or [],
        evidence_count=evidence_count,
        first_evidence_at=first,
        latest_evidence_at=now,
        last_evidence_at=now,
        confirmation_type=confirmation,  # type: ignore[arg-type]
    )


class TestStabilityScorer:
    """4C formula verification with documented examples."""

    def test_count_score_n1(self):
        assert count_score(1) == pytest.approx(7.087, abs=0.001)

    def test_count_score_n5(self):
        assert count_score(5) == pytest.approx(20.278, abs=0.001)

    def test_count_score_zero(self):
        assert count_score(0) == 0.0

    def test_cross_context_zero_for_one_session(self):
        assert cross_context_score(1) == 0.0

    def test_cross_context_cap_at_five_sessions(self):
        assert cross_context_score(5) == 25.0
        assert cross_context_score(10) == 25.0

    def test_confirmation_score_long_term(self):
        assert confirmation_score("long_term") == 25.0

    def test_status_mapping(self):
        assert status_from_score(10) == "candidate"
        assert status_from_score(40) == "weak"
        assert status_from_score(60) == "active"
        assert status_from_score(75) == "stable"
        assert status_from_score(95) == "confirmed"


class TestProfileLifecycleManager:
    """Lifecycle transitions, promotions, and conflict resolution."""

    def test_current_fact_stays_current(self):
        mgr = ProfileLifecycleManager()
        fact = make_fact(time_level="current")
        level, status = mgr.promote_or_demote(fact)
        assert level == "current"

    def test_low_score_stays_candidate(self):
        mgr = ProfileLifecycleManager()
        fact = make_fact(evidence_count=1, days_span=0, sessions=["s1"])
        level, status = mgr.promote_or_demote(fact)
        assert level == "candidate"

    def test_high_score_promotes_to_long_term(self):
        mgr = ProfileLifecycleManager()
        fact = make_fact(
            evidence_count=10,
            days_span=200,
            sessions=[f"s{i}" for i in range(5)],
            confirmation="long_term",
        )
        level, status = mgr.promote_or_demote(fact)
        assert level == "long-term"
        assert fact.stability_score >= 85

    def test_status_transition_to_weakened(self):
        mgr = ProfileLifecycleManager()
        now = datetime.now(timezone.utc)
        fact = make_fact()
        fact.status = "active"
        fact.latest_evidence_at = now - timedelta(days=100)
        new_status = mgr.transition_status(fact, now)
        assert new_status == "weakened"

    def test_status_transition_to_expired(self):
        mgr = ProfileLifecycleManager()
        now = datetime.now(timezone.utc)
        fact = make_fact()
        fact.status = "weakened"
        fact.latest_evidence_at = now - timedelta(days=200)
        new_status = mgr.transition_status(fact, now)
        assert new_status == "expired"

    def test_conflict_explicit_overrides_behavior(self):
        mgr = ProfileLifecycleManager()
        behavior = make_fact(
            content="price sensitive",
            source="behavior",
            time_level="candidate",
        )
        explicit = make_fact(
            content="not price sensitive",
            source="explicit_statement",
            time_level="current",
        )
        decisions = mgr.resolve_conflicts(explicit, [behavior])
        assert any(a == "downgrade" for _, a, _ in decisions)

    def test_offline_analyze_produces_events(self):
        mgr = ProfileLifecycleManager()
        facts = [
            make_fact(
                content="prefers light devices",
                evidence_count=5,
                days_span=200,
                sessions=[f"s{i}" for i in range(5)],
                confirmation="long_term",
            )
        ]
        report = mgr.offline_analyze("user-1", facts)
        assert report["analyzed_count"] == 1
        assert any(e["action"] == "promotion" for e in report["events"])

    def test_finalize_conversation_merges_duplicates(self):
        mgr = ProfileLifecycleManager()
        facts = [
            make_fact(content="likes coffee", sessions=["s1"]),
            make_fact(content="likes coffee", sessions=["s1"]),
        ]
        report = mgr.finalize_conversation("user-1", facts)
        assert report["merged_count"] == 1


class TestProfileFactModel:
    """Pydantic model validation for extended ProfileFact."""

    def test_default_lifecycle_fields(self):
        fact = ProfileFact(dimension=ProfileDimension.GOAL, content="wants a phone")
        assert fact.time_level == "candidate"
        assert fact.status == "active"
        assert fact.stability_score == 0.0
