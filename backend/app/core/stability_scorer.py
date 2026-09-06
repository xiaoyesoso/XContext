"""4C stability scoring for user-profile facts.

Count, Continuity, Cross-context, Confirmation — combined into a 0-100
stability score that decides whether a profile fact is a fleeting signal
or a durable preference.
"""

from datetime import datetime, timezone
from math import exp, log


MAX_COUNT_SCORE = 25.0
MAX_CONTINUITY_SCORE = 25.0
MAX_CROSS_CONTEXT_SCORE = 25.0
MAX_CONFIRMATION_SCORE = 25.0


def count_score(n: int) -> float:
    """Score for independent occurrences of a preference.

    Saturation curve: 25 * (1 - exp(-n / 3)).
    """
    if n <= 0:
        return 0.0
    return round(MAX_COUNT_SCORE * (1 - exp(-n / 3)), 3)


def continuity_score(
    first_evidence_at: datetime,
    latest_evidence_at: datetime,
    horizon_days: int = 180,
) -> float:
    """Score for how long a preference has persisted in days.

    Uses 25 * min(1, ln(1 + d) / ln(1 + H)) where d is the span in days.
    """
    if first_evidence_at is None or latest_evidence_at is None:
        return 0.0
    delta = latest_evidence_at - first_evidence_at
    days = max(0, delta.total_seconds() / 86400.0)
    if days <= 0:
        return 0.0
    if horizon_days <= 0:
        return MAX_CONTINUITY_SCORE
    return round(
        MAX_CONTINUITY_SCORE * min(1.0, log(1 + days) / log(1 + horizon_days)),
        3,
    )


def cross_context_score(sessions: int) -> float:
    """Score for consistency across independent sessions/scenarios.

    25 * min(1, max(0, s - 1) / 4). Five independent sessions hit the cap.
    """
    if sessions <= 1:
        return 0.0
    return round(MAX_CROSS_CONTEXT_SCORE * min(1.0, (sessions - 1) / 4.0), 3)


def confirmation_score(kind: str) -> float:
    """Score for explicit user confirmation of a profile fact."""
    mapping = {
        "none": 0.0,
        "implicit": 2.5,       # no objection, but not explicit
        "hesitant": 10.0,      # "差不多"
        "scenario": 20.0,      # confirmed for a specific scenario
        "long_term": 25.0,     # confirmed as a durable preference
    }
    return mapping.get(kind, 0.0)


def score_4c(
    evidence_count: int,
    first_evidence_at: datetime | None,
    latest_evidence_at: datetime | None,
    independent_sessions: int,
    confirmation_kind: str,
    horizon_days: int = 180,
) -> dict[str, float]:
    """Compute the full 4C score breakdown and total.

    Returns a dict with the four sub-scores and the total stability_score.
    """
    s_count = count_score(evidence_count)
    s_continuity = continuity_score(
        first_evidence_at, latest_evidence_at, horizon_days
    )
    s_cross = cross_context_score(independent_sessions)
    s_confirmation = confirmation_score(confirmation_kind)
    total = round(
        min(100.0, s_count + s_continuity + s_cross + s_confirmation), 3
    )
    return {
        "score_count": s_count,
        "score_continuity": s_continuity,
        "score_cross_context": s_cross,
        "score_confirmation": s_confirmation,
        "stability_score": total,
    }


def status_from_score(stability_score: float) -> str:
    """Map a 0-100 stability score to a usage status."""
    if stability_score < 30:
        return "candidate"
    if stability_score < 50:
        return "weak"
    if stability_score < 70:
        return "active"
    if stability_score < 85:
        return "stable"
    return "confirmed"


def days_between(now: datetime, then: datetime) -> float:
    """Return the number of days between two datetimes (non-negative)."""
    if then is None:
        return float("inf")
    return max(0.0, (now - then).total_seconds() / 86400.0)


def utc_now() -> datetime:
    """Return timezone-aware UTC now."""
    return datetime.now(timezone.utc)
