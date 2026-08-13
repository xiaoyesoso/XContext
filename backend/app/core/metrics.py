from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.models import BudgetMode


@dataclass
class WindowMetrics:
    """Metrics collected during a single window composition."""

    session_id: str
    retrieved_count: int = 0
    selected_count: int = 0
    compressed_count: int = 0
    ordered_count: int = 0
    total_context_tokens: int = 0
    window_tokens: int = 0
    budget_mode: Optional[BudgetMode] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class MetricsCollector:
    """In-memory collector for context-window metrics per session."""

    def __init__(self):
        self._store: dict[str, list[WindowMetrics]] = {}

    def record(self, session_id: str, metrics: WindowMetrics) -> None:
        """Record metrics for a session."""
        self._store.setdefault(session_id, []).append(metrics)

    def get_latest(self, session_id: str) -> WindowMetrics | None:
        """Return the most recent metrics for a session."""
        records = self._store.get(session_id, [])
        return records[-1] if records else None

    def get_history(self, session_id: str) -> list[WindowMetrics]:
        """Return all recorded metrics for a session."""
        return list(self._store.get(session_id, []))
