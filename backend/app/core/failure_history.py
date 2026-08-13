"""Failure history tracking for feedback-driven context selection."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from app.models import CompressionLevel, ContextType


@dataclass
class FailureRecord:
    """A single failure record linking a task type to a missing context type."""

    task_type: str
    missing_context_type: ContextType
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FailureHistoryTracker:
    """In-memory tracker for context-selection failures.

    When a class of tasks historically failed because a particular context
    type was missing, the Selector / Compressor can query this tracker to
    elevate the priority or minimum compression level for that type.
    """

    def __init__(self):
        self._records: list[FailureRecord] = []

    def record(self, task_type: str, missing_context_type: ContextType) -> None:
        """Record that a task failed due to a missing context type."""
        self._records.append(
            FailureRecord(task_type=task_type, missing_context_type=missing_context_type)
        )

    def get_missing_types(self, task_type: str) -> set[ContextType]:
        """Return the set of context types that were missing in past failures for the given task type."""
        return {
            r.missing_context_type
            for r in self._records
            if r.task_type == task_type
        }

    def minimum_level_for(
        self, context_type: ContextType, task_type: Optional[str] = None
    ) -> Optional[CompressionLevel]:
        """Return the minimum compression level for a context type.

        If the context type has been implicated in past failures (optionally
        filtered by task type), return L3 as the minimum to preserve more
        detail. Otherwise return None (no elevation).
        """
        if task_type is not None:
            missing = self.get_missing_types(task_type)
            if context_type in missing:
                return CompressionLevel.L3
        else:
            if any(r.missing_context_type == context_type for r in self._records):
                return CompressionLevel.L3
        return None

    def clear(self) -> None:
        """Remove all failure records."""
        self._records.clear()
