from abc import ABC, abstractmethod
from datetime import datetime, timezone
from uuid import uuid4

from app.core.tokenizer import estimate_tokens
from app.models import ContextAuthority, ContextItem, ContextSource, ContextType


class Summarizer(ABC):
    """Abstract base for generating summaries from context items."""

    @abstractmethod
    async def summarize(self, items: list[ContextItem]) -> ContextItem:
        """Return a summary context item covering the given items."""


class MockSummarizer(Summarizer):
    """Placeholder summarizer for local development and testing.

    In production, replace this with a call to an external LLM API.
    """

    async def summarize(self, items: list[ContextItem]) -> ContextItem:
        topics = [item.content_as_string()[:40] for item in items]
        summary_text = "Summary of " + ", ".join(topics)
        return ContextItem(
            id=str(uuid4()),
            type=ContextType.SUMMARY,
            content=summary_text,
            source=ContextSource.AGENT,
            scope=items[-1].scope if items else None,
            authority=ContextAuthority.INFERRED,
            confidence=0.8,
            priority=0,
            token_cost=estimate_tokens(summary_text),
            layer=items[-1].layer if items else "session",
            created_at=datetime.now(timezone.utc),
        )
