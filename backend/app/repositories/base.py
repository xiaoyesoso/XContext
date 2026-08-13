from abc import ABC, abstractmethod

from app.models import ContextItem


class ContextRepository(ABC):
    """Abstract repository for context item persistence."""

    @abstractmethod
    async def add(self, session_id: str, item: ContextItem) -> ContextItem:
        """Persist a context item and return it."""

    @abstractmethod
    async def get(self, session_id: str, item_id: str) -> ContextItem | None:
        """Retrieve a single context item by id."""

    @abstractmethod
    async def list(self, session_id: str) -> list[ContextItem]:
        """List all context items for a session."""

    @abstractmethod
    async def delete(self, session_id: str, item_id: str) -> bool:
        """Delete a context item. Return True if it existed."""

    @abstractmethod
    async def clear(self, session_id: str) -> None:
        """Remove all context items for a session."""
