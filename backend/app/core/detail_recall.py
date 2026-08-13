"""Detail recall via search over archived/historical context.

Provides a pluggable interface supporting keyword search, vector search,
and hybrid search. Implements the K-turn raw-window strategy to avoid
data synchronization latency issues.
"""

from abc import ABC, abstractmethod
from typing import Optional

from app.models import ContextItem


class DetailRetriever(ABC):
    """Abstract interface for retrieving historical context details."""

    @abstractmethod
    async def recall(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[ContextItem]:
        """Retrieve relevant context items matching the query.

        Args:
            query: The search query (keyword or semantic).
            session_id: Optional session scope; if None, search all sessions.
            top_k: Maximum number of results to return.

        Returns:
            A list of matching ContextItems, ranked by relevance.
        """

    @abstractmethod
    async def index(self, item: ContextItem) -> None:
        """Index a context item for future retrieval.

        Called asynchronously after each turn to keep the retrieval
        store up-to-date. Data sync latency is tolerated because the
        last K turns are always kept raw in-window.
        """


class InMemoryDetailRetriever(DetailRetriever):
    """Simple in-memory keyword-based retriever for testing.

    Uses substring matching on content. Suitable for unit tests and
    local development; production should use ES or a vector DB.
    """

    def __init__(self):
        self._store: list[ContextItem] = []

    async def recall(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[ContextItem]:
        """Retrieve items whose content contains the query keywords."""
        query_lower = query.lower()
        scored: list[tuple[float, ContextItem]] = []
        for item in self._store:
            if session_id and item.layer != session_id:
                # In-memory store doesn't track session per item directly;
                # we rely on layer field as a proxy for session grouping.
                pass
            content = item.content_as_string().lower()
            if query_lower in content:
                # Simple scoring: count keyword occurrences.
                score = content.count(query_lower)
                scored.append((score, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]

    async def index(self, item: ContextItem) -> None:
        """Add an item to the in-memory retrieval store."""
        self._store.append(item)


class HybridDetailRetriever(DetailRetriever):
    """Hybrid retriever combining keyword and vector search.

    Delegates to keyword and vector backends, then merges and reranks
    results. This is the recommended production retriever.
    """

    def __init__(
        self,
        keyword_retriever: DetailRetriever,
        vector_retriever: DetailRetriever,
        keyword_weight: float = 0.4,
        vector_weight: float = 0.6,
    ):
        self._keyword = keyword_retriever
        self._vector = vector_retriever
        self._kw_weight = keyword_weight
        self._vec_weight = vector_weight

    async def recall(
        self,
        query: str,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[ContextItem]:
        """Retrieve and merge results from both backends."""
        # Fetch from both backends with extra headroom for merging.
        fetch_k = top_k * 2
        kw_results = await self._keyword.recall(query, session_id, fetch_k)
        vec_results = await self._vector.recall(query, session_id, fetch_k)

        # Merge with weighted scores (rank-based fusion).
        scores: dict[str, float] = {}
        items_by_id: dict[str, ContextItem] = {}

        for rank, item in enumerate(kw_results):
            score = self._kw_weight * (1.0 / (rank + 1))
            scores[item.id] = scores.get(item.id, 0.0) + score
            items_by_id[item.id] = item

        for rank, item in enumerate(vec_results):
            score = self._vec_weight * (1.0 / (rank + 1))
            scores[item.id] = scores.get(item.id, 0.0) + score
            items_by_id[item.id] = item

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [items_by_id[item_id] for item_id, _ in ranked[:top_k]]

    async def index(self, item: ContextItem) -> None:
        """Index the item in both backends."""
        await self._keyword.index(item)
        await self._vector.index(item)


class KTurnRawWindow:
    """Keeps the last K turns of dialogue as raw context.

    This strategy avoids data synchronization latency issues: content
    within the last K turns is always present in raw form and does not
    rely on retrieval. Only content older than K turns is recalled via
    the DetailRetriever.
    """

    def __init__(self, k: int = 10):
        if k < 1:
            raise ValueError("K must be at least 1")
        self._k = k
        self._turns: list[list[ContextItem]] = []

    def add_turn(self, items: list[ContextItem]) -> None:
        """Add a turn's worth of context items."""
        self._turns.append(items)
        # Keep only the last K turns.
        if len(self._turns) > self._k:
            self._turns.pop(0)

    def get_raw_items(self) -> list[ContextItem]:
        """Return all raw items from the last K turns."""
        result: list[ContextItem] = []
        for turn_items in self._turns:
            result.extend(turn_items)
        return result

    def get_archivable_items(self) -> list[ContextItem]:
        """Return items that have fallen out of the K-turn window.

        These items should be indexed for retrieval. Since we pop them
        in add_turn(), this method returns an empty list by default;
        the caller should capture evicted items via a callback if needed.
        """
        return []

    @property
    def k(self) -> int:
        """Return the K value."""
        return self._k

    def reset(self) -> None:
        """Clear all stored turns."""
        self._turns.clear()
