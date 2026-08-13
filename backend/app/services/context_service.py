from typing import Optional

from app.core.compression import DynamicCompressor
from app.core.engine import ContextEngine
from app.core.metrics import MetricsCollector
from app.core.ordering import CacheAwareOrderer
from app.core.pipeline import (
    DefaultSelector,
    HybridOrderer,
    PlainTextInjector,
    Retriever,
    Selector,
    SlidingWindowOrderer,
    SummaryOrderer,
)
from app.core.selectors import DynamicSelector, RetrievalSelector
from app.core.summarizer import MockSummarizer, Summarizer
from app.models import (
    ComposeRequest,
    ComposeResponse,
    ContextItem,
    ContextItemCreateRequest,
    WindowStrategy,
)
from app.repositories.base import ContextRepository


class ContextService:
    """Application service for context item and window management."""

    def __init__(
        self,
        repository: ContextRepository,
        metrics_collector: Optional[MetricsCollector] = None,
        summarizer: Optional[Summarizer] = None,
    ):
        self._repository = repository
        self._metrics_collector = metrics_collector
        self._summarizer = summarizer or MockSummarizer()

    async def create_item(
        self, session_id: str, request: ContextItemCreateRequest
    ) -> ContextItem:
        """Create a new context item in the given session."""
        item = ContextItem(
            type=request.type,
            content=request.content,
            source=request.source,
            scope=request.scope,
            authority=request.authority,
            confidence=request.confidence,
            priority=request.priority,
            layer=request.layer,
            expires_at=request.expires_at,
        )
        return await self.create_item_direct(session_id, item)

    async def create_item_direct(self, session_id: str, item: ContextItem) -> ContextItem:
        """Persist a fully built context item."""
        return await self._repository.add(session_id, item)

    async def list_items(self, session_id: str) -> list[ContextItem]:
        """List all context items for a session."""
        return await self._repository.list(session_id)

    async def get_item(self, session_id: str, item_id: str) -> ContextItem | None:
        """Fetch a single context item by id."""
        return await self._repository.get(session_id, item_id)

    async def delete_item(self, session_id: str, item_id: str) -> bool:
        """Delete a context item by id."""
        return await self._repository.delete(session_id, item_id)

    async def compose_window(self, request: ComposeRequest) -> ComposeResponse:
        """Compose a context window for a session using the requested strategy."""
        retriever: Retriever = _RepositoryAdapter(self._repository, request.session_id)
        selector = self._build_selector(request)
        orderer = self._build_orderer(request)
        injector = PlainTextInjector()
        compressor = self._build_compressor(request)
        engine = ContextEngine(
            retriever=retriever,
            selector=selector,
            orderer=orderer,
            injector=injector,
            compressor=compressor,
            metrics_collector=self._metrics_collector,
        )
        items, prompt, total, budget_mode = await engine.compose(
            session_id=request.session_id,
            task_state=request.task_state,
            token_budget=request.token_budget,
            scenario=request.scenario,
        )
        return ComposeResponse(
            session_id=request.session_id,
            strategy=request.strategy,
            items=items,
            prompt_fragment=prompt,
            total_tokens=total,
            item_count=len(items),
            budget_mode=budget_mode,
        )

    def _build_selector(self, request: ComposeRequest) -> Selector:
        if request.task_state is not None:
            return RetrievalSelector(base_selector=DynamicSelector())
        if request.strategy == WindowStrategy.DYNAMIC:
            return DynamicSelector()
        return DefaultSelector()

    def _build_compressor(self, request: ComposeRequest):
        if request.token_budget is not None or request.strategy == WindowStrategy.DYNAMIC:
            return DynamicCompressor()
        return None

    def _build_orderer(self, request: ComposeRequest):
        window_size = request.window_size or 3
        if request.strategy == WindowStrategy.DYNAMIC:
            return CacheAwareOrderer(max_tokens=request.max_tokens)
        if request.strategy == WindowStrategy.SLIDING:
            return SlidingWindowOrderer(max_tokens=request.max_tokens)
        if request.strategy == WindowStrategy.SUMMARY:
            return SummaryOrderer(
                max_tokens=request.max_tokens,
                window_size=window_size,
                summarizer=self._summarizer,
            )
        if request.strategy == WindowStrategy.HYBRID:
            return HybridOrderer(
                max_tokens=request.max_tokens,
                window_size=window_size,
                summarizer=self._summarizer,
            )
        raise ValueError(f"Unsupported strategy: {request.strategy}")


class _RepositoryAdapter(Retriever):
    """Adapts a ContextRepository to the pipeline Retriever interface."""

    def __init__(self, repository: ContextRepository, session_id: str):
        self._repository = repository
        self._session_id = session_id

    async def retrieve(self, session_id: str) -> list[ContextItem]:
        return await self._repository.list(self._session_id)
