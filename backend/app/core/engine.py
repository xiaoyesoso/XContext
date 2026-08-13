from typing import Optional

from app.core.metrics import MetricsCollector, WindowMetrics
from app.core.pipeline import (
    Compressor,
    Injector,
    NoOpCompressor,
    Orderer,
    Retriever,
    Selector,
)
from app.core.tokenizer import estimate_item_tokens
from app.models import BudgetMode, ContextItem, TaskState, TokenBudget


class ContextEngine:
    """Orchestrates Retrieve -> Select -> [Compress] -> Order -> Inject.

    When ``task_state`` and ``token_budget`` are provided, the engine uses
    the dynamic pipeline path: budget-mode-aware batch compression and
    cache-aware ordering. Otherwise it falls back to the legacy path.
    """

    def __init__(
        self,
        retriever: Retriever,
        selector: Selector,
        orderer: Orderer,
        injector: Injector,
        compressor: Optional[Compressor] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self.retriever = retriever
        self.selector = selector
        self.orderer = orderer
        self.injector = injector
        self.compressor = compressor or NoOpCompressor()
        self.metrics_collector = metrics_collector

    async def compose(
        self,
        session_id: str,
        task_state: Optional[TaskState] = None,
        token_budget: Optional[TokenBudget] = None,
        scenario: Optional[str] = None,
    ) -> tuple[list[ContextItem], str, int, Optional[BudgetMode]]:
        """Compose and inject the context window.

        Returns:
            A tuple of (items, prompt_fragment, total_tokens, budget_mode).
        """
        all_items = await self.retriever.retrieve(session_id)

        # Selection: use task_state-aware selection if the selector supports it
        selected_items = self._select(all_items, task_state)

        # Compression: dynamic batch if task_state/token_budget provided
        budget_mode: Optional[BudgetMode] = None
        if token_budget is not None and hasattr(self.compressor, "compress_batch"):
            from app.core.budget import BudgetModeResolver
            budget_mode = BudgetModeResolver().resolve(token_budget)
            compressed_items = self.compressor.compress_batch(
                selected_items,
                budget_mode=budget_mode,
                task_state=task_state,
                scenario=scenario,
            )
        else:
            compressed_items = [self.compressor.compress(item) for item in selected_items]

        ordered_items = await self.orderer.order(compressed_items)
        total = sum(
            item.token_cost or estimate_item_tokens(item.content_as_string())
            for item in ordered_items
        )
        prompt = self.injector.inject(ordered_items)

        if self.metrics_collector is not None:
            total_context = sum(
                item.token_cost or estimate_item_tokens(item.content_as_string())
                for item in all_items
            )
            self.metrics_collector.record(
                session_id,
                WindowMetrics(
                    session_id=session_id,
                    retrieved_count=len(all_items),
                    selected_count=len(selected_items),
                    compressed_count=len(compressed_items),
                    ordered_count=len(ordered_items),
                    total_context_tokens=total_context,
                    window_tokens=total,
                    budget_mode=budget_mode,
                ),
            )

        return ordered_items, prompt, total, budget_mode

    def _select(
        self, items: list[ContextItem], task_state: Optional[TaskState]
    ) -> list[ContextItem]:
        """Call the selector, passing task_state if the selector supports it."""
        try:
            return self.selector.select(items, task_state)  # type: ignore[call-arg]
        except TypeError:
            return self.selector.select(items)
