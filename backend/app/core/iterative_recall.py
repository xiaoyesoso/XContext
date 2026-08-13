"""Iterative detail recall: LLM-driven detail retrieval.

Instead of guessing what details are needed before the LLM call, this
module implements an iterative loop where the LLM itself evaluates
context sufficiency and requests additional details via dedicated
Actions. Includes max-iteration guards and window-overflow handling.
"""

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from openai import AsyncOpenAI

from app.config import settings
from app.core.conflict_resolution import ConflictResolver
from app.core.detail_recall import DetailRetriever
from app.models import ContextItem


class RecallActionType(str, Enum):
    """Types of detail recall actions the LLM can invoke."""

    RECALL_DETAIL = "recall_detail"
    RECALL_ENTITY = "recall_entity"
    RECALL_TOOL_HISTORY = "recall_tool_history"


class IterationOutcome(str, Enum):
    """Outcome of an iterative recall session."""

    SUFFICIENT = "sufficient"
    MAX_ITERATIONS_REACHED = "max_iterations_reached"
    WINDOW_OVERFLOW = "window_overflow"


@dataclass
class IterationResult:
    """Result of an iterative detail recall session."""

    outcome: IterationOutcome
    final_items: list[ContextItem] = field(default_factory=list)
    recalled_items: list[ContextItem] = field(default_factory=list)
    iterations: int = 0
    detail_catalog: list[str] = field(default_factory=list)


_SUFFICIENCY_PROMPT = """\
You are a context sufficiency evaluator for an Agent system.

Given the current context and a catalog of available details, determine \
whether the context is sufficient to proceed, or if additional details \
are needed.

Current context:
{context}

Available detail catalog (indexed but not yet loaded):
{catalog}

Respond with a JSON object:
{{
  "sufficient": true/false,
  "needed_details": ["catalog_entry_1", "catalog_entry_2"],
  "reason": "brief explanation"
}}

If sufficient is true, needed_details should be empty.
"""


class IterativeRecallLoop:
    """Implements the iterative detail recall flow.

    1. Send initial context + detail catalog to LLM.
    2. LLM evaluates sufficiency and requests details if needed.
    3. Invoke recall Actions to retrieve requested details.
    4. Merge recalled details, resolve conflicts, re-call LLM.
    5. Repeat until sufficient or max iterations reached.
    """

    def __init__(
        self,
        retriever: DetailRetriever,
        conflict_resolver: Optional[ConflictResolver] = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        max_iterations: int = 3,
    ):
        self._retriever = retriever
        self._conflict_resolver = conflict_resolver or ConflictResolver()
        self._client = AsyncOpenAI(
            api_key=api_key or settings.api_key,
            base_url=base_url or settings.base_url,
        )
        self._model = model or settings.flash_llm_model
        self._max_iterations = max_iterations

    async def run(
        self,
        initial_items: list[ContextItem],
        detail_catalog: list[str],
        session_id: Optional[str] = None,
        window_budget_tokens: Optional[int] = None,
    ) -> IterationResult:
        """Execute the iterative recall loop.

        Args:
            initial_items: The context items already in the window.
            detail_catalog: Index entries of available but unloaded details.
            session_id: Optional session scope for retrieval.
            window_budget_tokens: Optional token budget; if recall would
                exceed it, trigger window-overflow handling.

        Returns:
            IterationResult with the final item set and outcome.
        """
        current_items = list(initial_items)
        all_recalled: list[ContextItem] = []

        for iteration in range(1, self._max_iterations + 1):
            # Build context summary for the LLM.
            context_text = self._summarize_context(current_items)
            prompt = _SUFFICIENCY_PROMPT.format(
                context=context_text,
                catalog="\n".join(f"- {entry}" for entry in detail_catalog),
            )

            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": "You are a context sufficiency evaluator."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=256,
            )

            raw = (response.choices[0].message.content or "").strip()
            evaluation = _parse_sufficiency_response(raw)

            if evaluation.get("sufficient", True):
                return IterationResult(
                    outcome=IterationOutcome.SUFFICIENT,
                    final_items=current_items,
                    recalled_items=all_recalled,
                    iterations=iteration,
                    detail_catalog=detail_catalog,
                )

            # Retrieve requested details.
            needed = evaluation.get("needed_details", [])
            if not needed:
                return IterationResult(
                    outcome=IterationOutcome.SUFFICIENT,
                    final_items=current_items,
                    recalled_items=all_recalled,
                    iterations=iteration,
                    detail_catalog=detail_catalog,
                )

            recalled_this_round: list[ContextItem] = []
            for query in needed:
                results = await self._retriever.recall(query, session_id, top_k=3)
                recalled_this_round.extend(results)

            if not recalled_this_round:
                # No results found; stop to avoid infinite loop.
                return IterationResult(
                    outcome=IterationOutcome.SUFFICIENT,
                    final_items=current_items,
                    recalled_items=all_recalled,
                    iterations=iteration,
                    detail_catalog=detail_catalog,
                )

            # Check window budget before merging.
            if window_budget_tokens is not None:
                total_tokens = sum(i.token_cost for i in current_items)
                new_tokens = sum(i.token_cost for i in recalled_this_round)
                if total_tokens + new_tokens > window_budget_tokens:
                    # Window overflow: try to compress or drop existing details.
                    handled = await self._handle_overflow(
                        current_items, recalled_this_round, window_budget_tokens
                    )
                    if not handled:
                        return IterationResult(
                            outcome=IterationOutcome.WINDOW_OVERFLOW,
                            final_items=current_items,
                            recalled_items=all_recalled,
                            iterations=iteration,
                            detail_catalog=detail_catalog,
                        )

            # Merge and resolve conflicts.
            all_recalled.extend(recalled_this_round)
            merged = current_items + recalled_this_round
            current_items = self._conflict_resolver.resolve(merged)

        return IterationResult(
            outcome=IterationOutcome.MAX_ITERATIONS_REACHED,
            final_items=current_items,
            recalled_items=all_recalled,
            iterations=self._max_iterations,
            detail_catalog=detail_catalog,
        )

    async def _handle_overflow(
        self,
        current_items: list[ContextItem],
        new_items: list[ContextItem],
        budget: int,
    ) -> bool:
        """Attempt to handle window overflow by dropping low-priority items.

        Returns True if the overflow was resolved, False otherwise.
        """
        # Drop lowest-priority items until we fit.
        sorted_items = sorted(current_items, key=lambda x: (x.priority, x.confidence))
        total = sum(i.token_cost for i in current_items + new_items)
        while sorted_items and total > budget:
            dropped = sorted_items.pop(0)
            total -= dropped.token_cost
            current_items.remove(dropped)
        return total <= budget

    def _summarize_context(self, items: list[ContextItem]) -> str:
        """Build a compact text summary of the current context items."""
        lines: list[str] = []
        for item in items[:20]:  # Cap to avoid prompt explosion.
            content = item.content_as_string()
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"[{item.type.value}] {content}")
        return "\n".join(lines)


def _parse_sufficiency_response(raw: str) -> dict:
    """Parse the LLM's sufficiency evaluation response."""
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass
    # Default to sufficient if parsing fails.
    return {"sufficient": True, "needed_details": [], "reason": "parse error"}
