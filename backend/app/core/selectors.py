"""Dynamic selectors: rule-based, retrieval-based, and model-based selection."""

from typing import Optional
from uuid import uuid4

from app.core.pipeline import Selector
from app.core.tokenizer import estimate_tokens
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
    TaskState,
)


class NegativeContextHandler:
    """Converts denied items into one-sentence anti-guidance reminders."""

    def convert(self, item: ContextItem) -> Optional[ContextItem]:
        """Return a compressed reminder item, or None if not a denied item."""
        if item.authority != ContextAuthority.DENIED:
            return None

        content = item.content_as_string()
        # Take the first sentence as the reminder
        first_sentence = content
        for sep in (". ", "。", "! ", "? ", "\n"):
            idx = content.find(sep)
            if idx > 0:
                first_sentence = content[: idx + len(sep)].strip()
                break

        reminder = f"[REJECTED] Do not repeat: {first_sentence}"
        return ContextItem(
            id=str(uuid4()),
            type=ContextType.SUMMARY,
            content=reminder,
            source=ContextSource.AGENT,
            scope=ContextScope.CURRENT_SESSION,
            authority=ContextAuthority.DENIED,
            confidence=0.5,
            priority=0,
            token_cost=estimate_tokens(reminder),
            layer=item.layer,
        )


class DynamicSelector(Selector):
    """Rule-based selector with negative-context support.

    Unlike DefaultSelector which drops denied items entirely, this selector
    converts them to one-sentence reminders and includes them in the output.
    """

    def __init__(self, negative_handler: Optional[NegativeContextHandler] = None):
        self._negative_handler = negative_handler or NegativeContextHandler()

    def select(self, items: list[ContextItem]) -> list[ContextItem]:
        active: list[ContextItem] = []
        reminders: list[ContextItem] = []

        for item in items:
            if item.is_expired():
                continue
            if item.authority == ContextAuthority.DENIED:
                reminder = self._negative_handler.convert(item)
                if reminder is not None:
                    reminders.append(reminder)
            else:
                active.append(item)

        authority_rank = {
            ContextAuthority.HARD_RULE: 0,
            ContextAuthority.CONFIRMED: 1,
            ContextAuthority.INFERRED: 2,
            ContextAuthority.ASSUMED: 3,
        }
        active.sort(
            key=lambda item: (
                -item.priority,
                authority_rank.get(item.authority, 99),
                -item.confidence,
            )
        )

        # Reminders go after active items but before injection
        return active + reminders


class RetrievalSelector(Selector):
    """Keyword-based retrieval selector.

    Scores items by keyword overlap with the task state (goal, current_step,
    missing_context). Items with zero overlap are still included but ranked
    lower. Off by default — enable by passing task_state.
    """

    def __init__(self, base_selector: Optional[Selector] = None):
        self._base = base_selector or DynamicSelector()

    def select(self, items: list[ContextItem], task_state: Optional[TaskState] = None) -> list[ContextItem]:
        filtered = self._base.select(items)

        if task_state is None:
            return filtered

        keywords = self._extract_keywords(task_state)
        if not keywords:
            return filtered

        def score(item: ContextItem) -> int:
            content_lower = item.content_as_string().lower()
            return sum(1 for kw in keywords if kw in content_lower)

        filtered.sort(key=lambda item: (-score(item), -item.priority))
        return filtered

    def _extract_keywords(self, task_state: TaskState) -> set[str]:
        text_parts = []
        if task_state.goal:
            text_parts.append(task_state.goal)
        if task_state.current_step:
            text_parts.append(task_state.current_step)
        if task_state.missing_context:
            text_parts.extend(task_state.missing_context)

        keywords: set[str] = set()
        for part in text_parts:
            for word in part.lower().split():
                if len(word) > 2:
                    keywords.add(word)
        return keywords


class ModelBasedSelector(Selector):
    """Skeleton for LLM-based selection.

    Uses a Prompt template and an optional LLMAction decorator to select items.
    Off by default — requires an LLM client to be injected.
    """

    SELECTION_PROMPT_TEMPLATE = (
        "You are a context selector. Given the following context items and "
        "the current task state, select the item IDs that should be included "
        "in the context window.\n\n"
        "Task state:\n{task_state}\n\n"
        "Items:\n{items}\n\n"
        "Return a comma-separated list of item IDs."
    )

    def __init__(self, base_selector: Optional[Selector] = None, llm_client=None):
        self._base = base_selector or DynamicSelector()
        self._llm_client = llm_client

    def select(self, items: list[ContextItem], task_state: Optional[TaskState] = None) -> list[ContextItem]:
        # Without an LLM client, fall back to rule-based selection
        if self._llm_client is None or task_state is None:
            return self._base.select(items)

        # LLM-based selection is async; callers should use select_async
        # For sync interface, fall back to base
        return self._base.select(items)

    async def select_async(
        self, items: list[ContextItem], task_state: Optional[TaskState] = None
    ) -> list[ContextItem]:
        """Async selection using the LLM client."""
        if self._llm_client is None or task_state is None:
            return self._base.select(items)

        # Build the prompt
        items_text = "\n".join(
            f"[{item.id}] ({item.type.value}) {item.content_as_string()[:100]}"
            for item in items
        )
        prompt = self.SELECTION_PROMPT_TEMPLATE.format(
            task_state=task_state.model_dump_json(),
            items=items_text,
        )

        # Call the LLM (implementation depends on the client)
        response = await self._llm_client.complete(prompt)
        selected_ids = {
            sid.strip() for sid in response.split(",") if sid.strip()
        }

        return [item for item in items if item.id in selected_ids]
