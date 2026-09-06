"""Action-based retrieval for context sources.

Instead of letting services write raw SQL, Redis commands, or vector queries,
every context source is exposed through a uniform ContextAction interface.
The policy orchestrator calls actions to collect candidate context items.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.tokenizer import estimate_item_tokens
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
)


@dataclass
class ActionServices:
    """Services available to actions during policy execution."""

    context_service: Any = None
    profile_service: Any = None
    summary_service: Any = None


@dataclass
class ActionContext:
    """Runtime context passed to every action."""

    session_id: str
    user_id: str
    scenario: str
    user_message: str
    remaining_tokens: int
    items: list[ContextItem] = field(default_factory=list)
    state: dict = field(default_factory=dict)
    services: ActionServices = field(default_factory=ActionServices)


@dataclass
class ActionResult:
    """Result of executing a context action."""

    action_name: str
    items: list[ContextItem] = field(default_factory=list)
    raw_tokens: int = 0
    compressed_tokens: int = 0
    compressed: bool = False
    dropped: bool = False
    metadata: dict = field(default_factory=dict)


class ContextAction(ABC):
    """Abstract base for all context retrieval actions."""

    name: str = ""

    @abstractmethod
    async def execute(self, ctx: ActionContext) -> ActionResult:
        """Return candidate context items for this action."""


class _ItemsAction(ContextAction):
    """Base action that filters and ranks existing context items by type."""

    item_types: tuple[ContextType, ...] = ()
    authority_order = {
        ContextAuthority.HARD_RULE: 0,
        ContextAuthority.CONFIRMED: 1,
        ContextAuthority.INFERRED: 2,
        ContextAuthority.ASSUMED: 3,
        ContextAuthority.DENIED: 4,
    }

    def _filter_items(self, ctx: ActionContext) -> list[ContextItem]:
        items = [
            i for i in ctx.items
            if i.type in self.item_types and not i.is_expired()
        ]
        # Apply simple keyword filters if configured.
        filters = ctx.state.get("filters", [])
        if filters:
            keywords = {k.lower() for k in filters if k}
            items = [
                i for i in items
                if any(k in i.content_as_string().lower() for k in keywords)
            ]
        # Stable ordering: authority first, then priority desc, then recency.
        items.sort(
            key=lambda i: (
                self.authority_order.get(i.authority, 99),
                -i.priority,
                i.created_at.timestamp() if i.created_at else 0,
            )
        )
        return items


class ConstraintAction(_ItemsAction):
    """Retrieve hard constraints and denied rules."""

    name = "constraint"
    item_types = (ContextType.CONSTRAINT, ContextType.FACT)

    async def execute(self, ctx: ActionContext) -> ActionResult:
        items = [
            i for i in self._filter_items(ctx)
            if i.authority in (ContextAuthority.HARD_RULE, ContextAuthority.DENIED)
            or "约束" in i.content_as_string()
            or "不能" in i.content_as_string()
            or "不要" in i.content_as_string()
            or "绝不" in i.content_as_string()
        ]
        raw_tokens = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        return ActionResult(
            action_name=self.name,
            items=items,
            raw_tokens=raw_tokens,
            compressed_tokens=raw_tokens,
            metadata={"scope": "hard_constraints"},
        )


class KeyFactsAction(_ItemsAction):
    """Retrieve confirmed facts that are critical for the current turn."""

    name = "key_facts"
    item_types = (ContextType.FACT, ContextType.PROFILE)

    async def execute(self, ctx: ActionContext) -> ActionResult:
        items = [
            i for i in self._filter_items(ctx)
            if i.authority == ContextAuthority.CONFIRMED
            or i.priority >= 8
        ]
        raw_tokens = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        return ActionResult(
            action_name=self.name,
            items=items,
            raw_tokens=raw_tokens,
            compressed_tokens=raw_tokens,
            metadata={"scope": "confirmed_facts"},
        )


class UserProfileAction(_ItemsAction):
    """Retrieve profile items for the current user."""

    name = "user_profile"
    item_types = (ContextType.PROFILE,)

    async def execute(self, ctx: ActionContext) -> ActionResult:
        items = self._filter_items(ctx)
        raw_tokens = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        return ActionResult(
            action_name=self.name,
            items=items,
            raw_tokens=raw_tokens,
            compressed_tokens=raw_tokens,
            metadata={"scope": "user_profile"},
        )


class DialogueHistoryAction(_ItemsAction):
    """Retrieve recent dialogue turns."""

    name = "dialogue_history"
    item_types = (ContextType.USER_INPUT, ContextType.MODEL_OUTPUT)

    async def execute(self, ctx: ActionContext) -> ActionResult:
        items = self._filter_items(ctx)
        # Prefer the most recent turns when space is tight.
        items = items[-6:] if len(items) > 6 else items
        raw_tokens = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        return ActionResult(
            action_name=self.name,
            items=items,
            raw_tokens=raw_tokens,
            compressed_tokens=raw_tokens,
            metadata={"scope": "dialogue_history"},
        )


class SummaryAction(_ItemsAction):
    """Retrieve summary items injected by the summary service."""

    name = "summary"
    item_types = (ContextType.SUMMARY,)

    async def execute(self, ctx: ActionContext) -> ActionResult:
        items = self._filter_items(ctx)
        raw_tokens = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        return ActionResult(
            action_name=self.name,
            items=items,
            raw_tokens=raw_tokens,
            compressed_tokens=raw_tokens,
            metadata={"scope": "summaries"},
        )


class DetailRecallAction(ContextAction):
    """Recall evicted details matching the current user message keywords."""

    name = "detail_recall"

    async def execute(self, ctx: ActionContext) -> ActionResult:
        from app.core.drift import extract_keywords

        keywords = extract_keywords(ctx.user_message)
        if not keywords or ctx.services.summary_service is None:
            return ActionResult(action_name=self.name, metadata={"keywords": keywords})

        k_state = ctx.state.get("k_turn", {})
        exclude_ids = set(k_state.get("raw_item_ids", []))
        hits = await ctx.services.summary_service.recall_by_keywords(
            ctx.session_id,
            ctx.items,
            keywords,
            top_k=3,
            exclude_ids=exclude_ids,
        )
        items: list[ContextItem] = []
        for hit in hits:
            item = ContextItem(
                id=f"policy-recall-{hit['id'][:8]}-{ctx.session_id[:8]}",
                type=ContextType(hit["type"]),
                content=hit["content"],
                source=ContextSource.INTERNAL,
                scope=ContextScope.CURRENT_STEP,
                authority=ContextAuthority.INFERRED,
                token_cost=hit.get("token_cost"),
            )
            items.append(item)
        raw_tokens = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        return ActionResult(
            action_name=self.name,
            items=items,
            raw_tokens=raw_tokens,
            compressed_tokens=raw_tokens,
            metadata={"keywords": keywords, "hit_count": len(hits)},
        )


class TaskStateAction(ContextAction):
    """Inject the current task state as a context item."""

    name = "task_state"

    async def execute(self, ctx: ActionContext) -> ActionResult:
        task_state = ctx.state.get("task_state")
        if not task_state:
            return ActionResult(action_name=self.name)
        content = (
            task_state if isinstance(task_state, str) else str(task_state)
        )
        item = ContextItem(
            type=ContextType.TOOL_RESULT,
            content=content,
            source=ContextSource.INTERNAL,
            scope=ContextScope.CURRENT_TASK,
            authority=ContextAuthority.CONFIRMED,
        )
        tokens = estimate_item_tokens(content)
        return ActionResult(
            action_name=self.name,
            items=[item],
            raw_tokens=tokens,
            compressed_tokens=tokens,
            metadata={"scope": "task_state"},
        )


class ActionRegistry:
    """Registry of all available context actions."""

    def __init__(self) -> None:
        self._actions: dict[str, ContextAction] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        for cls in (
            ConstraintAction,
            KeyFactsAction,
            UserProfileAction,
            DialogueHistoryAction,
            SummaryAction,
            DetailRecallAction,
            TaskStateAction,
        ):
            action = cls()
            self._actions[action.name] = action

    def register(self, action: ContextAction) -> None:
        """Register a custom action."""
        self._actions[action.name] = action

    def get(self, name: str) -> Optional[ContextAction]:
        """Fetch an action by name."""
        return self._actions.get(name)

    def list_names(self) -> list[str]:
        """Return names of all registered actions."""
        return sorted(self._actions.keys())

    def validate_policy(self, policy: Any) -> list[str]:
        """Return a list of validation errors for the given policy."""
        errors: list[str] = []
        seen: set[str] = set()
        for idx, segment in enumerate(getattr(policy, "segments", [])):
            action_name = segment.action_name
            if action_name in seen:
                errors.append(f"segment {idx}: duplicate action '{action_name}'")
            seen.add(action_name)
            if action_name not in self._actions:
                errors.append(
                    f"segment {idx}: unknown action '{action_name}'"
                )
        return errors
