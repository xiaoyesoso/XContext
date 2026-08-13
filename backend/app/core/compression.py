"""Dynamic multi-level compression (L0-L4) with scenario variants and cascading rules."""

from typing import Optional

from app.core.failure_history import FailureHistoryTracker
from app.core.pipeline import Compressor
from app.core.tokenizer import estimate_tokens
from app.models import (
    BudgetMode,
    CompressionLevel,
    ContextAuthority,
    ContextItem,
    ContextType,
    Importance,
    TaskState,
)


class ItemImportanceClassifier:
    """Classifies a ContextItem into one of four importance levels."""

    def classify(self, item: ContextItem) -> Importance:
        if item.authority == ContextAuthority.HARD_RULE:
            return Importance.CRITICAL
        if item.type == ContextType.USER_INPUT:
            return Importance.CRITICAL
        if item.authority == ContextAuthority.CONFIRMED:
            return Importance.HIGH
        if item.priority >= 10:
            return Importance.HIGH
        if item.authority == ContextAuthority.INFERRED:
            return Importance.MEDIUM
        if item.type in (ContextType.TOOL_RESULT, ContextType.PROFILE):
            return Importance.MEDIUM
        return Importance.LOW


# Decision table: [importance][budget_mode] -> CompressionLevel
_DECISION_TABLE: dict[Importance, dict[BudgetMode, CompressionLevel]] = {
    Importance.CRITICAL: {
        BudgetMode.FULL: CompressionLevel.L4,
        BudgetMode.BALANCED: CompressionLevel.L4,
        BudgetMode.COMPACT: CompressionLevel.L4,
        BudgetMode.MINIMAL: CompressionLevel.L4,
    },
    Importance.HIGH: {
        BudgetMode.FULL: CompressionLevel.L4,
        BudgetMode.BALANCED: CompressionLevel.L4,
        BudgetMode.COMPACT: CompressionLevel.L3,
        BudgetMode.MINIMAL: CompressionLevel.L2,
    },
    Importance.MEDIUM: {
        BudgetMode.FULL: CompressionLevel.L4,
        BudgetMode.BALANCED: CompressionLevel.L3,
        BudgetMode.COMPACT: CompressionLevel.L2,
        BudgetMode.MINIMAL: CompressionLevel.L1,
    },
    Importance.LOW: {
        BudgetMode.FULL: CompressionLevel.L3,
        BudgetMode.BALANCED: CompressionLevel.L2,
        BudgetMode.COMPACT: CompressionLevel.L1,
        BudgetMode.MINIMAL: CompressionLevel.L0,
    },
}


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate text to approximately max_tokens, appending '...' if cut."""
    tokens = text.split()
    if len(tokens) <= max_tokens:
        return text
    return " ".join(tokens[:max_tokens]) + "..."


def _first_sentence(text: str) -> str:
    """Extract the first sentence from text."""
    for sep in (". ", "。", "! ", "? ", "\n"):
        idx = text.find(sep)
        if idx > 0:
            return text[: idx + len(sep)].strip()
    return _truncate_to_tokens(text, 30)


def apply_compression(item: ContextItem, level: CompressionLevel) -> Optional[ContextItem]:
    """Return a copy of the item with content transformed to the given level.

    Returns None for L0 (drop). For L4 the item is returned unchanged
    (except for the compression_level marker).
    """
    if level == CompressionLevel.L0:
        return None

    if level == CompressionLevel.L4:
        return item.model_copy(update={"compression_level": level})

    content = item.content_as_string()

    if level == CompressionLevel.L1:
        new_content = _truncate_to_tokens(content, 10)
    elif level == CompressionLevel.L2:
        new_content = _first_sentence(content)
    elif level == CompressionLevel.L3:
        new_content = _truncate_to_tokens(content, 100)
    else:
        new_content = content

    new_cost = estimate_tokens(new_content)
    return item.model_copy(
        update={
            "content": new_content,
            "token_cost": new_cost,
            "compression_level": level,
        }
    )


class DynamicCompressor(Compressor):
    """Compressor that selects L0-L4 per item based on importance, budget mode,
    task requirements, scenario variants, and cascading rules.

    Use :meth:`compress_batch` for the full dynamic pipeline. The single-item
    :meth:`compress` falls back to L4 (no compression) for backward compatibility.
    """

    def __init__(
        self,
        classifier: Optional[ItemImportanceClassifier] = None,
        failure_tracker: Optional[FailureHistoryTracker] = None,
        scenario_variants: Optional[dict[str, dict[str, CompressionLevel]]] = None,
    ):
        self._classifier = classifier or ItemImportanceClassifier()
        self._failure_tracker = failure_tracker
        # scenario_variants: {scenario_name: {item_id: CompressionLevel}}
        self._scenario_variants = scenario_variants or {}

    def compress(self, item: ContextItem) -> ContextItem:
        """Single-item compression — returns item at L4 (backward compatible)."""
        return apply_compression(item, CompressionLevel.L4)  # type: ignore[return-value]

    def compress_batch(
        self,
        items: list[ContextItem],
        budget_mode: BudgetMode,
        task_state: Optional[TaskState] = None,
        scenario: Optional[str] = None,
    ) -> list[ContextItem]:
        """Compress a batch of items with full dynamic logic.

        Steps:
        1. Classify importance per item.
        2. Select compression level from the decision table.
        3. Override with scenario variant if available.
        4. Elevate minimum level based on failure history.
        5. Apply cascading rules for correlated items.
        6. Transform content and filter out L0 drops.
        """
        task_type = task_state.current_step if task_state else None

        # Step 1-4: select level per item
        levels: dict[str, CompressionLevel] = {}
        for item in items:
            importance = self._classifier.classify(item)
            level = _DECISION_TABLE[importance][budget_mode]

            # Scenario variant override
            if scenario and scenario in self._scenario_variants:
                variant = self._scenario_variants[scenario].get(item.id)
                if variant is not None:
                    level = variant

            # Failure history elevation
            if self._failure_tracker:
                min_level = self._failure_tracker.minimum_level_for(
                    item.type, task_type
                )
                if min_level is not None:
                    level = self._elevate(level, min_level)

            levels[item.id] = level

        # Step 5: cascading rules for correlated items
        levels = self._apply_cascading_rules(items, levels)

        # Step 6: transform and filter
        result: list[ContextItem] = []
        for item in items:
            level = levels[item.id]
            compressed = apply_compression(item, level)
            if compressed is not None:
                result.append(compressed)

        return result

    def _elevate(
        self, current: CompressionLevel, minimum: CompressionLevel
    ) -> CompressionLevel:
        """Return the higher (less compressed) of two levels."""
        order = [
            CompressionLevel.L0,
            CompressionLevel.L1,
            CompressionLevel.L2,
            CompressionLevel.L3,
            CompressionLevel.L4,
        ]
        if order.index(current) < order.index(minimum):
            return minimum
        return current

    def _apply_cascading_rules(
        self, items: list[ContextItem], levels: dict[str, CompressionLevel]
    ) -> dict[str, CompressionLevel]:
        """Apply cascading rules for correlated items.

        Items sharing the same ``correlation_group`` are treated as
        complementary: they are kept at the same (highest) compression level
        so the model can cross-reference them.
        """
        groups: dict[str, list[ContextItem]] = {}
        for item in items:
            if item.correlation_group:
                groups.setdefault(item.correlation_group, []).append(item)

        for group_items in groups.values():
            # Use the highest (least compressed) level in the group
            group_levels = [levels[item.id] for item in group_items]
            best = max(
                group_levels,
                key=lambda l: [
                    CompressionLevel.L0,
                    CompressionLevel.L1,
                    CompressionLevel.L2,
                    CompressionLevel.L3,
                    CompressionLevel.L4,
                ].index(l),
            )
            for item in group_items:
                levels[item.id] = best

        return levels
