"""Scenario-aware profile loading (Decision 12.5).

Loads only the subset of the user profile relevant to the current task
scenario, avoiding profile bloat in the context window:

- The global profile is always loaded (kept small by design).
- Category profiles are loaded only when the scenario matches.
- Relationship data is loaded by mention, not in full.
- Explicit dislikes / hard requirements are injected as high-authority
  constraint items.
"""

from dataclasses import dataclass, field
from typing import Optional

from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
    ProfileDimension,
    ProfileTier,
)


@dataclass
class ProfileLoadResult:
    """Outcome of a scenario-aware profile load."""

    items: list[ContextItem] = field(default_factory=list)
    loaded_by_dimension: dict[str, int] = field(default_factory=dict)
    skipped_count: int = 0
    profile_tokens: int = 0


# Scenario keywords that map a scenario name to relevant profile dimensions
# and (optionally) product categories.
_SCENARIO_MAP: dict[str, dict] = {
    "refund": {
        "dimensions": {ProfileDimension.PREFERENCE},
        "categories": set(),
    },
    "recommendation": {
        "dimensions": {ProfileDimension.PREFERENCE, ProfileDimension.DECISION},
        "categories": {"*"},  # any category is relevant
    },
    "education": {
        "dimensions": {ProfileDimension.GOAL, ProfileDimension.CAPABILITY},
        "categories": set(),
    },
    "writing": {
        "dimensions": {ProfileDimension.CAPABILITY, ProfileDimension.PREFERENCE},
        "categories": set(),
    },
    "social": {
        "dimensions": {ProfileDimension.RELATIONSHIP},
        "categories": set(),
    },
}


class ProfileSelector:
    """Selects a scenario-relevant subset of profile context items."""

    def __init__(
        self,
        scenario_map: Optional[dict[str, dict]] = None,
        max_profile_items: int = 20,
    ):
        self._scenario_map = scenario_map or _SCENARIO_MAP
        self._max_profile_items = max_profile_items

    def select(
        self,
        profile_items: list[ContextItem],
        scenario: Optional[str] = None,
        mentioned_entities: Optional[list[str]] = None,
    ) -> ProfileLoadResult:
        """Load a scenario-relevant subset of profile items.

        Args:
            profile_items: all profile-typed context items for the user.
            scenario: current task scenario (e.g., refund, recommendation).
            mentioned_entities: names/persons mentioned in the current turn;
                relationship items are loaded only for these.
        """
        result = ProfileLoadResult()
        mentioned = {m.lower() for m in (mentioned_entities or [])}
        scenario_config = self._scenario_map.get(scenario or "", None)
        relevant_dimensions: Optional[set] = (
            scenario_config["dimensions"] if scenario_config else None
        )

        for item in profile_items:
            if len(result.items) >= self._max_profile_items:
                result.skipped_count += 1
                continue

            if not self._is_relevant(
                item, relevant_dimensions, mentioned, scenario_config
            ):
                result.skipped_count += 1
                continue

            result.items.append(self._prepare(item))
            dim = item.profile_dimension.value if item.profile_dimension else "unknown"
            result.loaded_by_dimension[dim] = result.loaded_by_dimension.get(dim, 0) + 1
            result.profile_tokens += item.token_cost

        return result

    # ---------------------------------------------------------------- helpers

    def _is_relevant(
        self,
        item: ContextItem,
        relevant_dimensions: Optional[set],
        mentioned: set[str],
        scenario_config: Optional[dict],
    ) -> bool:
        """Decide whether a profile item is relevant to the scenario."""
        dimension = item.profile_dimension
        tier = item.profile_tier

        # Global-tier preference dislikes/hard rules are always relevant:
        # they act as constraints across all scenarios.
        if (
            tier == ProfileTier.GLOBAL
            and item.authority == ContextAuthority.HARD_RULE
        ):
            return True

        # Relationship data is loaded only when the person is mentioned.
        if dimension == ProfileDimension.RELATIONSHIP:
            return bool(mentioned) and self._mentions_any(item, mentioned)

        # No scenario config: keep global items, skip the rest.
        if relevant_dimensions is None:
            return tier == ProfileTier.GLOBAL

        if dimension is not None and dimension not in relevant_dimensions:
            return False

        # Category-tier items require a category-aware scenario.
        if tier == ProfileTier.CATEGORY:
            return bool(scenario_config and "*" in scenario_config.get("categories", set()))

        return True

    @staticmethod
    def _mentions_any(item: ContextItem, mentioned: set[str]) -> bool:
        """Check whether the item content mentions any of the given names."""
        content = item.content_as_string().lower()
        return any(name in content for name in mentioned)

    @staticmethod
    def _prepare(item: ContextItem) -> ContextItem:
        """Promote explicit dislikes / hard requirements to constraints."""
        if item.authority != ContextAuthority.HARD_RULE:
            return item
        # Re-tag as a constraint item with current-task scope so downstream
        # selectors and compressors treat it as a hard rule.
        return item.model_copy(
            update={
                "type": ContextType.CONSTRAINT,
                "scope": ContextScope.CURRENT_TASK,
                "source": ContextSource.USER,
            }
        )
