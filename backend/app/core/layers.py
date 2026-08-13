from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from app.models import ContextAuthority, ContextItem


class PromotionAction(str, Enum):
    """Supported promotion actions between layers."""

    COPY = "copy"
    MOVE = "move"


class LayerConfig(BaseModel):
    """Configuration for a single context layer."""

    name: str
    description: str = ""
    next_layer: str | None = None
    action: PromotionAction = PromotionAction.COPY
    ttl_seconds: int | None = None


class PromotionRule(BaseModel):
    """A rule that decides whether an item can be promoted to the next layer."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    predicate: Callable[[ContextItem], bool]


DEFAULT_LAYERS: list[LayerConfig] = [
    LayerConfig(
        name="working",
        description="Context for the current step or few model calls.",
        next_layer="session",
        ttl_seconds=300,
    ),
    LayerConfig(
        name="session",
        description="Context valid for the current conversation.",
        next_layer="long_term",
    ),
    LayerConfig(
        name="long_term",
        description="Stable user profile and confirmed facts.",
        next_layer="archive",
    ),
    LayerConfig(
        name="archive",
        description="Cold storage for audit and re-analysis.",
        next_layer=None,
    ),
]


def default_promotion_rules() -> list[PromotionRule]:
    """Return the default promotion rules for layered context."""
    return [
        PromotionRule(
            name="confirmed_fact",
            predicate=lambda item: item.authority == ContextAuthority.CONFIRMED,
        ),
        PromotionRule(
            name="high_priority",
            predicate=lambda item: item.priority >= 100,
        ),
    ]


class LayerManager:
    """Manages context layers and promotion/demotion rules."""

    def __init__(
        self,
        layers: list[LayerConfig] | None = None,
        rules: list[PromotionRule] | None = None,
    ):
        self._layers = {layer.name: layer for layer in (layers or DEFAULT_LAYERS)}
        self._rules = rules or default_promotion_rules()

    def list_layers(self) -> list[LayerConfig]:
        """Return all configured layers."""
        return list(self._layers.values())

    def get_layer(self, name: str) -> LayerConfig | None:
        """Return a layer by name."""
        return self._layers.get(name)

    def evaluate_promotion(self, item: ContextItem) -> str | None:
        """Return the target layer name if any rule matches, otherwise None."""
        current_layer = self._layers.get(item.layer)
        if current_layer is None or current_layer.next_layer is None:
            return None
        if any(rule.predicate(item) for rule in self._rules):
            return current_layer.next_layer
        return None

    def promote(self, item: ContextItem) -> ContextItem | None:
        """Promote an item to the next layer if it qualifies.

        Returns a new item with updated layer and version, or None if no
        promotion is possible.
        """
        target_layer = self.evaluate_promotion(item)
        if target_layer is None:
            return None
        current_layer = self._layers[item.layer]
        data: dict[str, Any] = item.model_dump()
        data["layer"] = target_layer
        data["version"] = item.version + 1
        if current_layer.action == PromotionAction.MOVE:
            # Mark the original item as archived/denied by returning a new item.
            data["id"] = str(uuid4())
        return ContextItem(**data)
