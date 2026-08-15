"""E-commerce category preference modeling (Decision 12.4).

Three-tier profile architecture:
- Global: cross-category traits.
- Category: per product category preferences (the most important tier).
- Current shopping: this specific purchase context.

Includes the advanced price-preference design: per-category historical price
percentiles computed from order history, with sibling-category fallback when
a category has no direct history.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PreferenceType(str, Enum):
    """Kinds of category preferences."""

    PRICE = "price"
    BRAND = "brand"
    ATTRIBUTE = "attribute"
    DECISION_WEIGHT = "decision_weight"


class PreferenceMode(str, Enum):
    """Nature of the preference."""

    LIKE = "like"
    DISLIKE = "dislike"
    HARD_REQUIREMENT = "hard_requirement"


class PreferenceSource(str, Enum):
    """How the preference was derived."""

    EXPLICIT = "explicit"
    ORDER_HISTORY = "order_history"
    RETURN_HISTORY = "return_history"
    SIBLING_CATEGORY = "sibling_category"
    INFERRED = "inferred"


class CategoryPreferenceRecord(BaseModel):
    """A single category-level preference entry (Decision 12.4 schema)."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    category_id: str
    preference_type: PreferenceType
    attribute_key: str
    # Scalar, range, list, or JSON-compatible value.
    preference_value: str
    preference_mode: PreferenceMode = PreferenceMode.LIKE
    # 0.0-1.0; used for ranking and conflict resolution.
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    source: PreferenceSource = PreferenceSource.INFERRED
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class OrderItem(BaseModel):
    """A single purchased SKU used for price-percentile computation."""

    category_id: str
    sku_id: str
    price: float = Field(ge=0.0)
    # Price percentile of the SKU within its category at purchase time
    # (e.g., 0.9 means top 10% price band). If unknown, set to None and
    # provide the category price distribution instead.
    price_percentile: Optional[float] = None
    purchased_at: Optional[datetime] = None


class CategoryPreferenceStore:
    """In-memory store for category preferences with percentile computation."""

    def __init__(self, sibling_map: Optional[dict[str, list[str]]] = None):
        self._records: dict[str, CategoryPreferenceRecord] = {}
        # category_id -> sibling category ids (fallback candidates).
        self._sibling_map: dict[str, list[str]] = sibling_map or {}

    # ------------------------------------------------------------------ CRUD

    def upsert(self, record: CategoryPreferenceRecord) -> CategoryPreferenceRecord:
        """Create or update a category preference record."""
        existing = self._find(
            record.user_id,
            record.category_id,
            record.preference_type,
            record.attribute_key,
        )
        if existing is None:
            self._records[record.id] = record
            return record

        updated = record.model_copy(
            update={
                "id": existing.id,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._records[updated.id] = updated
        return updated

    def list_for_user(
        self,
        user_id: str,
        category_id: Optional[str] = None,
    ) -> list[CategoryPreferenceRecord]:
        """List preferences for a user, optionally filtered by category."""
        records = [r for r in self._records.values() if r.user_id == user_id]
        if category_id is not None:
            records = [r for r in records if r.category_id == category_id]
        return records

    def get_price_preference(
        self, user_id: str, category_id: str
    ) -> CategoryPreferenceRecord | None:
        """Get the price preference for a category, with sibling fallback."""
        direct = self._find(
            user_id, category_id, PreferenceType.PRICE, "price_percentile"
        )
        if direct is not None:
            return direct
        return self._borrow_from_sibling(user_id, category_id)

    # ------------------------------------------------------- percentile logic

    def compute_price_percentile(
        self, user_id: str, orders: list[OrderItem]
    ) -> list[CategoryPreferenceRecord]:
        """Compute per-category price percentiles from order history.

        Returns the created/updated price-preference records. Percentiles are
        averaged per category to summarize the user's typical price band.
        """
        by_category: dict[str, list[float]] = {}
        for order in orders:
            if order.price_percentile is None:
                continue
            by_category.setdefault(order.category_id, []).append(order.price_percentile)

        created: list[CategoryPreferenceRecord] = []
        for category_id, percentiles in by_category.items():
            average = sum(percentiles) / len(percentiles)
            record = CategoryPreferenceRecord(
                user_id=user_id,
                category_id=category_id,
                preference_type=PreferenceType.PRICE,
                attribute_key="price_percentile",
                preference_value=f"{average:.2f}",
                preference_mode=PreferenceMode.LIKE,
                strength=min(1.0, len(percentiles) / 5.0),
                source=PreferenceSource.ORDER_HISTORY,
                confidence=min(0.95, 0.5 + 0.1 * len(percentiles)),
            )
            created.append(self.upsert(record))
        return created

    # ---------------------------------------------------------------- helpers

    def _find(
        self,
        user_id: str,
        category_id: str,
        preference_type: PreferenceType,
        attribute_key: str,
    ) -> CategoryPreferenceRecord | None:
        """Find an existing record matching the natural key."""
        for record in self._records.values():
            if (
                record.user_id == user_id
                and record.category_id == category_id
                and record.preference_type == preference_type
                and record.attribute_key == attribute_key
            ):
                return record
        return None

    def _borrow_from_sibling(
        self, user_id: str, category_id: str
    ) -> CategoryPreferenceRecord | None:
        """Fall back to a sibling category's price preference.

        Confidence is lowered because the borrowed value is only a prior,
        not direct history.
        """
        siblings = self._sibling_map.get(category_id, [])
        for sibling in siblings:
            sibling_record = self._find(
                user_id, sibling, PreferenceType.PRICE, "price_percentile"
            )
            if sibling_record is None:
                continue
            borrowed = sibling_record.model_copy(
                update={
                    "id": str(uuid4()),
                    "category_id": category_id,
                    "source": PreferenceSource.SIBLING_CATEGORY,
                    # Borrowed priors carry reduced confidence.
                    "confidence": sibling_record.confidence * 0.6,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._records[borrowed.id] = borrowed
            return borrowed
        return None
