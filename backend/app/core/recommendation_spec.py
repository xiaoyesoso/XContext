"""Spec-driven recommendation and acceptable-ad boundary (Decision 12.6).

Translates user preferences plus the current request into a structured
recommendation specification. Ads may deviate slightly from the spec but
must stay within an acceptable-ad boundary, computed with tunable slack.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

from app.core.category_preference import (
    CategoryPreferenceRecord,
    PreferenceMode,
    PreferenceType,
)


@dataclass
class RecommendationSpec:
    """Structured specification derived from profile + current request."""

    user_id: str
    category_id: Optional[str] = None
    price_range: Optional[tuple[float, float]] = None
    required_features: list[str] = field(default_factory=list)
    excluded_brands: list[str] = field(default_factory=list)
    preferred_brands: list[str] = field(default_factory=list)
    source_summary: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API responses."""
        return {
            "user_id": self.user_id,
            "category_id": self.category_id,
            "price_range": list(self.price_range) if self.price_range else None,
            "required_features": self.required_features,
            "excluded_brands": self.excluded_brands,
            "preferred_brands": self.preferred_brands,
            "source_summary": self.source_summary,
        }


@dataclass
class AcceptableAdBoundary:
    """Acceptable deviation range for sponsored results."""

    spec: RecommendationSpec
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    slack_ratio: float = 0.2

    def allows(self, price: Optional[float] = None, brand: Optional[str] = None) -> bool:
        """Check whether an ad candidate is within the boundary."""
        if brand is not None and brand.lower() in {
            b.lower() for b in self.spec.excluded_brands
        }:
            return False
        if price is None or self.spec.price_range is None:
            return True
        if self.min_price is not None and price < self.min_price:
            return False
        if self.max_price is not None and price > self.max_price:
            return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dict for API responses."""
        return {
            "spec": self.spec.to_dict(),
            "min_price": self.min_price,
            "max_price": self.max_price,
            "slack_ratio": self.slack_ratio,
        }


class RecommendationSpecBuilder:
    """Derives a RecommendationSpec from preferences + current request."""

    def build(
        self,
        user_id: str,
        preferences: list[CategoryPreferenceRecord],
        category_id: Optional[str] = None,
        request_price_range: Optional[tuple[float, float]] = None,
        request_required_features: Optional[list[str]] = None,
    ) -> RecommendationSpec:
        """Build a spec, with the explicit request overriding profile data."""
        spec = RecommendationSpec(user_id=user_id, category_id=category_id)

        for record in preferences:
            self._apply_record(spec, record)

        # The current request wins over inferred profile data.
        if request_price_range is not None:
            spec.price_range = request_price_range
            spec.source_summary.append("explicit request price range")
        if request_required_features:
            spec.required_features = list(
                dict.fromkeys(spec.required_features + request_required_features)
            )
            spec.source_summary.append("explicit request features")

        return spec

    @staticmethod
    def _apply_record(spec: RecommendationSpec, record: CategoryPreferenceRecord) -> None:
        """Apply a single preference record to the spec."""
        if record.preference_type == PreferenceType.PRICE:
            if record.attribute_key == "price_percentile":
                spec.source_summary.append(
                    f"price_percentile={record.preference_value} "
                    f"({record.source.value})"
                )
            else:
                try:
                    low, high = record.preference_value.split("-")
                    spec.price_range = (float(low), float(high))
                    spec.source_summary.append(f"profile price range {record.preference_value}")
                except ValueError:
                    pass
        elif record.preference_type == PreferenceType.BRAND:
            brand = record.preference_value
            if record.preference_mode == PreferenceMode.DISLIKE:
                spec.excluded_brands.append(brand)
            else:
                spec.preferred_brands.append(brand)
        elif record.preference_type == PreferenceType.ATTRIBUTE:
            if record.preference_mode == PreferenceMode.HARD_REQUIREMENT:
                spec.required_features.append(
                    f"{record.attribute_key}={record.preference_value}"
                )
            else:
                spec.source_summary.append(
                    f"{record.attribute_key}={record.preference_value}"
                )


class AcceptableAdBoundaryBuilder:
    """Computes the acceptable-ad boundary from a spec with tunable slack."""

    def __init__(self, default_slack_ratio: float = 0.2):
        self._default_slack = default_slack_ratio

    def build(
        self,
        spec: RecommendationSpec,
        slack_ratio: Optional[float] = None,
    ) -> AcceptableAdBoundary:
        """Widen the spec price range by the slack ratio.

        Example: a 300-500 spec with 0.2 slack accepts ads in 240-600;
        an ad at 1200 is rejected as an extreme outlier.
        """
        slack = slack_ratio if slack_ratio is not None else self._default_slack
        boundary = AcceptableAdBoundary(spec=spec, slack_ratio=slack)
        if spec.price_range is not None:
            low, high = spec.price_range
            span = high - low
            boundary.min_price = max(0.0, low - slack * max(span, low))
            boundary.max_price = high + slack * max(span, high)
        return boundary
