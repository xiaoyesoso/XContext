"""Application service for the user-profile subsystem (Decision 12).

Orchestrates UserProfileExtractor, RelationshipProfileStore,
CategoryPreferenceStore, ProfileSelector, and the recommendation-spec
builders on top of the shared ContextService repository.
"""

from typing import Optional

from app.core.category_preference import (
    CategoryPreferenceRecord,
    CategoryPreferenceStore,
    OrderItem,
)
from app.core.metrics import MetricsCollector, ProfileMetrics
from app.core.profile_selector import ProfileSelector
from app.core.recommendation_spec import (
    AcceptableAdBoundary,
    AcceptableAdBoundaryBuilder,
    RecommendationSpec,
    RecommendationSpecBuilder,
)
from app.core.relationship_profile import (
    Person,
    RelationshipEvent,
    RelationshipProfileStore,
)
from app.core.user_profile import (
    MockUserProfileExtractor,
    ProfileFact,
    UserProfileExtractor,
    fact_to_context_item,
)
from app.models import ContextItem, ContextType, ProfileDimension


class UserProfileService:
    """Facade for user-profile management APIs."""

    def __init__(
        self,
        extractor: Optional[UserProfileExtractor | MockUserProfileExtractor] = None,
        relationship_store: Optional[RelationshipProfileStore] = None,
        category_store: Optional[CategoryPreferenceStore] = None,
        profile_selector: Optional[ProfileSelector] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ):
        self._extractor = extractor or MockUserProfileExtractor()
        self._relationships = relationship_store or RelationshipProfileStore()
        self._categories = category_store or CategoryPreferenceStore(
            sibling_map=_DEFAULT_SIBLINGS
        )
        self._selector = profile_selector or ProfileSelector()
        self._metrics = metrics_collector
        self._spec_builder = RecommendationSpecBuilder()
        self._boundary_builder = AcceptableAdBoundaryBuilder()

    # ------------------------------------------------------------- extraction

    async def extract_facts(
        self, user_id: str, items: list[ContextItem]
    ) -> list[ProfileFact]:
        """Extract profile facts from context items and record metrics."""
        facts = await self._extractor.extract(items)
        if self._metrics is not None:
            by_dimension: dict[str, int] = {}
            for fact in facts:
                key = fact.dimension.value
                by_dimension[key] = by_dimension.get(key, 0) + 1
            self._metrics.record_profile(
                user_id,
                ProfileMetrics(
                    session_id=user_id,
                    extracted_count=len(facts),
                    extracted_by_dimension=by_dimension,
                ),
            )
        return facts

    def persist_fact(self, user_id: str, fact: ProfileFact) -> ContextItem:
        """Convert a profile fact into a context item (caller persists it)."""
        return fact_to_context_item(fact)

    def list_profile_items(
        self, items: list[ContextItem]
    ) -> list[ContextItem]:
        """Filter context items down to profile-typed entries."""
        return [i for i in items if i.type == ContextType.PROFILE]

    # ---------------------------------------------------------- relationships

    def upsert_person(self, person: Person) -> Person:
        """Create or update a relationship person record."""
        return self._relationships.upsert_person(person)

    def get_person(self, person_id: str) -> Person | None:
        """Fetch a person record by id."""
        return self._relationships.get_person(person_id)

    def list_persons(self, user_id: str) -> list[Person]:
        """List all person records for a user."""
        return self._relationships.list_persons(user_id)

    def add_event(self, event: RelationshipEvent) -> RelationshipEvent:
        """Record a relationship-shaping event."""
        return self._relationships.add_event(event)

    def resolve_or_create_person(self, user_id: str, name: str) -> Person:
        """Find a person by name/alias, creating a stub record if unknown."""
        matches = self._relationships.find_persons_by_name(user_id, name)
        if matches:
            return matches[0]
        return self._relationships.upsert_person(Person(user_id=user_id, name=name))

    def list_events(
        self, user_id: str, person_id: Optional[str] = None
    ) -> list[RelationshipEvent]:
        """List relationship events, optionally filtered by person."""
        return self._relationships.list_events(user_id, person_id)

    # ------------------------------------------------------ category prefs

    def upsert_preference(
        self, record: CategoryPreferenceRecord
    ) -> CategoryPreferenceRecord:
        """Create or update a category preference."""
        return self._categories.upsert(record)

    def list_preferences(
        self, user_id: str, category_id: Optional[str] = None
    ) -> list[CategoryPreferenceRecord]:
        """List category preferences for a user."""
        return self._categories.list_for_user(user_id, category_id)

    def compute_price_percentiles(
        self, user_id: str, orders: list[OrderItem]
    ) -> list[CategoryPreferenceRecord]:
        """Compute per-category price percentiles from order history."""
        return self._categories.compute_price_percentile(user_id, orders)

    def get_price_preference(
        self, user_id: str, category_id: str
    ) -> Optional[CategoryPreferenceRecord]:
        """Get a price preference with sibling-category fallback."""
        return self._categories.get_price_preference(user_id, category_id)

    # --------------------------------------------------- scenario-aware load

    def select_profile_items(
        self,
        profile_items: list[ContextItem],
        scenario: Optional[str] = None,
        mentioned_entities: Optional[list[str]] = None,
    ):
        """Load the scenario-relevant profile subset."""
        return self._selector.select(profile_items, scenario, mentioned_entities)

    # ------------------------------------------------------ recommendation

    def build_recommendation_spec(
        self,
        user_id: str,
        category_id: Optional[str],
        request_price_range: Optional[tuple[float, float]] = None,
        request_required_features: Optional[list[str]] = None,
    ) -> RecommendationSpec:
        """Derive a recommendation spec from profile + current request."""
        preferences = self.list_preferences(user_id, category_id)
        return self._spec_builder.build(
            user_id=user_id,
            preferences=preferences,
            category_id=category_id,
            request_price_range=request_price_range,
            request_required_features=request_required_features,
        )

    def build_acceptable_ads(
        self,
        spec: RecommendationSpec,
        slack_ratio: Optional[float] = None,
    ) -> AcceptableAdBoundary:
        """Compute the acceptable-ad boundary for a spec."""
        return self._boundary_builder.build(spec, slack_ratio=slack_ratio)


# Static sibling-category taxonomy: pants borrows from shirts, etc.
_DEFAULT_SIBLINGS: dict[str, list[str]] = {
    "pants": ["shirts", "jackets"],
    "skirts": ["pants", "shirts"],
    "jackets": ["shirts"],
    "phone_case": ["phones"],
    "laptop_bag": ["laptops"],
}
