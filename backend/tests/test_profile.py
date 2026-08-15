"""Tests for Phase 11: user profile / persona modeling.

Covers:
- Five-dimension profile fact extraction (mock extractor)
- Profile fact to context-item conversion (dislikes become hard rules)
- Relationship person/event modeling with fact/opinion separation and
  directional attitudes
- Category preference store with price percentile + sibling fallback
- Scenario-aware profile loading
- Recommendation spec derivation and acceptable-ad boundary
- Profile APIs
"""

import pytest

from app.core.category_preference import (
    CategoryPreferenceRecord,
    CategoryPreferenceStore,
    OrderItem,
    PreferenceMode,
    PreferenceSource,
    PreferenceType,
)
from app.core.profile_selector import ProfileSelector
from app.core.recommendation_spec import (
    AcceptableAdBoundaryBuilder,
    RecommendationSpecBuilder,
)
from app.core.relationship_profile import (
    EventSource,
    Person,
    PersonIdentity,
    RelationshipEvent,
    RelationshipProfileStore,
)
from app.core.user_profile import (
    MockUserProfileExtractor,
    fact_to_context_item,
    parse_profile_facts,
)
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextType,
    ProfileDimension,
    ProfileTier,
)


def _make_item(
    content: str,
    item_type: ContextType = ContextType.USER_INPUT,
    **kwargs,
) -> ContextItem:
    """Build a context item with sensible defaults for tests."""
    return ContextItem(
        type=item_type,
        content=content,
        source=kwargs.pop("source", "user"),
        scope=kwargs.pop("scope", "current_session"),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Dimension extraction
# ---------------------------------------------------------------------------


class TestMockUserProfileExtractor:
    @pytest.mark.asyncio
    async def test_goal_extracted(self):
        extractor = MockUserProfileExtractor()
        facts = await extractor.extract([_make_item("My goal is to pass P7 interview")])
        assert any(f.dimension == ProfileDimension.GOAL for f in facts)

    @pytest.mark.asyncio
    async def test_capability_extracted(self):
        extractor = MockUserProfileExtractor()
        facts = await extractor.extract([_make_item("I am a beginner, not familiar with Rust")])
        assert any(f.dimension == ProfileDimension.CAPABILITY for f in facts)

    @pytest.mark.asyncio
    async def test_preference_with_dislike_flag(self):
        extractor = MockUserProfileExtractor()
        facts = await extractor.extract([_make_item("I like concise answers but I hate emoji")])
        prefs = [f for f in facts if f.dimension == ProfileDimension.PREFERENCE]
        assert prefs, "expected at least one preference fact"
        assert any(f.is_dislike for f in prefs)

    @pytest.mark.asyncio
    async def test_decision_dimension_extracted(self):
        extractor = MockUserProfileExtractor()
        facts = await extractor.extract([_make_item("I usually decide by budget first")])
        assert any(f.dimension == ProfileDimension.DECISION for f in facts)

    @pytest.mark.asyncio
    async def test_relationship_dimension_extracted(self):
        extractor = MockUserProfileExtractor()
        facts = await extractor.extract([_make_item("My manager 老王 trusts me")])
        assert any(f.dimension == ProfileDimension.RELATIONSHIP for f in facts)

    @pytest.mark.asyncio
    async def test_empty_items_return_empty(self):
        extractor = MockUserProfileExtractor()
        facts = await extractor.extract([])
        assert facts == []


class TestParseProfileFacts:
    def test_parses_valid_json(self):
        raw = '[{"dimension": "goal", "content": "pass exam", "confidence": 0.9}]'
        facts = parse_profile_facts(raw)
        assert len(facts) == 1
        assert facts[0].dimension == ProfileDimension.GOAL
        assert facts[0].confidence == 0.9

    def test_parses_markdown_fenced_json(self):
        raw = '```json\n[{"dimension": "preference", "content": "likes tea"}]\n```'
        facts = parse_profile_facts(raw)
        assert len(facts) == 1

    def test_invalid_json_returns_empty(self):
        assert parse_profile_facts("not json") == []

    def test_unknown_dimension_skipped(self):
        raw = '[{"dimension": "bogus", "content": "x"}]'
        assert parse_profile_facts(raw) == []


class TestFactToContextItem:
    def test_dislike_becomes_hard_rule(self):
        from app.core.user_profile import ProfileFact

        fact = ProfileFact(
            dimension=ProfileDimension.PREFERENCE,
            content="Never recommend brand X",
            is_dislike=True,
        )
        item = fact_to_context_item(fact)
        assert item.type == ContextType.PROFILE
        assert item.authority == ContextAuthority.HARD_RULE
        assert item.priority == 10
        assert item.profile_dimension == ProfileDimension.PREFERENCE
        assert item.layer == "long_term"

    def test_normal_fact_is_confirmed(self):
        from app.core.user_profile import ProfileFact

        fact = ProfileFact(
            dimension=ProfileDimension.GOAL,
            content="Wants to reach P7 level",
        )
        item = fact_to_context_item(fact)
        assert item.authority == ContextAuthority.CONFIRMED
        assert item.priority == 5


# ---------------------------------------------------------------------------
# Relationship profile
# ---------------------------------------------------------------------------


class TestRelationshipProfileStore:
    def test_upsert_person_merges_aliases(self):
        store = RelationshipProfileStore()
        person = Person(
            user_id="u1",
            name="老王",
            aliases=["Wang"],
            identity=PersonIdentity.MANAGER,
        )
        created = store.upsert_person(person)

        updated_person = person.model_copy(update={"aliases": ["Boss Wang"]})
        updated = store.upsert_person(updated_person)

        assert updated.person_id == created.person_id
        assert set(updated.aliases) == {"Wang", "Boss Wang"}

    def test_find_persons_by_alias(self):
        store = RelationshipProfileStore()
        store.upsert_person(
            Person(user_id="u1", name="老王", aliases=["Wang", "老板"])
        )
        matches = store.find_persons_by_name("u1", "老板")
        assert len(matches) == 1

    def test_find_isolated_per_user(self):
        store = RelationshipProfileStore()
        store.upsert_person(Person(user_id="u1", name="老王"))
        assert store.find_persons_by_name("u2", "老王") == []

    def test_event_updates_attitude_via_relation_effects(self):
        store = RelationshipProfileStore()
        person = store.upsert_person(
            Person(user_id="u1", name="小李", identity=PersonIdentity.FRIEND)
        )
        store.add_event(
            RelationshipEvent(
                user_id="u1",
                participants={person.person_id: "target"},
                objective_fact="小李 shared my secret with others",
                user_interpretation="小李 betrayed my trust",
                user_emotion="angry",
                user_emotion_intensity=0.8,
                relation_effects={person.person_id: "trust↓"},
            )
        )
        refreshed = store.get_person(person.person_id)
        assert "trust↓" in refreshed.user_attitude
        assert any("secret" in e for e in refreshed.evidence)

    def test_fact_and_opinion_stored_separately(self):
        store = RelationshipProfileStore()
        event = RelationshipEvent(
            user_id="u1",
            objective_fact="Manager criticized the plan in the meeting",
            user_interpretation="Manager is targeting me",
        )
        stored = store.add_event(event)
        assert stored.objective_fact != stored.user_interpretation
        assert "targeting" in stored.user_interpretation

    def test_opinion_detector(self):
        assert RelationshipProfileStore.assert_opinion_not_fact("老王又在群里阴阳怪气")
        assert not RelationshipProfileStore.assert_opinion_not_fact("老王 signed the contract")

    def test_directional_attitudes_not_confused(self):
        person = Person(
            user_id="u1",
            name="老王",
            user_attitude="user is annoyed",
            person_attitude_toward_user="老王 is supportive",
        )
        assert person.user_attitude != person.person_attitude_toward_user

    def test_list_events_filtered_by_person(self):
        store = RelationshipProfileStore()
        a = store.upsert_person(Person(user_id="u1", name="A"))
        b = store.upsert_person(Person(user_id="u1", name="B"))
        store.add_event(
            RelationshipEvent(
                user_id="u1",
                participants={a.person_id: "target"},
                objective_fact="Event with A",
            )
        )
        store.add_event(
            RelationshipEvent(
                user_id="u1",
                participants={b.person_id: "target"},
                objective_fact="Event with B",
            )
        )
        events_for_a = store.list_events("u1", person_id=a.person_id)
        assert len(events_for_a) == 1
        assert events_for_a[0].objective_fact == "Event with A"

    def test_fuzzy_time_preserved(self):
        event = RelationshipEvent(
            user_id="u1",
            objective_fact="Something happened",
            occurred_at_fuzzy="last week",
            source=EventSource.USER_WITNESS,
        )
        assert event.occurred_at_fuzzy == "last week"
        assert event.occurred_at is None


# ---------------------------------------------------------------------------
# Category preference
# ---------------------------------------------------------------------------


class TestCategoryPreferenceStore:
    def test_upsert_replaces_same_natural_key(self):
        store = CategoryPreferenceStore()
        record = CategoryPreferenceRecord(
            user_id="u1",
            category_id="phones",
            preference_type=PreferenceType.BRAND,
            attribute_key="brand",
            preference_value="Apple",
        )
        created = store.upsert(record)
        updated = record.model_copy(update={"preference_value": "Samsung"})
        result = store.upsert(updated)

        assert result.id == created.id
        records = store.list_for_user("u1", "phones")
        assert len(records) == 1
        assert records[0].preference_value == "Samsung"

    def test_price_percentile_from_orders(self):
        store = CategoryPreferenceStore()
        created = store.compute_price_percentile(
            "u1",
            [
                OrderItem(category_id="phones", sku_id="s1", price=999, price_percentile=0.9),
                OrderItem(category_id="phones", sku_id="s2", price=799, price_percentile=0.8),
            ],
        )
        assert len(created) == 1
        assert created[0].attribute_key == "price_percentile"
        assert created[0].preference_value == "0.85"
        assert created[0].source == PreferenceSource.ORDER_HISTORY

    def test_sibling_fallback(self):
        store = CategoryPreferenceStore(
            sibling_map={"pants": ["shirts"]}
        )
        store.compute_price_percentile(
            "u1",
            [OrderItem(category_id="shirts", sku_id="s1", price=59, price_percentile=0.7)],
        )
        borrowed = store.get_price_preference("u1", "pants")

        assert borrowed is not None
        assert borrowed.category_id == "pants"
        assert borrowed.source == PreferenceSource.SIBLING_CATEGORY
        # Borrowed priors carry reduced confidence (0.7 * 0.6 here; capped
        # by the percentile confidence rule, so just check it is lower).
        shirts = store.get_price_preference("u1", "shirts")
        assert borrowed.confidence < shirts.confidence

    def test_no_sibling_returns_none(self):
        store = CategoryPreferenceStore(sibling_map={})
        assert store.get_price_preference("u1", "pants") is None

    def test_dislike_mode_stored(self):
        store = CategoryPreferenceStore()
        store.upsert(
            CategoryPreferenceRecord(
                user_id="u1",
                category_id="phones",
                preference_type=PreferenceType.BRAND,
                attribute_key="brand",
                preference_value="BrandX",
                preference_mode=PreferenceMode.DISLIKE,
            )
        )
        records = store.list_for_user("u1", "phones")
        assert records[0].preference_mode == PreferenceMode.DISLIKE


# ---------------------------------------------------------------------------
# Scenario-aware profile loading
# ---------------------------------------------------------------------------


def _profile_item(
    content: str,
    dimension: ProfileDimension,
    tier: ProfileTier = ProfileTier.GLOBAL,
    authority: ContextAuthority = ContextAuthority.CONFIRMED,
) -> ContextItem:
    return ContextItem(
        type=ContextType.PROFILE,
        content=content,
        source="user",
        scope="current_user",
        authority=authority,
        profile_dimension=dimension,
        profile_tier=tier,
        token_cost=5,
    )


class TestProfileSelector:
    def test_refund_scenario_skips_unrelated_dimensions(self):
        selector = ProfileSelector()
        items = [
            _profile_item("Education goal", ProfileDimension.GOAL),
            _profile_item("No loud packaging", ProfileDimension.PREFERENCE,
                          authority=ContextAuthority.HARD_RULE),
            _profile_item("Phone camera priority", ProfileDimension.PREFERENCE,
                          tier=ProfileTier.CATEGORY),
        ]
        result = selector.select(items, scenario="refund")
        loaded_contents = [i.content_as_string() for i in result.items]
        assert "Education goal" not in loaded_contents
        # Global hard rules stay relevant in every scenario.
        assert "No loud packaging" in loaded_contents

    def test_relationship_loaded_only_by_mention(self):
        selector = ProfileSelector()
        laowang = _profile_item("老王 is my manager", ProfileDimension.RELATIONSHIP)
        xiaoli = _profile_item("小李 betrayed trust", ProfileDimension.RELATIONSHIP)

        result = selector.select([laowang, xiaoli], scenario="social",
                                 mentioned_entities=["老王"])
        assert len(result.items) == 1
        assert "老王" in result.items[0].content_as_string()

    def test_no_scenario_keeps_global_only(self):
        selector = ProfileSelector()
        items = [
            _profile_item("Global preference", ProfileDimension.PREFERENCE),
            _profile_item("Category preference", ProfileDimension.PREFERENCE,
                          tier=ProfileTier.CATEGORY),
        ]
        result = selector.select(items, scenario=None)
        assert len(result.items) == 1
        assert result.items[0].profile_tier == ProfileTier.GLOBAL

    def test_education_scenario_loads_goal_and_capability(self):
        selector = ProfileSelector()
        items = [
            _profile_item("Target P7", ProfileDimension.GOAL),
            _profile_item("Beginner in Rust", ProfileDimension.CAPABILITY),
            _profile_item("Likes concise output", ProfileDimension.PREFERENCE),
        ]
        result = selector.select(items, scenario="education")
        dims = {i.profile_dimension for i in result.items}
        assert ProfileDimension.GOAL in dims
        assert ProfileDimension.CAPABILITY in dims
        assert ProfileDimension.PREFERENCE not in dims

    def test_hard_rule_promoted_to_constraint(self):
        selector = ProfileSelector()
        item = _profile_item(
            "Never recommend brand X", ProfileDimension.PREFERENCE,
            authority=ContextAuthority.HARD_RULE,
        )
        result = selector.select([item], scenario="recommendation")
        assert len(result.items) == 1
        prepared = result.items[0]
        assert prepared.type == ContextType.CONSTRAINT
        assert prepared.authority == ContextAuthority.HARD_RULE

    def test_max_items_cap(self):
        selector = ProfileSelector(max_profile_items=2)
        items = [
            _profile_item(f"Goal {i}", ProfileDimension.GOAL) for i in range(5)
        ]
        result = selector.select(items, scenario="education")
        assert len(result.items) == 2
        assert result.skipped_count == 3


# ---------------------------------------------------------------------------
# Recommendation spec & acceptable ads
# ---------------------------------------------------------------------------


class TestRecommendationSpec:
    def _preferences(self):
        return [
            CategoryPreferenceRecord(
                user_id="u1",
                category_id="phones",
                preference_type=PreferenceType.PRICE,
                attribute_key="price_range",
                preference_value="300-500",
            ),
            CategoryPreferenceRecord(
                user_id="u1",
                category_id="phones",
                preference_type=PreferenceType.BRAND,
                attribute_key="brand",
                preference_value="BrandX",
                preference_mode=PreferenceMode.DISLIKE,
            ),
            CategoryPreferenceRecord(
                user_id="u1",
                category_id="phones",
                preference_type=PreferenceType.ATTRIBUTE,
                attribute_key="screen",
                preference_value="oled",
                preference_mode=PreferenceMode.HARD_REQUIREMENT,
            ),
        ]

    def test_spec_derived_from_preferences(self):
        builder = RecommendationSpecBuilder()
        spec = builder.build("u1", self._preferences(), category_id="phones")
        assert spec.price_range == (300.0, 500.0)
        assert spec.excluded_brands == ["BrandX"]
        assert "screen=oled" in spec.required_features

    def test_request_overrides_profile(self):
        builder = RecommendationSpecBuilder()
        spec = builder.build(
            "u1",
            self._preferences(),
            category_id="phones",
            request_price_range=(200.0, 400.0),
        )
        assert spec.price_range == (200.0, 400.0)

    def test_boundary_default_slack(self):
        builder = RecommendationSpecBuilder()
        spec = builder.build("u1", self._preferences(), category_id="phones")
        boundary = AcceptableAdBoundaryBuilder().build(spec)
        assert boundary.min_price == pytest.approx(240.0)
        assert boundary.max_price == pytest.approx(600.0)

    def test_ad_within_boundary_allowed(self):
        builder = RecommendationSpecBuilder()
        spec = builder.build("u1", self._preferences(), category_id="phones")
        boundary = AcceptableAdBoundaryBuilder().build(spec)
        assert boundary.allows(price=550.0) is True
        assert boundary.allows(price=1200.0) is False
        assert boundary.allows(price=100.0) is False

    def test_ad_excluded_brand_rejected(self):
        builder = RecommendationSpecBuilder()
        spec = builder.build("u1", self._preferences(), category_id="phones")
        boundary = AcceptableAdBoundaryBuilder().build(spec)
        assert boundary.allows(price=400.0, brand="BrandX") is False
        assert boundary.allows(price=400.0, brand="BrandY") is True

    def test_custom_slack_ratio(self):
        builder = RecommendationSpecBuilder()
        spec = builder.build("u1", self._preferences(), category_id="phones")
        boundary = AcceptableAdBoundaryBuilder().build(spec, slack_ratio=0.5)
        assert boundary.min_price == pytest.approx(150.0)
        assert boundary.max_price == pytest.approx(750.0)


# ---------------------------------------------------------------------------
# Profile APIs
# ---------------------------------------------------------------------------


class TestProfileAPI:
    def test_person_crud(self, client):
        resp = client.post(
            "/profiles/u1/relationships/persons",
            json={
                "name": "老王",
                "identity": "manager",
                "user_attitude": "user is unhappy with 老王",
            },
        )
        assert resp.status_code == 200
        person = resp.json()
        assert person["name"] == "老王"

        listing = client.get("/profiles/u1/relationships/persons")
        assert listing.status_code == 200
        assert len(listing.json()) == 1

    def test_event_creation_applies_effects(self, client):
        # Create the person first.
        person = client.post(
            "/profiles/u1/relationships/persons",
            json={"name": "小李", "identity": "friend"},
        ).json()

        event = client.post(
            "/profiles/u1/relationships/events",
            json={
                "participants": {"小李": "target"},
                "objective_fact": "小李 shared my secret",
                "user_interpretation": "betrayed my trust",
                "user_emotion": "angry",
                "relation_effects": {"小李": "trust↓"},
            },
        ).json()

        assert event["objective_fact"] == "小李 shared my secret"
        persons = client.get("/profiles/u1/relationships/persons").json()
        target = next(p for p in persons if p["person_id"] == person["person_id"])
        assert "trust↓" in target["user_attitude"]

    def test_preference_crud_and_sibling_fallback(self, client):
        resp = client.post(
            "/profiles/u1/preferences/percentiles",
            json={
                "orders": [
                    {"category_id": "shirts", "sku_id": "s1", "price": 59,
                     "price_percentile": 0.7},
                ]
            },
        )
        assert resp.status_code == 200

        # Pants has no direct history; falls back to shirts.
        borrowed = client.get("/profiles/u1/preferences/pants/price")
        assert borrowed.status_code == 200
        body = borrowed.json()
        assert body["source"] == "sibling_category"

    def test_price_preference_not_found(self, client):
        resp = client.get("/profiles/u1/preferences/laptops/price")
        assert resp.status_code == 404

    def test_recommendation_spec_endpoint(self, client):
        client.post(
            "/profiles/u1/preferences/phones",
            json={
                "preference_type": "price",
                "attribute_key": "price_range",
                "preference_value": "300-500",
            },
        )
        resp = client.post(
            "/profiles/u1/recommendation-spec",
            json={"category_id": "phones"},
        )
        assert resp.status_code == 200
        spec = resp.json()
        assert spec["price_range"] == [300.0, 500.0]

    def test_acceptable_ads_endpoint(self, client):
        resp = client.post(
            "/profiles/u1/acceptable-ads",
            json={"price_range": [300, 500], "slack_ratio": 0.2},
        )
        assert resp.status_code == 200
        boundary = resp.json()
        assert boundary["min_price"] == pytest.approx(240.0)
        assert boundary["max_price"] == pytest.approx(600.0)

    def test_ad_check_endpoint(self, client):
        resp = client.post(
            "/profiles/u1/acceptable-ads/check",
            json={
                "price_range": [300, 500],
                "candidate_price": 1200,
            },
        )
        assert resp.status_code == 200
        assert resp.json()["allowed"] is False

    def test_extract_endpoint(self, client):
        # Ingest a context item first.
        client.post(
            "/context/items?session_id=s-profile",
            json={
                "type": "user_input",
                "content": "My goal is to pass the P7 interview",
                "source": "user",
                "scope": "current_session",
            },
        )
        resp = client.post(
            "/profiles/u1/extract",
            json={"session_id": "s-profile"},
        )
        assert resp.status_code == 200
        facts = resp.json()
        assert len(facts) >= 1
        assert facts[0]["fact"]["dimension"] == "goal"

    def test_extract_missing_session(self, client):
        resp = client.post(
            "/profiles/u1/extract",
            json={"session_id": "no-such-session"},
        )
        assert resp.status_code == 404

    def test_profile_summary(self, client):
        # Unique user id to avoid singleton state leakage across tests.
        client.post(
            "/profiles/u-summary/relationships/persons",
            json={"name": "老王"},
        )
        resp = client.get("/profiles/u-summary")
        assert resp.status_code == 200
        body = resp.json()
        assert body["user_id"] == "u-summary"
        assert len(body["persons"]) == 1
