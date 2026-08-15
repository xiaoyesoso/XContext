from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.core.category_preference import (
    CategoryPreferenceRecord,
    OrderItem,
)
from app.core.relationship_profile import (
    Person,
    RelationshipEvent,
)
from app.core.user_profile import ProfileFact
from app.dependencies import get_context_service, get_user_profile_service
from app.models import ContextItem, ProfileDimension

router = APIRouter(prefix="/profiles", tags=["profiles"])


# ------------------------------------------------------------------- schemas


class ExtractRequest(BaseModel):
    """Request body for extracting profile facts from session items."""

    session_id: str = Field(min_length=1)


class PersonCreateRequest(BaseModel):
    """Request body for creating/updating a person record."""

    person_id: str | None = None
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    identity: str = "other"
    attributes: list[str] = Field(default_factory=list)
    user_attitude: str = ""
    person_attitude_toward_user: str = ""
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class EventCreateRequest(BaseModel):
    """Request body for recording a relationship event."""

    occurred_at_fuzzy: str | None = None
    # person name -> role (initiator / target / observer).
    participants: dict[str, str] = Field(default_factory=dict)
    objective_fact: str = Field(min_length=1)
    user_interpretation: str = ""
    user_emotion: str = ""
    user_emotion_intensity: float = Field(ge=0.0, le=1.0, default=0.0)
    relation_effects: dict[str, str] = Field(default_factory=dict)
    source: str = "user_witness"
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    status: str = "pending"


class PreferenceCreateRequest(BaseModel):
    """Request body for creating/updating a category preference.

    ``category_id`` comes from the path parameter and is ignored if also
    present in the body.
    """

    category_id: str | None = None
    preference_type: str
    attribute_key: str
    preference_value: str
    preference_mode: str = "like"
    strength: float = Field(ge=0.0, le=1.0, default=0.5)
    source: str = "explicit"
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)


class PercentileRequest(BaseModel):
    """Request body for offline price-percentile computation."""

    orders: list[OrderItem] = Field(min_length=1)


class SpecRequest(BaseModel):
    """Request body for deriving a recommendation spec."""

    category_id: str | None = None
    price_range: list[float] | None = None
    required_features: list[str] | None = None


class AdBoundaryRequest(BaseModel):
    """Request body for computing an acceptable-ad boundary."""

    category_id: str | None = None
    price_range: list[float] | None = None
    required_features: list[str] | None = None
    slack_ratio: float | None = Field(default=None, ge=0.0, le=1.0)


class AdCheckRequest(BaseModel):
    """Request body for checking an ad candidate against the boundary."""

    category_id: str | None = None
    price_range: list[float] | None = None
    slack_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    candidate_price: float | None = None
    candidate_brand: str | None = None


# ------------------------------------------------------------------ endpoints


@router.get("/{user_id}")
async def get_profile(user_id: str) -> dict:
    """Return a structured profile summary (persons + preferences)."""
    service = get_user_profile_service()
    persons = service.list_persons(user_id)
    preferences = service.list_preferences(user_id)
    return {
        "user_id": user_id,
        "persons": [p.model_dump() for p in persons],
        "preferences": [p.model_dump() for p in preferences],
    }


@router.post("/{user_id}/extract")
async def extract_profile(user_id: str, request: ExtractRequest) -> list[dict]:
    """Extract profile facts from session context items via the mock extractor."""
    context_service = get_context_service()
    service = get_user_profile_service()

    items = await context_service.list_items(request.session_id)
    if not items:
        raise HTTPException(
            status_code=404,
            detail=f"No context items found for session {request.session_id}",
        )

    facts = await service.extract_facts(user_id, items)
    persisted: list[dict] = []
    for fact in facts:
        item = service.persist_fact(user_id, fact)
        await context_service.create_item_direct(request.session_id, item)
        persisted.append(
            {
                "fact": fact.model_dump(),
                "context_item_id": item.id,
            }
        )
    return persisted


@router.get("/{user_id}/dimension/{dimension}")
async def get_dimension(user_id: str, dimension: str, session_id: str) -> list[ContextItem]:
    """Return profile context items for a specific dimension."""
    try:
        dim = ProfileDimension(dimension)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Unknown dimension: {dimension}")

    context_service = get_context_service()
    service = get_user_profile_service()
    items = await context_service.list_items(session_id)
    profile_items = service.list_profile_items(items)
    return [i for i in profile_items if i.profile_dimension == dim]


@router.get("/{user_id}/relationships/persons")
async def list_persons(user_id: str) -> list[Person]:
    """List relationship person records for a user."""
    return get_user_profile_service().list_persons(user_id)


@router.post("/{user_id}/relationships/persons")
async def upsert_person(user_id: str, request: PersonCreateRequest) -> Person:
    """Create or update a relationship person record."""
    person = Person(
        user_id=user_id,
        name=request.name,
        aliases=request.aliases,
        identity=request.identity,  # type: ignore[arg-type]
        attributes=request.attributes,
        user_attitude=request.user_attitude,
        person_attitude_toward_user=request.person_attitude_toward_user,
        evidence=request.evidence,
        confidence=request.confidence,
    )
    if request.person_id:
        person = person.model_copy(update={"person_id": request.person_id})
    return get_user_profile_service().upsert_person(person)


@router.get("/{user_id}/relationships/events")
async def list_events(user_id: str, person_id: str | None = None) -> list[RelationshipEvent]:
    """List relationship events for a user, optionally by person."""
    return get_user_profile_service().list_events(user_id, person_id)


@router.post("/{user_id}/relationships/events")
async def create_event(user_id: str, request: EventCreateRequest) -> RelationshipEvent:
    """Record a relationship-shaping event."""
    service = get_user_profile_service()

    # Resolve participant names and relation-effect keys to person ids,
    # creating unknown persons as stubs.
    participants: dict[str, str] = {}
    resolved_ids: dict[str, str] = {}
    all_names = set(request.participants) | set(request.relation_effects)
    for name in all_names:
        person = service.resolve_or_create_person(user_id, name)
        resolved_ids[name] = person.person_id
    for name, role in request.participants.items():
        participants[resolved_ids[name]] = role

    relation_effects = {
        resolved_ids[name]: effect
        for name, effect in request.relation_effects.items()
    }

    event = RelationshipEvent(
        user_id=user_id,
        occurred_at_fuzzy=request.occurred_at_fuzzy,
        participants=participants,
        objective_fact=request.objective_fact,
        user_interpretation=request.user_interpretation,
        user_emotion=request.user_emotion,
        user_emotion_intensity=request.user_emotion_intensity,
        relation_effects=relation_effects,
        source=request.source,  # type: ignore[arg-type]
        confidence=request.confidence,
        status=request.status,  # type: ignore[arg-type]
    )
    return service.add_event(event)


@router.post("/{user_id}/preferences/percentiles")
async def compute_percentiles(
    user_id: str, request: PercentileRequest
) -> list[CategoryPreferenceRecord]:
    """Compute per-category price percentiles from order history (offline)."""
    return get_user_profile_service().compute_price_percentiles(user_id, request.orders)


@router.get("/{user_id}/preferences/{category_id}")
async def list_preferences(user_id: str, category_id: str) -> list[CategoryPreferenceRecord]:
    """List category-level preferences for a user."""
    return get_user_profile_service().list_preferences(user_id, category_id)


@router.post("/{user_id}/preferences/{category_id}")
async def upsert_preference(
    user_id: str, category_id: str, request: PreferenceCreateRequest
) -> CategoryPreferenceRecord:
    """Create or update a category preference record."""
    record = CategoryPreferenceRecord(
        user_id=user_id,
        category_id=category_id,
        preference_type=request.preference_type,  # type: ignore[arg-type]
        attribute_key=request.attribute_key,
        preference_value=request.preference_value,
        preference_mode=request.preference_mode,  # type: ignore[arg-type]
        strength=request.strength,
        source=request.source,  # type: ignore[arg-type]
        confidence=request.confidence,
    )
    return get_user_profile_service().upsert_preference(record)


@router.get("/{user_id}/preferences/{category_id}/price")
async def get_price_preference(user_id: str, category_id: str) -> CategoryPreferenceRecord:
    """Get the price preference for a category, with sibling fallback."""
    record = get_user_profile_service().get_price_preference(user_id, category_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"No price preference available for category {category_id}",
        )
    return record


@router.post("/{user_id}/recommendation-spec")
async def build_spec(user_id: str, request: SpecRequest) -> dict:
    """Derive a recommendation spec from profile + current request."""
    price_range = None
    if request.price_range and len(request.price_range) == 2:
        price_range = (request.price_range[0], request.price_range[1])
    spec = get_user_profile_service().build_recommendation_spec(
        user_id=user_id,
        category_id=request.category_id,
        request_price_range=price_range,
        request_required_features=request.required_features,
    )
    return spec.to_dict()


@router.post("/{user_id}/acceptable-ads")
async def build_ad_boundary(user_id: str, request: AdBoundaryRequest) -> dict:
    """Compute the acceptable-ad boundary for a recommendation spec."""
    price_range = None
    if request.price_range and len(request.price_range) == 2:
        price_range = (request.price_range[0], request.price_range[1])
    service = get_user_profile_service()
    spec = service.build_recommendation_spec(
        user_id=user_id,
        category_id=request.category_id,
        request_price_range=price_range,
        request_required_features=request.required_features,
    )
    boundary = service.build_acceptable_ads(spec, slack_ratio=request.slack_ratio)
    return boundary.to_dict()


@router.post("/{user_id}/acceptable-ads/check")
async def check_ad(user_id: str, request: AdCheckRequest) -> dict:
    """Check whether an ad candidate falls within the acceptable boundary."""
    price_range = None
    if request.price_range and len(request.price_range) == 2:
        price_range = (request.price_range[0], request.price_range[1])
    service = get_user_profile_service()
    spec = service.build_recommendation_spec(
        user_id=user_id,
        category_id=request.category_id,
        request_price_range=price_range,
    )
    boundary = service.build_acceptable_ads(spec, slack_ratio=request.slack_ratio)
    return {
        "allowed": boundary.allows(
            price=request.candidate_price, brand=request.candidate_brand
        ),
        "boundary": boundary.to_dict(),
    }
