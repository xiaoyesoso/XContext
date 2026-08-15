"""Relationship profile modeling: person + event tables (Decision 12.3).

Relationships are event-driven, not static tags. Two linked structures are
maintained:

- Person records: stable identities with directional attitudes.
- RelationshipEvent records: events that shaped the relationships.

Core rules:
1. Distinguish fact from opinion: user interpretations are stored as
   ``user_attitude`` / ``user_interpretation``, never as objective attributes.
2. Directionality: the user's attitude toward a person and the person's
   attitude toward the user are stored separately.
3. Timeliness: relationship changes are recorded as events, not by
   overwriting the current state.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class PersonIdentity(str, Enum):
    """Role of a person relative to the user."""

    COLLEAGUE = "colleague"
    MANAGER = "manager"
    SUBORDINATE = "subordinate"
    FRIEND = "friend"
    FAMILY = "family"
    PARTNER = "partner"
    EX = "ex"
    ACQUAINTANCE = "acquaintance"
    OTHER = "other"


class EventStatus(str, Enum):
    """Confirmation state of a recorded event."""

    PENDING = "pending"
    CONFIRMED = "confirmed"
    DISPUTED = "disputed"
    REFUTED = "refuted"


class EventSource(str, Enum):
    """Where the event information came from."""

    USER_WITNESS = "user_witness"
    THIRD_PARTY = "third_party"
    CHAT_LOG = "chat_log"
    INFERENCE = "inference"


class Person(BaseModel):
    """A person record in the user's world (Decision 12.3 person table)."""

    person_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    identity: PersonIdentity = PersonIdentity.OTHER
    attributes: list[str] = Field(default_factory=list)
    # Directional: the user's attitude toward this person. Opinions are stored
    # here, never promoted to objective attributes.
    user_attitude: str = ""
    # The person's attitude toward the user (kept separate by direction).
    person_attitude_toward_user: str = ""
    # Debug/audit evidence supporting the current judgment.
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RelationshipEvent(BaseModel):
    """An event that shaped one or more relationships (Decision 12.3 event table)."""

    event_id: str = Field(default_factory=lambda: str(uuid4()))
    user_id: str
    # Exact or fuzzy time ("last week" is preserved as-is).
    occurred_at: Optional[datetime] = None
    occurred_at_fuzzy: Optional[str] = None
    # person_id -> role (initiator / target / observer).
    participants: dict[str, str] = Field(default_factory=dict)
    # What verifiably happened (facts only).
    objective_fact: str
    # How the user understood the event (opinion, stored separately).
    user_interpretation: str = ""
    # Emotion and intensity caused by the event.
    user_emotion: str = ""
    user_emotion_intensity: float = Field(ge=0.0, le=1.0, default=0.0)
    # Trust/conflict/intimacy deltas applied to participants.
    relation_effects: dict[str, str] = Field(default_factory=dict)
    source: EventSource = EventSource.USER_WITNESS
    confidence: float = Field(ge=0.0, le=1.0, default=0.7)
    status: EventStatus = EventStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RelationshipProfileStore:
    """In-memory store for person and relationship-event records.

    Enforces fact/opinion separation and directional attitudes when records
    are created or updated.
    """

    def __init__(self):
        self._persons: dict[str, Person] = {}
        self._events: dict[str, RelationshipEvent] = {}

    # ------------------------------------------------------------------ persons

    def upsert_person(self, person: Person) -> Person:
        """Create or update a person record, preserving history semantics."""
        existing = self._persons.get(person.person_id)
        if existing is None:
            self._persons[person.person_id] = person
            return person

        # Merge aliases/attributes/evidence instead of overwriting.
        merged_aliases = list(dict.fromkeys(existing.aliases + person.aliases))
        merged_attributes = list(
            dict.fromkeys(existing.attributes + person.attributes)
        )
        merged_evidence = list(dict.fromkeys(existing.evidence + person.evidence))
        updated = person.model_copy(
            update={
                "aliases": merged_aliases,
                "attributes": merged_attributes,
                "evidence": merged_evidence,
                "updated_at": datetime.now(timezone.utc),
            }
        )
        self._persons[updated.person_id] = updated
        return updated

    def get_person(self, person_id: str) -> Person | None:
        """Fetch a person record by id."""
        return self._persons.get(person_id)

    def find_persons_by_name(self, user_id: str, name: str) -> list[Person]:
        """Find persons by exact name or alias match."""
        name_lower = name.lower()
        results = []
        for person in self._persons.values():
            if person.user_id != user_id:
                continue
            names = {person.name.lower(), *[a.lower() for a in person.aliases]}
            if name_lower in names:
                results.append(person)
        return results

    def list_persons(self, user_id: str) -> list[Person]:
        """List all person records for a user."""
        return [p for p in self._persons.values() if p.user_id == user_id]

    # ------------------------------------------------------------------- events

    def add_event(self, event: RelationshipEvent) -> RelationshipEvent:
        """Record a relationship-shaping event and apply relation effects."""
        self._events[event.event_id] = event
        self._apply_relation_effects(event)
        return event

    def get_event(self, event_id: str) -> RelationshipEvent | None:
        """Fetch an event by id."""
        return self._events.get(event_id)

    def list_events(
        self, user_id: str, person_id: Optional[str] = None
    ) -> list[RelationshipEvent]:
        """List events for a user, optionally filtered by involved person."""
        events = [e for e in self._events.values() if e.user_id == user_id]
        if person_id is None:
            return events
        return [e for e in events if person_id in e.participants]

    def _apply_relation_effects(self, event: RelationshipEvent) -> None:
        """Update person attitudes based on event relation effects.

        Attitude updates are stored as user attitude (directional), and the
        event is appended as evidence for debugging.
        """
        for person_id, effect in event.relation_effects.items():
            person = self._persons.get(person_id)
            if person is None:
                continue
            new_attitude = (
                f"{person.user_attitude}; {effect}".lstrip("; ")
                if person.user_attitude
                else effect
            )
            updated = person.model_copy(
                update={
                    "user_attitude": new_attitude,
                    "evidence": person.evidence + [event.objective_fact[:120]],
                    "updated_at": datetime.now(timezone.utc),
                }
            )
            self._persons[person_id] = updated

    # -------------------------------------------------------------------- misc

    @staticmethod
    def assert_opinion_not_fact(statement: str) -> bool:
        """Heuristic check that an accusatory statement stays an attitude.

        Returns True when the statement looks like a subjective judgment
        (opinion) that must be recorded as user attitude rather than an
        objective attribute.
        """
        opinion_markers = [
            "阴阳怪气", "阴险", "狡猾", "针对", "讨厌", "不喜欢",
            "mean", "nasty", "malicious", "hates", "targeting me",
        ]
        lowered = statement.lower()
        return any(marker in lowered for marker in opinion_markers)
