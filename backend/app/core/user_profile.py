"""User profile / persona modeling (Decision 12).

Extracts profile facts from context items and classifies them into five
dimensions: goal, capability, preference, decision, and relationship.
Profile facts are stored as typed ContextItems so the same pipeline
(Select / Compress / Order / Inject) applies to profiles as to any other
context.
"""

from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.core.tokenizer import estimate_tokens
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
    ProfileDimension,
    ProfileTier,
)


class ProfileFact(BaseModel):
    """A single extracted profile fact with dimension and source back-reference."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    dimension: ProfileDimension
    content: str
    tier: ProfileTier = ProfileTier.GLOBAL
    source_item_id: Optional[str] = None
    # Explicit dislikes / hard requirements become high-authority constraints.
    is_dislike: bool = False
    is_hard_requirement: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def token_cost(self) -> int:
        """Estimate the token cost of this profile fact."""
        return estimate_tokens(self.content)


_EXTRACTION_PROMPT = """\
You are a user profile extractor for an Agent context system.

Analyze the following conversation content and extract profile facts about
the user. Classify each fact into exactly one of these five dimensions:

- goal: What the user wants to achieve (target scores, deadlines, career levels)
- capability: What the user can understand and do (expertise, language, technical depth)
- preference: What the user likes and dislikes (brands, styles, tone, formats)
- decision: How the user usually chooses (factor ordering, risk tolerance, heuristics)
- relationship: Who matters to the user and how (people, roles, attitudes, dynamics)

Rules:
1. Mark dislikes (is_dislike=true) when the user explicitly rejects something.
2. Mark hard requirements (is_hard_requirement=true) for non-negotiable constraints.
3. Distinguish user opinion from objective fact; store opinions as user attitude.
4. Only extract facts useful for personalizing subsequent Agent behavior.
5. Be concise: one fact per entry, no explanations.

Return a JSON array. Each element:
{"dimension": "<dimension>", "content": "<fact>", "is_dislike": <bool>,
 "is_hard_requirement": <bool>, "confidence": <0.0-1.0>}

Content to analyze:
"""


class UserProfileExtractor:
    """Extracts and classifies profile facts using an LLM (flash/mini model)."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self._client = AsyncOpenAI(
            api_key=api_key or settings.api_key,
            base_url=base_url or settings.base_url,
        )
        self._model = model or settings.flash_llm_model

    async def extract(self, items: list[ContextItem]) -> list[ProfileFact]:
        """Extract profile facts from a list of context items."""
        if not items:
            return []

        content_parts = []
        for item in items:
            prefix = f"[{item.type.value}]"
            content_parts.append(f"{prefix} {item.content_as_string()}")

        prompt = _EXTRACTION_PROMPT + "\n".join(content_parts)
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a user profile extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )
        raw_text = (response.choices[0].message.content or "").strip()
        return parse_profile_facts(raw_text)


class MockUserProfileExtractor:
    """Mock extractor using keyword matching (no LLM call).

    Suitable for tests and local development without API access.
    """

    _KEYWORDS: dict[ProfileDimension, list[str]] = {
        ProfileDimension.GOAL: [
            "goal", "target", "want to", "aim", "plan to achieve",
            "target score", "targeting",
        ],
        ProfileDimension.CAPABILITY: [
            "i know", "i can", "familiar with", "expert in",
            "beginner", "newbie", "not familiar", "i don't understand",
        ],
        ProfileDimension.PREFERENCE: [
            "i like", "i prefer", "i love", "i hate", "don't like",
            "dislike", "never recommend", "avoid",
        ],
        ProfileDimension.DECISION: [
            "i usually", "i always first", "i decide by", "priority is",
            "most important factor", "budget first", "brand first",
        ],
        ProfileDimension.RELATIONSHIP: [
            "my colleague", "my manager", "my friend", "my boss",
            "my partner", "my family", "my ex", "trust", "colleague",
        ],
    }

    _DISLIKE_KEYWORDS = ["i hate", "don't like", "dislike", "never recommend", "avoid"]
    _HARD_REQUIREMENT_KEYWORDS = ["must", "never", "always", "only", "required"]

    async def extract(self, items: list[ContextItem]) -> list[ProfileFact]:
        """Extract profile facts via keyword matching (no LLM call)."""
        facts: list[ProfileFact] = []
        for item in items:
            text = item.content_as_string().lower()
            for dimension, keywords in self._KEYWORDS.items():
                for kw in keywords:
                    if kw in text:
                        facts.append(
                            ProfileFact(
                                dimension=dimension,
                                content=item.content_as_string()[:120],
                                source_item_id=item.id,
                                is_dislike=any(k in text for k in self._DISLIKE_KEYWORDS),
                                is_hard_requirement=any(
                                    k in text for k in self._HARD_REQUIREMENT_KEYWORDS
                                ),
                                confidence=0.7,
                            )
                        )
                        break
        return facts


def parse_profile_facts(raw_text: str) -> list[ProfileFact]:
    """Parse an LLM JSON response into ProfileFact objects."""
    import json

    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [line for line in lines if not line.startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    facts: list[ProfileFact] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        dimension_str = entry.get("dimension", "")
        content = entry.get("content", "")
        try:
            dimension = ProfileDimension(dimension_str)
        except ValueError:
            continue
        if not content:
            continue
        facts.append(
            ProfileFact(
                dimension=dimension,
                content=content,
                is_dislike=bool(entry.get("is_dislike", False)),
                is_hard_requirement=bool(entry.get("is_hard_requirement", False)),
                confidence=float(entry.get("confidence", 0.8)),
            )
        )
    return facts


def fact_to_context_item(fact: ProfileFact) -> ContextItem:
    """Convert a ProfileFact into a typed ContextItem for the pipeline.

    Explicit dislikes and hard requirements are promoted to hard_rule
    authority so selectors treat them as constraints.
    """
    if fact.is_dislike or fact.is_hard_requirement:
        authority = ContextAuthority.HARD_RULE
        priority = 10
    else:
        authority = ContextAuthority.CONFIRMED
        priority = 5

    return ContextItem(
        type=ContextType.PROFILE,
        content=fact.content,
        source=ContextSource.USER,
        scope=ContextScope.CURRENT_USER,
        authority=authority,
        confidence=fact.confidence,
        priority=priority,
        token_cost=fact.token_cost(),
        layer="long_term",
        profile_dimension=fact.dimension,
        profile_tier=fact.tier,
    )
