"""Key fact extraction and classification.

Extracts atomic facts from conversation content and classifies them into
six categories: goals, hard constraints, soft preferences, user-confirmed
facts, decision facts, and entity information.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from app.config import settings
from app.core.tokenizer import estimate_tokens
from app.models import ContextItem


class KeyFactCategory(str, Enum):
    """Six categories of key facts that influence Agent decisions."""

    GOAL = "goal"
    HARD_CONSTRAINT = "hard_constraint"
    SOFT_PREFERENCE = "soft_preference"
    CONFIRMED_FACT = "confirmed_fact"
    DECISION = "decision"
    ENTITY = "entity"


class KeyFact(BaseModel):
    """A single extracted key fact with category and source back-reference."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    category: KeyFactCategory
    content: str
    source_item_id: Optional[str] = None
    session_id: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.8)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def token_cost(self) -> int:
        """Estimate the token cost of this key fact."""
        return estimate_tokens(self.content)


_EXTRACTION_PROMPT = """\
You are a key fact extractor for an Agent context system.

Analyze the following conversation content and extract atomic key facts.
Classify each fact into exactly one of these categories:

- goal: User intent modifications (e.g., "emphasize technical complexity", "I want something cheaper")
- hard_constraint: Must-be-satisfied rules that cause failures if lost (e.g., budget limits, compliance)
- soft_preference: UX-affecting preferences, not mandatory but should be satisfied (e.g., "keep it brief")
- confirmed_fact: Facts explicitly affirmed or denied by the user (both positive and negative)
- decision: What decisions were made and why (similar to Plan/ReAct reasoning traces)
- entity: Domain-specific entities (projects, products, people, companies, APIs, documents, versions)

Rules:
1. Only extract facts that influence subsequent decisions, tool calls, or output.
2. If a fact does not affect Agent behavior, do not include it.
3. Be concise: one fact per entry, no explanations.
4. Preserve the original meaning; do not infer beyond what is stated.

Return a JSON array. Each element: {"category": "<category>", "content": "<fact>", "confidence": <0.0-1.0>}

Content to analyze:
"""


class KeyFactExtractor:
    """Extracts and classifies key facts using an LLM.

    Uses a lightweight flash/mini model since this is information extraction,
    not complex reasoning.
    """

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

    async def extract(self, items: list[ContextItem]) -> list[KeyFact]:
        """Extract key facts from a list of context items."""
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
                {"role": "system", "content": "You are a key fact extraction assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
        )

        raw_text = (response.choices[0].message.content or "").strip()
        return _parse_key_facts(raw_text, items)


class MockKeyFactExtractor:
    """Mock extractor for local development and testing.

    Uses simple keyword matching instead of an LLM call.
    """

    _KEYWORDS = {
        KeyFactCategory.GOAL: ["goal", "target", "objective", "want", "need", "aim"],
        KeyFactCategory.HARD_CONSTRAINT: ["must", "cannot", "budget", "limit", "deadline", "require"],
        KeyFactCategory.SOFT_PREFERENCE: ["prefer", "like", "would rather", "brief", "concise"],
        KeyFactCategory.CONFIRMED_FACT: ["confirmed", "yes", "no", "agree", "disagree", "deny"],
        KeyFactCategory.DECISION: ["decided", "chose", "selected", "plan", "action"],
        KeyFactCategory.ENTITY: ["order", "product", "user", "company", "project", "api"],
    }

    async def extract(self, items: list[ContextItem]) -> list[KeyFact]:
        """Extract key facts via keyword matching (no LLM call)."""
        facts: list[KeyFact] = []
        for item in items:
            text = item.content_as_string().lower()
            for category, keywords in self._KEYWORDS.items():
                for kw in keywords:
                    if kw in text:
                        facts.append(
                            KeyFact(
                                category=category,
                                content=f"[{category.value}] {item.content_as_string()[:80]}",
                                source_item_id=item.id,
                                confidence=0.7,
                            )
                        )
                        break
        return facts


def _parse_key_facts(raw_text: str, source_items: list[ContextItem]) -> list[KeyFact]:
    """Parse LLM JSON response into KeyFact objects."""
    import json

    # Strip markdown code fences if present.
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.startswith("```")]
        text = "\n".join(lines)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    facts: list[KeyFact] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        category_str = entry.get("category", "")
        content = entry.get("content", "")
        confidence = entry.get("confidence", 0.8)
        try:
            category = KeyFactCategory(category_str)
        except ValueError:
            continue
        if not content:
            continue
        source_id = source_items[0].id if source_items else None
        facts.append(
            KeyFact(
                category=category,
                content=content,
                source_item_id=source_id,
                confidence=float(confidence),
            )
        )
    return facts
