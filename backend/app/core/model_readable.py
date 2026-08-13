"""Model-readable summaries: high-density compression for LLM consumption.

Strips human-facing filler (politeness, transitions, redundancy) while
preserving goals, entities, facts, constraints, decisions, and states.
Uses a lightweight flash/mini model for generation.
"""

from openai import AsyncOpenAI

from app.config import settings
from app.core.tokenizer import estimate_tokens


_COMPRESSION_PROMPT = """\
You are an Agent context compressor.

Compress the input content into a high-density context for subsequent LLM use, \
NOT a natural-language summary for humans.

Requirements:
1. Remove politeness, transitional sentences, conversational filler, redundant explanations, and emotional-value expressions.
2. Preserve: goals, entities, key facts, hard constraints, soft preferences, user confirmations, user denials, numbers, times, decisions, rationales, risks, missing information, and evidence sources.
3. Short phrases, lists, and key-value pairs are acceptable; natural-language fluency is NOT required.
4. Never change the original meaning.
5. Never convert uncertain information into stated facts.
6. Never convert recommendations into executed decisions.
7. Never convert Assistant speculation into user-confirmed facts.
8. Hard constraints and user denials MUST be explicitly retained.

Content to compress:
"""


class ModelReadableCompressor:
    """Produces model-readable high-density summaries.

    Unlike traditional prose summaries written for human readability, this
    compressor targets LLM consumption: it strips low-information filler
    and retains only high-signal content in a compact, structured form.
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

    async def compress(self, content: str) -> str:
        """Compress a content string into a model-readable summary."""
        if not content.strip():
            return ""

        prompt = _COMPRESSION_PROMPT + content

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a context compression assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=256,
        )

        return (response.choices[0].message.content or "").strip()

    async def compress_with_cost(self, content: str) -> tuple[str, int]:
        """Compress content and return (compressed_text, token_cost)."""
        compressed = await self.compress(content)
        return compressed, estimate_tokens(compressed)


class MockModelReadableCompressor:
    """Mock compressor for local development and testing.

    Applies simple rule-based filler removal instead of an LLM call.
    """

    _FILLER_PATTERNS = [
        "这是一个很好的问题。",
        "我理解你的意思。",
        "我理解您的意思。",
        "下面我从三个方面来分析。",
        "That's a great question.",
        "I understand your concern.",
        "Let me break this down into three points.",
        "嗯",
        "啊",
        "这个",
        "其实",
        "然后",
        "就是说",
        "怎么讲呢",
        "我觉得吧",
        "差不多",
        "大概",
        "可能",
        "反正",
        "I think",
        "basically",
        "probably",
    ]

    async def compress(self, content: str) -> str:
        """Compress content via simple filler removal (no LLM call)."""
        result = content
        for pattern in self._FILLER_PATTERNS:
            result = result.replace(pattern, "")
        # Collapse whitespace.
        result = " ".join(result.split())
        return result

    async def compress_with_cost(self, content: str) -> tuple[str, int]:
        """Compress content and return (compressed_text, token_cost)."""
        compressed = await self.compress(content)
        return compressed, estimate_tokens(compressed)
