from datetime import datetime, timezone
from uuid import uuid4

from openai import AsyncOpenAI

from app.config import settings
from app.core.summarizer import Summarizer
from app.core.tokenizer import estimate_tokens
from app.models import ContextAuthority, ContextItem, ContextSource, ContextType


class OpenAISummarizer(Summarizer):
    """Summarizer that calls an OpenAI-compatible API (e.g., SiliconFlow)."""

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

    async def summarize(self, items: list[ContextItem]) -> ContextItem:
        """Generate a concise summary of the provided context items."""
        texts = []
        for item in items:
            prefix = f"[{item.type.value}]"
            texts.append(f"{prefix} {item.content_as_string()}")
        prompt = (
            "Summarize the following conversation messages into a concise paragraph. "
            "Preserve key facts, decisions, and constraints.\n\n"
            + "\n".join(texts)
        )

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": "You are a helpful summarization assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=256,
        )
        summary_text = response.choices[0].message.content or ""

        # Use actual token count for the summary content, not the API completion tokens.
        # API completion_tokens may include special tokens and is not suitable for
        # context-window budgeting.
        summary_content = summary_text.strip()
        token_cost = estimate_tokens(summary_content)

        return ContextItem(
            id=str(uuid4()),
            type=ContextType.SUMMARY,
            content=summary_content,
            source=ContextSource.AGENT,
            scope=items[-1].scope if items else None,
            authority=ContextAuthority.INFERRED,
            confidence=0.8,
            priority=0,
            token_cost=token_cost,
            layer=items[-1].layer if items else "session",
            created_at=datetime.now(timezone.utc),
        )
