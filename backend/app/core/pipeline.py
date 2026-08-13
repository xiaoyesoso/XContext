from abc import ABC, abstractmethod

from app.core.tokenizer import estimate_item_tokens
from app.models import ContextAuthority, ContextItem, ContextType


def _item_cost(item: ContextItem) -> int:
    """Return the token cost of an item, estimating if not provided."""
    return item.token_cost or estimate_item_tokens(item.content_as_string())


class Retriever(ABC):
    """Abstract base for fetching candidate context items."""

    @abstractmethod
    async def retrieve(self, session_id: str) -> list[ContextItem]:
        """Return all candidate items for a session."""


class Selector(ABC):
    """Abstract base for filtering and selecting relevant context items."""

    @abstractmethod
    def select(self, items: list[ContextItem]) -> list[ContextItem]:
        """Return the filtered and ranked list of items."""


class Compressor(ABC):
    """Abstract base for reducing item granularity to fit the budget."""

    @abstractmethod
    def compress(self, item: ContextItem) -> ContextItem:
        """Return a compressed copy of the item."""


class Orderer(ABC):
    """Abstract base for deciding the final order of items in the window."""

    @abstractmethod
    async def order(self, items: list[ContextItem]) -> list[ContextItem]:
        """Return ordered items."""


class Injector(ABC):
    """Abstract base for serializing items into the prompt format."""

    @abstractmethod
    def inject(self, items: list[ContextItem]) -> str:
        """Return the serialized prompt fragment."""


class DefaultSelector(Selector):
    """Default selector that drops expired and denied items.

    Remaining items are ranked by priority (descending), authority
    (hard_rule first), and confidence (descending).
    """

    def select(self, items: list[ContextItem]) -> list[ContextItem]:
        active = [
            item
            for item in items
            if not item.is_expired() and item.authority != ContextAuthority.DENIED
        ]
        authority_rank = {
            ContextAuthority.HARD_RULE: 0,
            ContextAuthority.CONFIRMED: 1,
            ContextAuthority.INFERRED: 2,
            ContextAuthority.ASSUMED: 3,
            ContextAuthority.DENIED: 4,
        }
        active.sort(
            key=lambda item: (
                -item.priority,
                authority_rank.get(item.authority, 99),
                -item.confidence,
            )
        )
        return active


class SlidingWindowOrderer(Orderer):
    """Keep the most recent items, capped by a token budget."""

    def __init__(self, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.max_tokens = max_tokens

    async def order(self, items: list[ContextItem]) -> list[ContextItem]:
        sorted_items = sorted(items, key=lambda item: item.created_at)
        selected: list[ContextItem] = []
        total = 0
        for item in reversed(sorted_items):
            cost = _item_cost(item)
            if total + cost > self.max_tokens and selected:
                break
            selected.append(item)
            total += cost
        return list(reversed(selected))


class PlainTextInjector(Injector):
    """Serialize items as plain text blocks."""

    def inject(self, items: list[ContextItem]) -> str:
        blocks = []
        for item in items:
            header = f"[{item.type.value}] source={item.source.value} scope={item.scope.value}"
            blocks.append(f"{header}\n{item.content_as_string()}\n")
        return "\n".join(blocks)


class NoOpCompressor(Compressor):
    """Compressor that returns the item unchanged."""

    def compress(self, item: ContextItem) -> ContextItem:
        return item


class SummaryOrderer(Orderer):
    """Replace older blocks of messages with summaries, keep recent raw items.

    Every `window_size` messages form a block. The most recent block is kept
    raw; older blocks are replaced by a summary item.
    """

    def __init__(self, max_tokens: int, window_size: int, summarizer):
        if window_size < 1:
            raise ValueError("window_size must be positive")
        self.max_tokens = max_tokens
        self.window_size = window_size
        self._summarizer = summarizer

    async def order(self, items: list[ContextItem]) -> list[ContextItem]:
        sorted_items = sorted(items, key=lambda item: item.created_at)
        if len(sorted_items) <= self.window_size:
            return await SlidingWindowOrderer(self.max_tokens).order(sorted_items)

        # Partition into blocks of window_size from oldest to newest.
        blocks: list[list[ContextItem]] = []
        for i in range(0, len(sorted_items), self.window_size):
            blocks.append(sorted_items[i : i + self.window_size])

        # Keep the last block raw; summarize the rest.
        summarized: list[ContextItem] = []
        for block in blocks[:-1]:
            summary = await self._summarizer.summarize(block)
            # Align summary timestamp with the block end so ordering is stable.
            summary.created_at = block[-1].created_at
            summarized.append(summary)
        summarized.extend(blocks[-1])

        return await SlidingWindowOrderer(self.max_tokens).order(summarized)


class HybridOrderer(Orderer):
    """Keep user inputs raw; summarize model outputs except the most recent ones.

    The most recent `raw_output_count` model outputs are kept raw. Older model
    outputs are summarized. User inputs are always kept raw.
    """

    def __init__(
        self,
        max_tokens: int,
        window_size: int,
        summarizer,
    ):
        self.max_tokens = max_tokens
        self.window_size = window_size
        self.raw_output_count = max(1, window_size // 2)
        self._summarizer = summarizer

    async def order(self, items: list[ContextItem]) -> list[ContextItem]:
        sorted_items = sorted(items, key=lambda item: item.created_at)
        user_inputs = [item for item in sorted_items if item.type == ContextType.USER_INPUT]
        model_outputs = [item for item in sorted_items if item.type == ContextType.MODEL_OUTPUT]

        raw_outputs = model_outputs[-self.raw_output_count :]
        old_outputs = model_outputs[: -self.raw_output_count]

        summaries: list[ContextItem] = []
        for i in range(0, len(old_outputs), self.window_size):
            block = old_outputs[i : i + self.window_size]
            summary = await self._summarizer.summarize(block)
            summary.created_at = block[-1].created_at
            summaries.append(summary)

        combined = sorted(user_inputs + summaries + raw_outputs, key=lambda item: item.created_at)
        return await SlidingWindowOrderer(self.max_tokens).order(combined)
