"""Conflict resolution for recalled details.

When multiple recalled details contradict each other, conflicts must be
resolved deterministically. Two strategies are supported:

1. Last-write-wins (recency): newer information overrides older.
2. Authority precedence: higher-authority sources override lower.

The "reinforcement vs conflict" distinction is handled via configurable
rules: a stricter/specific rule is treated as reinforcement (both kept),
not a conflict.
"""

from enum import Enum

from app.models import ContextAuthority, ContextItem


class ConflictResolutionStrategy(str, Enum):
    """Strategies for resolving conflicts among recalled details."""

    LAST_WRITE_WINS = "last_write_wins"
    AUTHORITY_PRECEDENCE = "authority_precedence"


# Authority rank: higher number = higher authority.
_AUTHORITY_RANK: dict[ContextAuthority, int] = {
    ContextAuthority.HARD_RULE: 5,
    ContextAuthority.CONFIRMED: 4,
    ContextAuthority.INFERRED: 3,
    ContextAuthority.ASSUMED: 2,
    ContextAuthority.DENIED: 1,
}


def _authority_rank(item: ContextItem) -> int:
    """Return the numeric authority rank of an item."""
    return _AUTHORITY_RANK.get(item.authority, 0)


def _is_reinforcement(item_a: ContextItem, item_b: ContextItem) -> bool:
    """Heuristic: detect if item_b reinforces (is stricter than) item_a.

    A reinforcement is a stricter or more specific version of the same
    rule, not a contradiction. This simple heuristic checks if one
    content string is a substring of the other or if both share
    significant keyword overlap.
    """
    text_a = item_a.content_as_string().lower()
    text_b = item_b.content_as_string().lower()

    # Substring relationship implies specificity.
    if text_a in text_b or text_b in text_a:
        return True

    # Keyword overlap: if >60% of words match, likely reinforcement.
    words_a = set(text_a.split())
    words_b = set(text_b.split())
    if not words_a or not words_b:
        return False
    overlap = len(words_a & words_b)
    min_len = min(len(words_a), len(words_b))
    return overlap / min_len > 0.6


class ConflictResolver:
    """Resolves contradictions among recalled context items.

    Uses the specified strategy to pick a winner when items conflict.
    Reinforcements (stricter/more specific versions) are kept alongside
    the original rather than replacing it.
    """

    def __init__(self, strategy: ConflictResolutionStrategy = ConflictResolutionStrategy.LAST_WRITE_WINS):
        self._strategy = strategy

    def resolve(self, items: list[ContextItem]) -> list[ContextItem]:
        """Resolve conflicts among a list of items.

        Items that reinforce each other are all kept. Items that truly
        conflict are reduced to a single winner based on the strategy.
        """
        if len(items) <= 1:
            return list(items)

        # Group items by a rough topic key (first 20 chars of content).
        groups: dict[str, list[ContextItem]] = {}
        for item in items:
            key = item.content_as_string()[:20].lower().strip()
            groups.setdefault(key, []).append(item)

        resolved: list[ContextItem] = []
        for group_items in groups.values():
            resolved.extend(self._resolve_group(group_items))
        return resolved

    def _resolve_group(self, items: list[ContextItem]) -> list[ContextItem]:
        """Resolve conflicts within a single group of related items."""
        if len(items) <= 1:
            return items

        # Check for reinforcement relationships.
        kept: list[ContextItem] = []
        used = [False] * len(items)
        for i, item_a in enumerate(items):
            if used[i]:
                continue
            used[i] = True
            kept.append(item_a)
            for j in range(i + 1, len(items)):
                if used[j]:
                    continue
                if _is_reinforcement(item_a, items[j]):
                    # Keep the more specific (longer) version.
                    if len(items[j].content_as_string()) > len(item_a.content_as_string()):
                        kept[-1] = items[j]
                    used[j] = True

        # For remaining unused items (true conflicts), pick a winner.
        conflicts = [items[i] for i in range(len(items)) if not used[i]]
        if conflicts:
            winner = self._pick_winner(conflicts)
            kept.append(winner)
        return kept

    def _pick_winner(self, items: list[ContextItem]) -> ContextItem:
        """Pick a single winner from conflicting items."""
        if self._strategy == ConflictResolutionStrategy.LAST_WRITE_WINS:
            # Newer items win; fall back to authority if timestamps are equal.
            return max(items, key=lambda x: (x.created_at, _authority_rank(x)))
        # AUTHORITY_PRECEDENCE: higher authority wins; fall back to recency.
        return max(items, key=lambda x: (_authority_rank(x), x.created_at))


class RecallMetrics:
    """Tracks metrics for detail recall operations."""

    def __init__(self):
        self.recall_count: int = 0
        self.hit_count: int = 0
        self.conflict_count: int = 0
        self.total_latency_ms: float = 0.0

    @property
    def hit_rate(self) -> float:
        """Return the recall hit rate (0.0–1.0)."""
        if self.recall_count == 0:
            return 0.0
        return self.hit_count / self.recall_count

    @property
    def avg_latency_ms(self) -> float:
        """Return the average recall latency in milliseconds."""
        if self.recall_count == 0:
            return 0.0
        return self.total_latency_ms / self.recall_count

    def record_recall(self, result_count: int, latency_ms: float) -> None:
        """Record a single recall operation."""
        self.recall_count += 1
        if result_count > 0:
            self.hit_count += 1
        self.total_latency_ms += latency_ms

    def record_conflict(self, count: int = 1) -> None:
        """Record conflict resolution events."""
        self.conflict_count += count
