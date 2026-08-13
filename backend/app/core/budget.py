"""Budget mode resolution based on remaining-token ratio."""

from app.models import BudgetMode, TokenBudget


class BudgetModeResolver:
    """Maps a TokenBudget's remaining ratio to a BudgetMode.

    Default thresholds (configurable via constructor):
        Full      > 50%
        Balanced  20-50%
        Compact   5-20%
        Minimal   < 5%
    """

    def __init__(
        self,
        full_threshold: float = 0.50,
        balanced_threshold: float = 0.20,
        compact_threshold: float = 0.05,
    ):
        if not (0 < compact_threshold <= balanced_threshold <= full_threshold < 1):
            raise ValueError("Thresholds must satisfy 0 < compact <= balanced <= full < 1")
        self.full_threshold = full_threshold
        self.balanced_threshold = balanced_threshold
        self.compact_threshold = compact_threshold

    def resolve(self, token_budget: TokenBudget) -> BudgetMode:
        """Return the BudgetMode for the given token budget."""
        ratio = token_budget.ratio()
        if ratio > self.full_threshold:
            return BudgetMode.FULL
        if ratio > self.balanced_threshold:
            return BudgetMode.BALANCED
        if ratio > self.compact_threshold:
            return BudgetMode.COMPACT
        return BudgetMode.MINIMAL
