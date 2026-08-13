from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


class ContextType(str, Enum):
    """Supported context item types."""

    CONSTRAINT = "constraint"
    FACT = "fact"
    TOOL_RESULT = "tool_result"
    USER_INPUT = "user_input"
    MODEL_OUTPUT = "model_output"
    SUMMARY = "summary"
    PROFILE = "profile"


class ContextSource(str, Enum):
    """Where the context item comes from."""

    USER = "user"
    AGENT = "agent"
    TOOL = "tool"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ContextScope(str, Enum):
    """Validity scope of the context item."""

    CURRENT_STEP = "current_step"
    CURRENT_TASK = "current_task"
    CURRENT_SESSION = "current_session"
    CURRENT_USER = "current_user"
    GLOBAL = "global"


class ContextAuthority(str, Enum):
    """Trust level of the context item."""

    HARD_RULE = "hard_rule"
    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    ASSUMED = "assumed"
    DENIED = "denied"


class CompressionLevel(str, Enum):
    """Five compression levels from drop to raw."""

    L0 = "l0"  # Drop the item entirely
    L1 = "l1"  # Entity / keyword extraction (~10 tokens)
    L2 = "l2"  # One-sentence summary (~30 tokens)
    L3 = "l3"  # Structured summary (~100 tokens or 30% of content)
    L4 = "l4"  # Raw, no compression


class ContextItem(BaseModel):
    """Unified schema for a single piece of context."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    type: ContextType
    content: Any
    source: ContextSource
    scope: ContextScope
    authority: ContextAuthority = ContextAuthority.ASSUMED
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    priority: int = 0
    token_cost: int = Field(ge=0, default=0)
    layer: str = "session"
    version: int = Field(ge=1, default=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    compression_level: Optional[CompressionLevel] = None
    correlation_group: Optional[str] = None

    @field_validator("content", mode="before")
    @classmethod
    def _ensure_string_content(cls, value: Any) -> Any:
        """Allow any content type; downstream serializers handle string conversion."""
        return value

    def is_expired(self) -> bool:
        """Return True when the item has passed its expiration time."""
        if self.expires_at is None:
            return False
        return datetime.now(timezone.utc) > self.expires_at

    def content_as_string(self) -> str:
        """Return the content as a string for token estimation or injection."""
        if isinstance(self.content, str):
            return self.content
        return str(self.content)


class WindowStrategy(str, Enum):
    """Available window composition strategies."""

    SLIDING = "sliding"
    SUMMARY = "summary"
    HYBRID = "hybrid"
    DYNAMIC = "dynamic"


class BudgetMode(str, Enum):
    """Window budget modes driven by remaining-token ratio."""

    FULL = "full"        # > 50% remaining
    BALANCED = "balanced"  # 20-50% remaining
    COMPACT = "compact"  # 5-20% remaining
    MINIMAL = "minimal"  # < 5% remaining


class Importance(str, Enum):
    """Item importance classification for compression decisions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class TaskState(BaseModel):
    """Current task state provided by the Agent runtime."""

    current_step: Optional[str] = None
    goal: Optional[str] = None
    progress: Optional[str] = None
    missing_context: Optional[list[str]] = None


class TokenBudget(BaseModel):
    """Token budget for window composition."""

    total: int = Field(gt=0, default=4096)
    reserved: int = Field(ge=0, default=1024)
    remaining: Optional[int] = None

    def effective_remaining(self) -> int:
        """Return remaining tokens, computing from total - reserved if not set."""
        if self.remaining is not None:
            return max(0, self.remaining)
        return max(0, self.total - self.reserved)

    def ratio(self) -> float:
        """Return the remaining-to-total ratio (0.0–1.0)."""
        if self.total <= 0:
            return 0.0
        return self.effective_remaining() / self.total


class ComposeRequest(BaseModel):
    """Request body for composing a context window."""

    session_id: str = Field(min_length=1)
    strategy: WindowStrategy = WindowStrategy.SLIDING
    max_tokens: int = Field(gt=0, default=4096)
    window_size: Optional[int] = Field(default=None, ge=1)
    task_state: Optional[TaskState] = None
    token_budget: Optional[TokenBudget] = None
    scenario: Optional[str] = None


class ComposeResponse(BaseModel):
    """Response body for a composed context window."""

    session_id: str
    strategy: WindowStrategy
    items: list[ContextItem]
    prompt_fragment: str
    total_tokens: int
    item_count: int
    budget_mode: Optional[BudgetMode] = None


class ContextItemCreateRequest(BaseModel):
    """Request body for creating a context item."""

    type: ContextType
    content: Any
    source: ContextSource
    scope: ContextScope
    authority: ContextAuthority = ContextAuthority.ASSUMED
    confidence: float = Field(ge=0.0, le=1.0, default=1.0)
    priority: int = 0
    layer: str = "session"
    expires_at: Optional[datetime] = None
