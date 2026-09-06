"""Policy-driven context composition.

A policy is an ordered list of segments. Each segment names a ContextAction
and declares its priority, compression level, and optional filters. The
orchestrator executes segments in priority order, fits them into the token
budget, and emits a prompt fragment plus execution metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4

from app.core.context_actions import (
    ActionContext,
    ActionRegistry,
    ActionResult,
    ActionServices,
)
from app.core.tokenizer import estimate_item_tokens
from app.models import (
    ContextItem,
    ContextPolicy,
    PolicyExecution,
    PolicyMetrics,
    PolicySegment,
    PolicyState,
    SegmentExecution,
)


# Compression level -> max token cap per item (rough heuristic).
_COMPRESSION_CAPS = {
    0: None,       # raw
    1: 200,        # light
    2: 80,         # summary
    3: 20,         # index
}


@dataclass
class OrchestratorConfig:
    """Runtime configuration for the policy orchestrator."""

    budget_tokens: int = 4096
    reserved_tokens: int = 1024

    def effective_budget(self) -> int:
        """Return tokens available for context."""
        return max(0, self.budget_tokens - self.reserved_tokens)


class PolicyOrchestrator:
    """Execute context policies and compose prompt fragments."""

    DEFAULT_POLICY_ID = "default"

    def __init__(
        self,
        registry: ActionRegistry | None = None,
        services: ActionServices | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self._registry = registry or ActionRegistry()
        self._services = services or ActionServices()
        self._config = config or OrchestratorConfig()
        self._policies: dict[str, ContextPolicy] = {}
        self._executions: dict[str, PolicyExecution] = {}
        self._selected_items: dict[str, list[ContextItem]] = {}
        self._default_policy = self._build_default_policy()
        self._policies[self.DEFAULT_POLICY_ID] = self._default_policy

    # ------------------------------------------------------------------ policy mgmt

    def _build_default_policy(self) -> ContextPolicy:
        """Built-in fallback policy used when no scenario-specific policy exists."""
        return ContextPolicy(
            policy_id=self.DEFAULT_POLICY_ID,
            name="default",
            scenario="*",
            default=True,
            segments=[
                PolicySegment(action_name="constraint", priority=10),
                PolicySegment(action_name="key_facts", priority=9),
                PolicySegment(action_name="user_profile", priority=8),
                PolicySegment(action_name="dialogue_history", priority=7),
                PolicySegment(action_name="summary", priority=5),
                PolicySegment(action_name="detail_recall", priority=4),
            ],
        )

    def list_policies(self) -> list[ContextPolicy]:
        """Return all registered policies including the default."""
        return list(self._policies.values())

    def get_policy(self, policy_id: str) -> ContextPolicy | None:
        """Fetch a policy by id."""
        return self._policies.get(policy_id)

    def get_default_policy(self) -> ContextPolicy:
        """Return the default policy."""
        return self._default_policy

    def upsert_policy(self, policy: ContextPolicy) -> list[str]:
        """Validate and store a policy. Returns validation errors if any."""
        errors = self._registry.validate_policy(policy)
        if errors:
            return errors
        # Only one policy can be default; clear others.
        if policy.default:
            for p in self._policies.values():
                p.default = False
            policy.default = True
        self._policies[policy.policy_id] = policy
        return []

    def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy. The default policy cannot be deleted."""
        if policy_id == self.DEFAULT_POLICY_ID:
            return False
        return self._policies.pop(policy_id, None) is not None

    def select_policy(self, scenario: str | None) -> ContextPolicy:
        """Pick the best matching policy for a scenario."""
        if scenario:
            for policy in self._policies.values():
                if policy.scenario == scenario and not policy.default:
                    return policy
        for policy in self._policies.values():
            if policy.default:
                return policy
        return self._default_policy

    # ------------------------------------------------------------------ execution

    async def execute(
        self,
        session_id: str,
        user_id: str,
        scenario: str | None,
        user_message: str,
        items: list[ContextItem],
        state: dict | None = None,
        task_state: Any = None,
        token_budget: int | None = None,
    ) -> PolicyExecution:
        """Execute the selected policy and return a full execution record."""
        policy = self.select_policy(scenario)
        budget = (
            max(0, token_budget - self._config.reserved_tokens)
            if token_budget is not None
            else self._config.effective_budget()
        )

        runtime_state = dict(state or {})
        runtime_state["task_state"] = task_state
        runtime_state["k_turn"] = state.get("k_turn") if state else None

        ctx = ActionContext(
            session_id=session_id,
            user_id=user_id,
            scenario=scenario or "default",
            user_message=user_message,
            remaining_tokens=budget,
            items=items,
            state=runtime_state,
            services=self._services,
        )

        # Execute segments in priority order (highest first).
        sorted_segments = sorted(
            (s for s in policy.segments if s.enabled),
            key=lambda s: -s.priority,
        )

        results: list[ActionResult] = []
        raw_total = 0
        used_total = 0
        remaining = budget
        dropped_segments = 0

        for segment in sorted_segments:
            action = self._registry.get(segment.action_name)
            if action is None:
                dropped_segments += 1
                results.append(
                    ActionResult(
                        action_name=segment.action_name,
                        dropped=True,
                        metadata={"error": "action not found"},
                    )
                )
                continue

            action_ctx = self._narrow_context(ctx, segment)
            try:
                result = await action.execute(action_ctx)
            except Exception as exc:  # noqa: BLE001
                result = ActionResult(
                    action_name=segment.action_name,
                    dropped=True,
                    metadata={"error": str(exc)},
                )

            # Apply max_items filter.
            if segment.max_items is not None and result.items:
                result.items = result.items[: segment.max_items]
                result.raw_tokens = sum(
                    estimate_item_tokens(i.content_as_string())
                    for i in result.items
                )
                result.compressed_tokens = result.raw_tokens

            raw_total += result.raw_tokens
            result, remaining = self._fit_to_budget(result, remaining, segment)
            if result.dropped:
                dropped_segments += 1
            used_total += result.compressed_tokens
            results.append(result)

        # Re-order selected items by priority to form the prompt fragment.
        selected_items = self._order_items(results, policy.segments)
        self._selected_items[session_id] = selected_items

        metrics = self._compute_metrics(
            results, budget, used_total, dropped_segments
        )
        execution = PolicyExecution(
            session_id=session_id,
            policy_id=policy.policy_id,
            policy_name=policy.name,
            scenario=policy.scenario,
            segments=[self._to_segment_execution(r, policy.segments) for r in results],
            total_raw_tokens=raw_total,
            total_used_tokens=used_total,
            budget_tokens=budget,
            metrics=metrics,
        )
        self._executions[session_id] = execution
        return execution

    def _narrow_context(
        self, ctx: ActionContext, segment: PolicySegment
    ) -> ActionContext:
        """Create a context view scoped to the segment configuration."""
        narrowed = ActionContext(
            session_id=ctx.session_id,
            user_id=ctx.user_id,
            scenario=ctx.scenario,
            user_message=ctx.user_message,
            remaining_tokens=ctx.remaining_tokens,
            items=ctx.items,
            state=dict(ctx.state),
            services=ctx.services,
        )
        narrowed.state["filters"] = segment.filters
        return narrowed

    def _fit_to_budget(
        self,
        result: ActionResult,
        remaining: int,
        segment: PolicySegment,
    ) -> tuple[ActionResult, int]:
        """Compress or drop a result to fit the remaining token budget."""
        if result.dropped:
            return result, remaining

        raw_tokens = sum(
            estimate_item_tokens(i.content_as_string()) for i in result.items
        )
        result.raw_tokens = raw_tokens

        # First try the segment's configured compression level.
        compressed = self._apply_compression(result.items, segment.compression_level)
        compressed_tokens = sum(
            estimate_item_tokens(i.content_as_string()) for i in compressed
        )

        if compressed_tokens <= remaining:
            result.items = compressed
            result.compressed_tokens = compressed_tokens
            result.compressed = segment.compression_level > 0
            return result, remaining - compressed_tokens

        # If still too large, try heavier compression up to level 3.
        for level in range(segment.compression_level + 1, 4):
            compressed = self._apply_compression(result.items, level)
            compressed_tokens = sum(
                estimate_item_tokens(i.content_as_string()) for i in compressed
            )
            if compressed_tokens <= remaining:
                result.items = compressed
                result.compressed_tokens = compressed_tokens
                result.compressed = True
                return result, remaining - compressed_tokens

        # Still too large: drop the segment entirely.
        result.items = []
        result.compressed_tokens = 0
        result.compressed = False
        result.dropped = True
        return result, remaining

    def _apply_compression(
        self, items: list[ContextItem], level: int
    ) -> list[ContextItem]:
        """Return a compressed copy of the items at the given level."""
        cap = _COMPRESSION_CAPS.get(level)
        if cap is None:
            return items

        compressed: list[ContextItem] = []
        for item in items:
            text = item.content_as_string()
            if estimate_item_tokens(text) <= cap:
                compressed.append(item)
                continue
            # Naive compression: truncate and append ellipsis.
            truncated = text[: cap * 4] + " ..." if len(text) > cap * 4 else text
            while estimate_item_tokens(truncated) > cap and len(truncated) > 20:
                truncated = truncated[: -len(truncated) // 10]
            new_item = item.model_copy(update={"content": truncated})
            compressed.append(new_item)
        return compressed

    def _order_items(
        self, results: list[ActionResult], segments: list[PolicySegment]
    ) -> list[ContextItem]:
        """Order selected items by policy priority (highest first)."""
        priority_by_action = {s.action_name: s.priority for s in segments}
        ordered: list[tuple[int, ContextItem]] = []
        for result in results:
            if result.dropped or not result.items:
                continue
            priority = priority_by_action.get(result.action_name, 0)
            for item in result.items:
                ordered.append((-priority, item))
        ordered.sort(key=lambda x: x[0])
        return [item for _, item in ordered]

    def _compute_metrics(
        self,
        results: list[ActionResult],
        budget: int,
        used_total: int,
        dropped_segments: int,
    ) -> PolicyMetrics:
        """Compute effectiveness metrics for the execution."""
        critical_total = 0
        critical_retained = 0
        recall_hits = 0
        for result in results:
            if result.action_name in ("constraint", "key_facts"):
                critical_total += 1
                if not result.dropped:
                    critical_retained += 1
            if result.action_name == "detail_recall":
                recall_hits += result.metadata.get("hit_count", 0)

        critical_retention = (
            critical_retained / critical_total if critical_total else 1.0
        )
        utilization = min(1.0, used_total / budget) if budget else 0.0
        return PolicyMetrics(
            window_utilization=utilization,
            critical_retention=critical_retention,
            recall_hit_count=recall_hits,
            dropped_segments=dropped_segments,
        )

    def _to_segment_execution(
        self, result: ActionResult, segments: list[PolicySegment]
    ) -> SegmentExecution:
        priority = next(
            (s.priority for s in segments if s.action_name == result.action_name),
            0,
        )
        return SegmentExecution(
            action_name=result.action_name,
            priority=priority,
            raw_tokens=result.raw_tokens,
            compressed_tokens=result.compressed_tokens,
            item_count=len(result.items),
            compressed=result.compressed,
            dropped=result.dropped,
            metadata=result.metadata,
        )

    # ------------------------------------------------------------------ observation

    def get_execution(self, session_id: str) -> PolicyExecution | None:
        """Return the latest execution for a session."""
        return self._executions.get(session_id)

    def get_selected_items(self, session_id: str) -> list[ContextItem]:
        """Return the items selected by the latest execution."""
        return self._selected_items.get(session_id, [])

    def get_state(self, session_id: str) -> PolicyState | None:
        """Return a lightweight state view for the frontend."""
        execution = self._executions.get(session_id)
        if execution is None:
            policy = self._default_policy
            return PolicyState(
                policy_id=policy.policy_id,
                policy_name=policy.name,
                scenario=policy.scenario,
            )
        return PolicyState(
            policy_id=execution.policy_id,
            execution_id=execution.execution_id,
            policy_name=execution.policy_name,
            scenario=execution.scenario,
            metrics=execution.metrics,
            segments=execution.segments,
        )

    def compose_prompt(self, items: list[ContextItem]) -> str:
        """Convert selected items into a plain-text prompt fragment."""
        parts: list[str] = []
        for item in items:
            label = item.type.value.upper()
            parts.append(f"[{label}] {item.content_as_string()}")
        return "\n\n".join(parts)
