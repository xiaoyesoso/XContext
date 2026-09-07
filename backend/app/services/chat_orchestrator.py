"""Conversation-driven orchestration for summaries, profiles, and recall.

Architecture (product decision):
- Construction is ASYNC during the conversation: at the end of each turn
  (end_of_turn trigger), multi-type summary extraction and five-dimension
  profile extraction run as background tasks.
- Injection is SYNC during the conversation: at the start of the next turn,
  completed summaries, extracted profile facts, and keyword-recalled
  details are injected into the session context before the window is
  composed, so the pipeline sees them transparently.
"""

import asyncio
import re
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4

from app.models import (
    ComposeRequest,
    ComposeResponse,
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
    PolicyExecution,
    TaskState,
)
from app.services.profile_service import UserProfileService
from app.services.summary_service import SummaryService

from app.core.checkpoint import CheckpointManager
from app.core.drift import ContextVerifier, extract_keywords
from app.core.handoff import HandoffBuilder, HandoffStore
from app.core.policy_orchestrator import PolicyOrchestrator
from app.core.task_events import EventStore, EventType
from app.core.tokenizer import estimate_item_tokens

_DISLIKE_RE = re.compile(r"不(?:要|喜欢|考虑|接受)[:：\s]*([A-Za-z0-9\u4e00-\u9fff ]{1,20})")
_BUDGET_RE = re.compile(r"(?:预算|价位|以内|不超过|budget)[^0-9]{0,6}([0-9]{3,6})|([0-9]{3,6})[^0-9]{0,4}(?:以内|以内|预算|块|元)")


def extract_dislike_brand(content: str) -> Optional[str]:
    """Parse a brand name out of an explicit-dislike profile fact."""
    match = _DISLIKE_RE.search(content or "")
    if not match:
        return None
    brand = match.group(1).strip()
    # Keep it short: brand names are 1-12 chars.
    return brand[:12] if brand else None


def extract_budget(content: str) -> Optional[int]:
    """Parse a budget ceiling (CNY) out of a profile fact."""
    match = _BUDGET_RE.search(content or "")
    if not match:
        return None
    value = match.group(1) or match.group(2)
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ChatOrchestrator:
    """Coordinates async construction and sync injection around chat turns."""

    K_TURN_K = 3
    RECALL_TOP_K = 3

    def __init__(
        self,
        context_service,
        summary_service: SummaryService,
        profile_service: UserProfileService,
        event_store: EventStore | None = None,
        checkpoint_manager: CheckpointManager | None = None,
        handoff_store: HandoffStore | None = None,
        policy_orchestrator: PolicyOrchestrator | None = None,
    ):
        self._context = context_service
        self._summary = summary_service
        self._profile = profile_service

        self._event_store = event_store or EventStore()
        self._checkpoint_manager = checkpoint_manager or CheckpointManager(
            self._event_store
        )
        self._handoff_store = handoff_store or HandoffStore()
        self._policy_orchestrator = policy_orchestrator or PolicyOrchestrator()
        self._turn_token_budgets: dict[str, int] = {}

        self._background_tasks: set[asyncio.Task] = set()
        # session_id -> latest multi-type summary extraction result
        self._summary_results: dict[str, dict] = {}
        # session_id -> last sync-injection report
        self._injections: dict[str, dict] = {}
        # session_id -> last K-turn state dict
        self._k_turn_states: dict[str, dict] = {}
        # session_id -> last auto-recall results
        self._recalls: dict[str, list[dict]] = {}
        # session_id -> ids of auto-injected recall items (replaced each turn)
        self._recall_item_ids: dict[str, list[str]] = {}
        # session_id -> ids of auto-injected summary items (replaced each turn)
        self._summary_item_ids: dict[str, list[str]] = {}
        # session_id -> orchestrator task records
        self._tasks: dict[str, list[dict]] = {}
        # session_id -> derived recommendation spec + boundary
        self._specs: dict[str, dict] = {}
        # user_id -> last extracted profile facts
        self._profile_facts: dict[str, list] = {}
        # session_id -> current turn counter for drift control
        self._turns: dict[str, int] = {}
        # session_id -> latest policy execution
        self._policy_executions: dict[str, PolicyExecution] = {}

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def user_for_session(session_id: str) -> str:
        """Derive the user id owning a chat session."""
        return f"user-{session_id}"

    def _register_task(self, session_id: str, kind: str, item_count: int) -> dict:
        record = {
            "task_id": str(uuid4()),
            "kind": kind,
            "trigger": "end_of_turn",
            "state": "running",
            "item_count": item_count,
            "error": None,
            "result": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "completed_at": None,
        }
        self._tasks.setdefault(session_id, []).append(record)
        # Keep the task log bounded.
        self._tasks[session_id] = self._tasks[session_id][-20:]
        return record

    @staticmethod
    def _finish_task(record: dict, result=None, error: Optional[str] = None) -> None:
        record["state"] = "failed" if error else "completed"
        record["error"] = error
        record["result"] = result
        record["completed_at"] = datetime.now(timezone.utc).isoformat()

    # -------------------------------------------------- start of turn (SYNC)

    async def prepare_turn(
        self,
        session_id: str,
        user_message: str,
        scenario: Optional[str] = None,
        token_budget: Optional[int] = None,
        task_state: Optional[TaskState] = None,
    ) -> dict:
        """Synchronously inject summaries, profile facts, and recalled details.

        Runs at the start of every turn, right after the user message is
        ingested and before the context window is composed.
        """
        self._turns[session_id] = self._turns.get(session_id, 0) + 1
        turn = self._turns[session_id]
        if token_budget is not None:
            self._turn_token_budgets[session_id] = token_budget

        # Record user input event and detect constraint changes.
        self._event_store.append(
            session_id,
            EventType.USER_INPUT,
            payload={"text": user_message},
            turn=turn,
        )
        changed = self._detect_constraint_change(user_message)
        if changed:
            self._event_store.append(
                session_id,
                EventType.CONSTRAINT_CHANGED,
                payload=changed,
                turn=turn,
            )

        items = await self._context.list_items(session_id)
        report: dict = {
            "summary_types": [],
            "profile_item_count": 0,
            "recalls": [],
            "keywords": extract_keywords(user_message),
        }

        # 1. Sync K-turn raw window; details outside it are recall targets.
        k_state = self._summary.sync_k_turn_window(session_id, items, self.K_TURN_K)
        self._k_turn_states[session_id] = k_state
        report["k_turn"] = k_state

        # 2. Inject completed summaries (replace previous auto-injected ones).
        summary_result = self._summary_results.get(session_id)
        if summary_result is not None:
            report["summary_types"] = await self._inject_summaries(
                session_id, summary_result
            )
            items = await self._context.list_items(session_id)

        # 3. Profile facts were persisted as PROFILE items by the background
        #    extraction; they flow into the window through the normal pipeline.
        report["profile_item_count"] = sum(
            1 for i in items if i.type == ContextType.PROFILE
        )

        # 4. Keyword recall over evicted details (outside the K-turn window).
        raw_ids = set(k_state.get("raw_item_ids", []))
        recalls = await self._recall_and_inject(
            session_id, user_message, items, exclude_ids=raw_ids
        )
        report["recalls"] = recalls
        self._recalls[session_id] = recalls

        # Run policy orchestration to decide what enters the window.
        policy_execution = await self._policy_orchestrator.execute(
            session_id=session_id,
            user_id=self.user_for_session(session_id),
            scenario=scenario,
            user_message=user_message,
            items=items,
            state={"k_turn": k_state},
            task_state=task_state,
            token_budget=self._turn_token_budgets.get(session_id),
        )
        self._policy_executions[session_id] = policy_execution
        report["policy_execution_id"] = policy_execution.execution_id

        self._injections[session_id] = report

        # Trigger a checkpoint when context usage is high or strategy signals it.
        if self._checkpoint_manager.should_checkpoint(
            turn,
            usage_ratio=self._estimate_usage_ratio(items),
            denied_violation=False,
        ):
            self._run_checkpoint_later(session_id, items, turn)

        return report

    async def compose_window(self, request: ComposeRequest) -> ComposeResponse:
        """Compose a context window using the policy orchestrator.

        Falls back to the legacy ContextService pipeline when no policy
        execution is available for the session.
        """
        execution = self._policy_executions.get(request.session_id)
        if execution is None:
            return await self._context.compose_window(request)

        items = self._policy_orchestrator.get_selected_items(request.session_id)
        prompt = self._policy_orchestrator.compose_prompt(items)
        total_tokens = sum(
            estimate_item_tokens(i.content_as_string()) for i in items
        )
        return ComposeResponse(
            session_id=request.session_id,
            strategy=request.strategy,
            items=items,
            prompt_fragment=prompt,
            total_tokens=total_tokens,
            item_count=len(items),
            budget_mode=None,
        )

    def _estimate_usage_ratio(self, items: list[ContextItem]) -> float:
        """Rough token-usage ratio for checkpoint triggering."""
        total = sum(estimate_item_tokens(i.content_as_string()) for i in items)
        # Assume a typical 8k window for drift-control purposes.
        return min(1.0, total / 8192.0)

    def _run_checkpoint_later(self, session_id: str, items: list[ContextItem], turn: int) -> None:
        """Schedule a checkpoint without blocking the chat response."""
        def run_checkpoint():
            self._checkpoint_manager.run_checkpoint(
                session_id,
                items,
                goal=self._extract_goal(session_id),
                task_state=None,
                turn=turn,
            )

        try:
            loop = asyncio.get_running_loop()
            loop.call_soon(run_checkpoint)
        except RuntimeError:
            run_checkpoint()

    def _extract_goal(self, session_id: str) -> Optional[str]:
        """Pull the original goal from the first user input event."""
        for event in self._event_store.list_events(session_id):
            if event.type == EventType.USER_INPUT:
                return event.payload.get("text")
        return None

    @staticmethod
    def _detect_constraint_change(message: str) -> Optional[dict]:
        """Detect simple 'change X to Y' patterns in Chinese user messages."""
        import re
        patterns = [
            r"(?:预算|价格|价位)[^0-9]{0,6}改[为成]?\s*([0-9]{3,6})",
            r"(?:改|改成|改为|调整到|更新为)\s*([^，。]{2,20})\s*(?:为|成|到)\s*([^，。]{2,20})",
        ]
        for pat in patterns:
            match = re.search(pat, message)
            if match:
                groups = match.groups()
                if len(groups) == 2:
                    return {"old": groups[0].strip(), "new": groups[1].strip()}
                return {"new": groups[0].strip()}
        return None

    async def _inject_summaries(self, session_id: str, result: dict) -> list[str]:
        """Replace auto-injected summary items with the latest result."""
        # Remove items injected by the previous turn.
        for item_id in self._summary_item_ids.get(session_id, []):
            await self._context.delete_item(session_id, item_id)

        summaries = result.get("summaries", {})
        injected: list[str] = []
        new_ids: list[str] = []
        for kind in ("conversation", "model_readable"):
            payload = summaries.get(kind)
            if not payload or not payload.get("content"):
                continue
            item = ContextItem(
                id=f"auto-summary-{kind}-{session_id}",
                type=ContextType.SUMMARY,
                content=payload["content"],
                source=ContextSource.INTERNAL,
                scope=ContextScope.CURRENT_SESSION,
                authority=ContextAuthority.INFERRED,
                token_cost=payload.get("token_cost"),
            )
            await self._context.create_item_direct(session_id, item)
            new_ids.append(item.id)
            injected.append(kind)
        self._summary_item_ids[session_id] = new_ids
        return injected

    async def _recall_and_inject(
        self,
        session_id: str,
        user_message: str,
        items: list[ContextItem],
        exclude_ids: set[str],
    ) -> list[dict]:
        """Recall evicted details matching the message and inject them."""
        keywords = extract_keywords(user_message)
        if not keywords:
            # Drop the previous turn's recall items; nothing new to inject.
            for item_id in self._recall_item_ids.get(session_id, []):
                await self._context.delete_item(session_id, item_id)
            self._recall_item_ids[session_id] = []
            return []

        recalls = await self._summary.recall_by_keywords(
            session_id,
            items,
            keywords,
            top_k=self.RECALL_TOP_K,
            exclude_ids=exclude_ids,
        )

        # Replace the previous turn's injected recall items.
        for item_id in self._recall_item_ids.get(session_id, []):
            await self._context.delete_item(session_id, item_id)

        new_ids: list[str] = []
        for hit in recalls:
            item = ContextItem(
                id=f"auto-recall-{hit['id'][:8]}-{session_id[:8]}",
                type=ContextType(hit["type"]),
                content=hit["content"],
                source=ContextSource.INTERNAL,
                scope=ContextScope.CURRENT_STEP,
                authority=ContextAuthority.INFERRED,
                token_cost=hit.get("token_cost"),
            )
            await self._context.create_item_direct(session_id, item)
            new_ids.append(item.id)
        self._recall_item_ids[session_id] = new_ids
        return recalls

    # ---------------------------------------------------- end of turn (ASYNC)

    def finalize_turn(self, session_id: str, scenario: Optional[str] = None) -> None:
        """Schedule background construction tasks (end_of_turn trigger)."""
        task = asyncio.create_task(self._finalize_async(session_id, scenario))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _finalize_async(self, session_id: str, scenario: Optional[str]) -> None:
        user_id = self.user_for_session(session_id)
        items = await self._context.list_items(session_id)
        if not items:
            return

        # Run construction tasks concurrently so a slow summary LLM does not
        # block profile extraction or spec derivation.
        # Profile extraction only analyzes the recent conversation tail so its
        # prompt stays bounded as the session grows; fact deduplication still
        # considers every profile fact stored in this session.
        conversation_tail = [
            i for i in items if i.type != ContextType.PROFILE
        ][-8:]
        profile_contents = {
            i.content for i in items if i.type == ContextType.PROFILE
        }
        await asyncio.gather(
            self._run_summary_extraction(session_id, items),
            self._run_profile_extraction(
                session_id, user_id, conversation_tail, scenario, profile_contents
            ),
            self._run_spec_derivation(session_id, user_id, scenario),
            return_exceptions=True,
        )

    async def _run_summary_extraction(
        self, session_id: str, items: list[ContextItem]
    ) -> None:
        record = self._register_task(session_id, "summary_extract", len(items))
        try:
            result = await self._summary.extract_summaries(session_id, items)
            self._summary_results[session_id] = result
            self._finish_task(
                record,
                result={
                    "types": result.get("types"),
                    "compression_ratio": result.get("compression_ratio"),
                },
            )
        except Exception as exc:  # noqa: BLE001 - background task isolation
            self._finish_task(record, error=str(exc))

    def _maybe_create_snapshot(self, session_id: str, turn: int) -> None:
        """Create a trusted snapshot every SNAPSHOT_INTERVAL turns."""
        SNAPSHOT_INTERVAL = 10
        if turn > 0 and turn % SNAPSHOT_INTERVAL == 0:
            items = asyncio.get_event_loop().run_until_complete(
                self._context.list_items(session_id)
            )
            constraints = [
                i.content_as_string()
                for i in items
                if i.authority.value in ("hard_rule", "denied")
            ]
            facts = [
                i.content_as_string()
                for i in items
                if i.type.value == "fact" and i.authority.value == "confirmed"
            ]
            self._event_store.create_snapshot(
                session_id,
                goal=self._extract_goal(session_id),
                hard_constraints=constraints,
                confirmed_facts=facts,
                task_state={"turn": turn, "item_count": len(items)},
                turn=turn,
            )

    async def _run_profile_extraction(
        self,
        session_id: str,
        user_id: str,
        items: list[ContextItem],
        scenario: Optional[str],
        profile_contents: set[str],
    ) -> None:
        record = self._register_task(session_id, "profile_extract", len(items))
        try:
            facts = await self._profile.extract_facts(
                user_id,
                items,
                time_level="current",
                source="conversation_inference",
                session_id=session_id,
                domain=scenario,
            )
            # Lifecycle classification: explicit dislikes / hard requirements
            # are current-level explicit statements; soft inferences stay as
            # candidate observations until they repeat across sessions.
            for fact in facts:
                if fact.is_dislike or fact.is_hard_requirement:
                    fact.source = "explicit_statement"  # type: ignore[assignment]
                    fact.confirmation_type = "scenario"
                    fact.time_level = "current"  # type: ignore[assignment]
                else:
                    fact.source = "conversation_inference"  # type: ignore[assignment]
                    fact.time_level = "candidate"  # type: ignore[assignment]
            existing = set(profile_contents)
            new_count = 0
            for fact in facts:
                if fact.content in existing:
                    continue
                item = self._profile.persist_fact(user_id, fact)
                await self._context.create_item_direct(session_id, item)
                existing.add(fact.content)
                new_count += 1
            self._profile_facts[user_id] = facts
            self._finish_task(
                record,
                result={"new_facts": new_count, "total_facts": len(facts)},
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_task(record, error=str(exc))

        # Record agent reply event after profile extraction so the event stream
        # can see what was generated in this turn.
        self._event_store.append(
            session_id,
            EventType.AGENT_REPLY,
            payload={"new_facts": new_count},
            turn=self._turns.get(session_id, 1),
        )
        self._maybe_create_snapshot(session_id, self._turns.get(session_id, 1))

    async def _run_spec_derivation(
        self, session_id: str, user_id: str, scenario: Optional[str]
    ) -> None:
        if scenario != "recommendation":
            return
        items = await self._context.list_items(session_id)
        facts = self._profile_facts.get(user_id, [])
        record = self._register_task(session_id, "spec_derivation", len(items))
        try:
            spec_state = await self._derive_spec(user_id, facts)
            self._specs[session_id] = spec_state
            self._finish_task(
                record,
                result={"price_range": spec_state["spec"]["price_range"]},
            )
        except Exception as exc:  # noqa: BLE001
            self._finish_task(record, error=str(exc))

    async def _derive_spec(self, user_id: str, facts: list) -> dict:
        """Derive a recommendation spec + acceptable-ad boundary from facts."""
        from app.core.category_preference import CategoryPreferenceRecord

        # Mirror explicit dislikes into category-level preferences so the
        # spec builder treats them as excluded brands.
        for fact in facts:
            if getattr(fact, "is_dislike", False):
                brand = extract_dislike_brand(fact.content)
                if brand:
                    self._profile.upsert_preference(
                        CategoryPreferenceRecord(
                            user_id=user_id,
                            category_id="phones",
                            preference_type="brand",
                            attribute_key="brand",
                            preference_value=brand,
                            preference_mode="dislike",
                            strength=0.9,
                            confidence=fact.confidence,
                        )
                    )

        # Derive the price range from any budget-related fact.
        budget = None
        for fact in facts:
            budget = extract_budget(fact.content)
            if budget:
                break
        if budget is None:
            budget = 4000
        price_range = (round(budget * 0.75), budget)

        spec = self._profile.build_recommendation_spec(
            user_id, "phones", request_price_range=price_range
        )
        boundary = self._profile.build_acceptable_ads(spec)
        return {
            "spec": spec.to_dict(),
            "boundary": boundary.to_dict(),
        }

    # ----------------------------------------------------------------- state

    async def get_state(self, session_id: str) -> dict:
        """Return the full observation state for the frontend panels."""
        user_id = self.user_for_session(session_id)
        items = await self._context.list_items(session_id)
        facts = self._profile_facts.get(user_id, [])
        return {
            "session_id": session_id,
            "user_id": user_id,
            "injection": self._injections.get(session_id),
            "summary": self._summary_results.get(session_id),
            "tasks": list(reversed(self._tasks.get(session_id, []))),
            "k_turn": self._k_turn_states.get(session_id),
            "recalls": self._recalls.get(session_id, []),
            "profile": {
                "facts": [f.model_dump(mode="json") for f in facts],
                "persons": [p.model_dump(mode="json") for p in self._profile.list_persons(user_id)],
                "events": [e.model_dump(mode="json") for e in self._profile.list_events(user_id)],
                "preferences": [
                    p.model_dump(mode="json")
                    for p in self._profile.list_preferences(user_id)
                ],
                "spec": (self._specs.get(session_id) or {}).get("spec"),
                "boundary": (self._specs.get(session_id) or {}).get("boundary"),
            },
            "profile_lifecycle": self._build_lifecycle_view(user_id),
            "profile_item_count": sum(
                1 for i in items if i.type == ContextType.PROFILE
            ),
            "drift_control": self._build_drift_control_view(session_id),
            "policy": self._build_policy_view(session_id),
        }

    def _build_policy_view(self, session_id: str) -> dict | None:
        """Return policy state for the frontend observation panel."""
        state = self._policy_orchestrator.get_state(session_id)
        if state is None:
            return None
        return state.model_dump(mode="json")

    def _build_drift_control_view(self, session_id: str) -> dict:
        """Return task-health state for the frontend observation panel."""
        events = self._event_store.list_events(session_id)
        snapshots = self._event_store.list_snapshots(session_id)
        checkpoints = [
            e for e in events
            if e.type.value in ("checkpoint_passed", "checkpoint_failed")
        ]
        drifts = [
            e for e in events if e.type.value == "drift_detected"
        ]
        last_checkpoint = checkpoints[-1].payload if checkpoints else None
        last_drift = drifts[-1].payload if drifts else None
        return {
            "events_count": self._event_store.count(session_id),
            "snapshots_count": len(snapshots),
            "checkpoints_count": len(checkpoints),
            "drifts_count": len(drifts),
            "last_checkpoint": last_checkpoint,
            "last_drift_report": last_drift,
            "latest_snapshot_id": snapshots[-1].snapshot_id if snapshots else None,
            "rebuilt_state": self._event_store.rebuild_state(session_id),
        }

    def _build_lifecycle_view(self, user_id: str) -> dict:
        """Group profile facts by time level for the observation panel."""
        facts = self._profile.list_facts(user_id)
        return {
            "current": [
                f.model_dump(mode="json") for f in facts if f.time_level == "current"
            ],
            "candidate": [
                f.model_dump(mode="json") for f in facts if f.time_level == "candidate"
            ],
            "long_term": [
                f.model_dump(mode="json") for f in facts if f.time_level == "long-term"
            ],
            "events": self._profile.list_lifecycle_events(user_id)[:20],
        }

    async def wait_for_background_tasks(self, timeout: float = 10.0) -> None:
        """Await pending background tasks (used by tests and graceful checks)."""
        tasks = [t for t in self._background_tasks if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
