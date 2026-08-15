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
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
)
from app.services.profile_service import UserProfileService
from app.services.summary_service import SummaryService

# Keywords that carry no recall signal.
_STOPWORDS = {
    "你好", "您好", "谢谢", "请问", "一下", "可以", "这个", "那个",
    "我想", "我要", "帮我", "直接", "告诉", "什么", "怎么", "顺便",
    "看看", "哪些", "哪个", "还是", "就是", "然后", "一份", "目前",
    "现在", "最近", "非常", "特别", "真的", "应该", "可能", "主要",
    "超过", "以内", "左右", "大概", "大约",
    "the", "and", "for", "you", "please", "help", "can", "with",
}

# Grammatical (particle) characters used to split Chinese runs into chunks.
_PARTICLE_CHARS = "的是了吗呢吧啊嘛呀我你他她它们个台件部很也挺都还就才再最跟和与或要会能不"

_CN_RUN_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{1,15}|[0-9$¥][0-9.,]{2,10}")
_DISLIKE_RE = re.compile(r"不(?:要|喜欢|考虑|接受)[:：\s]*([A-Za-z0-9\u4e00-\u9fff ]{1,20})")
_BUDGET_RE = re.compile(r"(?:预算|价位|以内|不超过|budget)[^0-9]{0,6}([0-9]{3,6})|([0-9]{3,6})[^0-9]{0,4}(?:以内|以内|预算|块|元)")


def _clean_token(token: str) -> str:
    """Strip stopword prefixes/suffixes from a Chinese token."""
    changed = True
    while changed and token:
        changed = False
        for stop in _STOPWORDS:
            if len(token) > len(stop) and token.startswith(stop):
                token = token[len(stop):]
                changed = True
            if len(token) > len(stop) and token.endswith(stop):
                token = token[: -len(stop)]
                changed = True
    return token


def extract_keywords(message: str, limit: int = 4) -> list[str]:
    """Extract distinctive keywords from a user message for detail recall."""
    if not message:
        return []
    tokens: list[str] = []
    # Full Chinese runs, split on particles into content chunks.
    for run in _CN_RUN_RE.findall(message):
        for chunk in re.split(f"[{_PARTICLE_CHARS}]", run):
            chunk = chunk.strip()
            if len(chunk) >= 2:
                tokens.append(chunk)
    # Latin words and numbers/prices.
    tokens.extend(_TOKEN_RE.findall(message))

    seen: set[str] = set()
    keywords: list[str] = []
    for token in tokens:
        token = _clean_token(token)
        if not token or token.lower() in _STOPWORDS:
            continue
        if token.isdigit() and len(token) < 3:
            continue
        if token in seen:
            continue
        seen.add(token)
        keywords.append(token)
        if len(keywords) >= limit:
            break
    return keywords


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
    ):
        self._context = context_service
        self._summary = summary_service
        self._profile = profile_service

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

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def user_for_session(session_id: str) -> str:
        """Derive the demo user id owning a chat session."""
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
    ) -> dict:
        """Synchronously inject summaries, profile facts, and recalled details.

        Runs at the start of every turn, right after the user message is
        ingested and before the context window is composed.
        """
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

        self._injections[session_id] = report
        return report

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

        # --- Task 1: multi-type summary extraction -------------------------
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

        # --- Task 2: five-dimension profile extraction ---------------------
        facts: list = []
        record = self._register_task(session_id, "profile_extract", len(items))
        try:
            facts = await self._profile.extract_facts(user_id, items)
            existing = {
                i.content for i in items if i.type == ContextType.PROFILE
            }
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

        # --- Task 3: recommendation spec derivation (scenario-aware) --------
        if scenario == "recommendation":
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
            "profile_item_count": sum(
                1 for i in items if i.type == ContextType.PROFILE
            ),
        }

    async def wait_for_background_tasks(self, timeout: float = 10.0) -> None:
        """Await pending background tasks (used by tests and graceful checks)."""
        tasks = [t for t in self._background_tasks if not t.done()]
        if tasks:
            await asyncio.wait(tasks, timeout=timeout)
