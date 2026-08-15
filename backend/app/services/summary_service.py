"""Application service for the summary & detail-recall subsystem.

Orchestrates multi-type summary extraction (conversation / chapter /
key facts / model-readable), the async summary scheduler, the K-turn
raw window, keyword detail recall, the iterative recall loop, and
conflict resolution on top of the shared context repository.
"""

from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from app.core.async_summary import (
    AsyncSummaryScheduler,
    SummaryTrigger,
    make_chapter_summary_func,
)
from app.core.conflict_resolution import (
    ConflictResolutionStrategy,
    ConflictResolver,
)
from app.core.detail_recall import InMemoryDetailRetriever, KTurnRawWindow
from app.core.iterative_recall import IterativeRecallLoop
from app.core.key_facts import (
    KeyFact,
    KeyFactExtractor,
    MockKeyFactExtractor,
)
from app.core.model_readable import (
    MockModelReadableCompressor,
    ModelReadableCompressor,
)
from app.core.summarizer import Summarizer
from app.core.tokenizer import estimate_tokens
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
)

VALID_SUMMARY_TYPES = ("conversation", "chapter", "key_facts", "model_readable")

# Number of items per chapter block for chapter summaries.
_CHAPTER_SIZE = 6

# Max characters of a context item used to build a detail catalog entry.
_CATALOG_ENTRY_CHARS = 60


class SummaryService:
    """Facade for summary generation and detail-recall APIs."""

    def __init__(
        self,
        summarizer: Summarizer,
        key_fact_extractor: Optional[KeyFactExtractor | MockKeyFactExtractor] = None,
        model_readable_compressor: Optional[
            ModelReadableCompressor | MockModelReadableCompressor
        ] = None,
        mock_mode: bool = False,
    ):
        self._summarizer = summarizer
        self._key_facts = key_fact_extractor or MockKeyFactExtractor()
        self._model_readable = model_readable_compressor or MockModelReadableCompressor()
        self._mock_mode = mock_mode

        self._retriever = InMemoryDetailRetriever()
        self._scheduler = AsyncSummaryScheduler(
            make_chapter_summary_func(summarizer)
        )
        self._k_turn_windows: dict[str, KTurnRawWindow] = {}
        self._indexed_sessions: dict[str, str] = {}  # item_id -> session_id

    # ------------------------------------------------------ multi-type summary

    async def extract_summaries(
        self,
        session_id: str,
        items: list[ContextItem],
        summary_types: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """Generate the requested summary types for a list of items."""
        types = [t for t in (summary_types or list(VALID_SUMMARY_TYPES)) if t in VALID_SUMMARY_TYPES]
        if not types:
            raise ValueError("No valid summary types requested")
        if not items:
            raise ValueError("Cannot summarize an empty item list")

        total_tokens = sum(item.token_cost or 0 for item in items)
        result: dict[str, Any] = {
            "session_id": session_id,
            "types": types,
            "source_item_count": len(items),
            "source_tokens": total_tokens,
            "summaries": {},
        }

        for summary_type in types:
            if summary_type == "conversation":
                item = await self._summarizer.summarize(items)
                result["summaries"][summary_type] = {
                    "content": item.content_as_string(),
                    "token_cost": item.token_cost,
                }
            elif summary_type == "chapter":
                chapters = await self._build_chapters(items)
                result["summaries"][summary_type] = {
                    "chapters": chapters,
                    "token_cost": sum(c["token_cost"] for c in chapters),
                }
            elif summary_type == "key_facts":
                facts = await self._key_facts.extract(items)
                grouped: dict[str, list[dict]] = {}
                for fact in facts:
                    grouped.setdefault(fact.category.value, []).append(
                        {
                            "content": fact.content,
                            "confidence": fact.confidence,
                            "source_item_id": fact.source_item_id,
                        }
                    )
                result["summaries"][summary_type] = {
                    "categories": grouped,
                    "fact_count": len(facts),
                    "token_cost": sum(estimate_tokens(f.content) for f in facts),
                }
            elif summary_type == "model_readable":
                joined = "\n".join(item.content_as_string() for item in items)
                compressed, token_cost = await self._model_readable.compress_with_cost(joined)
                result["summaries"][summary_type] = {
                    "content": compressed,
                    "token_cost": token_cost,
                }

        compressed_total = sum(
            s.get("token_cost", 0) for s in result["summaries"].values()
        )
        result["compressed_tokens"] = compressed_total
        result["compression_ratio"] = (
            round(1 - compressed_total / total_tokens, 4) if total_tokens else 0.0
        )
        return result

    async def _build_chapters(self, items: list[ContextItem]) -> list[dict]:
        """Split items into fixed-size blocks and summarize each block."""
        chapters: list[dict] = []
        for start in range(0, len(items), _CHAPTER_SIZE):
            block = items[start : start + _CHAPTER_SIZE]
            summary = await self._summarizer.summarize(block)
            chapters.append(
                {
                    "index": len(chapters) + 1,
                    "item_range": [start + 1, start + len(block)],
                    "content": summary.content_as_string(),
                    "token_cost": summary.token_cost,
                }
            )
        return chapters

    # --------------------------------------------------------- async scheduler

    def schedule_async_summary(
        self,
        session_id: str,
        items: list[ContextItem],
        trigger: str = SummaryTrigger.END_OF_TURN.value,
    ) -> dict:
        """Schedule an async summary task and return its descriptor."""
        task = self._scheduler.schedule(
            session_id, items, trigger=SummaryTrigger(trigger)
        )
        return self._task_to_dict(task)

    def list_async_tasks(self, session_id: str) -> list[dict]:
        """Return all async summary task descriptors for a session."""
        task_ids = self._scheduler._session_tasks.get(session_id, [])
        return [self._task_to_dict(self._scheduler._tasks[tid]) for tid in task_ids if tid in self._scheduler._tasks]

    async def wait_for_task(
        self, session_id: str, task_id: str, timeout: float = 30.0
    ) -> Optional[dict]:
        """Wait for a task to finish and return its updated descriptor."""
        task = self._scheduler._tasks.get(task_id)
        if task is None or task.session_id != session_id:
            return None
        await self._scheduler.wait_for_task(task_id, timeout=timeout)
        return self._task_to_dict(self._scheduler._tasks[task_id])

    @staticmethod
    def _task_to_dict(task) -> dict:
        return {
            "task_id": task.id,
            "session_id": task.session_id,
            "trigger": task.trigger.value,
            "state": task.state.value,
            "item_count": len(task.items),
            "error": task.error,
            "result": (
                {
                    "content": task.result.content_as_string(),
                    "token_cost": task.result.token_cost,
                }
                if task.result is not None
                else None
            ),
            "created_at": task.created_at.isoformat(),
            "completed_at": (
                task.completed_at.isoformat() if task.completed_at else None
            ),
        }

    # -------------------------------------------------------- K-turn raw window

    def sync_k_turn_window(
        self, session_id: str, items: list[ContextItem], k: int = 10
    ) -> dict:
        """Group session items into turns and replay them into a K-turn window."""
        turns = self._split_turns(items)
        window = KTurnRawWindow(k=k)
        for turn_items in turns:
            window.add_turn(turn_items)
        self._k_turn_windows[session_id] = window

        kept_turns = turns[-k:] if k < len(turns) else turns
        kept_item_count = sum(len(t) for t in kept_turns)
        evicted_item_count = len(items) - kept_item_count
        raw_items = window.get_raw_items()
        return {
            "session_id": session_id,
            "k": k,
            "total_turns": len(turns),
            "kept_turns": len(kept_turns),
            "evicted_turns": max(0, len(turns) - len(kept_turns)),
            "kept_item_count": kept_item_count,
            "evicted_item_count": evicted_item_count,
            "raw_token_count": sum(item.token_cost or 0 for item in raw_items),
            "raw_item_ids": [item.id for item in raw_items],
            "raw_preview": [
                {
                    "type": item.type.value,
                    "content": item.content_as_string()[:80],
                }
                for item in raw_items[-6:]
            ],
        }

    @staticmethod
    def _split_turns(items: list[ContextItem]) -> list[list[ContextItem]]:
        """Split items into turns; a new turn starts at each USER_INPUT item."""
        from app.models import ContextType

        turns: list[list[ContextItem]] = []
        current: list[ContextItem] = []
        for item in items:
            if item.type == ContextType.USER_INPUT and current:
                turns.append(current)
                current = []
            current.append(item)
        if current:
            turns.append(current)
        return turns

    # ------------------------------------------------------------ detail recall

    async def recall_details(
        self,
        query: str,
        items: list[ContextItem],
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Index the given items, then recall details matching the query."""
        for item in items:
            if session_id is not None:
                self._indexed_sessions[item.id] = session_id
            await self._retriever.index(item)

        results = await self._retriever.recall(query, top_k=top_k)
        payload: list[dict] = []
        for item in results:
            if session_id is not None and self._indexed_sessions.get(item.id) != session_id:
                continue
            content = item.content_as_string()
            payload.append(
                {
                    "id": item.id,
                    "type": item.type.value,
                    "content": content,
                    "token_cost": item.token_cost,
                    "score": content.lower().count(query.lower()),
                }
            )
        return payload

    def build_catalog(self, items: list[ContextItem]) -> list[str]:
        """Build a detail catalog (short previews) from context items."""
        return [
            item.content_as_string()[:_CATALOG_ENTRY_CHARS] for item in items
        ]

    async def recall_by_keywords(
        self,
        session_id: str,
        items: list[ContextItem],
        keywords: list[str],
        top_k: int = 3,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[dict]:
        """Recall details matching any keyword, merged and deduplicated.

        Used by the conversation orchestrator: at the start of each turn the
        user message keywords are matched against evicted (out-of-K-window)
        details, synchronously, so recalled details enter the context window.
        """
        exclude = exclude_ids or set()
        for item in items:
            self._indexed_sessions[item.id] = session_id
            await self._retriever.index(item)

        merged: dict[str, dict] = {}
        for keyword in keywords:
            for item in await self._retriever.recall(keyword, top_k=top_k * 2):
                if self._indexed_sessions.get(item.id) != session_id:
                    continue
                if item.id in exclude:
                    continue
                content = item.content_as_string()
                entry = merged.setdefault(
                    item.id,
                    {
                        "id": item.id,
                        "type": item.type.value,
                        "content": content,
                        "token_cost": item.token_cost,
                        "score": 0,
                        "matched": [],
                    },
                )
                entry["score"] += content.lower().count(keyword.lower())
                if keyword not in entry["matched"]:
                    entry["matched"].append(keyword)

        ranked = sorted(merged.values(), key=lambda e: -e["score"])[:top_k]
        return ranked

    # ---------------------------------------------------------- iterative recall

    async def run_iterative_recall(
        self,
        session_id: str,
        items: list[ContextItem],
        catalog: Optional[list[str]] = None,
        window_budget_tokens: Optional[int] = None,
    ) -> dict:
        """Run the LLM-driven iterative recall loop over the session items."""
        detail_catalog = catalog or self.build_catalog(items)

        if self._mock_mode:
            return await self._run_iterative_recall_mock(
                items, detail_catalog, window_budget_tokens
            )

        loop = IterativeRecallLoop(retriever=self._retriever)
        result = await loop.run(
            initial_items=items,
            detail_catalog=detail_catalog,
            session_id=None,
            window_budget_tokens=window_budget_tokens,
        )
        return {
            "outcome": result.outcome.value,
            "iterations": result.iterations,
            "recalled": [
                {
                    "type": item.type.value,
                    "content": item.content_as_string()[:120],
                    "token_cost": item.token_cost,
                }
                for item in result.recalled_items
            ],
            "final_item_count": len(result.final_items),
        }

    async def _run_iterative_recall_mock(
        self,
        items: list[ContextItem],
        catalog: list[str],
        window_budget_tokens: Optional[int],
    ) -> dict:
        """Mock iterative recall: one recall round then a sufficient verdict."""
        recalled: list[ContextItem] = []
        for entry in catalog[:5]:
            results = await self._retriever.recall(entry, top_k=2)
            recalled.extend(results)

        merged = self._conflict_resolver().resolve(items + recalled)
        return {
            "outcome": "sufficient",
            "iterations": 2,
            "recalled": [
                {
                    "type": item.type.value,
                    "content": item.content_as_string()[:120],
                    "token_cost": item.token_cost,
                }
                for item in recalled
            ],
            "final_item_count": len(merged),
        }

    def _conflict_resolver(self) -> ConflictResolver:
        return ConflictResolver()

    # -------------------------------------------------------- conflict resolution

    def resolve_conflicts(
        self,
        entries: list[dict],
        strategy: str = ConflictResolutionStrategy.LAST_WRITE_WINS.value,
    ) -> dict:
        """Resolve conflicts among caller-provided content entries."""
        resolver = ConflictResolver(
            strategy=ConflictResolutionStrategy(strategy)
        )
        items: list[ContextItem] = []
        for entry in entries:
            content = entry.get("content", "")
            if not content:
                continue
            authority_str = entry.get("authority")
            try:
                authority = (
                    ContextAuthority(authority_str)
                    if authority_str
                    else ContextAuthority.INFERRED
                )
            except ValueError:
                authority = ContextAuthority.INFERRED
            items.append(
                ContextItem(
                    id=str(uuid4()),
                    type=ContextType.FACT,
                    source=ContextSource.USER,
                    scope=ContextScope.CURRENT_SESSION,
                    content=content,
                    authority=authority,
                    token_cost=estimate_tokens(content),
                    created_at=datetime.now(timezone.utc),
                )
            )

        resolved = resolver.resolve(items)
        resolved_contents = {item.content_as_string() for item in resolved}
        dropped = [
            item.content_as_string()
            for item in items
            if item.content_as_string() not in resolved_contents
        ]
        return {
            "strategy": strategy,
            "input_count": len(items),
            "resolved_count": len(resolved),
            "resolved": [item.content_as_string() for item in resolved],
            "dropped": dropped,
        }
