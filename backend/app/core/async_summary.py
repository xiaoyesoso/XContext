"""Async summary scheduler.

Manages background summary generation with multiple trigger strategies:
end-of-turn, start-of-next-turn, and watchdog/periodic. Includes a
wait-for-completion fallback for async summaries not yet ready when needed.
"""

import asyncio
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Awaitable, Callable, Optional
from uuid import uuid4

from app.core.tokenizer import estimate_tokens
from app.models import (
    ContextAuthority,
    ContextItem,
    ContextScope,
    ContextSource,
    ContextType,
)


class SummaryTrigger(str, Enum):
    """When an async summary task is triggered."""

    END_OF_TURN = "end_of_turn"
    START_OF_NEXT_TURN = "start_of_next_turn"
    WATCHDOG = "watchdog"


class SummaryTaskState(str, Enum):
    """Lifecycle states of an async summary task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SummaryTask:
    """A single async summary generation task with completion tracking."""

    def __init__(
        self,
        task_id: str,
        session_id: str,
        trigger: SummaryTrigger,
        items: list[ContextItem],
    ):
        self.id = task_id
        self.session_id = session_id
        self.trigger = trigger
        self.items = items
        self.state: SummaryTaskState = SummaryTaskState.PENDING
        self.result: Optional[ContextItem] = None
        self.error: Optional[str] = None
        self.created_at: datetime = datetime.now(timezone.utc)
        self.completed_at: Optional[datetime] = None
        self._event: asyncio.Event = asyncio.Event()

    def mark_running(self) -> None:
        """Mark the task as running."""
        self.state = SummaryTaskState.RUNNING

    def mark_completed(self, result: ContextItem) -> None:
        """Mark the task as completed and wake waiters."""
        self.result = result
        self.state = SummaryTaskState.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self._event.set()

    def mark_failed(self, error: str) -> None:
        """Mark the task as failed and wake waiters."""
        self.error = error
        self.state = SummaryTaskState.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self._event.set()

    async def wait_for_completion(self, timeout: float = 30.0) -> Optional[ContextItem]:
        """Wait for the task to complete, returning the result or None."""
        try:
            await asyncio.wait_for(self._event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None
        return self.result


SummaryFunc = Callable[[list[ContextItem]], Awaitable[ContextItem]]


class AsyncSummaryScheduler:
    """Schedules and tracks async summary generation tasks.

    Supports three trigger strategies (end-of-turn, start-of-next-turn,
    watchdog). When a subsequent turn requires a summary that is still being
    generated, the pipeline can call wait_for_completion() to block until
    the summary is ready.
    """

    def __init__(self, summary_func: Optional[SummaryFunc] = None):
        self._summary_func = summary_func
        self._tasks: dict[str, SummaryTask] = {}
        self._session_tasks: dict[str, list[str]] = defaultdict(list)

    def set_summary_func(self, func: SummaryFunc) -> None:
        """Set the function used to generate summaries."""
        self._summary_func = func

    def schedule(
        self,
        session_id: str,
        items: list[ContextItem],
        trigger: SummaryTrigger = SummaryTrigger.END_OF_TURN,
    ) -> SummaryTask:
        """Schedule a new async summary task.

        Returns the task handle. The task starts running immediately if a
        summary function is configured.
        """
        if not items:
            raise ValueError("Cannot schedule summary for empty item list")

        task_id = str(uuid4())
        task = SummaryTask(task_id, session_id, trigger, items)
        self._tasks[task_id] = task
        self._session_tasks[session_id].append(task_id)

        if self._summary_func is not None:
            asyncio.create_task(self._run_task(task))
        return task

    async def _run_task(self, task: SummaryTask) -> None:
        """Execute the summary generation for a task."""
        task.mark_running()
        try:
            if self._summary_func is None:
                raise RuntimeError("No summary function configured")
            result = await self._summary_func(task.items)
            task.mark_completed(result)
        except Exception as exc:
            task.mark_failed(str(exc))

    async def wait_for_task(self, task_id: str, timeout: float = 30.0) -> Optional[ContextItem]:
        """Wait for a specific task to complete."""
        task = self._tasks.get(task_id)
        if task is None:
            return None
        if task.state in (SummaryTaskState.COMPLETED, SummaryTaskState.FAILED):
            return task.result
        return await task.wait_for_completion(timeout)

    def get_pending_tasks(self, session_id: str) -> list[SummaryTask]:
        """Return all pending/running tasks for a session."""
        task_ids = self._session_tasks.get(session_id, [])
        return [
            self._tasks[tid]
            for tid in task_ids
            if tid in self._tasks
            and self._tasks[tid].state in (SummaryTaskState.PENDING, SummaryTaskState.RUNNING)
        ]

    async def wait_for_session(self, session_id: str, timeout: float = 30.0) -> list[ContextItem]:
        """Wait for all pending tasks of a session and return their results."""
        pending = self.get_pending_tasks(session_id)
        results: list[ContextItem] = []
        for task in pending:
            result = await task.wait_for_completion(timeout)
            if result is not None:
                results.append(result)
        return results

    def clear_session(self, session_id: str) -> None:
        """Remove all task records for a session."""
        task_ids = self._session_tasks.pop(session_id, [])
        for tid in task_ids:
            self._tasks.pop(tid, None)


def make_chapter_summary_func(
    summarizer,
) -> SummaryFunc:
    """Wrap a Summarizer instance into a SummaryFunc for the scheduler."""

    async def func(items: list[ContextItem]) -> ContextItem:
        return await summarizer.summarize(items)

    return func
