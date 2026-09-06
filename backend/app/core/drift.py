"""Drift detection for long-running tasks (Context Verifier).

Implements the six drift types from the long-task design doc:
goal / constraint / state / evidence drift, detail loss, and
error accumulation. Rule-based checks run by default; an optional
LLM-backed deep check can be layered on top later.
"""

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field
from uuid import uuid4

from app.models import ContextAuthority, ContextItem, ContextType


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
    """Extract distinctive keywords from a message for detail recall."""
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


class DriftType(str, Enum):
    """The six drift categories a long task can suffer from."""

    GOAL_DRIFT = "goal_drift"
    CONSTRAINT_DRIFT = "constraint_drift"
    STATE_DRIFT = "state_drift"
    EVIDENCE_DRIFT = "evidence_drift"
    DETAIL_LOSS = "detail_loss"
    ERROR_ACCUMULATION = "error_accumulation"


class DriftFinding(BaseModel):
    """A single detected drift with impact scope and repair advice."""

    drift_type: DriftType
    severity: Literal["low", "medium", "high"] = "medium"
    description: str
    affected_item_ids: list[str] = Field(default_factory=list)
    suggestion: str = ""


class DriftReport(BaseModel):
    """Structured output of a Context Verifier run."""

    report_id: str = Field(default_factory=lambda: str(uuid4()))
    session_id: str
    turn: int = 0
    healthy: bool = True
    findings: list[DriftFinding] = Field(default_factory=list)
    degraded: bool = False  # True when an LLM check fell back to rules
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def refresh_health(self) -> "DriftReport":
        """Recompute the healthy flag from findings."""
        self.healthy = len(self.findings) == 0
        return self


# Keywords signalling an absolute hard constraint in Chinese/English text.
_HARD_CONSTRAINT_MARKERS = ("绝对不能", "绝不能", "必须", "不能", "禁止", "一定要", "must", "never")
# Weakened phrasings that indicate a hard constraint lost its strength.
_WEAKENED_MARKERS = ("尽量", "尽可能", "最好", "考虑", "preferably", "try to")
# Failure signals inside an observation/tool result.
_FAILURE_MARKERS = ("失败", "错误", "报错", "异常", "failed", "failure", "error", "exception")


def _content(item: ContextItem) -> str:
    return item.content_as_string() if hasattr(item, "content_as_string") else str(item.content)


class ContextVerifier:
    """Rule-based drift detector over the current window and event stream."""

    def __init__(self, goal_overlap_threshold: float = 0.15):
        self.goal_overlap_threshold = goal_overlap_threshold

    # ------------------------------------------------------------------ main

    def verify(
        self,
        session_id: str,
        items: list[ContextItem],
        events: Optional[list] = None,
        goal: Optional[str] = None,
        turn: int = 0,
    ) -> DriftReport:
        """Run all rule checks and return a structured drift report."""
        events = events or []
        report = DriftReport(session_id=session_id, turn=turn)
        if not items:
            return report.refresh_health()

        findings: list[DriftFinding] = []
        findings.extend(self._check_goal_drift(items, goal))
        findings.extend(self._check_constraint_drift(items))
        findings.extend(self._check_state_drift(items, events))
        findings.extend(self._check_evidence_drift(items, events))
        findings.extend(self._check_detail_loss(items))
        findings.extend(self._check_error_accumulation(items))
        report.findings = findings
        return report.refresh_health()

    # ---------------------------------------------------------------- checks

    def _check_goal_drift(
        self, items: list[ContextItem], goal: Optional[str]
    ) -> list[DriftFinding]:
        """Goal drift: recent agent output no longer overlaps the original goal."""
        goal_text = goal
        if not goal_text:
            user_inputs = [i for i in items if i.type == ContextType.USER_INPUT]
            if user_inputs:
                goal_text = _content(user_inputs[0])
        if not goal_text:
            return []

        agent_outputs = [i for i in items if i.type == ContextType.MODEL_OUTPUT]
        if len(agent_outputs) < 2:
            return []

        goal_tokens = self._significant_tokens(goal_text)
        if not goal_tokens:
            return []
        recent = agent_outputs[-2:]
        for output in recent:
            overlap = goal_tokens & self._significant_tokens(_content(output))
            if len(overlap) / len(goal_tokens) < self.goal_overlap_threshold:
                return [
                    DriftFinding(
                        drift_type=DriftType.GOAL_DRIFT,
                        severity="high",
                        description=(
                            "Recent agent output shares almost no keywords with the "
                            "original goal; the task may be drifting."
                        ),
                        affected_item_ids=[output.id],
                        suggestion="re-inject the original goal and consider a local replan",
                    )
                ]
        return []

    def _check_constraint_drift(self, items: list[ContextItem]) -> list[DriftFinding]:
        """Constraint drift: denied/ hard-rule items missing from the window."""
        findings: list[DriftFinding] = []
        # A denied item still present is fine (it acts as a "do not repeat"
        # reminder); drift means a hard constraint vanished entirely while the
        # task continues. We approximate "vanished" via detail-loss check; here
        # we flag denied content whose counterpart hard rule is absent.
        denied = [i for i in items if i.authority == ContextAuthority.DENIED]
        hard_rules = [i for i in items if i.authority == ContextAuthority.HARD_RULE]
        if denied and not hard_rules:
            findings.append(
                DriftFinding(
                    drift_type=DriftType.CONSTRAINT_DRIFT,
                    severity="medium",
                    description=(
                        f"{len(denied)} denied item(s) present but no active hard-rule "
                        "constraint remains in the window; constraints may be lost."
                    ),
                    affected_item_ids=[i.id for i in denied[:5]],
                    suggestion="recall the original hard constraints and re-inject them",
                )
            )
        return findings

    def _check_state_drift(
        self, items: list[ContextItem], events: list
    ) -> list[DriftFinding]:
        """State drift: failure signals present but nothing recorded as failed."""
        failure_items = [
            i
            for i in items
            if any(marker in _content(i) for marker in _FAILURE_MARKERS)
            and i.type in (ContextType.TOOL_RESULT, ContextType.MODEL_OUTPUT)
        ]
        if not failure_items:
            return []
        failed_events = [
            e for e in events if getattr(e, "type", None) is not None
            and e.type.value == "step_failed"
        ]
        step_events = [
            e for e in events if getattr(e, "type", None) is not None
            and e.type.value in ("step_completed", "step_failed")
        ]
        if failure_items and not failed_events and step_events:
            return [
                DriftFinding(
                    drift_type=DriftType.STATE_DRIFT,
                    severity="high",
                    description=(
                        "Failure wording found in observations but the event stream "
                        "records no failed step; task state may be inaccurate."
                    ),
                    affected_item_ids=[i.id for i in failure_items[:5]],
                    suggestion="re-check the failed step status and correct task state",
                )
            ]
        return []

    def _check_evidence_drift(
        self, items: list[ContextItem], events: list
    ) -> list[DriftFinding]:
        """Evidence drift: confirmed facts without a supporting observation event."""
        confirmed = [
            i
            for i in items
            if i.authority == ContextAuthority.CONFIRMED
            and i.type in (ContextType.FACT, ContextType.TOOL_RESULT)
        ]
        if not confirmed:
            return []
        evidence_events = [
            e for e in events if getattr(e, "type", None) is not None
            and e.type.value in ("observation", "fact_confirmed")
        ]
        if confirmed and events and not evidence_events:
            return [
                DriftFinding(
                    drift_type=DriftType.EVIDENCE_DRIFT,
                    severity="medium",
                    description=(
                        f"{len(confirmed)} confirmed fact(s) lack any supporting "
                        "observation/fact_confirmed event; evidence may be stale."
                    ),
                    affected_item_ids=[i.id for i in confirmed[:5]],
                    suggestion="recall the original evidence or mark the facts as assumed",
                )
            ]
        return []

    def _check_detail_loss(self, items: list[ContextItem]) -> list[DriftFinding]:
        """Detail loss: summaries weakened an absolute hard-constraint phrasing."""
        findings: list[DriftFinding] = []
        summaries = [i for i in items if i.type == ContextType.SUMMARY]
        hard_inputs = [
            i
            for i in items
            if i.type == ContextType.USER_INPUT
            and any(m in _content(i) for m in _HARD_CONSTRAINT_MARKERS)
        ]
        if not summaries or not hard_inputs:
            return findings
        summary_text = "\n".join(_content(s) for s in summaries)
        for source in hard_inputs:
            source_text = _content(source)
            hard_marker = next(
                (m for m in _HARD_CONSTRAINT_MARKERS if m in source_text), None
            )
            if not hard_marker:
                continue
            weakened = any(m in summary_text for m in _WEAKENED_MARKERS)
            marker_kept = hard_marker in summary_text
            if weakened and not marker_kept:
                findings.append(
                    DriftFinding(
                        drift_type=DriftType.DETAIL_LOSS,
                        severity="high",
                        description=(
                            f"Hard constraint marker '{hard_marker}' was weakened in "
                            "the summary; the original requirement may be violated."
                        ),
                        affected_item_ids=[source.id] + [s.id for s in summaries[:3]],
                        suggestion="recall the raw user input and restore the hard constraint",
                    )
                )
        return findings

    def _check_error_accumulation(
        self, items: list[ContextItem]
    ) -> list[DriftFinding]:
        """Error accumulation: failure signals spread across multiple items."""
        failure_items = [
            i
            for i in items
            if any(marker in _content(i) for marker in _FAILURE_MARKERS)
        ]
        if len(failure_items) >= 3:
            ids = [i.id for i in failure_items]
            return [
                DriftFinding(
                    drift_type=DriftType.ERROR_ACCUMULATION,
                    severity="medium",
                    description=(
                        f"Failure signals appear in {len(failure_items)} items; "
                        "an early error may be compounding downstream."
                    ),
                    affected_item_ids=ids[:6],
                    suggestion="trace the first occurrence, restore a trusted snapshot if needed",
                )
            ]
        return []

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _significant_tokens(text: str, min_len: int = 2) -> set[str]:
        """Cheap token set for overlap checks (CJK runs + latin words)."""
        import re

        tokens = set(re.findall(r"[A-Za-z][A-Za-z0-9_-]{1,20}", text))
        for run in re.findall(r"[\u4e00-\u9fff]{2,}", text):
            # Sliding 2-grams keep it robust without a segmenter.
            tokens.update(run[i : i + 2] for i in range(len(run) - 1))
        return {t for t in tokens if len(t) >= min_len}
