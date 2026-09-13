"""The agent: four memory tiers plus a hybrid retrieval strategy.

HYBRID = UPFRONT + JUST-IN-TIME
-------------------------------
Two loading strategies, used for different things, because they fail
differently.

**Upfront**, once when the session opens: user preferences, task progress,
the top episodic facts, and the last compaction summary. Small — a few hundred
tokens — and it must be present before the first word, because it is what makes
the agent recognise the person. You cannot retrieve "the user prefers Celsius"
in response to a question that never mentions temperature, and a system that
tries will feel amnesiac in exactly the situations where continuity matters.

**Just-in-time**, per turn and only when needed: knowledge-base chunks for
*this* question. The corpus is too large to preload and, more importantly,
mostly irrelevant to any given turn. Loading it upfront would dilute the window
with documents about firmware when the question is about antennas.

The split is: **upfront for what is small and always relevant; just-in-time for
what is large and conditionally relevant.** Getting it backwards gives you
either an amnesiac agent or a diluted one.

The JIT half is gated. A question like "what did we decide yesterday?" is
answered from memory and retrieving for it wastes a search and pollutes the
window with documents nobody asked about.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .config import (CONTEXT_BUDGET_TOKENS, RERANK_K, SESSION_COMPACT_AFTER,
                     generation_client)
from .context import ContextReport, compact_history, mean_relevance, optimise
from .generation import Answer, generate
from .memory import LongTermMemory, SessionMemory, WorkingMemory
from .retrieval import KnowledgeIndex, Retrieved
from .tokens import count_tokens, truncate_to_tokens


@dataclass
class TurnResult:
    query: str
    answer: Answer
    chunks: list[Retrieved] = field(default_factory=list)
    facts_recalled: list[str] = field(default_factory=list)
    facts_promoted: list[str] = field(default_factory=list)
    retrieved: bool = True
    retrieval_reason: str = ""
    report: ContextReport | None = None
    compacted: bool = False
    context_tokens: int = 0
    latency_ms: float = 0.0

    def trace(self) -> str:
        bits = [f"retrieval={'jit' if self.retrieved else 'skipped'}"
                f" ({self.retrieval_reason})"]
        if self.report:
            bits.append(self.report.summary())
        if self.facts_recalled:
            bits.append(f"memory: {len(self.facts_recalled)} recalled")
        if self.facts_promoted:
            bits.append(f"promoted: {len(self.facts_promoted)}")
        if self.compacted:
            bits.append("session compacted")
        bits.append(f"{self.context_tokens} ctx tokens · {self.latency_ms:.0f}ms")
        return "\n    ".join(bits)


# Questions that are about the conversation or the user, not the product.
_MEMORY_ONLY = re.compile(
    r"\b(what did (we|i|you)|did (we|i) (say|decide|agree)|remind me|"
    r"my preference|remember|earlier you|last time|previously|"
    r"where (did|were) we|recap|so far)\b", re.I)

_TRIVIAL = re.compile(
    r"^\s*(thanks?|thank you|ok(ay)?|got it|great|cheers|yep|no|yes)[!. ]*$", re.I)

# Preferences are stored under a stable key so that restating one in different
# words UPDATES it rather than accumulating a second, contradictory entry. A
# free-text key would let "keep answers short" and "i prefer short answers"
# coexist, and the upfront pack would then carry both.
_PREF_TOPICS = [
    ("units", ("celsius", "fahrenheit", "metric", "imperial")),
    ("answer_length", ("short", "terse", "brief", "detailed", "concise")),
    ("tone", ("formal", "informal", "plain")),
]


def _preference_key(text: str) -> str:
    lowered = text.lower()
    for key, markers in _PREF_TOPICS:
        if any(m in lowered for m in markers):
            return key
    slug = re.sub(r"[^a-z0-9]+", "_", lowered).strip("_")
    return slug[:32] or "preference"


class Agent:
    def __init__(self, session_id: str, user_id: str = "default",
                 index: KnowledgeIndex | None = None, verbose: bool = False):
        self.session = SessionMemory(session_id, user_id)
        self.longterm = LongTermMemory(user_id)
        self.working = WorkingMemory()
        self.index = index or KnowledgeIndex()
        if self.index.size == 0:
            self.index.load()
        self.client, self.model = generation_client()
        self.verbose = verbose
        self.upfront_pack: str = ""
        self.upfront_tokens: int = 0
        self.carried_progress: dict = {}

    # -- upfront half of the hybrid strategy -------------------------------
    def open_session(self, task: str | None = None) -> str:
        """Assemble the upfront context pack. Called once, at session start."""
        prefs = self.session.preferences()
        progress = self.session.progress()
        summaries = self.session.summaries()

        # A task outlives the conversation that started it. If this session has
        # no task of its own, carry the user's most recent one forward.
        carried = False
        if not progress.get("task"):
            prior = self.session.last_progress()
            if prior:
                progress, carried = prior, True
        self.carried_progress = progress

        lines: list[str] = []
        if prefs:
            lines.append("User preferences: " +
                         "; ".join(f"{k}={v}" for k, v in sorted(prefs.items())))
        if progress.get("task"):
            label = "Open task (carried over)" if carried else "Open task"
            lines.append(f"{label}: {progress['task']} [{progress['status']}]")
        if progress.get("findings"):
            lines.append("Established so far: " +
                         "; ".join(progress["findings"][-5:]))
        if progress.get("next_steps"):
            lines.append("Next steps: " + "; ".join(progress["next_steps"][:3]))
        if summaries:
            lines.append(f"Earlier: {summaries[-1]}")

        # Seed episodic facts. Queried against the task when there is one,
        # otherwise against the user's general profile.
        probe = task or progress.get("task") or "user preferences and site details"
        for fact in self.longterm.recall(probe, k=5, min_confidence=0.6):
            lines.append(fact.attribution())

        if task:
            self.working.set_task(task)
            self.session.set_progress(task=task)

        self.upfront_pack = "\n".join(lines)
        self.upfront_tokens = count_tokens(self.upfront_pack)
        return self.upfront_pack

    # -- JIT gate ----------------------------------------------------------
    def should_retrieve(self, query: str,
                        promoted: int = 0) -> tuple[bool, str]:
        """Decide whether this turn needs the knowledge base.

        Four reasons to skip. The last one is the interesting one: when a turn
        is the user *telling* the agent about themselves — "we run 4.3, our
        site is NW-41, keep answers short" — there is nothing to look up. The
        earlier version searched anyway and answered a question nobody asked,
        with confident citations about firmware rollback. Retrieving on a
        statement does not merely waste a search; it manufactures an
        irrelevant answer and puts citations on it.
        """
        if _TRIVIAL.match(query):
            return False, "no informational content"
        if _MEMORY_ONLY.search(query):
            return False, "answerable from memory"
        if len(query.split()) <= 2:
            return False, "too short to retrieve on"
        if promoted >= 2 and "?" not in query:
            return False, "profile statement — recorded, nothing asked"
        return True, "knowledge question"

    # -- one turn ----------------------------------------------------------
    def ask(self, query: str, k: int = RERANK_K,
            budget: int = CONTEXT_BUDGET_TOKENS) -> TurnResult:
        started = time.perf_counter()

        self.session.append("user", query)
        promoted = self.longterm.consider(query, "user", self.session.session_id)

        # Preferences also land in session memory so they survive as structured
        # keys, not only as free-text facts.
        for fact in promoted:
            if fact.kind == "preference":
                self.session.set_preference(_preference_key(fact.text), fact.text)

        do_retrieve, reason = self.should_retrieve(query, len(promoted))

        chunks: list[Retrieved] = []
        report: ContextReport | None = None
        if do_retrieve:
            candidates = self.index.search(query, k=k * 3)
            chunks, report = optimise(candidates, budget=budget)
            self.working.observe(f"retrieved {len(chunks)} chunks for: {query[:60]}")

        facts = self.longterm.recall(query, k=3, min_confidence=0.5)
        memory_lines = [f.attribution() for f in facts]

        history = [{"role": t.role, "content": t.content}
                   for t in self.session.history(limit=6)][:-1]

        preamble = []
        if self.upfront_pack:
            preamble.append(self.upfront_pack)
        if self.working.render():
            preamble.append("Working memory:\n" + self.working.render())

        if reason == "answerable from memory" and self.client is None:
            # "What did we decide yesterday?" must be answered from session
            # progress and long-term facts. Falling through to the extractive
            # path here returned "nothing in the knowledge base" — technically
            # true and completely wrong, because the knowledge base was never
            # the place to look. Skipping retrieval obliges you to supply the
            # alternative source, not to answer as though none exists.
            answer = self._memory_answer(facts)
        else:
            answer = generate(query, chunks, preamble + memory_lines, history,
                              self.client, self.model,
                              noted=[f.text for f in promoted])

        self.session.append("assistant", answer.text, answer.citations)

        # Record a finding so the next session's upfront pack is richer.
        # Trimmed on a word boundary — a finding cut mid-word ("What shoul")
        # is noise in tomorrow's context window.
        if answer.citations:
            topic = truncate_to_tokens(query.rstrip("?. "), 12)
            self.session.add_finding(
                f"{topic} → {', '.join(answer.citations[:3])}")

        compacted = self._maybe_compact()

        context_tokens = (self.upfront_tokens
                          + sum(count_tokens(c.text) for c in chunks)
                          + count_tokens("\n".join(memory_lines))
                          + self.working.tokens)

        return TurnResult(
            query=query, answer=answer, chunks=chunks,
            facts_recalled=memory_lines,
            facts_promoted=[f.text for f in promoted],
            retrieved=do_retrieve, retrieval_reason=reason, report=report,
            compacted=compacted, context_tokens=context_tokens,
            latency_ms=(time.perf_counter() - started) * 1000,
        )

    def _memory_answer(self, facts) -> Answer:
        """Answer a 'where were we?' question from the memory tiers alone."""
        progress = self.session.progress()
        if not progress.get("task"):
            progress = getattr(self, "carried_progress", None) or progress
        parts: list[str] = []
        if progress.get("task"):
            parts.append(f"Open task: {progress['task']} ({progress['status']}).")
        for finding in progress.get("findings", [])[-4:]:
            parts.append(f"- {finding}")
        for step in progress.get("next_steps", [])[:3]:
            parts.append(f"- next: {step}")
        summaries = self.session.summaries()
        if summaries:
            parts.append(summaries[-1])
        if facts:
            parts.append("I also have on file: " +
                         "; ".join(f.text for f in facts[:3]) + ".")
        if not parts:
            parts.append("Nothing recorded yet for this task — this is the "
                         "first thing you've told me.")
        return Answer("\n".join(parts), [], "memory")

    # -- compaction --------------------------------------------------------
    def _maybe_compact(self) -> bool:
        if self.session.turn_count() < SESSION_COMPACT_AFTER * 2:
            return False
        turns = [{"role": t.role, "content": t.content}
                 for t in self.session.history()]
        result = compact_history(turns, keep_recent=4,
                                 client=self.client, model=self.model)
        if result is None:
            return False
        self.session.add_summary(result.summary, result.compacted_turns)
        return True

    # -- session close -----------------------------------------------------
    def close_session(self, status: str = "open",
                      next_steps: list[str] | None = None) -> dict:
        """Promote durable facts and record where the task stands.

        This is the hinge of multi-session continuity. Everything the next
        session loads upfront is written here.
        """
        promoted = self.longterm.promote_session(self.session)
        self.session.set_progress(status=status, next_steps=next_steps or [])
        return {
            "promoted": len(promoted),
            "facts": self.longterm.stats(),
            "session": self.session.stats(),
        }

    # -- introspection -----------------------------------------------------
    def memory_report(self) -> dict:
        return {
            "working": self.working.stats(),
            "session": self.session.stats(),
            "long_term": self.longterm.stats(),
            "knowledge_chunks": self.index.size,
            "upfront_tokens": self.upfront_tokens,
        }


def context_relevance(result: TurnResult) -> float:
    return mean_relevance(result.chunks)
