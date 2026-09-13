"""Tier 1 — working memory.

Scope: the current task. Lifetime: until the task ends or the process exits.

The whole point of this tier is that it is **bounded and evicting**. Tool
output is verbose and mostly stale within two steps — a device listing, a log
excerpt, a diagnostic dump. An agent that accumulates all of it drifts into a
context window full of its own exhaust, and the symptom is subtle: it does not
crash, it just gets worse at the task while looking busy.

Eviction is oldest-first among unpinned entries. Pinned entries — the task
statement and its constraints — survive eviction, because losing *what you are
doing* while keeping the output of step three is the exact inversion of what
you want.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import WORKING_MEMORY_TOKENS
from ..tokens import count_tokens


@dataclass
class Entry:
    kind: str            # task | observation | tool_output | note
    content: str
    pinned: bool = False
    ts: float = field(default_factory=time.time)

    @property
    def tokens(self) -> int:
        return count_tokens(self.content)


class WorkingMemory:
    def __init__(self, budget: int = WORKING_MEMORY_TOKENS):
        self.budget = budget
        self.entries: list[Entry] = []
        self.evicted: list[str] = []

    # -- writes -----------------------------------------------------------
    def set_task(self, statement: str) -> None:
        """Replace the task descriptor. Pinned, so it cannot be evicted."""
        self.entries = [e for e in self.entries if e.kind != "task"]
        self.entries.insert(0, Entry("task", statement, pinned=True))
        self._evict()

    def add_tool_output(self, tool: str, output: str, max_tokens: int = 200) -> None:
        """Record a tool result, truncated on the way in.

        Truncating at write time rather than at read time is deliberate: a
        10,000-token dump that sits in memory until the next eviction pass has
        already displaced everything else."""
        from ..tokens import truncate_to_tokens
        body = truncate_to_tokens(output.strip(), max_tokens)
        self.entries.append(Entry("tool_output", f"{tool} → {body}"))
        self._evict()

    def note(self, text: str, pinned: bool = False) -> None:
        self.entries.append(Entry("note", text, pinned=pinned))
        self._evict()

    def observe(self, text: str) -> None:
        self.entries.append(Entry("observation", text))
        self._evict()

    # -- reads ------------------------------------------------------------
    @property
    def tokens(self) -> int:
        return sum(e.tokens for e in self.entries)

    def task(self) -> str | None:
        for e in self.entries:
            if e.kind == "task":
                return e.content
        return None

    def render(self) -> str:
        if not self.entries:
            return ""
        lines = []
        for e in self.entries:
            prefix = {"task": "TASK", "tool_output": "TOOL",
                      "observation": "OBS", "note": "NOTE"}.get(e.kind, "•")
            lines.append(f"{prefix}: {e.content}")
        return "\n".join(lines)

    def clear(self, keep_task: bool = True) -> None:
        task = self.task() if keep_task else None
        self.entries = []
        self.evicted = []
        if task:
            self.set_task(task)

    # -- eviction ---------------------------------------------------------
    def _evict(self) -> None:
        while self.tokens > self.budget:
            victim = next((i for i, e in enumerate(self.entries) if not e.pinned),
                          None)
            if victim is None:
                # Everything is pinned and still over budget. Refusing to evict
                # a pin is the right call; the caller pinned too much.
                return
            self.evicted.append(self.entries.pop(victim).content[:80])

    def stats(self) -> dict:
        return {"entries": len(self.entries), "tokens": self.tokens,
                "budget": self.budget, "evicted": len(self.evicted)}
