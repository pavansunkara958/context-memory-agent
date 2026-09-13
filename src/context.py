"""Context optimisation: filtering, deduplication, compaction, budgeting.

Retrieval decides what *could* go in the context window. This module decides
what actually does, and it matters more than it looks.

The failure mode it prevents is not running out of room. It is **dilution**.
A model given six relevant chunks and fourteen loosely-related ones answers
worse than one given six, because attention is finite and every irrelevant
token competes with a relevant one. More context is not more information.

Three mechanisms, in order:

  filter      Drop anything below a relevance floor. Cheap, and it removes the
              long tail of "vaguely on topic" that retrieval always returns.
  deduplicate Drop near-copies. Three chunks that all describe the rollback
              procedure occupy three slots and contribute one fact. Worse, in
              a long context repetition reads as emphasis, so a duplicated
              minor point can outweigh a unique critical one.
  compact     When the conversation outgrows its budget, replace old turns with
              a summary that preserves decisions and open threads. Truncation
              loses the beginning of the conversation, which is usually where
              the task was defined.

All three are deterministic and report what they removed, so their effect is
measurable in the ablation rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import (CONTEXT_BUDGET_TOKENS, DEDUP_THRESHOLD, SIM_FLOOR,
                     SPARSE_FLOOR_RATIO)
from .embeddings import cosine, embed
from .retrieval import Retrieved
from .tokens import count_tokens, truncate_to_tokens


@dataclass
class ContextReport:
    """What the optimiser did. Reported, not hidden — a context pipeline you
    cannot inspect is one you cannot debug when the answer is wrong."""
    retrieved: int = 0
    after_filter: int = 0
    after_dedup: int = 0
    packed: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    dropped_low_relevance: list[str] = field(default_factory=list)
    dropped_duplicate: list[tuple[str, str]] = field(default_factory=list)
    dropped_budget: list[str] = field(default_factory=list)

    @property
    def reduction(self) -> float:
        if not self.tokens_in:
            return 0.0
        return round(1 - self.tokens_out / self.tokens_in, 3)

    def summary(self) -> str:
        return (f"{self.retrieved} retrieved → {self.after_filter} relevant → "
                f"{self.after_dedup} unique → {self.packed} packed  "
                f"({self.tokens_in}→{self.tokens_out} tokens, "
                f"{int(self.reduction * 100)}% reduction)")


# ---------------------------------------------------------------------------
def filter_by_relevance(items: list[Retrieved],
                        floor: float = SIM_FLOOR) -> tuple[list[Retrieved], list[str]]:
    """Drop chunks the dense retriever considers off-topic.

    A chunk found by BM25 is kept regardless of its dense score. Exact-token
    matches — an error code, a version string, a CLI flag — are precisely the
    case where the embedding is uninformative but the match is exactly right.

    This condition was originally written as `"sparse" in retrievers and
    dense_score == 0.0`, which looked equivalent and was not. A chunk found by
    BOTH retrievers has a non-zero dense score, so it failed the first branch,
    and if that score sat below the floor it failed the second too — meaning a
    chunk was PENALISED for being found twice. The effect was invisible in unit
    tests and obvious in the ablation: hybrid retrieval scored identically to
    dense alone while BM25 alone more than doubled it. Agreement between two
    retrievers is the strongest signal available; it must never subtract.

    A BM25 hit still has to be a *good* BM25 hit. "Found by the sparse
    retriever at all" is too permissive — BM25 returns anything sharing one
    common token, so an unconditional pass lets its entire tail into the window
    and the filter stops filtering. The bar is relative to the best score for
    this query, because BM25 scores are not comparable between queries.
    """
    best_bm25 = max((r.bm25_score for r in items), default=0.0)
    sparse_floor = best_bm25 * SPARSE_FLOOR_RATIO

    kept, dropped = [], []
    for r in items:
        strong_lexical = r.bm25_score > 0 and r.bm25_score >= sparse_floor
        if strong_lexical or r.dense_score >= floor:
            kept.append(r)
        else:
            dropped.append(r.citation())
    return kept, dropped


def deduplicate(items: list[Retrieved],
                threshold: float = DEDUP_THRESHOLD
                ) -> tuple[list[Retrieved], list[tuple[str, str]]]:
    """Remove near-duplicates, keeping the higher-ranked copy.

    Exact string equality is useless here — chunk overlap and restated
    procedures produce text that differs in wording but not in content. Cosine
    similarity over embeddings catches the semantic case, which is the one that
    actually wastes the window.
    """
    if len(items) < 2:
        return list(items), []

    vectors = embed([r.text for r in items])
    kept: list[int] = []
    dropped: list[tuple[str, str]] = []

    for i in range(len(items)):
        duplicate_of = None
        for j in kept:
            if cosine(vectors[i], vectors[j]) >= threshold:
                duplicate_of = j
                break
        if duplicate_of is None:
            kept.append(i)
        else:
            dropped.append((items[i].citation(), items[duplicate_of].citation()))

    return [items[i] for i in kept], dropped


def pack(items: list[Retrieved],
         budget: int = CONTEXT_BUDGET_TOKENS) -> tuple[list[Retrieved], list[str]]:
    """Fill the budget in rank order, stopping at the first chunk that does not
    fit. Deliberately not a knapsack: keeping strict rank order means the model
    sees the best evidence first, and a greedy 'squeeze in a small low-ranked
    chunk' would reorder relevance to save tokens."""
    packed, dropped, used = [], [], 0
    for r in items:
        cost = count_tokens(r.text) + count_tokens(r.attribution()) + 4
        if used + cost > budget:
            dropped.append(r.citation())
            continue
        packed.append(r)
        used += cost
    return packed, dropped


def optimise(items: list[Retrieved], *,
             floor: float = SIM_FLOOR,
             dedup_threshold: float = DEDUP_THRESHOLD,
             budget: int = CONTEXT_BUDGET_TOKENS,
             do_filter: bool = True,
             do_dedup: bool = True) -> tuple[list[Retrieved], ContextReport]:
    report = ContextReport(retrieved=len(items))
    report.tokens_in = sum(count_tokens(r.text) for r in items)

    stage = list(items)
    if do_filter:
        stage, dropped = filter_by_relevance(stage, floor)
        report.dropped_low_relevance = dropped
    report.after_filter = len(stage)

    if do_dedup:
        stage, dups = deduplicate(stage, dedup_threshold)
        report.dropped_duplicate = dups
    report.after_dedup = len(stage)

    stage, over = pack(stage, budget)
    report.dropped_budget = over
    report.packed = len(stage)
    report.tokens_out = sum(count_tokens(r.text) for r in stage)
    return stage, report


def render(items: list[Retrieved]) -> str:
    """Format chunks for the prompt with citations attached to each block.

    The citation sits with the text, not in a bibliography at the end. A model
    asked to cite from a trailing source list has to remember which fact came
    from which entry; a model reading `[RB-105#1]` immediately above the fact
    does not.
    """
    blocks = []
    for r in items:
        blocks.append(f"{r.attribution()}\n{r.text}")
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Conversation compaction
# ---------------------------------------------------------------------------
@dataclass
class Compaction:
    summary: str
    compacted_turns: int
    tokens_before: int
    tokens_after: int

    @property
    def ratio(self) -> float:
        if not self.tokens_before:
            return 1.0
        return round(self.tokens_after / self.tokens_before, 3)


def compact_history(turns: list[dict], keep_recent: int = 4,
                    summary_budget: int = 320,
                    client=None, model: str | None = None) -> Compaction | None:
    """Replace old turns with a summary, keeping the most recent ones verbatim.

    Recency is kept exact because that is where referring expressions point —
    "the one you mentioned", "that error". Summarising the recent turns breaks
    the pronouns the next turn depends on.

    The extractive path needs no model: it keeps the user's stated goals and
    any line carrying a decision or constraint. Lossier than an LLM summary,
    and free, deterministic and testable.
    """
    if len(turns) <= keep_recent:
        return None

    old, _recent = turns[:-keep_recent], turns[-keep_recent:]
    before = sum(count_tokens(t.get("content", "")) for t in old)

    if client is not None and model:
        transcript = "\n".join(
            f"{t.get('role')}: {t.get('content', '')}" for t in old)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content":
                     "Summarise this support conversation for an engineer "
                     "resuming it later. Preserve: the task, decisions taken, "
                     "facts established, and anything still unresolved. Drop "
                     "pleasantries. Be terse and concrete."},
                    {"role": "user", "content": transcript},
                ],
            )
            summary = (resp.choices[0].message.content or "").strip()
        except Exception:
            summary = _extractive_summary(old, summary_budget)
    else:
        summary = _extractive_summary(old, summary_budget)

    summary = truncate_to_tokens(summary, summary_budget)
    return Compaction(summary=summary, compacted_turns=len(old),
                      tokens_before=before, tokens_after=count_tokens(summary))


_SIGNAL = ("decided", "confirmed", "found", "cause", "root cause", "will",
           "need to", "must", "blocked", "next step", "prefers", "always",
           "never", "error", "exit code", "firmware", "version", "rma")


def _extractive_summary(turns: list[dict], budget: int) -> str:
    lines: list[str] = []
    for t in turns:
        content = (t.get("content") or "").strip()
        if not content:
            continue
        role = t.get("role", "?")
        if role == "user":
            lines.append(f"- user asked: {content[:160]}")
            continue
        for sentence in content.replace("\n", " ").split(". "):
            s = sentence.strip()
            if len(s) > 20 and any(k in s.lower() for k in _SIGNAL):
                lines.append(f"- {s[:160]}")
    text = "Earlier in this conversation:\n" + "\n".join(lines[:14])
    return truncate_to_tokens(text, budget)


def working_set_tokens(blocks: list[str]) -> int:
    return sum(count_tokens(b) for b in blocks)


def mean_relevance(items: list[Retrieved]) -> float:
    """Mean dense similarity of the packed context — the 'context relevance'
    metric the brief asks for. Rises when filtering works, falls when the
    window is padded."""
    scores = [r.dense_score for r in items if r.dense_score > 0]
    return round(float(np.mean(scores)), 4) if scores else 0.0
