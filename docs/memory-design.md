# Memory Design

## What makes a tier a tier

Four stores with different names is not a multi-tier memory system. What
separates a tier is **what causes a write and what causes a read**, and each
boundary here exists because the alternative fails in a specific way.

| | Working | Session | Long-term |
|---|---|---|---|
| Scope | one task | one conversation | one user, indefinitely |
| Storage | in process | SQLite | vector store |
| Written by | the current step | each turn | **promotion** |
| Read by | the next step | the next turn here | semantic recall |
| Addressed | positionally | by session id | by meaning |
| Bound | hard token budget | compaction | promotion rules |

---

## Tier 1 — Working memory

`src/memory/working.py`

Holds the task statement, recent tool output and in-flight observations for the
current task. Lives and dies with it.

**Bounded and evicting, because tool output is the problem.** A device listing
or a log excerpt is verbose and stale within two steps. An agent that keeps all
of it ends up with a context window full of its own exhaust, and the failure is
insidious: it does not crash, it just gets gradually worse at the task while
appearing busy.

Two rules:

- **Truncate on write, not on read.** A 10,000-token dump that sits in memory
  until the next eviction pass has already displaced everything else. Tool
  output is capped at 200 tokens as it goes in.
- **Pins survive eviction.** The task statement is pinned. Evicting *what you
  are doing* while keeping the output of step three is exactly backwards.

```
[PASS] stays within budget          21/120 tokens after 30 writes
[PASS] evicted the overflow         27 entries evicted
[PASS] the task survived eviction   pinned entries outlive unpinned ones
```

---

## Tier 2 — Session memory

`src/memory/session.py`

SQLite. Conversation history with citations, user preferences, task progress,
and compaction summaries.

**Addressed by key, never by embedding.** Within a conversation you want
recency and completeness — what did we *just* say, in order. Similarity search
is the wrong tool: it would return turn 3 when you asked about turn 11 because
they rhyme. Semantic search belongs one tier up, where the question is "have I
ever learned this?" rather than "what did we just say?".

**Preferences are scoped to the user, not the session.** That is what makes a
preference a preference — "metric units", "terse answers" should hold tomorrow
too. A preference scoped to a session is just a variable. Consequently
`clear()` deletes turns and progress but keeps preferences; forgetting how
someone likes to be spoken to because a conversation ended is a bug.
`clear_user()` exists for when you really do mean everything.

**Preferences are stored under a stable topic key.** "keep answers short" and
"I prefer short answers" both write to `answer_length`, so restating one
updates it. Keying on the raw text would let two phrasings of one preference
coexist, and the upfront pack would then carry both — including, eventually, two
that contradict.

**Task progress carries across sessions.** Progress is recorded per session,
but a task does not end when a conversation does. `last_progress()` returns the
user's most recent task from any session, and the upfront pack labels it
`carried over`. Without this, day 2 opened knowing the engineer's preferences
but not what they were working on — the less useful half of continuity.

---

## Tier 3 — Long-term memory

`src/memory/longterm.py`

A vector store with two collections: `knowledge` (the corpus) and `episodic`
(facts learned from conversations).

### Promotion is the whole design

Writing every utterance to long-term memory produces a store that grows
linearly with conversation and gets *less* useful as it grows, because recall
then competes against thousands of throwaway lines. A fact earns its place one
of two ways:

1. **Stated durably** — "we always work in Celsius", "our site is NW-41".
   Explicit, confidence ≥ 0.80, promoted on first sight.
2. **Observed repeatedly** across separate sessions. One mention is an
   accident; the same thing in two conversations is a pattern.

Anything else is held pending and expires with the process.

### Reinforcement, not duplication

Restating a known fact increments `times_seen` and records the new session
rather than creating a second row. Recall then ranks on similarity plus a small
bonus for repeated confirmation, so a fact stated in three conversations edges
ahead of an equally-similar one-off.

```
[PASS] restating does not duplicate   4 facts before, 4 after a restatement
[PASS] restating reinforces instead   times_seen=2
```

### Extraction is deterministic, on purpose

Pattern-based, not model-based. An LLM extractor would catch more, but it would
make the memory store non-reproducible run to run — and **a memory system you
cannot replay is one you cannot debug when it remembers something wrong.**
Patterns first; a model can be layered on later without touching the promotion
logic, which is where the interesting decisions live.

Two extraction details that matter more than they look:

**Clip at clause boundaries.** A greedy `.{4,90}` capture runs straight through
conjunctions: "we always work in celsius and our site is NW-41" became a single
malformed preference. One fact per fact — because facts are deduplicated and
recalled *individually*, and a fused pair matches neither probe well.

**Drop subsumed candidates.** Overlapping patterns fire on the same span at
different widths. "always work in celsius" and "in celsius" are one fact;
storing both gives recall two near-identical options while the shorter carries
less meaning.

### Never mine the assistant's own output

Only user turns are extracted. Mining the model's output is how a memory store
fills with the model's earlier guesses and then treats them as established fact
on the next retrieval — a feedback loop that is very hard to notice and very
hard to unwind.

```
[PASS] ignores assistant turns   mining its own output would make the agent
                                 believe its own guesses
```

---

## Tier 4 — Retrieval

Not storage. The path that pulls from long-term memory into the context window
on demand — see [retrieval-pipeline.md](retrieval-pipeline.md).

---

## Compaction

When a conversation outgrows its budget, old turns are replaced by a summary
and the four most recent are kept verbatim.

**Recency is kept exact because that is where referring expressions point** —
"the one you mentioned", "that error". Summarising recent turns breaks the
pronouns the next turn depends on.

**Truncation would be worse than summarising.** Dropping the oldest turns loses
the beginning of the conversation, which is usually where the task was defined.

The extractive path needs no model: it keeps the user's stated goals and any
line carrying a decision, cause or constraint. Lossier than an LLM summary, and
free, deterministic and testable.

```
[PASS] summary is smaller than the source   300 → 141 tokens (47%)
[PASS] summary keeps signal-bearing lines
[PASS] short conversations are left alone
```

---

## What is deliberately not built

**No forgetting curve.** Facts do not decay with time. Decay sounds principled
and is very hard to tune — too fast and the agent forgets a site id it was told
once; too slow and it is the same as never forgetting. Promotion already bounds
growth, which was the actual problem.

**No contradiction resolution.** If a user says "we run 4.3" and later "we run
5.0", both are stored and recall favours the more-confirmed one. Doing this
properly needs temporal reasoning about which fact superseded which, and
guessing wrong is worse than surfacing both.

**No LLM-based extraction.** Covered above: reproducibility is worth more here
than coverage.
