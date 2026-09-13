# Multi-Tier Memory & Retrieval Agent

Module 8 Lab — context engineering, four memory tiers, and hybrid RAG.

A field-engineering assistant that holds a single investigation together across
three days and three separate conversations. Working, session and long-term
memory; hybrid dense+sparse retrieval with reranking; context filtering,
deduplication and compaction; and a deterministic evaluation suite with an
ablation table.

```
76/76 offline checks passed — no API key, no network, no model download
```

| Requirement | Where |
|---|---|
| 1. Working memory | `src/memory/working.py` — bounded, token-budgeted, evicting |
| 2. Session memory | `src/memory/session.py` — SQLite: history, preferences, task progress |
| 3. Long-term memory | `src/memory/longterm.py` — vector store + **promotion**, not a log |
| 4. Retrieval system | `src/retrieval.py` — semantic search, RRF, cross-encoder rerank, attribution |
| 5. Hybrid strategy | `src/agent.py` — upfront context pack + gated just-in-time retrieval |
| 6. Context optimization | `src/context.py` — relevance floor, near-duplicate removal, compaction |
| 7. Memory evaluation | `src/evaluate.py` — recall@k, MRR, nDCG, cross-session recall, consistency |

Docs: [architecture](docs/architecture.md) · [memory design](docs/memory-design.md) ·
[retrieval pipeline](docs/retrieval-pipeline.md) ·
[multi-session transcript](docs/sample-multi-session.md) ·
[evaluation](docs/evaluation.md)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt      # numpy, python-dotenv, tiktoken
python -m src.main index
```

**No API key is required, and no heavy dependency either.** The core install is
three pure-Python packages.

### Optional upgrades

```bash
pip install -r requirements-optional.txt
```

Each is guarded by a fallback, so a failed install changes **quality, not
capability**:

| Package | Gives you | Falls back to | Impact if missing |
|---|---|---|---|
| `sentence-transformers` | semantic embeddings + cross-encoder | hash embedder + lexical reranker | dense retrieval and paraphrase matching get worse |
| `chromadb` | persistent vector DB | `NumpyStore` (exact cosine) | none at this corpus size — exact search is if anything *more* accurate |
| `openai` | model-written answers | extractive answerer | prose only; retrieval, memory and all metrics unaffected |

Check what actually loaded:

```bash
python -m src.main index
#   store      : numpy | chroma
#   embedder   : hash | sentence-transformers
#   reranker   : lexical-overlap | cross-encoder
```

**Why they are optional.** Both obvious dependencies pull native wheels that do
not exist everywhere: `chromadb` needs `onnxruntime` (no `cp314` wheel), and
`sentence-transformers` needs `torch` (no macOS x86_64 wheels after 2.2). A
pinned dependency that cannot install on the grader's machine is worse than an
optional one that degrades.

**Intel Mac or Python 3.13+?** Both extras are likely to fail. For real semantic
embeddings, use a Python 3.11 environment:

```bash
conda create -n fde python=3.11 -y && conda activate fde
pip install -r requirements.txt -r requirements-optional.txt
python -m src.main index && python -m src.main eval
```

Otherwise run on the core install — everything works, and the report records
which backend produced each number.

## Run

```bash
python -m scripts.mock_test        # 76 offline checks, ~1 second
python -m src.main index           # build the knowledge index
python -m src.main demo --fresh    # three sessions, one continuing task
python -m src.main eval            # metrics + ablation table
python -m src.main search "can we roll back to 4.2"   # inspect retrieval
python -m src.main chat ticket-9   # interactive; rerun to resume
python -m src.main memory          # what the agent remembers, and why
```

Run `mock_test` first. Every deterministic part of the system — the memory
tiers, chunking, BM25, rank fusion, dedup, compaction, and all the ranking
metrics — is verified there with no network. Anything that fails afterwards is
about model quality, not about the pipeline.

## The four tiers

What separates them is not storage technology; it is **what causes a write and
what causes a read**.

| Tier | Scope | Written by | Read by | Bounded? |
|---|---|---|---|---|
| Working | one task | the current step | the next step | **yes** — evicts |
| Session | one conversation | each turn | the next turn | compacts |
| Long-term | the user, forever | **promotion** | semantic recall | grows slowly |
| Knowledge | the corpus | indexing | just-in-time retrieval | static |

**Promotion is the part that matters.** Long-term memory is not a transcript
archive. A fact earns a place either by being stated durably ("we always work
in Celsius") or by recurring across separate sessions. Restating a known fact
*reinforces* it — `times_seen` increments — rather than creating a duplicate.
Without that rule, "long-term memory" is a chat log with a vector index, and it
gets less useful the more it holds.

## Hybrid retrieval strategy

Two loading strategies, for different things:

- **Upfront**, once per session (~110 tokens): preferences, carried-over task,
  findings so far, top episodic facts. You cannot retrieve "the user prefers
  Celsius" in response to a question that never mentions temperature.
- **Just-in-time**, per turn, and **gated**: knowledge-base chunks for this
  question only. The corpus is large and mostly irrelevant to any single turn.

The rule: *upfront for what is small and always relevant; just-in-time for what
is large and conditionally relevant.* Inverting it gives you either an amnesiac
agent or a diluted one.

The gate skips retrieval four ways — acknowledgements, memory-only questions,
two-word fragments, and **profile statements**:

```
engineer> We always work in celsius and I prefer short answers.
          Our site is NW-41 and we run firmware 4.3.
agent>    Noted — I'll remember: I prefer short answers; always work in
          celsius; Our site is NW-41; we run firmware 4.3.
          retrieval=skipped (profile statement — recorded, nothing asked)
```

## Continuity across sessions

Day 2 is a different process and a different session id. Everything it knows
came out of long-term memory:

```
── upfront context pack (111 tokens) ──
  User preferences: answer_length=I prefer short answers; units=always work in celsius
  Open task (carried over): Investigate why site NW-41 gateways stopped reporting
  Established so far: Half our gateways went quiet yesterday → RB-106#1; ...
  Next steps: fit replacement radios; confirm firmware plan
  [memory:entity] Our site is NW-41  (seen 2×)
  [memory:preference] always work in celsius  (seen 2×)

engineer> What did we establish yesterday?
agent>    Open task: Investigate why site NW-41 gateways stopped reporting (in progress).
          - Half our gateways went quiet yesterday → RB-106#1
          - The status light is solid amber on three of → RB-101#0, INC-2314#0
          - next: fit replacement radios
          retrieval=skipped (answerable from memory)
```

Nothing was re-asked. Full transcript in
[docs/sample-multi-session.md](docs/sample-multi-session.md).

## Context optimization

Per turn, typically **30–77% fewer tokens** reach the window:

```
18 retrieved → 4 relevant → 4 unique → 4 packed  (1565→356 tokens, 77% reduction)
```

The problem this solves is not running out of room. It is **dilution** —
attention is finite, and every irrelevant token competes with a relevant one.
Repetition is worse still: in a long context it reads as emphasis, so a
duplicated minor point can outweigh a unique critical one.

## Evaluation

`python -m src.main eval` runs six configurations over a 20-query labelled set:

| configuration | R@3 | MRR | nDCG |
|---|---|---|---|
| dense only | 0.417 | 0.525 | 0.435 |
| sparse (BM25) only | 0.908 | 0.963 | 0.942 |
| hybrid, no rerank | 0.858 | 0.868 | 0.876 |
| **hybrid + rerank** | **0.858** | **0.963** | **0.930** |

*Numbers above are from the **hash** embedding backend — a lexical fallback, not
a semantic model. They are a floor, not the headline. Install
`sentence-transformers` and rerun to get the real dense numbers; the report
records which backend produced it.*

Reranking is free recall and buys ordering: R@3 is unchanged while MRR rises
0.868 → 0.963. It moves the right answer up, which is exactly what a
cross-encoder is for.

Also measured: cross-session memory recall (4/4), context relevance, token
reduction, and response consistency across paraphrase pairs.

## Two bugs the ablation caught

Both passed their unit tests. Only the ablation table exposed them.

**Agreement was subtracting.** The relevance filter read
`"sparse" in retrievers and dense_score == 0.0`. A chunk found by *both*
retrievers has a non-zero dense score, so it failed that branch — and if the
score sat below the floor it failed the next one too. A chunk was penalised for
being found twice. Hybrid scored identically to dense alone while BM25 alone
more than doubled it.

**Chunking never chunked.** At 180 tokens every document fit in one chunk, so
overlap, ordinals and near-duplicate removal were dead code that still passed
its tests. At 110 tokens the corpus produces 39 chunks from 27 documents and
those paths are live.

The lesson both share: a unit test proves a function does what you wrote. An
ablation proves the component is *worth having*. Only the second one catches a
feature that silently does nothing.

## Layout

```
src/
  agent.py          the loop: upfront pack + gated JIT retrieval
  memory/
    working.py      tier 1 — bounded, evicting
    session.py      tier 2 — SQLite, durable
    longterm.py     tier 3 — vector store + promotion
  retrieval.py      BM25 + dense + RRF + rerank + attribution
  chunking.py       structure-aware, header-carrying
  context.py        filter, deduplicate, pack, compact
  embeddings.py     local embeddings + cross-encoder, with fallbacks
  vectorstore.py    Chroma, with a numpy store behind the same interface
  evaluate.py       metrics, ablation, memory and consistency suites
  generation.py     hosted model, or extractive
  corpus.py         27-document Helios knowledge base
evals/gold.json     20 labelled queries + 4 paraphrase pairs
scripts/mock_test.py  76 offline checks
```
# context-memory-agent
