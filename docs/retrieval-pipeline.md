# Retrieval Pipeline

```
query
  ├─ dense  (embeddings)  → top-20 by cosine
  └─ sparse (BM25)        → top-20 by term weight
            ↓
   reciprocal rank fusion
            ↓
   cross-encoder rerank (top candidates only)
            ↓
   relevance floor → near-duplicate removal → pack to budget
            ↓
   context window, each chunk carrying its citation
```

## 1. Chunking

`src/chunking.py` — 110 tokens, 25-token overlap, split on paragraph
boundaries.

**Split on paragraphs, not a fixed stride.** A chunk beginning mid-sentence
retrieves badly: its embedding is the average of half an idea and the start of
another. Packing whole paragraphs up to a budget keeps each chunk about one
thing.

**Prepend document identity to every chunk.** A chunk reading *"the buffer now
applies backpressure instead of discarding the oldest records"* is nearly
useless alone — backpressure in what, since when? The embedded text carries
`REL-5.0 · Helios firmware 5.0 release notes (release, 2026-02-26)`, so the
chunk answers version-qualified questions and arrives with its citation
already attached.

**Size chunks below typical document length, or the feature is dead code.** At
180 tokens every document in this corpus fit in a single chunk. Overlap,
ordinals and near-duplicate removal all still passed their unit tests while
never executing on real data. At 110 tokens the corpus yields **39 chunks from
27 documents** and those paths are live.

## 2. Two retrievers, because they fail differently

| | Good at | Blind to |
|---|---|---|
| Dense | paraphrase — "my gateway went quiet" → uplinks | version strings, error codes; they barely move an embedding |
| BM25 | exact tokens — `REL-5.0`, `exit code 3` | anything phrased differently from the document |

The gold set is labelled by type so the ablation can show this rather than
assert it:

```
recall@3 by query type
  dense only           exact=0.708  mixed=0.083  paraphrase=0.5
  sparse (BM25) only   exact=0.792  mixed=1.0    paraphrase=0.9
  hybrid + rerank      exact=0.917  mixed=0.833  paraphrase=0.85
```

Hybrid beats both on exact-token queries — the two retrievers agreeing is a
stronger signal than either alone.

*These are hash-backend numbers. The dense row is a lexical hashing embedder,
not a semantic model, so it understates dense retrieval badly. Install
`sentence-transformers` and rerun; the report records which backend produced
it.*

## 3. Reciprocal rank fusion

```
score(d) = Σ over retrievers  1 / (K + rank(d))        K = 60
```

**Why not weighted score blending?** Cosine similarity and BM25 scores are not
on the same scale, and BM25's scale shifts with corpus statistics. Any fixed
weighting is a constant that was tuned once and silently rots as the corpus
grows. RRF uses only *rank position*: no normalisation, no tuning, no drift.

## 4. Cross-encoder reranking

The cross-encoder reads query and passage **together** rather than comparing
two independently-produced vectors. That is why it resolves cases bi-encoder
similarity gets wrong, and why it costs enough to run last on few candidates.

Retrieve broad (20 per retriever), rerank narrow (top ~24 fused), keep 6.

What it buys, measured:

| | R@3 | MRR | nDCG@6 |
|---|---|---|---|
| hybrid, no rerank | 0.858 | 0.868 | 0.876 |
| hybrid + rerank | 0.858 | **0.963** | **0.930** |

**Recall is unchanged; ordering improves sharply.** That is exactly what a
reranker is for — it does not find new documents, it moves the right one to the
top. Recall@k alone cannot see this, which is why nDCG and MRR are in the table.

## 5. Relevance floor

Two thresholds, because the two retrievers need different ones:

- **Dense:** cosine ≥ 0.25. Scale-free and consistent.
- **Sparse:** BM25 ≥ 35% of the best BM25 score *for this query*. Relative, not
  absolute, because BM25 scores are not comparable between queries.

The floor is applied on retriever scores, never on the cross-encoder's. Those
are logits whose range shifts with the model, so a fixed threshold on them would
mean something different every time the reranker changed.

### The bug this is written around

The original condition was:

```python
if "sparse" in r.retrievers and r.dense_score == 0.0:   # wrong
```

A chunk found by **both** retrievers has a non-zero dense score, so it failed
that branch — and if its dense score sat below the floor it failed the next one
too. **A chunk was penalised for being found twice.**

Every unit test passed. The ablation table is what exposed it: hybrid scored
`0.442`, identical to dense alone, while BM25 alone scored `0.950`. Fixing it
moved hybrid to `0.875`.

Agreement between two independent retrievers is the strongest signal available.
It must never subtract.

The second half — requiring a BM25 hit to be a *good* BM25 hit — came from the
same table. An unconditional pass for any sparse hit let BM25's long tail
through and `token_reduction` fell to `0.0`: the filter had stopped filtering.

## 6. Near-duplicate removal

Cosine ≥ 0.92 between chunk embeddings; the higher-ranked copy survives.

Exact string equality is useless here — chunk overlap and restated procedures
produce text that differs in wording but not content. `RB-107` restates the
rollback procedure from `REL-5.0` almost verbatim; both are legitimately
retrieved for a rollback question, and both in the window is one fact taking two
slots.

## 7. Packing

Fill the budget in rank order; skip anything that does not fit; keep going.

**Deliberately not a knapsack.** Strict rank order means the model sees the best
evidence first. A greedy "squeeze in a small low-ranked chunk" reorders
relevance to save tokens, which is the wrong trade — the window is not the
scarce resource, attention is.

## 8. Source attribution

The citation travels with the chunk from retrieval to prompt to answer, and
sits **immediately above the text it belongs to**:

```
[RB-105#1] Clearing a full local buffer on HX-40 (runbook, 2026-02-19)
When the buffer fills, firmware 4.x discards the oldest records...
```

Not a bibliography at the end. A model reading a trailing source list has to
remember which fact came from which entry; a model reading `[RB-105#1]` directly
above the fact does not. Answers are parsed back for `[DOC-ID#n]` patterns, so
citation coverage is measurable rather than assumed.

## Configuration

| Parameter | Default | Why |
|---|---|---|
| `CHUNK_TOKENS` | 110 | below typical document length, so chunking actually splits |
| `CHUNK_OVERLAP` | 25 | keeps a procedure step with its warning |
| `RETRIEVE_K` | 20 | candidates per retriever |
| `RERANK_K` | 6 | survivors into the window |
| `SIM_FLOOR` | 0.25 | dense relevance floor |
| `SPARSE_FLOOR_RATIO` | 0.35 | fraction of best BM25 score for this query |
| `RRF_K` | 60 | standard fusion constant |
| `DEDUP_THRESHOLD` | 0.92 | high — removing a *distinct* chunk is worse than keeping a near-duplicate |
| `CONTEXT_BUDGET_TOKENS` | 2000 | retrieved context only, excluding memory and history |
