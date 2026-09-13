# Evaluation

`python -m src.main eval` — writes `logs/evaluation.json`.

Every metric here is **deterministic**. Nothing depends on a sampled model
response, which is what makes the ablation trustworthy: when a number moves
between two configurations, it moved because the configuration changed.

---

## Read this first: which backend produced these numbers

The results below come from the **hash embedding backend** — a deterministic
lexical hashing embedder, not a semantic model. It exists so the pipeline runs
with no network and no download, and the offline suite can assert real
behaviour in a second.

It is a **floor, not a headline.** The dense row in particular badly
understates dense retrieval, because hash embeddings cannot do paraphrase at
all. Every report records its backend in the `environment` block:

```json
"environment": {"embedder": "hash", "reranker": "lexical-overlap",
                "store": "numpy", "chunks": 39}
```

To produce the real numbers:

```bash
pip install sentence-transformers    # ~90MB model, downloaded once
python -m src.main index             # re-embed with the semantic model
python -m src.main eval
```

Comparing a hash-backend number against a sentence-transformers number is
meaningless, which is exactly why the backend is recorded rather than assumed.

---

## Retrieval accuracy — ablation

20 labelled queries, relevance judged at document level.

| configuration | R@3 | R@6 | P@3 | MRR | nDCG@6 |
|---|---|---|---|---|---|
| dense only | 0.417 | 0.417 | 0.233 | 0.525 | 0.435 |
| sparse (BM25) only | 0.908 | 0.983 | 0.500 | 0.963 | 0.942 |
| hybrid, no rerank | 0.858 | 0.983 | 0.483 | 0.868 | 0.876 |
| **hybrid + rerank** | 0.858 | 0.983 | 0.467 | **0.963** | **0.930** |
| full (+dedup+filter) | 0.858 | 0.983 | 0.467 | 0.963 | 0.930 |
| full, dedup off | 0.858 | 0.983 | 0.467 | 0.963 | 0.930 |

### By query type — the row that justifies hybrid

```
dense only           exact=0.708  mixed=0.083  paraphrase=0.5
sparse (BM25) only   exact=0.792  mixed=1.0    paraphrase=0.9
hybrid + rerank      exact=0.917  mixed=0.833  paraphrase=0.85
```

**Hybrid beats both constituents on exact-token queries** (0.917 vs 0.792 and
0.708). Two independent retrievers agreeing is a stronger signal than either
alone, and RRF turns that agreement into rank.

Sparse leads overall here only because the hash embedder is weak. A gold set
made of one query type would have hidden this entirely — it would have measured
half a pipeline and made hybrid retrieval look pointless.

### What reranking buys

| | R@3 | MRR | nDCG@6 |
|---|---|---|---|
| hybrid, no rerank | 0.858 | 0.868 | 0.876 |
| hybrid + rerank | 0.858 | **0.963** | **0.930** |

Recall is identical; **ordering improves sharply**. A reranker does not find new
documents, it moves the right one to the top. Recall@k cannot see that, which is
why MRR and nDCG are in the table at all.

---

## Context economy

```
                            chunks  relevance  token_reduction
dense only                    1.4     0.314        0.0
sparse (BM25) only            6.05    0.0          0.492
hybrid, no rerank             5.75    0.200        0.519
hybrid + rerank               5.95    0.202        0.505
```

Roughly **half the retrieved tokens never reach the window**. Per-turn figures
from the demo run between 32% and 77%.

The point is not saving money. It is **dilution**: attention is finite, and
every irrelevant token competes with a relevant one. Repetition is worse — in a
long context it reads as emphasis, so a duplicated minor point can outweigh a
unique critical one.

`dedup off` scores identically here because the corpus is small and its genuine
near-duplicate (`RB-107` restating `REL-5.0`'s rollback procedure) sits just
below the 0.92 threshold. Reported rather than tuned away: lowering the
threshold to make the row move would be fitting the metric, and removing a
*distinct* chunk is worse than keeping a near-duplicate.

---

## Memory quality — cross-session recall

Facts are stated in session 1. A **different** session then probes for them with
paraphrased questions and no shared process state.

```
facts promoted to long-term : 4
cross-session recall        : 4/4 (1.0)
  [OK ] what temperature units do we use   -> always work in celsius
  [OK ] which site do we work on           -> Our site is NW-41
  [OK ] what firmware are we on            -> we run firmware 4.3
  [OK ] how long should answers be         -> i prefer short answers
```

This is the metric that separates a memory system from a chat log. Anything
recalled here crossed a session boundary through the long-term store.

Note the probes deliberately share no vocabulary with the stored facts —
"what temperature units do we use" against "always work in celsius". A probe
that reuses the fact's own words tests string matching, not memory.

---

## Response consistency

Four paraphrase pairs. Two questions meaning the same thing should reach the
same evidence.

```
mean evidence overlap  : 0.5
mean answer similarity : 0.477

  C1  overlap=0.0   sim=0.197   buffer window
  C2  overlap=0.5   sim=0.703   rollback to 4.2
  C3  overlap=1.0   sim=0.738   boot loop / warranty
  C4  overlap=0.5   sim=0.270   antenna near metal
```

**Evidence overlap is the stricter signal and the one to trust.** Two answers
can read similarly while resting on different sources — that is precisely the
inconsistency that bites later, when one of the two sources turns out to be
version-specific.

C1 at 0.0 is the honest failure: "how long does the HX-40 buffer telemetry for"
and "what is the local buffering window on a gateway" share almost no
vocabulary, and a lexical embedder has no way to connect them. This is the
clearest single demonstration of what the hash backend costs, and it is the
number most likely to move on a semantic model.

---

## Metric definitions

| Metric | Definition | Answers |
|---|---|---|
| recall@k | relevant docs in top k ÷ total relevant | did we find it at all? |
| precision@k | relevant docs in top k ÷ k | how much noise came with it? |
| MRR | 1 ÷ rank of the first relevant doc | how fast is the first hit? |
| nDCG@k | discounted gain ÷ ideal gain | is the ordering good, not just the set? |
| context relevance | mean dense similarity of packed chunks | is what reached the window on topic? |
| token reduction | 1 − (packed tokens ÷ retrieved tokens) | how much did the optimiser remove? |
| cross-session recall | probes answered from long-term ÷ probes | does memory survive a session boundary? |
| evidence overlap | Jaccard of top-3 doc sets across a paraphrase pair | is the same question answered from the same sources? |

---

## Two bugs this suite found

Both passed every unit test. Only the ablation exposed them.

**1. Agreement was subtracting.** The relevance filter read
`"sparse" in retrievers and dense_score == 0.0`. A chunk found by *both*
retrievers has a non-zero dense score, so it failed that branch — and if the
score sat below the floor it failed the next too. A chunk was penalised for
being found twice.

Symptom in the table: hybrid `0.442`, dense alone `0.442`, BM25 alone `0.950`.
Hybrid contributing *exactly nothing* over one of its halves is not a tuning
problem, it is a wiring problem. Fixed: hybrid → `0.875`.

**2. Chunking never chunked.** At `CHUNK_TOKENS=180`, every document fit in one
chunk. Overlap, ordinals and near-duplicate removal were dead code that still
passed its tests. At 110 tokens the corpus produces 39 chunks from 27 documents
and `token_reduction` moved from `0.0` to `~0.5`.

**What they share:** a unit test proves a function does what you wrote. An
ablation proves a component is worth having. Only the second catches a feature
that silently does nothing — and a feature that silently does nothing is
strictly worse than an absent one, because you believe you have it.

---

## Reproducing

```bash
python -m scripts.mock_test      # 76 deterministic checks, ~1s
python -m src.main index
python -m src.main eval          # → logs/evaluation.json
```
