"""Evaluation: retrieval accuracy, context relevance, memory quality, consistency.

Every metric here is **deterministic**. No metric depends on a sampled model
response, which is what makes the ablation table trustworthy: when recall moves
between two configurations, it moved because the configuration changed, not
because the model happened to phrase something differently.

Four families, matching the brief's requirement 7:

  retrieval accuracy    recall@k, precision@k, MRR, nDCG@k against a labelled
                        gold set
  context relevance     mean similarity of what actually reached the window,
                        plus how much the optimiser removed
  memory quality        does a fact stated in session 1 survive into session 3
  response consistency  do paraphrases of one question produce the same
                        evidence and similar answers

The ablation is the part worth reading. Claiming "hybrid retrieval improves
results" is free; showing that turning off BM25 costs N points of recall on
exact-token queries and nothing on paraphrases is a finding.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ROOT
from .context import optimise
from .embeddings import backend_name, cosine, embed_one, reranker_name
from .retrieval import KnowledgeIndex

GOLD_PATH = ROOT / "evals" / "gold.json"


def load_gold(path: Path | None = None) -> dict:
    return json.loads(Path(path or GOLD_PATH).read_text())


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if k == 0:
        return 0.0
    return len(set(retrieved[:k]) & relevant) / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, doc in enumerate(retrieved, start=1):
        if doc in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Binary-relevance nDCG. Rewards putting relevant documents *early*,
    which recall@k alone does not — a system that buries the right answer at
    position 6 scores the same recall as one that leads with it."""
    dcg = sum(1.0 / math.log2(i + 1)
              for i, doc in enumerate(retrieved[:k], start=1) if doc in relevant)
    ideal = sum(1.0 / math.log2(i + 1)
                for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0


# ---------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    use_dense: bool = True
    use_sparse: bool = True
    use_rerank: bool = True
    do_dedup: bool = True
    do_filter: bool = True


@dataclass
class Result:
    config: str
    recall_at_3: float = 0.0
    recall_at_6: float = 0.0
    precision_at_3: float = 0.0
    mrr: float = 0.0
    ndcg_at_6: float = 0.0
    by_type: dict = field(default_factory=dict)
    mean_context_relevance: float = 0.0
    mean_chunks_packed: float = 0.0
    mean_token_reduction: float = 0.0

    def row(self) -> str:
        return (f"{self.config:<26} {self.recall_at_3:>7.3f} {self.recall_at_6:>7.3f} "
                f"{self.precision_at_3:>7.3f} {self.mrr:>7.3f} {self.ndcg_at_6:>7.3f}")


def _ranked_docs(hits) -> list[str]:
    """Chunk ranking → document ranking, first occurrence wins.

    The gold set labels documents, not chunks. Collapsing this way means a
    document whose three chunks all rank highly does not crowd out the rest of
    the list and inflate precision.
    """
    seen, out = set(), []
    for h in hits:
        if h.doc_id not in seen:
            seen.add(h.doc_id)
            out.append(h.doc_id)
    return out


def evaluate_config(index: KnowledgeIndex, gold: dict, cfg: Config,
                    k: int = 6) -> Result:
    res = Result(config=cfg.name)
    by_type: dict[str, list[float]] = {}
    r3 = r6 = p3 = mrr = ndcg = 0.0
    relevances, packed_counts, reductions = [], [], []

    queries = gold["queries"]
    for q in queries:
        relevant = set(q["relevant"])
        hits = index.search(q["query"], k=k * 2, use_dense=cfg.use_dense,
                            use_sparse=cfg.use_sparse, use_rerank=cfg.use_rerank)
        packed, report = optimise(hits, do_dedup=cfg.do_dedup,
                                  do_filter=cfg.do_filter)
        docs = _ranked_docs(packed)

        q_r3 = recall_at_k(docs, relevant, 3)
        r3 += q_r3
        r6 += recall_at_k(docs, relevant, 6)
        p3 += precision_at_k(docs, relevant, 3)
        mrr += reciprocal_rank(docs, relevant)
        ndcg += ndcg_at_k(docs, relevant, 6)

        by_type.setdefault(q.get("type", "mixed"), []).append(q_r3)

        scores = [h.dense_score for h in packed if h.dense_score > 0]
        if scores:
            relevances.append(float(np.mean(scores)))
        packed_counts.append(len(packed))
        reductions.append(report.reduction)

    n = len(queries)
    res.recall_at_3 = round(r3 / n, 4)
    res.recall_at_6 = round(r6 / n, 4)
    res.precision_at_3 = round(p3 / n, 4)
    res.mrr = round(mrr / n, 4)
    res.ndcg_at_6 = round(ndcg / n, 4)
    res.by_type = {t: round(float(np.mean(v)), 3) for t, v in sorted(by_type.items())}
    res.mean_context_relevance = round(float(np.mean(relevances)), 4) if relevances else 0.0
    res.mean_chunks_packed = round(float(np.mean(packed_counts)), 2)
    res.mean_token_reduction = round(float(np.mean(reductions)), 3)
    return res


ABLATIONS = [
    Config("dense only",          use_sparse=False, use_rerank=False),
    Config("sparse (BM25) only",  use_dense=False, use_rerank=False),
    Config("hybrid, no rerank",   use_rerank=False),
    Config("hybrid + rerank",     use_rerank=True),
    Config("full (+dedup+filter)", use_rerank=True, do_dedup=True, do_filter=True),
    Config("full, dedup off",     use_rerank=True, do_dedup=False),
]


def run_ablation(index: KnowledgeIndex, gold: dict) -> list[Result]:
    return [evaluate_config(index, gold, cfg) for cfg in ABLATIONS]


# ---------------------------------------------------------------------------
# Memory quality
# ---------------------------------------------------------------------------
def evaluate_memory(user_id: str = "eval-user") -> dict:
    """Does a fact stated in one session survive into a later one?

    This is the metric that separates a memory system from a chat log. It
    writes facts in session 1, opens a *fresh* session 3, and asks whether the
    upfront pack carries them — i.e. whether continuity is real or whether the
    agent is merely reading back its own scrollback.
    """
    from .memory import LongTermMemory, SessionMemory

    ltm = LongTermMemory(user_id)
    ltm.clear()

    stated = [
        ("we always work in celsius", "preference"),
        ("our site is NW-41", "entity"),
        ("we run firmware 4.3", "entity"),
        ("i prefer short answers", "preference"),
    ]

    s1 = SessionMemory("eval-s1", user_id)
    s1.clear()
    for text, _kind in stated:
        s1.append("user", text)
        ltm.consider(text, "user", "eval-s1")

    # A different session entirely. Nothing is carried in process state except
    # the long-term store, which is the point.
    s3 = SessionMemory("eval-s3", user_id)
    s3.clear()

    probes = [
        ("what temperature units do we use", "celsius"),
        ("which site do we work on", "NW-41"),
        ("what firmware are we on", "4.3"),
        ("how long should answers be", "short"),
    ]

    hits = 0
    details = []
    for probe, expected in probes:
        recalled = ltm.recall(probe, k=3)
        found = any(expected.lower() in f.text.lower() for f in recalled)
        hits += int(found)
        details.append({"probe": probe, "expected": expected, "recalled": found,
                        "top": recalled[0].text if recalled else None})

    return {
        "facts_stored": ltm.stats()["facts"],
        "probes": len(probes),
        "recalled": hits,
        "cross_session_recall": round(hits / len(probes), 3),
        "details": details,
    }


# ---------------------------------------------------------------------------
# Response consistency
# ---------------------------------------------------------------------------
def evaluate_consistency(index: KnowledgeIndex, gold: dict) -> dict:
    """Two phrasings of one question should reach the same evidence.

    Measured two ways: overlap of the retrieved document sets (Jaccard), and
    cosine similarity between the two answers. Evidence overlap is the stricter
    and more useful signal — two answers can read similarly while resting on
    different sources, which is exactly the inconsistency that bites later.
    """
    from .generation import extractive_answer

    rows = []
    for pair in gold.get("consistency_pairs", []):
        a_hits = index.search(pair["a"])
        b_hits = index.search(pair["b"])
        a_docs = set(_ranked_docs(a_hits)[:3])
        b_docs = set(_ranked_docs(b_hits)[:3])
        jaccard = (len(a_docs & b_docs) / len(a_docs | b_docs)) if (a_docs | b_docs) else 0.0

        a_ans = extractive_answer(pair["a"], a_hits, [])
        b_ans = extractive_answer(pair["b"], b_hits, [])
        sim = cosine(embed_one(a_ans.text), embed_one(b_ans.text))

        rows.append({"id": pair["id"], "evidence_overlap": round(jaccard, 3),
                     "answer_similarity": round(sim, 3),
                     "docs_a": sorted(a_docs), "docs_b": sorted(b_docs)})

    if not rows:
        return {"pairs": 0}
    return {
        "pairs": len(rows),
        "mean_evidence_overlap": round(
            float(np.mean([r["evidence_overlap"] for r in rows])), 3),
        "mean_answer_similarity": round(
            float(np.mean([r["answer_similarity"] for r in rows])), 3),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
def full_report(index: KnowledgeIndex | None = None) -> dict:
    index = index or KnowledgeIndex()
    if index.size == 0:
        index.load()
    if index.size == 0:
        raise SystemExit("Index is empty. Run:  python -m src.main index")

    gold = load_gold()
    results = run_ablation(index, gold)
    return {
        "environment": {"embedder": backend_name(), "reranker": reranker_name(),
                        "chunks": index.size, "store": index.store.name},
        "retrieval": [r.__dict__ for r in results],
        "memory": evaluate_memory(),
        "consistency": evaluate_consistency(index, gold),
    }


def print_report(report: dict) -> None:
    env = report["environment"]
    print("=" * 84)
    print("EVALUATION")
    print("=" * 84)
    print(f"  embedder={env['embedder']}  reranker={env['reranker']}  "
          f"store={env['store']}  chunks={env['chunks']}")

    print("\nRETRIEVAL — ablation")
    print(f"  {'configuration':<26} {'R@3':>7} {'R@6':>7} {'P@3':>7} "
          f"{'MRR':>7} {'nDCG':>7}")
    print("  " + "-" * 72)
    for row in report["retrieval"]:
        r = Result(**{k: v for k, v in row.items()})
        print("  " + r.row())

    print("\n  recall@3 by query type")
    for row in report["retrieval"]:
        types = ", ".join(f"{t}={v}" for t, v in row["by_type"].items())
        print(f"    {row['config']:<26} {types}")

    print("\n  context economy")
    for row in report["retrieval"]:
        print(f"    {row['config']:<26} chunks={row['mean_chunks_packed']:<6} "
              f"relevance={row['mean_context_relevance']:<8} "
              f"token_reduction={row['mean_token_reduction']}")

    mem = report["memory"]
    print(f"\nMEMORY QUALITY")
    print(f"  facts promoted to long-term : {mem['facts_stored']}")
    print(f"  cross-session recall        : {mem['recalled']}/{mem['probes']} "
          f"({mem['cross_session_recall']})")
    for d in mem["details"]:
        mark = "OK " if d["recalled"] else "MISS"
        print(f"    [{mark}] {d['probe']:<38} -> {d['top']}")

    con = report["consistency"]
    if con.get("pairs"):
        print(f"\nRESPONSE CONSISTENCY  ({con['pairs']} paraphrase pairs)")
        print(f"  mean evidence overlap  : {con['mean_evidence_overlap']}")
        print(f"  mean answer similarity : {con['mean_answer_similarity']}")
        for r in con["rows"]:
            print(f"    {r['id']}  overlap={r['evidence_overlap']:<6} "
                  f"sim={r['answer_similarity']:<6} {r['docs_a']} vs {r['docs_b']}")
    print("=" * 84)
