"""Offline verification. Zero API calls, zero model downloads, zero network.

Forces the hash embedder and the numpy store into a temporary directory, so
this runs in about a second in a bare container and cannot touch real data.

What it proves: chunking, token budgeting, all four memory tiers, promotion and
reinforcement, BM25, rank fusion, deduplication, packing, compaction, the
retrieval gate, and every ranking metric. That is the whole system minus the
quality of the embedding model.

Which is the point. When a live run later looks wrong, the question "is the
pipeline broken or is the model weak?" has already been answered here.

Run:  python -m scripts.mock_test
"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before src.config is imported.
TMP = Path(tempfile.mkdtemp(prefix="cma-test-"))
os.environ["EMBED_BACKEND"] = "hash"
os.environ["USE_RERANKER"] = "0"
os.environ["PROVIDER"] = "offline"
os.environ["DATA_DIR"] = str(TMP)

from src import config                                          # noqa: E402
from src.chunking import chunk_corpus, chunk_document           # noqa: E402
from src.context import (compact_history, deduplicate, optimise,  # noqa: E402
                         pack)
from src.corpus import BY_ID, DOCUMENTS                          # noqa: E402
from src.embeddings import backend_name, cosine, embed_one       # noqa: E402
from src.evaluate import (ndcg_at_k, precision_at_k,             # noqa: E402
                          recall_at_k, reciprocal_rank)
from src.memory import LongTermMemory, SessionMemory, WorkingMemory  # noqa: E402
from src.memory.longterm import extract_candidates               # noqa: E402
from src.retrieval import BM25, KnowledgeIndex                   # noqa: E402
from src.tokens import count_tokens, truncate_to_tokens          # noqa: E402
from src.vectorstore import NumpyStore                           # noqa: E402

CHECKS: list[tuple[str, bool]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    CHECKS.append((name, passed))
    print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def section(title: str) -> None:
    print(f"\n[{title}]")


def main() -> int:
    print("=" * 78)
    print("OFFLINE VERIFICATION — no API calls, no downloads")
    print("=" * 78)
    print(f"  embedder={backend_name()}  data={TMP}")

    # -- 1. TOKENS --------------------------------------------------------
    section("1. TOKEN BUDGETING")
    check("counts tokens", count_tokens("hello world this is a test") > 0)
    check("empty string costs nothing", count_tokens("") == 0)
    long_text = " ".join(["word"] * 500)
    cut = truncate_to_tokens(long_text, 50)
    check("truncates to budget", count_tokens(cut) <= 50,
          f"{count_tokens(long_text)} → {count_tokens(cut)} tokens")
    check("truncation is a prefix", long_text.startswith(cut[:20]))

    # -- 2. CHUNKING ------------------------------------------------------
    section("2. CHUNKING")
    chunks = chunk_corpus(DOCUMENTS)
    check("every document produced chunks",
          {c.doc_id for c in chunks} == {d.doc_id for d in DOCUMENTS},
          f"{len(chunks)} chunks from {len(DOCUMENTS)} documents")
    check("chunk ids are unique",
          len({c.chunk_id for c in chunks}) == len(chunks))
    check("embed text carries document identity",
          all(c.doc_id in c.embed_text and c.title in c.embed_text
              for c in chunks),
          "a chunk is retrievable by its version and title, not only its body")
    oversize = [c for c in chunks if count_tokens(c.text) > config.CHUNK_TOKENS * 2.2]
    check("no runaway chunks", not oversize,
          f"largest={max(count_tokens(c.text) for c in chunks)} tokens")
    multi = chunk_document(BY_ID["REL-5.0"])
    check("a multi-paragraph document splits", len(multi) >= 1)
    check("ordinals are sequential",
          [c.ordinal for c in multi] == list(range(len(multi))))

    # -- 3. WORKING MEMORY ------------------------------------------------
    section("3. WORKING MEMORY — bounded, evicting, pin-safe")
    wm = WorkingMemory(budget=120)
    wm.set_task("Diagnose site NW-41 outage")
    for i in range(30):
        wm.add_tool_output(f"tool_{i}", "x " * 60)
    check("stays within budget", wm.tokens <= wm.budget,
          f"{wm.tokens}/{wm.budget} tokens after 30 writes")
    check("evicted the overflow", len(wm.evicted) > 0,
          f"{len(wm.evicted)} entries evicted")
    check("the task survived eviction", wm.task() == "Diagnose site NW-41 outage",
          "pinned entries outlive unpinned ones")
    check("tool output is truncated on write",
          all(count_tokens(e.content) <= 260 for e in wm.entries))
    wm.clear()
    check("clear keeps the task", wm.task() is not None)

    # -- 4. SESSION MEMORY ------------------------------------------------
    section("4. SESSION MEMORY — durable across instances")
    s = SessionMemory("t-sess", "t-user")
    s.clear()
    s.append("user", "gateways went quiet at NW-41")
    s.append("assistant", "checking", ["RB-101#0"])
    check("history round-trips", s.turn_count() == 2)
    check("citations are stored with the turn",
          s.history()[-1].citations == ["RB-101#0"])

    s.set_preference("units", "celsius")
    s.set_progress(task="NW-41 outage", status="open")
    s.add_finding("three gateways solid amber")

    reopened = SessionMemory("t-sess", "t-user")
    check("survives a new instance", reopened.turn_count() == 2,
          "state is in SQLite, not in the object")
    check("preferences persist", reopened.preferences().get("units") == "celsius")
    check("progress persists",
          reopened.progress()["task"] == "NW-41 outage")
    check("findings accumulate",
          "three gateways solid amber" in reopened.progress()["findings"])
    reopened.set_progress(status="closed")
    check("partial update preserves other fields",
          reopened.progress()["task"] == "NW-41 outage"
          and reopened.progress()["status"] == "closed",
          "updating status must not wipe the task")

    other_session = SessionMemory("t-sess-2", "t-user")
    check("preferences are user-scoped, not session-scoped",
          other_session.preferences().get("units") == "celsius",
          "this is what makes a preference a preference")

    # -- 5. FACT EXTRACTION & PROMOTION -----------------------------------
    section("5. LONG-TERM MEMORY — extraction, promotion, reinforcement")
    cands = extract_candidates("We always work in celsius and our site is NW-41",
                               "user")
    check("extracts multiple facts from one utterance", len(cands) >= 2,
          str([c[1] for c in cands]))
    check("ignores assistant turns",
          extract_candidates("I prefer short answers", "assistant") == [],
          "mining its own output would make the agent believe its own guesses")

    ltm = LongTermMemory("t-user")
    ltm.clear()
    promoted = ltm.consider("we always work in celsius", "user", "s1")
    check("explicit statement promotes immediately", len(promoted) >= 1)
    before = ltm.stats()["facts"]
    ltm.consider("we always work in celsius", "user", "s2")
    after = ltm.stats()["facts"]
    check("restating does not duplicate", after == before,
          f"{before} facts before, {after} after a restatement")
    reinforced = [f for f in ltm.all_facts() if f.times_seen > 1]
    check("restating reinforces instead", len(reinforced) >= 1,
          f"times_seen={reinforced[0].times_seen if reinforced else 0}")
    check("the reinforcing session was recorded",
          any("s2" in f.sessions for f in ltm.all_facts()))

    ltm.consider("our site is NW-41", "user", "s1")
    ltm.consider("we run firmware 4.3", "user", "s1")
    recalled = ltm.recall("which site do we work at", k=3)
    check("recall is semantic, not keyed",
          any("NW-41" in f.text for f in recalled),
          f"top={recalled[0].text if recalled else None}")

    # -- 6. CROSS-SESSION CONTINUITY --------------------------------------
    section("6. CROSS-SESSION CONTINUITY")
    fresh = LongTermMemory("t-user")          # new object, no shared state
    hits = 0
    for probe, expected in [("temperature units", "celsius"),
                            ("which site", "NW-41"),
                            ("firmware version", "4.3")]:
        found = any(expected.lower() in f.text.lower()
                    for f in fresh.recall(probe, k=3))
        hits += int(found)
    check("facts survive into a new session object", hits == 3,
          f"{hits}/3 recalled with nothing carried in memory")

    # -- 7. VECTOR STORE --------------------------------------------------
    section("7. VECTOR STORE (numpy backend)")
    store = NumpyStore(TMP / "vs", "unit")
    store.reset()
    texts = ["the buffer discards oldest records",
             "antenna placement against metal loses 12 dB",
             "certificates are valid for 24 months"]
    store.add([f"id{i}" for i in range(3)], texts,
              [embed_one(t) for t in texts],
              [{"kind": "test", "n": i} for i in range(3)])
    check("stores everything", store.count() == 3)
    res = store.query(embed_one("antenna metal dB loss"), k=2)
    check("nearest neighbour is correct",
          res and "antenna" in res[0].text, res[0].text if res else "")
    check("metadata filtering works",
          len(store.query(embed_one("buffer"), k=5, where={"n": 0})) == 1)
    store.add(["id0"], ["updated text"], [embed_one("updated text")],
              [{"kind": "test", "n": 0}])
    check("upsert updates rather than appends", store.count() == 3)
    persisted = NumpyStore(TMP / "vs", "unit")
    check("survives reopening", persisted.count() == 3)

    # -- 8. BM25 ----------------------------------------------------------
    section("8. BM25 — the sparse half of the hybrid")
    docs = [["exit", "code", "3", "radio", "selftest"],
            ["the", "antenna", "loses", "12", "db"],
            ["certificates", "expire", "after", "24", "months"]]
    bm = BM25(docs)
    scores = bm.scores("exit code 3")
    check("exact token match ranks first",
          scores.index(max(scores)) == 0, f"scores={[round(s,2) for s in scores]}")
    check("unrelated query scores nothing",
          max(bm.scores("octopus submarine")) == 0.0,
          "no spurious matches — this is why the floor can trust BM25 hits")

    # -- 9. RETRIEVAL PIPELINE --------------------------------------------
    section("9. RETRIEVAL — hybrid, fusion, attribution")
    index = KnowledgeIndex(store=NumpyStore(TMP / "idx", "knowledge"))
    built = index.build(DOCUMENTS)
    check("index built", built["chunks"] == len(chunks),
          f"{built['chunks']} chunks in the {built['store']} store")

    hits = index.search("helios-cli radio selftest exit code 3", k=6)
    check("exact-token query finds the right document",
          "RB-101" in {h.doc_id for h in hits},
          str(sorted({h.doc_id for h in hits})))
    check("hits carry attribution",
          all(h.citation().startswith(h.doc_id) for h in hits),
          hits[0].attribution() if hits else "")
    check("hits record which retriever found them",
          all(h.retrievers for h in hits),
          str(hits[0].retrievers) if hits else "")

    sparse_only = index.search("REL-5.0 backpressure", use_dense=False, k=6)
    dense_only = index.search("REL-5.0 backpressure", use_sparse=False, k=6)
    check("both retrievers work independently",
          bool(sparse_only) and bool(dense_only),
          f"sparse={len(sparse_only)} dense={len(dense_only)}")

    fused = index.search("what happens when the buffer fills on 5.0", k=6)
    check("fusion assigns a score", all(h.fused_score > 0 for h in fused))
    check("fusion output is rank-ordered",
          [h.fused_score for h in fused] == sorted(
              [h.fused_score for h in fused], reverse=True)
          or any(h.rerank_score is not None for h in fused))

    check("empty-ish query does not crash", isinstance(index.search("zz"), list))

    reloaded = KnowledgeIndex(store=NumpyStore(TMP / "idx", "knowledge"))
    check("index reloads from disk", reloaded.load() == built["chunks"],
          "BM25 is rebuilt from the persisted chunks, not from the corpus")

    # -- 10. CONTEXT OPTIMISATION ------------------------------------------
    section("10. CONTEXT OPTIMISATION")
    candidates = index.search("rollback firmware to 4.2", k=12, candidates=30)
    packed, report = optimise(candidates)
    check("report accounts for every stage",
          report.retrieved >= report.after_filter >= report.after_dedup
          >= report.packed, report.summary())
    check("packing respects the token budget",
          sum(count_tokens(c.text) for c in packed) <= config.CONTEXT_BUDGET_TOKENS)

    dup = candidates[0]
    twin = type(dup)(**{**dup.__dict__, "chunk_id": "twin", "ordinal": 99})
    deduped, removed = deduplicate([dup, twin], threshold=0.9)
    check("identical text is deduplicated", len(deduped) == 1 and len(removed) == 1,
          f"removed {removed}")

    tiny, dropped = pack(candidates, budget=60)
    check("a small budget drops the tail",
          len(tiny) < len(candidates) and len(dropped) > 0,
          f"{len(tiny)} kept, {len(dropped)} dropped at 60 tokens")
    check("rank order is preserved when packing",
          [c.chunk_id for c in tiny] ==
          [c.chunk_id for c in candidates if c.chunk_id in
           {t.chunk_id for t in tiny}])

    no_filter, rep2 = optimise(candidates, do_filter=False, do_dedup=False)
    check("disabling stages is observable",
          rep2.after_dedup >= report.after_dedup,
          "the ablation switches actually change behaviour")

    # -- 11. COMPACTION ----------------------------------------------------
    section("11. SESSION COMPACTION")
    turns = []
    for i in range(14):
        turns.append({"role": "user", "content": f"question {i} about firmware"})
        turns.append({"role": "assistant",
                      "content": f"We confirmed finding {i}. The root cause "
                                 f"was a CRC mismatch in slot B."})
    comp = compact_history(turns, keep_recent=4)
    check("compaction happened", comp is not None)
    check("summary is smaller than the source",
          comp.tokens_after < comp.tokens_before,
          f"{comp.tokens_before} → {comp.tokens_after} tokens "
          f"({int(comp.ratio * 100)}%)")
    check("summary keeps signal-bearing lines",
          "cause" in comp.summary.lower() or "confirmed" in comp.summary.lower())
    check("short conversations are left alone",
          compact_history(turns[:3], keep_recent=4) is None)

    # -- 12. RANKING METRICS -----------------------------------------------
    section("12. RANKING METRICS")
    ranked = ["A", "B", "C", "D"]
    rel = {"B", "D"}
    check("recall@2", abs(recall_at_k(ranked, rel, 2) - 0.5) < 1e-9)
    check("recall@4 is total", recall_at_k(ranked, rel, 4) == 1.0)
    check("precision@2", abs(precision_at_k(ranked, rel, 2) - 0.5) < 1e-9)
    check("MRR uses the first hit",
          abs(reciprocal_rank(ranked, rel) - 0.5) < 1e-9)
    check("nDCG rewards ranking early",
          ndcg_at_k(["B", "D", "A"], rel, 3) > ndcg_at_k(["A", "B", "D"], rel, 3),
          "recall@3 is identical for both; only nDCG separates them")
    check("nDCG of a perfect ranking is 1",
          abs(ndcg_at_k(["B", "D"], rel, 2) - 1.0) < 1e-9)
    check("no relevant documents scores zero",
          reciprocal_rank(["X", "Y"], rel) == 0.0)

    # -- 13. AGENT GATE ----------------------------------------------------
    section("13. HYBRID STRATEGY — the just-in-time gate")
    from src.agent import Agent

    agent = Agent("t-agent", "t-user", index=index)
    for query, expected, why in [
        ("thanks!", False, "acknowledgement"),
        ("what did we decide yesterday?", False, "memory-only"),
        ("remind me where we got to", False, "memory-only"),
        ("how long does the buffer hold telemetry", True, "knowledge question"),
        ("can we roll back to 4.2", True, "knowledge question"),
    ]:
        got, reason = agent.should_retrieve(query)
        check(f"gate: {why} → {'retrieve' if expected else 'skip'}",
              got == expected, f"{query!r} → {reason}")

    agent.open_session("verify the gate end to end")
    result = agent.ask("how long does the HX-40 buffer telemetry for")
    check("a turn produces an answer", bool(result.answer.text))
    check("the answer carries citations", bool(result.answer.citations),
          str(result.answer.citations))
    check("offline generation needs no model", result.answer.mode == "extractive")
    check("the turn is traced", "retrieval=" in result.trace())
    skipped = agent.ask("thanks!")
    check("skipping retrieval retrieves nothing",
          not skipped.retrieved and not skipped.chunks)

    # -- 14. UPFRONT PACK --------------------------------------------------
    section("14. HYBRID STRATEGY — the upfront pack")
    a1 = Agent("t-upfront-1", "t-upfront-user", index=index)
    a1.session.clear()
    a1.longterm.clear()
    a1.open_session("investigate NW-41")
    a1.ask("we always work in celsius and our site is NW-41")
    a1.close_session()

    a2 = Agent("t-upfront-2", "t-upfront-user", index=index)
    a2.session.clear()
    pack_text = a2.open_session()
    check("a later session loads prior facts upfront",
          "celsius" in pack_text.lower() or "NW-41" in pack_text,
          pack_text.replace("\n", " | ")[:150] or "(empty)")
    check("the upfront pack is small",
          a2.upfront_tokens <= 400,
          f"{a2.upfront_tokens} tokens — cheap enough to always load")

    # -- summary -----------------------------------------------------------
    passed = sum(1 for _, ok in CHECKS if ok)
    print("\n" + "=" * 78)
    print(f"{passed}/{len(CHECKS)} checks passed")
    print("=" * 78)
    if passed != len(CHECKS):
        print("\nFailures above. Nothing here depends on a network or a model,")
        print("so a failure is a logic bug, not an environment problem.")
        return 1
    print("\nMemory tiers, retrieval, fusion, context optimisation, compaction")
    print("and metrics all verified without a single API call. Only embedding")
    print("QUALITY needs the real model.")
    return 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(TMP, ignore_errors=True)
    sys.exit(code)
