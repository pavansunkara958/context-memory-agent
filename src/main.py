"""CLI.

  python -m src.main index                 build the knowledge index
  python -m src.main demo                  three-session continuity walkthrough
  python -m src.main eval                  full evaluation + ablation
  python -m src.main chat SESSION_ID       interactive, resumable
  python -m src.main memory                dump what the agent remembers
  python -m src.main search "query"        inspect the retrieval pipeline
"""
from __future__ import annotations

import argparse
import json
import sys

from .agent import Agent
from .config import ROOT, describe, ensure_dirs
from .corpus import DOCUMENTS, corpus_stats
from .embeddings import backend_name, reranker_name
from .evaluate import full_report, print_report
from .memory import LongTermMemory, SessionMemory
from .retrieval import KnowledgeIndex

DEMO_USER = "eng-dana"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------------------
def cmd_index(args) -> None:
    ensure_dirs()
    rule("BUILDING KNOWLEDGE INDEX")
    print(f"  {describe()}")
    index = KnowledgeIndex()
    stats = index.build(DOCUMENTS)
    print(f"  corpus     : {corpus_stats()}")
    print(f"  chunks     : {stats['chunks']}")
    print(f"  store      : {stats['store']}")
    print(f"  embedder   : {stats['embedder']}")
    print(f"  reranker   : {reranker_name()}")
    if stats["embedder"] == "hash":
        print("\n  NOTE: running on the hash embedder — lexical, not semantic.")
        print("  `pip install sentence-transformers` for real semantic retrieval.")


def cmd_search(args) -> None:
    index = KnowledgeIndex()
    index.load()
    if index.size == 0:
        raise SystemExit("Index is empty. Run:  python -m src.main index")

    from .context import optimise

    rule(f"SEARCH — {args.query!r}")
    hits = index.search(args.query, k=args.k)
    packed, report = optimise(hits)
    print(f"  {report.summary()}\n")
    for i, h in enumerate(packed, start=1):
        rr = f"rerank={h.rerank_score:.3f}  " if h.rerank_score is not None else ""
        print(f"  {i}. {h.attribution()}")
        print(f"     dense={h.dense_score:.3f}  bm25={h.bm25_score:.2f}  "
              f"{rr}via={'+'.join(h.retrievers)}")
        print(f"     {h.text[:170]}...\n")
    if report.dropped_duplicate:
        print("  removed as duplicates:")
        for dup, of in report.dropped_duplicate:
            print(f"    {dup} ≈ {of}")


# ---------------------------------------------------------------------------
DEMO_SESSIONS = [
    ("day-1", "Investigate why site NW-41 gateways stopped reporting", [
        "We always work in celsius and I prefer short answers. "
        "Our site is NW-41 and we run firmware 4.3.",
        "Half our gateways went quiet yesterday afternoon. What should I check first?",
        "The status light is solid amber on three of them and no uplinks at all.",
    ]),
    ("day-2", None, [
        "What did we establish yesterday?",
        "Ran the selftest on all three. Two returned exit code 3.",
        "Do those two need an RMA or can I fix them on site?",
    ]),
    ("day-3", None, [
        "Remind me where we got to.",
        "Replacements arrived. Before I fit them, we are also planning to move "
        "the fleet to 5.0 — anything that changes for buffering?",
        "Can we roll back to 4.2 if it goes badly?",
    ]),
]


def cmd_demo(args) -> None:
    index = KnowledgeIndex()
    index.load()
    if index.size == 0:
        raise SystemExit("Index is empty. Run:  python -m src.main index")

    if args.fresh:
        # Wipe long-term facts AND user preferences. Clearing only the session
        # turns would leave yesterday's preferences in place, and day 1 would
        # open with a populated context pack — appearing to demonstrate
        # continuity it had not yet earned.
        LongTermMemory(DEMO_USER).clear()
        SessionMemory("wipe", DEMO_USER).clear_user()
        for sid, _task, _ in DEMO_SESSIONS:
            SessionMemory(sid, DEMO_USER).clear()

    rule("MULTI-SESSION DEMO — three separate sessions, one continuing task")
    print(f"  {describe()}  user={DEMO_USER}")
    print("  Each session is a fresh process-level object. Anything that")
    print("  survives between them came out of long-term memory, not scrollback.")

    for session_id, task, messages in DEMO_SESSIONS:
        rule(f"SESSION {session_id}")
        agent = Agent(session_id, DEMO_USER, index=index)
        pack = agent.open_session(task)

        if pack:
            print("  ── upfront context pack "
                  f"({agent.upfront_tokens} tokens) ──")
            for line in pack.splitlines():
                print(f"    {line}")
        else:
            print("  ── upfront context pack: empty (first session, nothing known) ──")
        print()

        for message in messages:
            print(f"  engineer> {message}")
            result = agent.ask(message)
            print(f"\n  agent> {result.answer.text}\n")
            print(f"    {result.trace()}")
            if result.facts_promoted:
                for f in result.facts_promoted:
                    print(f"    + promoted to long-term: {f}")
            print()

        closing = agent.close_session(
            status="in progress",
            next_steps=["fit replacement radios", "confirm firmware plan"])
        print(f"  session closed → {json.dumps(closing['facts'])}")

    rule("WHAT PERSISTED")
    ltm = LongTermMemory(DEMO_USER)
    for fact in ltm.all_facts():
        print(f"  {fact.attribution()}  conf={fact.confidence:.2f}  "
              f"sessions={fact.sessions}")
    print("\n  Day 2 and day 3 opened with preferences, site, firmware and open")
    print("  task already loaded. Nothing was re-asked.")


# ---------------------------------------------------------------------------
def cmd_eval(args) -> None:
    report = full_report()
    print_report(report)
    out = ROOT / "logs" / "evaluation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\n  saved → {out.relative_to(ROOT)}")


def cmd_memory(args) -> None:
    agent = Agent(args.session or "inspect", args.user)
    rule("MEMORY REPORT")
    print(json.dumps(agent.memory_report(), indent=2))
    rule("LONG-TERM FACTS")
    for fact in agent.longterm.all_facts():
        print(f"  {fact.attribution()}  conf={fact.confidence:.2f}")


def cmd_chat(args) -> None:
    index = KnowledgeIndex()
    index.load()
    if index.size == 0:
        raise SystemExit("Index is empty. Run:  python -m src.main index")

    agent = Agent(args.session_id, args.user, index=index)
    pack = agent.open_session(args.task)
    rule(f"CHAT — session {args.session_id!r}  ({describe()})")
    if pack:
        print("  Loaded from memory:")
        for line in pack.splitlines():
            print(f"    {line}")
    print("  Ctrl-D to exit. Rerun with the same session id to resume.\n")

    while True:
        try:
            message = input("  you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not message:
            continue
        result = agent.ask(message)
        print(f"\n  agent> {result.answer.text}\n")
        if args.trace:
            print(f"    {result.trace()}\n")

    summary = agent.close_session()
    print(f"  closed. {summary['promoted']} facts promoted; "
          f"{summary['facts']['facts']} held in long-term memory.")


# ---------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description="Multi-tier memory & retrieval agent")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("index").set_defaults(func=cmd_index)

    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("-k", type=int, default=6)
    s.set_defaults(func=cmd_search)

    d = sub.add_parser("demo")
    d.add_argument("--fresh", action="store_true",
                   help="wipe memory first, so continuity is proven not assumed")
    d.set_defaults(func=cmd_demo)

    sub.add_parser("eval").set_defaults(func=cmd_eval)

    m = sub.add_parser("memory")
    m.add_argument("--user", default=DEMO_USER)
    m.add_argument("--session", default=None)
    m.set_defaults(func=cmd_memory)

    c = sub.add_parser("chat")
    c.add_argument("session_id")
    c.add_argument("--user", default=DEMO_USER)
    c.add_argument("--task", default=None)
    c.add_argument("--trace", action="store_true")
    c.set_defaults(func=cmd_chat)

    args = p.parse_args()
    if not getattr(args, "func", None):
        p.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
