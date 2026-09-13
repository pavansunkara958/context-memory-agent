"""Tier 3 — long-term memory.

Scope: everything the agent has learned about this user and their estate.
Lifetime: indefinite. Addressed by meaning, not by key.

Two collections live in the vector store:

  knowledge   The document corpus. Written once at index time, read constantly.
  episodic    Facts learned *from conversations* — preferences, site details,
              decisions, constraints. Written by promotion, read by recall.

**Promotion is the whole design.** Writing every utterance into long-term
memory produces a store that grows linearly with conversation and gets less
useful as it grows, because recall then competes against thousands of
throwaway lines. A fact earns its place one of two ways:

  1. It was *stated* durably — "we always work in Celsius", "our site is 41".
     Explicit, high confidence, promoted on first sight.
  2. It was *observed repeatedly* across separate sessions. One mention is an
     accident; the same thing in two conversations is a pattern.

Re-observation reinforces rather than duplicates: `times_seen` increments and
the source session is recorded. That gives recall a confidence signal and keeps
the store small enough to stay sharp.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field

from ..embeddings import cosine, embed, embed_one
from ..vectorstore import open_store

EPISODIC_COLLECTION = "episodic"
REINFORCE_THRESHOLD = 0.94   # above this, a "new" fact is the same fact
PROMOTE_CONFIDENCE = 0.80    # explicit statements clear this on first sight
PROMOTE_SIGHTINGS = 2        # otherwise it must recur across sessions


@dataclass
class Fact:
    fact_id: str
    text: str
    kind: str                 # preference | entity | constraint | decision
    user_id: str
    confidence: float
    times_seen: int = 1
    sessions: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    score: float = 0.0        # populated by recall()

    def attribution(self) -> str:
        seen = f"seen {self.times_seen}×" if self.times_seen > 1 else "stated once"
        return f"[memory:{self.kind}] {self.text}  ({seen})"


def _fact_id(user_id: str, text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.blake2b(f"{user_id}|{norm}".encode(),
                           digest_size=8).hexdigest()


# ---------------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------------
# Deterministic, auditable, and free. An LLM extractor would catch more, but it
# would also make the memory store non-reproducible run to run — and a memory
# system you cannot replay is one you cannot debug when it remembers something
# wrong. Patterns first; a model can be layered on later without changing the
# promotion logic.
_PATTERNS: list[tuple[str, str, float]] = [
    (r"\b(?:i|we)\s+(?:prefer|like|want)\s+(.{4,90})", "preference", 0.85),
    (r"\b(?:always|never)\s+(.{4,90})", "preference", 0.85),
    (r"\b(?:keep it|answer|reply)\s+(short|terse|brief|detailed)\b",
     "preference", 0.85),
    (r"\b(?:use|using|in)\s+(celsius|fahrenheit|metric|imperial)\b",
     "preference", 0.85),
    (r"\b(?:our|my|the)\s+site\s+is\s+([A-Za-z0-9\- ]{2,40})", "entity", 0.9),
    (r"\bsite\s+([A-Z]{2,4}-?\d{1,4})\b", "entity", 0.8),
    (r"\b(?:we|our fleet|the fleet)\s+(?:run|runs|is on|are on)\s+"
     r"(?:firmware\s+)?(\d+\.\d+(?:\.\d+)?)", "entity", 0.9),
    (r"\b(?:we have|we manage|we operate)\s+([\d,]+\s+\w{3,20})", "entity", 0.85),
    (r"\b(?:we|i)\s+(?:decided|agreed|will)\s+(.{6,90})", "decision", 0.8),
    (r"\b(?:cannot|can't|must not|not allowed to)\s+(.{4,90})",
     "constraint", 0.8),
]


# A greedy `.{4,90}` capture runs straight through conjunctions and swallows
# the next clause: "we always work in celsius and our site is NW-41" becomes a
# single malformed preference. Clipping at the first clause boundary keeps one
# fact per fact, which matters because facts are deduplicated and recalled
# individually — a fused pair matches neither probe well.
_CLAUSE_BREAK = re.compile(
    r"(?:[.!?;,]\s+)|(?:\s+(?:and|but|so|then|because|while)\s+)", re.I)


def _clip_clause(text: str) -> str:
    return _CLAUSE_BREAK.split(text, maxsplit=1)[0].strip(" .,;:")


def extract_candidates(text: str, role: str = "user") -> list[tuple[str, str, float]]:
    """Return (kind, fact_text, confidence) candidates from one utterance.

    Only user turns are mined. Mining the assistant's own output is how a
    memory store fills with the model's earlier guesses and then treats them as
    established fact on the next retrieval — a feedback loop that is very hard
    to notice and very hard to unwind.
    """
    if role != "user" or not text:
        return []

    out: list[tuple[str, str, float]] = []
    lowered = " ".join(text.split())
    for pattern, kind, conf in _PATTERNS:
        for m in re.finditer(pattern, lowered, flags=re.I):
            captured = _clip_clause(m.group(0))
            if 6 <= len(captured) <= 120:
                out.append((kind, captured, conf))
    # De-duplicate within the utterance, keeping the highest confidence.
    best: dict[str, tuple[str, str, float]] = {}
    for kind, t, c in out:
        key = t.lower()
        if key not in best or c > best[key][2]:
            best[key] = (kind, t, c)

    # Drop subsumed candidates. Overlapping patterns fire on the same span at
    # different widths — "always work in celsius" and "in celsius" are one
    # fact, and storing both gives recall two near-identical things to choose
    # between while the shorter one carries less meaning.
    kept = []
    for kind, t, c in best.values():
        low = t.lower()
        if any(low != other.lower() and low in other.lower()
               for _k, other, _c in best.values()):
            continue
        kept.append((kind, t, c))
    return kept


# ---------------------------------------------------------------------------
class LongTermMemory:
    def __init__(self, user_id: str = "default", store=None):
        self.user_id = user_id
        self.store = store or open_store(EPISODIC_COLLECTION)
        self._pending: dict[str, dict] = {}   # candidates awaiting a 2nd sighting

    # -- write ------------------------------------------------------------
    def _existing(self) -> list[Fact]:
        facts = []
        for h in self.store.get_all():
            m = h.meta
            if m.get("user_id") != self.user_id:
                continue
            facts.append(Fact(
                fact_id=h.id, text=h.text, kind=m.get("kind", "entity"),
                user_id=m.get("user_id", ""),
                confidence=float(m.get("confidence", 0.5)),
                times_seen=int(m.get("times_seen", 1)),
                sessions=json.loads(m.get("sessions", "[]")),
                created_at=float(m.get("created_at", 0)),
                updated_at=float(m.get("updated_at", 0)),
            ))
        return facts

    def remember(self, text: str, kind: str, confidence: float,
                 session_id: str) -> tuple[Fact, bool]:
        """Store a fact, or reinforce the existing one it duplicates.

        Returns (fact, created). Reinforcement is what keeps the store from
        accumulating five phrasings of the same preference.
        """
        vec = embed_one(text)
        for fact in self._existing():
            if cosine(vec, embed_one(fact.text)) >= REINFORCE_THRESHOLD:
                fact.times_seen += 1
                if session_id not in fact.sessions:
                    fact.sessions.append(session_id)
                fact.confidence = min(0.99, fact.confidence + 0.05)
                fact.updated_at = time.time()
                self._write(fact, embed_one(fact.text))
                return fact, False

        fact = Fact(fact_id=_fact_id(self.user_id, text), text=text, kind=kind,
                    user_id=self.user_id, confidence=confidence,
                    sessions=[session_id])
        self._write(fact, vec)
        return fact, True

    def _write(self, fact: Fact, vec) -> None:
        self.store.add(
            ids=[fact.fact_id],
            texts=[fact.text],
            embeddings=[vec],
            metadatas=[{
                "user_id": fact.user_id, "kind": fact.kind,
                "confidence": fact.confidence, "times_seen": fact.times_seen,
                "sessions": json.dumps(fact.sessions),
                "created_at": fact.created_at, "updated_at": fact.updated_at,
            }],
        )

    # -- promotion --------------------------------------------------------
    def consider(self, text: str, role: str, session_id: str) -> list[Fact]:
        """Mine an utterance and promote whatever qualifies.

        High-confidence explicit statements promote immediately. Everything
        else waits for a second sighting in a different session — the cheapest
        possible defence against promoting a one-off remark into a permanent
        belief about the user.
        """
        promoted: list[Fact] = []
        for kind, candidate, confidence in extract_candidates(text, role):
            key = " ".join(candidate.lower().split())
            if confidence >= PROMOTE_CONFIDENCE:
                fact, _ = self.remember(candidate, kind, confidence, session_id)
                promoted.append(fact)
                continue

            seen = self._pending.setdefault(
                key, {"kind": kind, "text": candidate, "sessions": set()})
            seen["sessions"].add(session_id)
            if len(seen["sessions"]) >= PROMOTE_SIGHTINGS:
                fact, _ = self.remember(candidate, kind, 0.7, session_id)
                promoted.append(fact)
                self._pending.pop(key, None)
        return promoted

    def promote_session(self, session) -> list[Fact]:
        """Run promotion over a whole stored conversation."""
        promoted = []
        for turn in session.history():
            promoted.extend(
                self.consider(turn.content, turn.role, session.session_id))
        return promoted

    # -- read -------------------------------------------------------------
    def recall(self, query: str, k: int = 5,
               min_confidence: float = 0.0) -> list[Fact]:
        hits = self.store.query(embed_one(query), k=k * 3,
                                where={"user_id": self.user_id})
        facts: list[Fact] = []
        for h in hits:
            m = h.meta
            conf = float(m.get("confidence", 0.5))
            if conf < min_confidence:
                continue
            f = Fact(fact_id=h.id, text=h.text, kind=m.get("kind", "entity"),
                     user_id=m.get("user_id", ""), confidence=conf,
                     times_seen=int(m.get("times_seen", 1)),
                     sessions=json.loads(m.get("sessions", "[]")))
            f.score = h.score
            facts.append(f)
        # Rank by similarity, but let repeatedly-confirmed facts edge ahead of
        # equally-similar one-offs.
        facts.sort(key=lambda f: -(f.score + 0.02 * min(f.times_seen, 5)))
        return facts[:k]

    def all_facts(self) -> list[Fact]:
        return sorted(self._existing(), key=lambda f: -f.updated_at)

    def clear(self) -> None:
        self.store.reset()
        self._pending.clear()

    def stats(self) -> dict:
        facts = self._existing()
        kinds: dict[str, int] = {}
        for f in facts:
            kinds[f.kind] = kinds.get(f.kind, 0) + 1
        return {"facts": len(facts), "by_kind": kinds,
                "pending": len(self._pending)}
