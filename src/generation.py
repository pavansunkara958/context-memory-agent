"""Answer generation — hosted model, or extractive offline.

The offline answerer is not a placeholder. It selects the sentences from the
retrieved context that best match the query and returns them with their
citations. That is a worse *answer* than a model would write, but it is a
faithful readout of what retrieval actually supplied — which makes it a useful
diagnostic. If the offline answer is wrong, retrieval is wrong, and no amount
of model quality would have rescued it.

It also means the entire system is demonstrable end to end with no API key.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .context import render
from .embeddings import tokenize
from .retrieval import Retrieved

SYSTEM_PROMPT = """You are a field-engineering assistant for the Helios IoT platform.

Answer only from the CONTEXT and MEMORY provided. If they do not contain the
answer, say so plainly and state what you would need — do not fill the gap from
general knowledge.

Cite every factual claim with the bracketed id of the chunk it came from, like
[RB-105#1]. A sentence that states a fact without a citation is a defect.

Version matters in this product: behaviour differs between firmware 4.x and
5.x. If an answer depends on version, say which version it holds for.

Respect stated user preferences in MEMORY. Be concise and concrete."""


@dataclass
class Answer:
    text: str
    citations: list[str]
    mode: str          # llm | extractive
    model: str | None = None


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 25]


def extractive_answer(query: str, chunks: list[Retrieved],
                      noted: list[str] | None = None) -> Answer:
    """Rank sentences from the retrieved chunks by term overlap with the query.

    `noted` is what was just written to memory, and it is used ONLY when there
    is nothing to retrieve — i.e. the user stated facts rather than asking a
    question. It is deliberately not appended to real answers: the earlier
    version pasted the whole memory preamble onto the end of every response,
    which made the agent's internal state part of its output.
    """
    if not chunks:
        if noted:
            return Answer(
                "Noted — I'll remember: " + "; ".join(noted) + ".",
                [], "extractive")
        return Answer(
            "I don't have anything in the knowledge base that covers that. "
            "Try naming the component, the error, or the firmware version.",
            [], "extractive")

    q_terms = set(tokenize(query))
    scored: list[tuple[float, int, str, str]] = []
    for rank, ch in enumerate(chunks):
        for sent in _sentences(ch.text):
            overlap = len(q_terms & set(tokenize(sent)))
            if not overlap:
                continue
            # Favour overlap, discount by rank so a weak sentence from the top
            # chunk does not outrank a strong one from the second.
            score = overlap / (1 + 0.35 * rank)
            scored.append((score, overlap, sent, ch.citation()))

    # A single shared word is usually a stopword-ish coincidence — "them",
    # "the unit". If anything matched on two or more terms, drop the
    # one-word matches entirely rather than letting them pad the answer.
    if any(o >= 2 for _s, o, _t, _c in scored):
        scored = [row for row in scored if row[1] >= 2]

    scored.sort(key=lambda x: -x[0])
    if not scored:
        top = chunks[0]
        return Answer(f"{_sentences(top.text)[0] if _sentences(top.text) else top.text} "
                      f"[{top.citation()}]", [top.citation()], "extractive")

    lines, used, seen = [], [], set()
    for _score, _overlap, sent, cite in scored[:4]:
        if sent in seen:
            continue
        seen.add(sent)
        lines.append(f"{sent} [{cite}]")
        if cite not in used:
            used.append(cite)

    return Answer(" ".join(lines), used, "extractive")


def generate(query: str, chunks: list[Retrieved], memory_lines: list[str],
             history: list[dict], client=None, model: str | None = None,
             noted: list[str] | None = None) -> Answer:
    if client is None or model is None:
        return extractive_answer(query, chunks, noted)

    context_block = render(chunks) or "(no documents retrieved)"
    memory_block = "\n".join(memory_lines) or "(nothing remembered yet)"

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history[-6:])
    messages.append({
        "role": "user",
        "content": (f"MEMORY:\n{memory_block}\n\n"
                    f"CONTEXT:\n{context_block}\n\n"
                    f"QUESTION: {query}"),
    })

    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        # Fall back rather than fail the turn. A generation outage should
        # degrade the answer, not lose the conversation and its memory writes.
        fallback = extractive_answer(query, chunks, noted)
        fallback.text = f"[generation unavailable: {type(exc).__name__}]\n{fallback.text}"
        return fallback

    cites = sorted(set(re.findall(r"\[([A-Z]{2,4}-[\w.]+#\d+)\]", text)))
    return Answer(text, cites, "llm", model)
