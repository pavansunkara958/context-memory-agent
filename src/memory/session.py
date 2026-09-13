"""Tier 2 — session memory.

Scope: one conversation. Lifetime: durable, survives process restarts.

Holds three things the brief asks for — conversation history, user preferences,
and task progress — plus the compaction summaries that keep a long session
inside its budget.

SQLite rather than a JSON file because sessions are written turn by turn and
read concurrently by the CLI and the evaluation harness; a file rewritten on
every turn corrupts the moment two things touch it.

Note what is *not* here: nothing is stored by embedding. Session memory is
addressed by key — this conversation, its turns in order — because within a
conversation you want recency and completeness, not similarity. Semantic search
belongs one tier up, where the question is "have I ever learned this?" rather
than "what did we just say?".
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from ..config import SESSION_DB, ensure_dirs

SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    user_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    citations   TEXT DEFAULT '[]',
    ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_turns_session ON turns(session_id, id);

CREATE TABLE IF NOT EXISTS preferences (
    user_id     TEXT NOT NULL,
    key         TEXT NOT NULL,
    value       TEXT NOT NULL,
    session_id  TEXT,
    ts          REAL NOT NULL,
    PRIMARY KEY (user_id, key)
);

CREATE TABLE IF NOT EXISTS progress (
    session_id  TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    task        TEXT DEFAULT '',
    status      TEXT DEFAULT 'open',
    findings    TEXT DEFAULT '[]',
    next_steps  TEXT DEFAULT '[]',
    ts          REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    summary     TEXT NOT NULL,
    turns       INTEGER NOT NULL,
    ts          REAL NOT NULL
);
"""


@dataclass
class Turn:
    role: str
    content: str
    citations: list[str]
    ts: float


class SessionMemory:
    def __init__(self, session_id: str, user_id: str = "default",
                 db_path: Path | None = None):
        ensure_dirs()
        self.session_id = session_id
        self.user_id = user_id
        self.db_path = Path(db_path or SESSION_DB)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    # -- history ----------------------------------------------------------
    def append(self, role: str, content: str,
               citations: list[str] | None = None) -> None:
        self.conn.execute(
            "INSERT INTO turns (session_id, user_id, role, content, citations, ts)"
            " VALUES (?,?,?,?,?,?)",
            (self.session_id, self.user_id, role, content,
             json.dumps(citations or []), time.time()),
        )
        self.conn.commit()

    def history(self, limit: int | None = None) -> list[Turn]:
        q = ("SELECT role, content, citations, ts FROM turns "
             "WHERE session_id = ? ORDER BY id")
        rows = self.conn.execute(q, (self.session_id,)).fetchall()
        turns = [Turn(r["role"], r["content"], json.loads(r["citations"]), r["ts"])
                 for r in rows]
        return turns[-limit:] if limit else turns

    def turn_count(self) -> int:
        r = self.conn.execute(
            "SELECT COUNT(*) c FROM turns WHERE session_id = ?",
            (self.session_id,)).fetchone()
        return int(r["c"])

    # -- preferences ------------------------------------------------------
    def set_preference(self, key: str, value: str) -> None:
        """Preferences are keyed on the USER, not the session.

        That is the point of them: "metric units", "terse answers", "site 41"
        should hold in tomorrow's conversation too. A preference scoped to a
        session is just a variable."""
        self.conn.execute(
            "INSERT INTO preferences (user_id, key, value, session_id, ts) "
            "VALUES (?,?,?,?,?) ON CONFLICT(user_id, key) DO UPDATE SET "
            "value=excluded.value, session_id=excluded.session_id, ts=excluded.ts",
            (self.user_id, key, value, self.session_id, time.time()),
        )
        self.conn.commit()

    def preferences(self) -> dict[str, str]:
        rows = self.conn.execute(
            "SELECT key, value FROM preferences WHERE user_id = ?",
            (self.user_id,)).fetchall()
        return {r["key"]: r["value"] for r in rows}

    # -- task progress ----------------------------------------------------
    def set_progress(self, task: str | None = None, status: str | None = None,
                     findings: list[str] | None = None,
                     next_steps: list[str] | None = None) -> None:
        current = self.progress()
        merged = {
            "task": task if task is not None else current["task"],
            "status": status if status is not None else current["status"],
            "findings": findings if findings is not None else current["findings"],
            "next_steps": (next_steps if next_steps is not None
                           else current["next_steps"]),
        }
        self.conn.execute(
            "INSERT INTO progress (session_id, user_id, task, status, findings,"
            " next_steps, ts) VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET task=excluded.task, "
            "status=excluded.status, findings=excluded.findings, "
            "next_steps=excluded.next_steps, ts=excluded.ts",
            (self.session_id, self.user_id, merged["task"], merged["status"],
             json.dumps(merged["findings"]), json.dumps(merged["next_steps"]),
             time.time()),
        )
        self.conn.commit()

    def add_finding(self, finding: str) -> None:
        findings = self.progress()["findings"]
        if finding not in findings:
            findings.append(finding)
        self.set_progress(findings=findings)

    def progress(self) -> dict:
        r = self.conn.execute(
            "SELECT task, status, findings, next_steps FROM progress "
            "WHERE session_id = ?", (self.session_id,)).fetchone()
        if not r:
            return {"task": "", "status": "open", "findings": [], "next_steps": []}
        return {"task": r["task"], "status": r["status"],
                "findings": json.loads(r["findings"]),
                "next_steps": json.loads(r["next_steps"])}

    # -- compaction summaries ---------------------------------------------
    def add_summary(self, summary: str, turns: int) -> None:
        self.conn.execute(
            "INSERT INTO summaries (session_id, summary, turns, ts) VALUES (?,?,?,?)",
            (self.session_id, summary, turns, time.time()),
        )
        self.conn.commit()

    def summaries(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT summary FROM summaries WHERE session_id = ? ORDER BY id",
            (self.session_id,)).fetchall()
        return [r["summary"] for r in rows]

    def last_progress(self, exclude_current: bool = True) -> dict | None:
        """The most recent task state for this USER, from any session.

        Task progress is recorded per session, but a task does not end when a
        conversation does — an investigation spans days. Without this, day 2
        opens knowing the user's preferences but not what they were working on,
        which is the least useful half of continuity.
        """
        q = ("SELECT session_id, task, status, findings, next_steps FROM progress"
             " WHERE user_id = ? AND task != ''")
        params: list = [self.user_id]
        if exclude_current:
            q += " AND session_id != ?"
            params.append(self.session_id)
        q += " ORDER BY ts DESC LIMIT 1"
        r = self.conn.execute(q, params).fetchone()
        if not r:
            return None
        return {"session_id": r["session_id"], "task": r["task"],
                "status": r["status"], "findings": json.loads(r["findings"]),
                "next_steps": json.loads(r["next_steps"])}

    # -- cross-session ----------------------------------------------------
    def sessions_for_user(self) -> list[str]:
        rows = self.conn.execute(
            "SELECT DISTINCT session_id FROM turns WHERE user_id = ? ORDER BY id",
            (self.user_id,)).fetchall()
        return [r["session_id"] for r in rows]

    def clear(self) -> None:
        """Clear this session. Preferences survive deliberately — they belong
        to the user, not the conversation, and deleting a session should not
        make the agent forget how the person likes to be spoken to."""
        for table in ("turns", "progress", "summaries"):
            self.conn.execute(f"DELETE FROM {table} WHERE session_id = ?",
                              (self.session_id,))
        self.conn.commit()

    def clear_user(self) -> None:
        """Forget the user entirely, preferences included.

        Needed for `demo --fresh` and for any evaluation that claims to start
        from nothing: a run that inherits yesterday's preferences proves
        continuity it did not actually demonstrate. It is also the honest
        implementation of a deletion request.
        """
        self.conn.execute("DELETE FROM preferences WHERE user_id = ?",
                          (self.user_id,))
        for table in ("turns", "progress", "summaries"):
            self.conn.execute(
                f"DELETE FROM {table} WHERE session_id IN "
                f"(SELECT DISTINCT session_id FROM turns WHERE user_id = ?)",
                (self.user_id,))
        self.conn.execute("DELETE FROM turns WHERE user_id = ?", (self.user_id,))
        self.conn.commit()

    def stats(self) -> dict:
        return {"session_id": self.session_id, "turns": self.turn_count(),
                "preferences": len(self.preferences()),
                "summaries": len(self.summaries())}
