"""Four memory tiers, separated by lifetime and by what they are for.

  WORKING     One task. Volatile, bounded, evicting. Holds the current
              workflow state and recent tool output.
  SESSION     One conversation. Durable across restarts. Holds history,
              stated preferences and task progress.
  LONG-TERM   Across conversations. Vector-indexed. Holds facts promoted out
              of sessions because they proved durable, plus the knowledge base.
  RETRIEVAL   Not storage at all — the path that pulls from long-term memory
              into the context window on demand.

The tiers are not three databases with different names. What separates them is
**what causes a write and what causes a read**. Working memory is written by
the current step and read by the next one. Session memory is written by a turn
and read at the start of the next turn in that conversation. Long-term memory
is written only by *promotion* — something observed often enough, or declared
durable enough, to outlive the conversation that produced it — and read by
semantic search, not by key.

Without promotion, "long-term memory" is just a session log with a vector
index, and it grows without bound while getting less useful.
"""
from .longterm import Fact, LongTermMemory
from .session import SessionMemory
from .working import WorkingMemory

__all__ = ["WorkingMemory", "SessionMemory", "LongTermMemory", "Fact"]
