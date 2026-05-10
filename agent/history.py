"""
agent/history.py — Conversation history management.

Responsibilities:
  - Store the full message list in insertion order
  - Enforce a rolling window so we never blow the context limit
  - Keep the system prompt + first user message always present
  - Provide a trimmed view that is sent to the LLM each turn

Rolling-window strategy
-----------------------
We keep:
  [system]  (injected at call time, not stored here)
  [turn 0 user]         ← original query — always kept
  [turn 1 assistant]    ← first agent response — always kept
  ... middle messages may be dropped when window is full ...
  [last N pairs]        ← always kept

When the window is full we drop the OLDEST middle pair (assistant + observation)
to make room. This preserves context while staying within limits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Role = Literal["user", "assistant"]


@dataclass
class Message:
    role: Role
    content: str

    def to_dict(self) -> dict:
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    """
    Manual message-array conversation history with rolling window.

    Args:
        max_messages: Maximum number of messages to send to the LLM.
                      When exceeded, oldest middle pairs are dropped.
                      Minimum sensible value is 4 (query, response, obs, response).
        always_keep_first: Keep the first user+assistant pair regardless of window.
    """

    def __init__(self, max_messages: int = 20, always_keep_first: bool = True):
        if max_messages < 4:
            raise ValueError("max_messages must be at least 4")
        self.max_messages = max_messages
        self.always_keep_first = always_keep_first
        self._messages: list[Message] = []

    # ------------------------------------------------------------------ #
    # Write
    # ------------------------------------------------------------------ #

    def add_user(self, content: str) -> None:
        self._messages.append(Message(role="user", content=content))

    def add_assistant(self, content: str) -> None:
        self._messages.append(Message(role="assistant", content=content))

    # ------------------------------------------------------------------ #
    # Read
    # ------------------------------------------------------------------ #

    def as_dicts(self) -> list[dict]:
        """
        Return the windowed message list as plain dicts for the API call.
        This is what gets passed to groq_api_call().
        """
        return [m.to_dict() for m in self._windowed()]

    def full_history(self) -> list[dict]:
        """Return every message ever added — for logging / debugging."""
        return [m.to_dict() for m in self._messages]

    def __len__(self) -> int:
        return len(self._messages)

    # ------------------------------------------------------------------ #
    # Rolling window logic
    # ------------------------------------------------------------------ #

    def _windowed(self) -> list[Message]:
        """
        Apply the rolling window and return the trimmed message list.

        Layout preserved:
          messages[0]   — first user message  (original query)  ALWAYS
          messages[1]   — first assistant msg (first response)  ALWAYS if always_keep_first
          ... dropped pairs when over limit ...
          messages[-N:] — most recent messages                  ALWAYS
        """
        msgs = self._messages

        if len(msgs) <= self.max_messages:
            return msgs

        if not self.always_keep_first:
            # Simple tail truncation
            return msgs[-self.max_messages :]

        # Keep first 2 messages + as many recent messages as fit
        pinned = msgs[:2]  # [query, first_response]
        tail_budget = self.max_messages - len(pinned)
        tail = msgs[2:]  # everything after the first pair

        if len(tail) <= tail_budget:
            return pinned + tail

        # Drop oldest middle messages in pairs (assistant + observation)
        # so we don't break the alternating user/assistant pattern
        trimmed_tail = tail[-tail_budget:]

        # Ensure the tail starts with a user message (observation)
        # so the alternating pattern is maintained
        while trimmed_tail and trimmed_tail[0].role != "user":
            trimmed_tail = trimmed_tail[1:]

        return pinned + trimmed_tail

    # ------------------------------------------------------------------ #
    # Debug
    # ------------------------------------------------------------------ #

    def stats(self) -> dict:
        return {
            "total_messages": len(self._messages),
            "windowed_messages": len(self._windowed()),
            "max_messages": self.max_messages,
            "dropped": max(0, len(self._messages) - self.max_messages),
        }

    def pretty_print(self, windowed: bool = True) -> None:
        """Print conversation to stdout for debugging."""
        msgs = self._windowed() if windowed else self._messages
        print(f"\n{'─'*60}")
        print(
            f"Conversation ({len(msgs)} messages shown / {len(self._messages)} total)"
        )
        print(f"{'─'*60}")
        for i, m in enumerate(msgs):
            prefix = "👤" if m.role == "user" else "🤖"
            snippet = m.content[:120].replace("\n", " ")
            print(
                f"[{i:02d}] {prefix} {m.role.upper()}: {snippet}{'...' if len(m.content) > 120 else ''}"
            )
        print(f"{'─'*60}\n")
