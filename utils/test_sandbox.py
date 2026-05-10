"""
tests/test_sandbox.py — Security + correctness tests for the Docker sandbox.

Run: python utils/test_sandbox.py   (no pytest needed)
"""

from __future__ import annotations

import sys
import os
import json

# Make the project root importable when run directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from agent.sandbox_client import SandboxClient, ExecResult
from agent.history import ConversationHistory

# ============================================================================
# HELPERS
# ============================================================================


def _pass(label: str) -> None:
    print(f" Pass:  {label}")


def _fail(label: str, reason: str) -> None:
    print(f" Fail:  {label}")
    print(f"      reason: {reason}")


def run_test(label: str, fn) -> bool:
    try:
        fn()
        _pass(label)
        return True
    except AssertionError as exc:
        _fail(label, str(exc))
        return False
    except Exception as exc:
        _fail(label, f"unexpected exception: {exc}")
        return False


# ============================================================================
# SANDBOX SECURITY TESTS
# ============================================================================


def make_sandbox_tests(sb: SandboxClient):

    def t_basic_exec():
        r = sb.run("x = 6 * 7\nprint(x)", {})
        assert r.ok, f"expected ok=True, got stderr={r.stderr!r}"
        assert "42" in r.stdout, f"expected '42' in stdout, got {r.stdout!r}"

    def t_state_in():
        """Variables passed in via state are accessible."""
        r = sb.run("print(x + 10)", {"x": 42})
        assert r.ok
        assert "52" in r.stdout, f"expected '52', got {r.stdout!r}"

    def t_state_out():
        """Variables assigned in code come back in state."""
        r = sb.run("answer = 100 * 3", {})
        assert r.ok
        assert r.state.get("answer") == 300, f"state={r.state}"

    def t_state_persists_across_calls():
        """Simulate two-iteration loop: step 2 sees step 1's variable."""
        r1 = sb.run("base = 1000", {})
        assert r1.ok
        r2 = sb.run("result = base * 2\nprint(result)", r1.state)
        assert r2.ok
        assert "2000" in r2.stdout, f"got {r2.stdout!r}"
        assert r2.state.get("result") == 2000

    def t_numpy_allowed():
        r = sb.run("import numpy as np\narr = np.array([1,2,3])\nprint(arr.sum())", {})
        assert r.ok, f"stderr={r.stderr}"
        assert "6" in r.stdout

    def t_pandas_allowed():
        r = sb.run(
            "import pandas as pd\n"
            "df = pd.DataFrame({'a':[1,2,3]})\n"
            "print(df['a'].sum())",
            {},
        )
        assert r.ok, f"stderr={r.stderr}"
        assert "6" in r.stdout

    # ── SECURITY: imports that MUST be blocked ───────────────────────────

    def t_block_os():
        r = sb.run("import os\nprint(os.listdir('/'))", {})
        assert (
            not r.ok
            or "SecurityError" in r.stderr
            or "PermissionError" in r.stderr
            or "None" in r.stderr
        ), f"os import should be blocked. stderr={r.stderr!r} stdout={r.stdout!r}"

    def t_block_subprocess():
        r = sb.run("import subprocess\nsubprocess.run(['id'])", {})
        assert (
            not r.ok or "SecurityError" in r.stderr or "None" in r.stderr
        ), f"subprocess should be blocked. stderr={r.stderr!r}"

    def t_block_socket():
        r = sb.run("import socket\nsocket.gethostbyname('google.com')", {})
        assert (
            not r.ok or "SecurityError" in r.stderr or "None" in r.stderr
        ), f"socket should be blocked. stderr={r.stderr!r}"

    def t_block_requests():
        r = sb.run("import requests\nrequests.get('https://example.com')", {})
        assert (
            not r.ok or "SecurityError" in r.stderr or "None" in r.stderr
        ), f"requests should be blocked. stderr={r.stderr!r}"

    def t_block_open():
        r = sb.run("f = open('/etc/passwd')", {})
        assert (
            not r.ok
        ), f"open() should be blocked. stdout={r.stdout!r} stderr={r.stderr!r}"

    def t_block_dunder_import():
        r = sb.run("os = __import__('os')\nprint(os.getcwd())", {})
        assert (
            not r.ok or "SecurityError" in r.stderr
        ), f"__import__ should be blocked. stderr={r.stderr!r}"

    def t_block_pickle():
        r = sb.run("import pickle\nprint(pickle.dumps({'a':1}))", {})
        assert (
            not r.ok or "SecurityError" in r.stderr or "None" in r.stderr
        ), f"pickle should be blocked. stderr={r.stderr!r}"

    # ── Error handling ───────────────────────────────────────────────────

    def t_runtime_error_captured():
        r = sb.run("1 / 0", {})
        assert not r.ok
        assert "ZeroDivisionError" in r.stderr

    def t_syntax_error_captured():
        r = sb.run("def broken(\nprint('x')", {})
        assert not r.ok
        assert "SyntaxError" in r.stderr

    def t_undefined_variable():
        r = sb.run("print(nonexistent_var)", {})
        assert not r.ok
        assert "NameError" in r.stderr

    return [
        ("Basic code execution", t_basic_exec),
        ("State passed in", t_state_in),
        ("State returned out", t_state_out),
        ("State persists across calls", t_state_persists_across_calls),
        ("numpy allowed", t_numpy_allowed),
        ("pandas allowed", t_pandas_allowed),
        ("BLOCK: os", t_block_os),
        ("BLOCK: subprocess", t_block_subprocess),
        ("BLOCK: socket", t_block_socket),
        ("BLOCK: requests", t_block_requests),
        ("BLOCK: open()", t_block_open),
        ("BLOCK: __import__", t_block_dunder_import),
        ("BLOCK: pickle", t_block_pickle),
        ("RuntimeError captured", t_runtime_error_captured),
        ("SyntaxError captured", t_syntax_error_captured),
        ("Undefined variable captured", t_undefined_variable),
    ]


# ============================================================================
# HISTORY TESTS  (no Docker needed)
# ============================================================================


def make_history_tests():

    def t_basic_add_and_retrieve():
        h = ConversationHistory(max_messages=10)
        h.add_user("hello")
        h.add_assistant("hi there")
        msgs = h.as_dicts()
        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "hello"}
        assert msgs[1] == {"role": "assistant", "content": "hi there"}

    def t_under_window_no_truncation():
        h = ConversationHistory(max_messages=6)
        for i in range(3):
            h.add_user(f"u{i}")
            h.add_assistant(f"a{i}")
        assert len(h.as_dicts()) == 6  # all kept

    def t_over_window_drops_middle():
        h = ConversationHistory(max_messages=4, always_keep_first=True)
        # Add 6 messages (3 pairs)
        h.add_user("query")  # [0] pinned
        h.add_assistant("resp0")  # [1] pinned
        h.add_user("obs1")
        h.add_assistant("resp1")
        h.add_user("obs2")
        h.add_assistant("resp2")

        windowed = h.as_dicts()
        # max_messages=4 → pinned[2] + tail[2]
        assert len(windowed) <= 4, f"got {len(windowed)}"
        # First message must still be the original query
        assert windowed[0]["content"] == "query"

    def t_first_pair_always_kept():
        h = ConversationHistory(max_messages=4, always_keep_first=True)
        h.add_user("ORIGINAL QUERY")
        h.add_assistant("FIRST RESPONSE")
        for i in range(10):
            h.add_user(f"obs{i}")
            h.add_assistant(f"resp{i}")

        windowed = h.as_dicts()
        assert (
            windowed[0]["content"] == "ORIGINAL QUERY"
        ), f"first message lost: {windowed[0]}"

    def t_full_history_unchanged():
        h = ConversationHistory(max_messages=4)
        for i in range(5):
            h.add_user(f"u{i}")
            h.add_assistant(f"a{i}")
        assert len(h.full_history()) == 10  # nothing dropped from full log

    def t_len_counts_all_messages():
        h = ConversationHistory()
        h.add_user("a")
        h.add_assistant("b")
        h.add_user("c")
        assert len(h) == 3

    def t_stats_shows_dropped():
        h = ConversationHistory(max_messages=4, always_keep_first=True)
        for i in range(6):
            h.add_user(f"u{i}")
            h.add_assistant(f"a{i}")
        stats = h.stats()
        assert stats["total_messages"] == 12
        assert stats["windowed_messages"] <= 4
        assert stats["dropped"] > 0

    return [
        ("Add and retrieve messages", t_basic_add_and_retrieve),
        ("No truncation under window size", t_under_window_no_truncation),
        ("Drops middle when over window", t_over_window_drops_middle),
        ("First pair always kept", t_first_pair_always_kept),
        ("Full history log unchanged", t_full_history_unchanged),
        ("len() counts all messages", t_len_counts_all_messages),
        ("stats() reports dropped count", t_stats_shows_dropped),
    ]


# ============================================================================
# MAIN
# ============================================================================


def main():
    print("\n" + "═" * 72)
    print("  FinSight — Sandbox + History Test Suite")
    print("═" * 72)

    passed = 0
    failed = 0

    # ── History tests (no Docker) ────────────────────────────────────────
    print("\n▸ ConversationHistory tests")
    for label, fn in make_history_tests():
        if run_test(label, fn):
            passed += 1
        else:
            failed += 1

    # ── Sandbox tests (Docker required) ─────────────────────────────────
    print("\n▸ Docker sandbox tests")
    try:
        sb = SandboxClient()
        for label, fn in make_sandbox_tests(sb):
            if run_test(label, fn):
                passed += 1
            else:
                failed += 1
    except RuntimeError as exc:
        print(f"\n  ⚠️  Skipping sandbox tests — Docker not ready:\n  {exc}")
        print("  Build the image first:")
        print("    cd sandbox && docker build -t finsight-sandbox .")

    # ── Summary ──────────────────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'═'*72}")
    print(f"  Results: {passed}/{total} passed", end="")
    print(" 🎉" if failed == 0 else f"  ({failed} failed)")
    print("═" * 72 + "\n")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
