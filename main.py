#!/usr/bin/env python3
"""
main.py — FinSight entry point.

Usage:
  python main.py                          # run the built-in demo query
  python main.py "your question here"     # run a custom query
  python main.py --build                  # build the Docker image and exit
"""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(__file__))

from agent.loop import run_agent
from agent.sandbox_client import SandboxClient, build_sandbox_image

DEMO_QUERY = (
    "I have a list of annual revenues: [120, 145, 160, 178, 210] (in $M). "
    "Calculate year-over-year growth rates, the CAGR over the full period, "
    "and identify the single best growth year."
)


def main():
    args = sys.argv[1:]

    # --build flag: build Docker image and exit
    if args and args[0] == "--build":
        ok = build_sandbox_image()
        sys.exit(0 if ok else 1)

    query = " ".join(args) if args else DEMO_QUERY

    # Initialise sandbox (checks image exists)
    try:
        sandbox = SandboxClient()
    except RuntimeError as exc:
        print(f"\n{exc}")
        print("Run:  python main.py --build   to build the Docker image first.")
        sys.exit(1)

    # Run agent loop
    result = run_agent(query, sandbox, verbose=True)

    print(f"\n{'═'*72}")
    print(result.summary())
    print(f"{'═'*72}\n")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
