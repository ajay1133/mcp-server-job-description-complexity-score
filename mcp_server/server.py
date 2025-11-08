#!/usr/bin/env python3
"""Entry point for the MCP-based Software Complexity scoring server.

Provides an MCP tool `score_complexity(requirement: str)` which returns a
schema-compliant complexity analysis for a free-form software requirement.
Use `--self-test` to quickly exercise the scoring logic locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer  # noqa: E402


scorer = SoftwareComplexityScorer()


def score_complexity_(requirement: str) -> dict:
    """Score software requirement complexity using trained models."""
    return scorer.analyze_text(requirement)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Software Complexity MCP Server")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick self test and exit",
    )
    parser.add_argument(
        "--example",
        type=str,
        default=("Build a Next.js SaaS with Stripe payments and user auth dashboard."),
        help="Example text for --self-test",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        result = scorer.analyze_text(args.example)
        print(json.dumps(result, indent=2))
        return 0

    mcp = FastMCP("complexity-scorer")
    mcp.tool()(score_complexity_)
    print("[complexity-scorer] Starting MCP server... (Ctrl+C to stop)")
    print("[complexity-scorer] Tool registered: score_complexity(requirement: str) -> dict")
    mcp.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
