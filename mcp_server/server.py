#!/usr/bin/env python3
import sys
import os
import argparse
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

# Global scorer (pure, has no side effects on import)
scorer = SoftwareComplexityScorer()

def score_complexity_(requirement: str) -> dict:
    """Score software requirement complexity using trained models.

    Args:
        requirement: Free-form text describing a software project/task.
    Returns:
        Schema-compliant complexity analysis or error structure.
    """
    return scorer.analyze_text(requirement)


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="Software Complexity MCP Server")
    parser.add_argument("--self-test", action="store_true", help="Run a quick self test and exit")
    parser.add_argument("--example", type=str, default="Build a Next.js SaaS with Stripe payments and user auth dashboard.", help="Example text for --self-test")
    args = parser.parse_args(argv)

    if args.self_test:
        result = scorer.analyze_text(args.example)
        print(json.dumps(result, indent=2))
        return 0

    # Build MCP server and register tools lazily to avoid side-effects during --self-test
    mcp = FastMCP("complexity-scorer")
    # Register tool via callable decorator style to delay binding until here
    mcp.tool()(score_complexity_)

    print("[complexity-scorer] Starting MCP server... (Ctrl+C to stop)")
    print("[complexity-scorer] Tool registered: score_complexity(requirement: str) -> dict")
    mcp.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
