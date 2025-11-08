#!/usr/bin/env python3
"""Simple CLI to analyze software requirements without starting the MCP server.

Usage:
  python run_requirements_cli.py --text "Build a Next.js SaaS with Stripe"
  python run_requirements_cli.py --file prompts.txt            # one prompt per line
  python run_requirements_cli.py                               # interactive REPL
"""

import argparse
import json
import sys
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer


def analyze_texts(texts):
    scorer = SoftwareComplexityScorer()
    for t in texts:
        t = (t or '').strip()
        if not t:
            continue
        print("\n" + "=" * 70)
        print(t)
        print("-" * 70)
        result = scorer.analyze_text(t)
        print(json.dumps(result, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(description='Analyze software requirement text via CLI')
    ap.add_argument('--text', type=str, help='Single requirement text to analyze')
    ap.add_argument('--file', type=str, help='Path to a file with one prompt per line')
    args = ap.parse_args(argv)

    if args.text:
        analyze_texts([args.text])
        return 0

    if args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        analyze_texts(lines)
        return 0

    # Interactive REPL
    print("Enter requirement text (empty line to exit):")
    while True:
        try:
            line = input('> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            break
        analyze_texts([line])

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
