#!/usr/bin/env python3
"""Demo showing self-learning behavior for unknown projects like 'perplexity copy'."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.ml_tech_extractor import MLTechExtractor  # noqa: E402
from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402


def demo_simple():
    print("=" * 80)
    print("SimpleTechExtractor - 'perplexity copy'")
    print("=" * 80)

    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("perplexity copy")

    print(json.dumps(result, indent=2))
    print()


def demo_ml():
    print("=" * 80)
    print("MLTechExtractor - 'perplexity copy'")
    print("=" * 80)

    extractor = MLTechExtractor()
    result = extractor.extract_technologies("perplexity copy")

    print(json.dumps(result, indent=2))
    print()


if __name__ == "__main__":
    demo_simple()
    demo_ml()
