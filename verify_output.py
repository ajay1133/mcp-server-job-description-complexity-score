#!/usr/bin/env python3
"""Quick verification of the new output schema."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
import json

scorer = SoftwareComplexityScorer()

# Test cases
tests = [
    "Build React e-commerce site with Stripe payments and auth",
    "Create machine learning recommendation system with Python and TensorFlow",
    "Need a plumber"
]

print("=" * 80)
print("OUTPUT SCHEMA VERIFICATION")
print("=" * 80)

for test in tests:
    print(f"\n>>> {test}")
    result = scorer.analyze_text(test)
    print(json.dumps(result, indent=2))

print("\n" + "=" * 80)
