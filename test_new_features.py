#!/usr/bin/env python3
"""Test the new system design and technology criticality features."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
import json

scorer = SoftwareComplexityScorer()

# Test 1: Real-time chat application
print("=" * 80)
print("Test 1: Real-time chat application")
print("=" * 80)
result = scorer.analyze_text('Build a real-time chat application with websockets')

if 'per_technology_analysis' in result:
    print(f"\nPer-Technology Time Share (first 5):")
    for t in result['per_technology_analysis'][:5]:
        print(f"  {t['technology']}: human {t['time_spent']['human_percent']}% / ai {t['time_spent']['ai_percent']}% mentioned={t['is_mentioned_in_prompt']}")

# Test 2: Simple blog
print("\n" + "=" * 80)
print("Test 2: Simple blog")
print("=" * 80)
result2 = scorer.analyze_text('Build a simple blog')

if 'per_technology_analysis' in result2:
    print(f"\nPer-Technology Time Share (first 5):")
    for t in result2['per_technology_analysis'][:5]:
        print(f"  {t['technology']}: human {t['time_spent']['human_percent']}% / ai {t['time_spent']['ai_percent']}% mentioned={t['is_mentioned_in_prompt']}")

# Test 3: Microservices platform
print("\n" + "=" * 80)
print("Test 3: E-commerce platform")
print("=" * 80)
result3 = scorer.analyze_text('Create an e-commerce platform with inventory, payments, and order management')

if 'per_technology_analysis' in result3:
    print(f"\nPer-Technology Time Share (first 10):")
    for t in result3['per_technology_analysis'][:10]:
        print(f"  {t['technology']}: human {t['time_spent']['human_percent']}% / ai {t['time_spent']['ai_percent']}% mentioned={t['is_mentioned_in_prompt']}")

print("\n" + "=" * 80)
print("All tests complete!")
print("=" * 80)
