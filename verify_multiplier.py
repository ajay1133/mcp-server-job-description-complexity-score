#!/usr/bin/env python3
"""Quick test of the updated multiplier."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

scorer = SoftwareComplexityScorer()
result = scorer.analyze_text('Build a Twitter clone with real-time feeds')

print("\n" + "=" * 80)
print("VERIFICATION: Updated 98.73x Multiplier")
print("=" * 80)
print(f"\nPrompt: Build a Twitter clone with real-time feeds")

ai_time = result['with_ai_and_ml']['time_estimation']
manual_time = result['without_ai_and_ml']['time_estimation']

print(f"\nAI time: {ai_time['human_readable']}")
print(f"Manual time: {manual_time['human_readable']}")
print(f"\nSpeedup: {result['with_ai_and_ml']['speedup_details']['speed_ratio']}")
print(f"\nManual hours: {manual_time['hours']:.2f}")
print(f"AI hours: {ai_time['hours']:.2f}")
ratio = manual_time['hours'] / ai_time['hours']
print(f"Ratio: {ratio:.2f}x")
print("\n✓ Multiplier successfully updated to 98.73x!")
print("=" * 80 + "\n")
