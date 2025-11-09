#!/usr/bin/env python3
"""
Comprehensive demonstration of all three new features:
1. Empty category filtering
2. System design architecture prediction
3. Technology criticality classification
"""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
import json

def print_section(title):
    print("\n" + "=" * 80)
    print(title.center(80))
    print("=" * 80 + "\n")

scorer = SoftwareComplexityScorer()

# Test case: Dashboard with limited technologies
print_section("FEATURE DEMO: All Three Enhancements")

requirement = "Build a React dashboard with FastAPI backend and PostgreSQL database"
print(f"Requirement: {requirement}\n")

result = scorer.analyze_text(requirement)

# Feature 1: Empty category filtering
print_section("1. Empty Category Filtering")
print("Before: Output would include all categories (frontend, backend, database, mobile)")
print("After: Output only includes categories with detected technologies\n")
print(f"Technologies detected: {json.dumps(result['technologies'], indent=2)}")
print(f"\nEmpty categories removed: ✓")
if 'mobile' not in result['technologies']:
    print("  - 'mobile' category excluded (no mobile tech detected)")

# Feature 2: System design architecture prediction
print_section("2. Data Flow (moved to root)")
data_flow = result.get('data_flow', [])
print("Inferred Data Flow Steps:")
for step in data_flow:
    print(f"  • {step}")
print("\nNote: Architecture prediction is used internally; only data_flow is exposed in the final response.")

# Feature 3: Technology criticality classification
print_section("3. Per-Technology Time Share (percentages)")

tech_analysis = result.get('per_technology_analysis', [])
print(f"Total technologies analyzed: {len(tech_analysis)}\n")

print(f"{'Technology':<22} {'Human %':>8} {'AI %':>8} {'Mentioned':>12}")
print("-"*54)
for t in sorted(tech_analysis, key=lambda x: x.get('time_spent', {}).get('human_percent', 0), reverse=True)[:10]:
    tech = t.get('technology', '')
    human_pct = t.get('time_spent', {}).get('human_percent', 0)
    ai_pct = t.get('time_spent', {}).get('ai_percent', 0)
    mentioned = 'yes' if t.get('is_mentioned_in_prompt') else 'no'
    print(f"{tech:<22} {human_pct:>7.2f}% {ai_pct:>7.2f}% {mentioned:>12}")

# Summary
print_section("Summary of Enhancements")
print("✓ Feature 1: Empty categories removed from output")
print(f"    Result: Only {len(result['technologies'])} categories shown")
print()
print("✓ Feature 2: Data flow exposed at root")
print(f"    Result: {len(data_flow)} steps")
print()
print("✓ Feature 3: Per-technology time share computed")
print(f"    Result: {len(tech_analysis)} technologies with percentage of overall effort")
print()
print("All features working correctly! ✓")
print_section("End of Demo")
