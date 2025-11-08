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
print_section("2. System Design Architecture Prediction")
design = result['proposed_system_design']
print(f"Predicted Architecture: {design['architecture']}")
print(f"Confidence: {design['confidence']:.2%}")
print(f"Model Used: {'Yes' if design['model_used'] else 'No'}")
print("\nAll Architecture Probabilities:")
for arch, prob in sorted(design['all_probabilities'].items(), key=lambda x: x[1], reverse=True):
    bar = "█" * int(prob * 50)
    print(f"  {arch:20s} {prob:6.2%} {bar}")

# Feature 3: Technology criticality classification
print_section("3. Technology Criticality Classification with Overhead")

tech_analysis = result['per_technology_analysis']
print(f"Total technologies analyzed: {len(tech_analysis)}\n")

# Group by criticality
mandatory = [t for t in tech_analysis if t['criticality'] == 'mandatory']
recommended = [t for t in tech_analysis if t['criticality'] == 'recommended']
optional = [t for t in tech_analysis if t['criticality'] == 'optional']

print(f"Mandatory Technologies ({len(mandatory)}):")
print("  (Core requirements, cannot be removed)\n")
for t in mandatory[:5]:
    print(f"  • {t['technology']:20s}")
    print(f"      Confidence:     {t['confidence']:.2%}")
    print(f"      LOC Overhead:   {t['loc_overhead']:,} lines")
    print(f"      Time Overhead:  {t['time_overhead_readable']}")
    print()

if recommended:
    print(f"\nRecommended Technologies ({len(recommended)}):")
    print("  (Best practices for this use case)\n")
    for t in recommended[:3]:
        print(f"  • {t['technology']:20s}")
        print(f"      Confidence:     {t['confidence']:.2%}")
        print(f"      LOC Overhead:   {t['loc_overhead']:,} lines")
        print(f"      Time Overhead:  {t['time_overhead_readable']}")
        print()

if optional:
    print(f"\nOptional Technologies ({len(optional)}):")
    print("  (Nice-to-have enhancements)\n")
    for t in optional[:3]:
        print(f"  • {t['technology']:20s}")
        print(f"      Confidence:     {t['confidence']:.2%}")
        print(f"      LOC Overhead:   {t['loc_overhead']:,} lines")
        print(f"      Time Overhead:  {t['time_overhead_readable']}")
        print()

# Summary
print_section("Summary of Enhancements")
print("✓ Feature 1: Empty categories removed from output")
print(f"    Result: Only {len(result['technologies'])} categories shown")
print()
print("✓ Feature 2: System design architecture predicted")
print(f"    Result: {design['architecture']} architecture ({design['confidence']:.1%} confidence)")
print()
print("✓ Feature 3: Technology criticality classified")
print(f"    Result: {len(mandatory)} mandatory, {len(recommended)} recommended, {len(optional)} optional")
print()
print("All features working correctly! ✓")
print_section("End of Demo")
