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

print(f"\nArchitecture Prediction:")
print(f"  Type: {result['proposed_system_design']['architecture']}")
print(f"  Confidence: {result['proposed_system_design']['confidence']}")
print(f"\n  All probabilities:")
for arch, prob in result['proposed_system_design']['all_probabilities'].items():
    print(f"    {arch}: {prob}")

print(f"\nTechnology Criticality Analysis (first 5):")
for t in result['per_technology_analysis'][:5]:
    print(f"  {t['technology']}:")
    print(f"    Criticality: {t['criticality']} (confidence: {t['confidence']})")
    print(f"    LOC Overhead: {t['loc_overhead']} lines")
    print(f"    Time Overhead: {t['time_overhead_readable']}")

# Test 2: Simple blog
print("\n" + "=" * 80)
print("Test 2: Simple blog")
print("=" * 80)
result2 = scorer.analyze_text('Build a simple blog')

print(f"\nArchitecture Prediction:")
print(f"  Type: {result2['proposed_system_design']['architecture']}")
print(f"  Confidence: {result2['proposed_system_design']['confidence']}")

# Test 3: Microservices platform
print("\n" + "=" * 80)
print("Test 3: E-commerce platform")
print("=" * 80)
result3 = scorer.analyze_text('Create an e-commerce platform with inventory, payments, and order management')

print(f"\nArchitecture Prediction:")
print(f"  Type: {result3['proposed_system_design']['architecture']}")
print(f"  Confidence: {result3['proposed_system_design']['confidence']}")

print(f"\nMandatory Technologies:")
mandatory = [t for t in result3['per_technology_analysis'] if t['criticality'] == 'mandatory']
print(f"  Count: {len(mandatory)}")
for t in mandatory[:5]:
    print(f"  - {t['technology']}: {t['loc_overhead']} LOC, {t['time_overhead_readable']}")

print(f"\nRecommended Technologies:")
recommended = [t for t in result3['per_technology_analysis'] if t['criticality'] == 'recommended']
print(f"  Count: {len(recommended)}")
for t in recommended[:3]:
    print(f"  - {t['technology']}: {t['loc_overhead']} LOC, {t['time_overhead_readable']}")

print(f"\nOptional Technologies:")
optional = [t for t in result3['per_technology_analysis'] if t['criticality'] == 'optional']
print(f"  Count: {len(optional)}")
for t in optional[:3]:
    print(f"  - {t['technology']}: {t['loc_overhead']} LOC, {t['time_overhead_readable']}")

print("\n" + "=" * 80)
print("All tests complete!")
print("=" * 80)
