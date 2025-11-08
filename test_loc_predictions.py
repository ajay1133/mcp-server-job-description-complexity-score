#!/usr/bin/env python3
"""Test LOC predictions to understand if they seem accurate."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

scorer = SoftwareComplexityScorer()

tests = [
    ("Build a simple todo app", "Simple Todo"),
    ("Build a React dashboard with Stripe payments", "React Dashboard"),
    ("Build a full e-commerce platform with React, Node.js, PostgreSQL, Redis, Stripe, AWS S3, authentication, admin panel", "E-commerce Platform"),
    ("Create a Twitter clone with real-time feeds, notifications, DMs, media uploads, user profiles, search", "Twitter Clone"),
    ("Build Netflix clone with video streaming, recommendations, user profiles, search, admin dashboard", "Netflix Clone"),
    ("Create a banking app with accounts, transactions, transfers, loans, credit cards, fraud detection", "Banking App"),
    ("Build a social media platform with posts, comments, likes, shares, messaging, groups, events", "Social Media"),
]

print("\n" + "=" * 100)
print("Lines of Code Predictions Analysis")
print("=" * 100)

for prompt, label in tests:
    result = scorer.analyze_text(prompt)
    loc = result["predicted_lines_of_code"]
    techs = result["technologies"]
    num_services = len(result["microservices"])
    
    # Count total technologies
    total_techs = sum(len(v) if isinstance(v, list) else 0 for v in techs.values())
    
    print(f"\n{label}:")
    print(f"  Prompt: {prompt[:80]}...")
    print(f"  Predicted LOC: {loc:,}")
    print(f"  Technologies: {total_techs} ({', '.join([t for sublist in techs.values() if isinstance(sublist, list) for t in sublist][:5])}...)")
    print(f"  Microservices: {num_services}")
    print(f"  LOC per technology: ~{loc // max(total_techs, 1):,}")
    print(f"  LOC per service: ~{loc // max(num_services, 1):,}")

print("\n" + "=" * 100)
print("\nREAL-WORLD COMPARISON:")
print("  - Simple Todo App: typically 500-2,000 LOC")
print("  - Medium Dashboard: typically 3,000-10,000 LOC")
print("  - E-commerce Platform: typically 20,000-50,000 LOC")
print("  - Twitter Clone: typically 30,000-100,000 LOC")
print("  - Netflix Clone: typically 50,000-150,000 LOC")
print("=" * 100 + "\n")
