#!/usr/bin/env python3
"""Demo: ML-based difficulty estimation for new technologies.

Shows how the system predicts difficulty WITHOUT retraining by:
1. Fetching real-world metrics (GitHub stars, StackOverflow questions, npm downloads)
2. Using feature-based ML (works for ANY tech, not just known ones)
3. Falling back gracefully when APIs unavailable
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.ml_difficulty_estimator import get_difficulty_estimator  # noqa: E402


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + ("=" * 70))
    print(f"  {title}")
    print('=' * 70)


def demo_known_tech():
    """Demo: Difficulty estimation for well-known tech."""
    print_section("1. Well-Known Technology (React)")

    estimator = get_difficulty_estimator()
    result = estimator.estimate_difficulty("react")

    print("\nTechnology: React")
    print(f"  Difficulty: {result['difficulty']}/10")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Source: {result['source']}")
    print(f"  Explanation: {result['explanation']}")

    if "features" in result:
        print("\n  Key Metrics:")
        features = result["features"]
        print(f"    Years in market: {features.get('years_in_market', 0):.1f}")
        print(f"    Maintenance score: {features.get('maintenance_score', 0):.1f}")
        print(f"    Maturity score: {features.get('maturity_score', 0):.1f}")


def demo_new_tech():
    """Demo: Difficulty estimation for newer tech (Bun)."""
    print_section("2. Newer Technology (Bun - JS Runtime)")

    estimator = get_difficulty_estimator()
    result = estimator.estimate_difficulty("bun")

    print("\nTechnology: Bun")
    print(f"  Difficulty: {result['difficulty']}/10")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Source: {result['source']}")
    print(f"  Explanation: {result['explanation']}")

    print("\n  ✅ NO RETRAINING NEEDED!")
    print("  The ML model uses features (stars, age, community size)")
    print("  that work for ANY technology, not hardcoded tech names.")


def demo_unknown_tech():
    """Demo: Fallback for completely unknown tech."""
    print_section("3. Unknown Technology (Future Tech)")

    estimator = get_difficulty_estimator()
    result = estimator.estimate_difficulty("quantum-react-2026")

    print("\nTechnology: quantum-react-2026 (doesn't exist yet)")
    print(f"  Difficulty: {result['difficulty']}/10 (fallback baseline)")
    print(f"  Confidence: {result['confidence']} (low)")
    print(f"  Source: {result['source']}")

    print("\n  ✅ GRACEFUL FALLBACK!")
    print("  System doesn't crash, returns sensible default (5.0)")


def demo_comparison():
    """Demo: Compare multiple technologies."""
    print_section("4. Technology Comparison")

    estimator = get_difficulty_estimator()

    techs = [
        ("react", "Mature frontend framework"),
        ("svelte", "Newer frontend framework"),
        ("bun", "New JS runtime"),
        ("htmx", "Hypermedia-driven approach"),
        ("new-framework", "Unknown/doesn't exist"),
    ]

    print("\n{:<20} {:<12} {:<12} {:<15}".format("Technology", "Difficulty", "Confidence", "Source"))
    print("-" * 70)

    for tech_name, description in techs:
        result = estimator.estimate_difficulty(tech_name)
        print(
            "{:<20} {:<12} {:<12} {:<15}".format(
                f"{tech_name}",
                f"{result['difficulty']:.1f}/10",
                f"{result['confidence']:.2f}",
                result["source"],
            )
        )

    print("\n  Key Insight:")
    print("  - Known techs: High confidence, accurate difficulty")
    print("  - Unknown techs: Low confidence, fallback to 5.0")
    print("  - All techs: System never crashes!")


def demo_how_it_works():
    """Demo: Explain how the ML approach works."""
    print_section("5. How It Works (No Retraining Required!)")

    print(
        """
  Traditional Approach (REQUIRES RETRAINING):
  ┌─────────────────────────────────────────────────────────┐
  │ Model: tech_name → difficulty                          │
  │                                                         │
  │ Training data:                                          │
  │   "react" → 5.2                                         │
  │   "vue" → 4.8                                           │
  │   "angular" → 6.5                                       │
  │                                                         │
  │ ❌ NEW TECH: "bun" → ??? (not in training data)        │
  │ ❌ SOLUTION: Retrain model with "bun" data             │
  └─────────────────────────────────────────────────────────┘

  Feature-Based ML Approach (NO RETRAINING):
  ┌─────────────────────────────────────────────────────────┐
  │ Model: features → difficulty                           │
  │                                                         │
  │ Features (work for ANY tech):                           │
  │   - github_stars_log                                    │
  │   - years_in_market                                     │
  │   - stackoverflow_questions_log                         │
  │   - maintenance_score                                   │
  │   - community_size                                      │
  │                                                         │
  │ ✅ NEW TECH: "bun"                                      │
  │    1. Fetch GitHub stars: 50,000                        │
  │    2. Fetch age: 2 years                                │
  │    3. Fetch SO questions: 1,200                         │
  │    4. Extract features → predict: 5.3                   │
  │                                                         │
  │ ✅ NO RETRAINING! Model generalizes to any tech        │
  └─────────────────────────────────────────────────────────┘

  External APIs Used:
  - GitHub API: stars, forks, issues, age, contributors
  - npm API: download counts, version history
  - StackOverflow API: question counts, tag popularity
  - PyPI (future): Python package stats

  Feature Engineering:
  - Log-scale metrics (stars, downloads) for normalization
  - Derived features: maintenance_score, maturity_score
  - Issue ratio: open_issues / stars (quality indicator)
  - Age normalization: newer = higher uncertainty

  Heuristic Weights (learned from data):
  - More stars → slightly easier (more resources/docs)
  - Older → easier (mature, stable APIs)
  - High SO questions → harder (complex or error-prone)
  - Good maintenance → easier (active support)
    """
    )


def main():
    """Run all demos."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║  ML Difficulty Estimation: No Retraining for New Technologies     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")

    demo_how_it_works()
    demo_known_tech()
    demo_new_tech()
    demo_unknown_tech()
    demo_comparison()

    print("\n" + "=" * 70)
    print("  Summary")
    print("=" * 70)
    print(
        """
✅ NO RETRAINING NEEDED when new technologies appear!
✅ Feature-based ML model generalizes to ANY technology
✅ Fetches real-world metrics from GitHub, npm, StackOverflow
✅ Intelligent heuristics when APIs unavailable
✅ Graceful fallback for completely unknown techs
✅ Cached results (24h TTL) to avoid rate limits

How It Differs from Traditional ML:
┌────────────────────────────────────────────────────────────┐
│ Traditional:  tech_name → difficulty                       │
│ ❌ New tech requires retraining                            │
│                                                            │
│ Feature-based:  (stars, age, community) → difficulty       │
│ ✅ New tech works immediately                              │
└────────────────────────────────────────────────────────────┘

Next Steps:
1. Add GitHub API token for higher rate limits
2. Train model on more historical data
3. Add PyPI metrics for Python libraries
4. Improve category inference from GitHub topics
5. Add confidence intervals for predictions
    """
    )


if __name__ == "__main__":
    main()
