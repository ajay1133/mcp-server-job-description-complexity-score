#!/usr/bin/env python3
"""Demo: How the system handles new technologies.

This demo shows how the TechRegistry gracefully handles:
1. Known technologies (baseline database)
2. Brand new technologies (fallback with sensible defaults)
3. Manually added technologies (custom additions)
4. Finding alternatives dynamically
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp_server.tech_registry import get_tech_registry  # noqa: E402


def print_section(title: str):
    """Print formatted section header."""
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def demo_known_tech():
    """Demo: Known technology from baseline."""
    print_section("1. Known Technology (Baseline Database)")

    registry = get_tech_registry()
    react_info = registry.get_tech_info("react")

    print("\nTechnology: React")
    print(f"  Difficulty: {react_info['difficulty']}/10")
    print(f"  Category: {react_info['category']}")
    print(f"  Keywords: {', '.join(react_info['keywords'])}")

    alternatives = registry.search_similar_techs("react", top_k=3)
    print(f"  Alternatives: {', '.join(alternatives)}")


def demo_new_tech_unknown():
    """Demo: Brand new technology (never seen before)."""
    print_section("2. Unknown Technology (Graceful Fallback)")

    registry = get_tech_registry()

    # Simulate a tech that will exist in 2026 but we don't know about yet
    new_tech = registry.get_tech_info("quantum-react-2026")

    print("\nTechnology: quantum-react-2026 (doesn't exist yet)")
    print(f"  Difficulty: {new_tech['difficulty']}/10 (default mid-range)")
    print(f"  Category: {new_tech['category']}")
    print(f"  Confidence: {new_tech['confidence']} (low = unknown)")
    print(f"  Source: {new_tech['source']}")
    print("\n  ✅ System doesn't crash - returns sensible defaults!")


def demo_add_new_tech():
    """Demo: Manually adding a new technology."""
    print_section("3. Manually Adding New Technologies")

    registry = get_tech_registry()

    # Add htmx (trending in 2024/2025)
    print("\nAdding htmx to registry...")
    registry.add_custom_tech(
        tech_name="htmx",
        difficulty=4.0,
        category="frontend",
        keywords=["htmx", "hypermedia", "hateoas"],
    )

    # Add Bun (new JS runtime)
    print("Adding Bun to registry...")
    registry.add_custom_tech(
        tech_name="bun",
        difficulty=5.3,
        category="backend",
        keywords=["bun", "bun.js", "bunjs"],
    )

    # Verify they're available
    htmx_info = registry.get_tech_info("htmx")
    bun_info = registry.get_tech_info("bun")

    print("\nhtmx info:")
    print(f"  Difficulty: {htmx_info['difficulty']}/10")
    print(f"  Category: {htmx_info['category']}")
    print(f"  Keywords: {', '.join(htmx_info['keywords'])}")

    print("\nBun info:")
    print(f"  Difficulty: {bun_info['difficulty']}/10")
    print(f"  Category: {bun_info['category']}")
    print(f"  Keywords: {', '.join(bun_info['keywords'])}")


def demo_real_world_scenario():
    """Demo: Real-world scenario with mixed known/unknown techs."""
    print_section("4. Real-World Scenario: Job Description")

    job_desc = """
    We're looking for a senior full-stack engineer with experience in:
    - React and TypeScript (required)
    - Bun or Node.js for backend
    - htmx for progressive enhancement
    - Postgres database
    - Docker and some new-framework-xyz we're experimenting with
    """

    print(f"\nJob Description:\n{job_desc}")

    registry = get_tech_registry()

    # Add the new techs (would happen via admin UI in production)
    registry.add_custom_tech("htmx", 4.0, "frontend", ["htmx"])
    registry.add_custom_tech("bun", 5.3, "backend", ["bun"])

    techs_to_check = ["react", "typescript", "bun", "node", "htmx", "postgres", "docker", "new-framework-xyz"]

    print("\n\nExtracted Technologies:")
    print("-" * 60)

    for tech in techs_to_check:
        info = registry.get_tech_info(tech)
        confidence_indicator = "✅" if info.get("confidence", 1.0) > 0.5 else "⚠️"
        print(f"\n{confidence_indicator} {tech.upper()}")
        print(f"   Difficulty: {info['difficulty']:.1f}/10")
        print(f"   Category: {info['category']}")
        print(f"   Confidence: {info.get('confidence', 1.0)}")

        if info.get("source") == "fallback":
            print("   ⚠️  Unknown tech - using fallback defaults")


def main():
    """Run all demos."""
    print("\n")
    print("╔════════════════════════════════════════════════════════════╗")
    print("║  Tech Registry Demo: Handling New Technologies            ║")
    print("╚════════════════════════════════════════════════════════════╝")

    demo_known_tech()
    demo_new_tech_unknown()
    demo_add_new_tech()
    demo_real_world_scenario()

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(
        """
✅ Known techs: Loaded from baseline database
✅ Unknown techs: Graceful fallback with defaults
✅ New techs: Can be added dynamically (no code deploy!)
✅ Alternatives: Found automatically by category
✅ Future-proof: System never breaks, just lower confidence

Next Steps:
1. Integrate with SimpleTechExtractor / MLTechExtractor
2. Add external API enrichment (GitHub stars, SO questions)
3. Build admin UI for adding new techs
4. Add ML-based difficulty prediction from metrics
    """
    )


if __name__ == "__main__":
    main()
