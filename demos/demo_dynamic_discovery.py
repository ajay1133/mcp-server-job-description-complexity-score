#!/usr/bin/env python3
"""
Demo: Dynamic Technology Discovery

This demonstrates how the system now handles unknown technologies
without needing to hardcode every single one in tech_db.

The TechRegistry dynamically discovers technologies using:
1. Pattern matching for common tech names
2. External data sources (GitHub, NPM, etc.)
3. Semantic similarity for alternatives
4. Difficulty estimation based on community metrics
"""

import json

from mcp_server.simple_tech_extractor import SimpleTechExtractor


def demo():
    extractor = SimpleTechExtractor()

    print("=" * 80)
    print("DEMO: Dynamic Technology Discovery")
    print("=" * 80)
    print("\nThis system can now discover technologies that aren't hardcoded!")
    print("It uses TechRegistry to fetch info from external sources and caches it.")

    # Test cases with various unknown technologies
    test_cases = [
        {
            "name": "Grafana (Monitoring Tool)",
            "prompt": "need dev experienced with 5+ years in grafana",
            "description": "Grafana wasn't in tech_db, but was discovered dynamically",
        },
        {
            "name": "Terraform (Infrastructure as Code)",
            "prompt": "Terraform expert with 3 years experience required",
            "description": "Terraform discovered and experience extracted",
        },
        {
            "name": "Prometheus (Monitoring)",
            "prompt": "looking for Prometheus monitoring specialist",
            "description": "Prometheus discovered, alternatives suggested",
        },
        {
            "name": "Datadog (Observability)",
            "prompt": "senior Datadog engineer needed",
            "description": "Seniority level converted to experience range",
        },
        {
            "name": "Jenkins (CI/CD)",
            "prompt": "need Jenkins automation engineer with 4+ years",
            "description": "Jenkins discovered with experience",
        },
        {
            "name": "Ansible (Configuration Management)",
            "prompt": "mid-level Ansible developer",
            "description": "Seniority converted to experience range",
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print("\n" + ("-" * 80))
        print(f"Test {i}: {test['name']}")
        print("-" * 80)
        print(f"Prompt: {test['prompt']}")
        print(f"Expected: {test['description']}")

        result = extractor.extract_technologies(test['prompt'], is_resume=False)

        print("\nResult:")
        if result['technologies']:
            for tech_name, tech_data in result['technologies'].items():
                print(f"  • Technology: {tech_name}")
                print(f"    Difficulty: {tech_data.get('difficulty', 'N/A')}")
                print(f"    Category: {tech_data.get('category', 'N/A')}")

                exp = tech_data.get('experience_mentioned')
                if exp:
                    print(f"    Experience: {exp}")

                if tech_data.get('alternatives'):
                    print(f"    Alternatives: {', '.join(tech_data['alternatives'].keys())}")
        else:
            print("  No technologies detected")

    print("\n" + "=" * 80)
    print("How It Works:")
    print("=" * 80)
    print(
        """
1. Pattern Matching: Regex patterns detect common tech names (grafana, terraform, etc.)
2. TechRegistry Lookup: Checks cache first, then fetches from external sources
3. Difficulty Estimation: Uses GitHub stars, NPM downloads, or heuristics
4. Alternative Discovery: Finds similar technologies using semantic search
5. Experience Extraction: Works the same way as hardcoded technologies
6. Caching: Results are cached locally to avoid repeated API calls

Benefits:
✓ No need to manually add every new technology
✓ Self-learning system that improves over time
✓ Falls back gracefully if external sources fail
✓ Consistent experience extraction across all techs
✓ Alternatives suggested automatically based on similarity
    """
    )

    print("=" * 80)
    print("Example: Full Output for Grafana")
    print("=" * 80)
    result = extractor.extract_technologies("need dev experienced with 5+ years in grafana", is_resume=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    demo()
