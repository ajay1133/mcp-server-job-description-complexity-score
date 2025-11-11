#!/usr/bin/env python3
"""Demo showing experience_mentioned field with different inputs."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402


def demo_with_years():
    print("=" * 80)
    print("Test 1: With explicit years (5 years)")
    print("=" * 80)

    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "Looking for React developer with 5 years experience",
        is_resume=False,
        prompt_override="Looking for React developer with 5 years experience",
    )

    print(json.dumps(result, indent=2))
    print()


def demo_with_seniority():
    print("=" * 80)
    print("Test 2: With seniority level (senior)")
    print("=" * 80)

    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "Senior React developer needed", is_resume=False, prompt_override="Senior React developer needed"
    )

    print(json.dumps(result, indent=2))
    print()


def demo_with_junior():
    print("=" * 80)
    print("Test 3: With junior level")
    print("=" * 80)

    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "Junior Node.js developer position", is_resume=False, prompt_override="Junior Node.js developer position"
    )

    print(json.dumps(result, indent=2))
    print()


def demo_no_experience():
    print("=" * 80)
    print("Test 4: No experience mentioned")
    print("=" * 80)

    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("React and Node developer needed", is_resume=False, prompt_override="")

    print(json.dumps(result, indent=2))
    print()


def demo_with_resume():
    print("=" * 80)
    print("Test 5: With resume (3 years React experience)")
    print("=" * 80)

    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "I have 3 years experience with React and Node", is_resume=True, prompt_override="Senior developer position"
    )

    print(json.dumps(result, indent=2))
    print()


if __name__ == "__main__":
    demo_with_years()
    demo_with_seniority()
    demo_with_junior()
    demo_no_experience()
    demo_with_resume()
