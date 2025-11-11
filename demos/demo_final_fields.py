#!/usr/bin/env python3
"""Final demo showing all the new field behaviors."""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402

print("=" * 80)
print("SUMMARY OF CHANGES")
print("=" * 80)
print()
print("1. Renamed 'mentioned_explicitly' → 'experience_mentioned'")
print("2. Values: number (e.g., 5) or string ('>= 5 years', '<= 2 years')")
print("3. Only included when experience is mentioned in prompt or resume")
print("4. 'experience_validated_via_github' only included when resume is passed")
print()
print("=" * 80)
print()

extractor = SimpleTechExtractor()

# Test 1: Explicit years
print("Test 1: Explicit years (5)")
print("-" * 80)
result = extractor.extract_technologies(
    "React developer with 5 years experience",
    is_resume=False,
    prompt_override="React developer with 5 years experience",
)
print(json.dumps(result["technologies"]["react"], indent=2))
print()

# Test 2: Senior = ">= 5 years"
print("Test 2: Senior level = '>= 5 years'")
print("-" * 80)
result = extractor.extract_technologies(
    "Senior Node.js developer", is_resume=False, prompt_override="Senior Node.js developer"
)
print(json.dumps(result["technologies"]["node"], indent=2))
print()

# Test 3: Mid = ">= 3 years"
print("Test 3: Mid level = '>= 3 years'")
print("-" * 80)
result = extractor.extract_technologies(
    "Mid-level Python developer", is_resume=False, prompt_override="Mid-level Python developer"
)
print(json.dumps(result["technologies"], indent=2))
print()

# Test 4: Junior = "<= 2 years"
print("Test 4: Junior level = '<= 2 years'")
print("-" * 80)
result = extractor.extract_technologies(
    "Junior React developer", is_resume=False, prompt_override="Junior React developer"
)
print(json.dumps(result["technologies"]["react"], indent=2))
print()

# Test 5: No experience mentioned - field not present
print("Test 5: No experience mentioned - field not present")
print("-" * 80)
result = extractor.extract_technologies("React and Node developer", is_resume=False, prompt_override="")
print(json.dumps(result["technologies"]["react"], indent=2))
print()

# Test 6: With resume - experience_validated_via_github included
print("Test 6: With resume - experience_validated_via_github included")
print("-" * 80)
result = extractor.extract_technologies(
    "I have 4 years React experience", is_resume=True, prompt_override="Senior position"
)
print(json.dumps(result["technologies"]["react"], indent=2))
print()

# Test 7: Without resume - experience_validated_via_github not included
print("Test 7: Without resume - experience_validated_via_github not included")
print("-" * 80)
result = extractor.extract_technologies(
    "React developer with 3 years experience",
    is_resume=False,
    prompt_override="React developer with 3 years experience",
)
print(json.dumps(result["technologies"]["react"], indent=2))
print()
