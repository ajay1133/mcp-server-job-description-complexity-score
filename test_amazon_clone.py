#!/usr/bin/env python3
"""Quick test for project description extraction."""

from mcp_server.simple_tech_extractor import SimpleTechExtractor

extractor = SimpleTechExtractor()

tests = [
    "amazon copy site",
    "build a netflix clone",
    "twitter like app",
    "uber for delivery",
    "create airbnb site",
    "spotify clone",
]

for test in tests:
    result = extractor.extract_technologies(test)
    techs = list(result["technologies"].keys())
    print(f"\n'{test}'")
    print(f"  → {len(techs)} technologies: {', '.join(techs[:5])}")
    if len(techs) == 0:
        print("  ❌ No match found")
