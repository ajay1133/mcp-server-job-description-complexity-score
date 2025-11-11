#!/usr/bin/env python3
"""Tests for dynamic technology registry."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.tech_registry import TechRegistry, get_tech_registry  # noqa: E402


def test_baseline_techs():
    """Test that baseline technologies are available."""
    registry = TechRegistry()

    # Known baseline techs
    react_info = registry.get_tech_info("react")
    assert react_info is not None
    assert react_info["category"] == "frontend"
    assert react_info["difficulty"] > 0

    node_info = registry.get_tech_info("node")
    assert node_info is not None
    assert node_info["category"] == "backend"


def test_new_tech_fallback():
    """Test that unknown tech gets sensible defaults."""
    registry = TechRegistry()

    # Brand new tech that doesn't exist yet (hypothetical)
    new_tech = registry.get_tech_info("hyperscript-2025")
    assert new_tech is not None
    assert new_tech["difficulty"] == 5.0  # Default mid-range
    assert new_tech["category"] == "other"
    assert new_tech["confidence"] == 0.3  # Low confidence
    assert new_tech["source"] == "fallback"


def test_add_custom_tech():
    """Test manually adding a new technology."""
    registry = TechRegistry()

    # Add a brand new tech
    registry.add_custom_tech(
        tech_name="htmx",
        difficulty=4.0,
        category="frontend",
        keywords=["htmx", "hypermedia"],
    )

    # Verify it's available
    htmx_info = registry.get_tech_info("htmx")
    assert htmx_info is not None
    assert htmx_info["difficulty"] == 4.0
    assert htmx_info["category"] == "frontend"
    assert "htmx" in htmx_info["keywords"]


def test_similar_techs():
    """Test finding similar technologies."""
    registry = TechRegistry()

    # Find alternatives to React
    alternatives = registry.search_similar_techs("react", top_k=3)
    assert len(alternatives) > 0
    # Should include other frontend frameworks
    assert any(alt in ["vue", "svelte", "angular"] for alt in alternatives)


def test_singleton():
    """Test that get_tech_registry returns same instance."""
    registry1 = get_tech_registry()
    registry2 = get_tech_registry()
    assert registry1 is registry2


def test_get_all_keywords():
    """Test getting all keywords for pattern matching."""
    registry = TechRegistry()
    keywords = registry.get_all_keywords()

    assert "react" in keywords
    assert "react" in keywords["react"]
    assert "reactjs" in keywords["react"]
