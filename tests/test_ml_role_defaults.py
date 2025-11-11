#!/usr/bin/env python3
"""
Tests for generic role phrase fallbacks in MLTechExtractor
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.ml_tech_extractor import MLTechExtractor  # noqa: E402


def test_ml_full_stack_defaults():
    extractor = MLTechExtractor()
    result = extractor.extract_technologies("need a full stack developer")
    techs = result["technologies"]
    assert "react" in techs
    assert "node" in techs
    assert "postgres" in techs
    assert "docker" in techs


def test_ml_frontend_defaults():
    extractor = MLTechExtractor()
    result = extractor.extract_technologies("hiring a frontend engineer")
    techs = result["technologies"]
    assert "react" in techs
    assert "typescript" in techs


def test_ml_semantic_polyglot():
    """Test semantic fallback for 'polyglot engineer' → fullstack."""
    extractor = MLTechExtractor()
    result = extractor.extract_technologies("need a polyglot engineer")
    techs = result["technologies"]
    # Should map to fullstack defaults
    assert "react" in techs
    assert "node" in techs
    assert "postgres" in techs
    assert "docker" in techs


def test_ml_semantic_product_engineer():
    """Test semantic fallback for 'product engineer' → fullstack."""
    extractor = MLTechExtractor()
    result = extractor.extract_technologies("looking for a product engineer")
    techs = result["technologies"]
    assert "react" in techs
    assert "node" in techs


def test_ml_semantic_platform_engineer():
    """Test semantic fallback for 'platform engineer' → devops."""
    extractor = MLTechExtractor()
    result = extractor.extract_technologies("hiring a platform engineer")
    techs = result["technologies"]
    assert "docker" in techs
    assert "kubernetes" in techs
    assert "aws" in techs
