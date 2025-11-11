#!/usr/bin/env python3
"""
Tests for generic role phrase fallbacks in SimpleTechExtractor
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402


def test_full_stack_defaults():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("need a full stack developer")
    techs = result["technologies"]
    # Should inject representative stack
    assert "react" in techs
    assert "node" in techs
    assert "postgres" in techs
    assert "docker" in techs


def test_frontend_defaults():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("hiring a frontend engineer")
    techs = result["technologies"]
    assert "react" in techs
    assert "typescript" in techs


def test_backend_defaults():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("looking for a backend developer")
    techs = result["technologies"]
    assert "node" in techs or "python_fastapi" in techs
    assert "postgres" in techs


def test_devops_defaults():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("need a devops engineer")
    techs = result["technologies"]
    assert "docker" in techs
    assert "kubernetes" in techs
    assert "aws" in techs
