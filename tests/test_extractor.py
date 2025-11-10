#!/usr/bin/env python3
"""
Test suite for SimpleTechExtractor
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.simple_tech_extractor import SimpleTechExtractor


def test_basic_extraction():
    """Test basic technology detection"""
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("Looking for React and Node.js developer")
    
    assert "react" in result["technologies"]
    assert "node" in result["technologies"]
    # New schema: no mentioned_in_prompt field, technologies are simply detected
    assert "difficulty" in result["technologies"]["react"]
    assert "difficulty" in result["technologies"]["node"]


def test_difficulty_ratings():
    """Test difficulty ratings are present"""
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("React developer needed")
    
    react_info = result["technologies"]["react"]
    assert "difficulty" in react_info
    # experience_required only present if explicitly mentioned
    assert react_info["difficulty"] > 0


def test_alternatives():
    """Test alternatives are provided"""
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("React expert required")
    
    react_info = result["technologies"]["react"]
    assert "alternatives" in react_info
    assert len(react_info["alternatives"]) > 0
    assert "vue" in react_info["alternatives"]
    assert react_info["alternatives"]["vue"]["difficulty"] > 0
    # Alternatives no longer include experience_required by default


def test_multiple_technologies():
    """Test multiple technology detection"""
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "Full-Stack Engineer with React, Node.js, PostgreSQL, and Docker"
    )
    
    techs = result["technologies"]
    assert "react" in techs
    assert "node" in techs
    assert "postgres" in techs
    assert "docker" in techs


def test_empty_prompt():
    """Test empty prompt returns no technologies"""
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("")
    
    assert result["technologies"] == {}


def test_no_tech_prompt():
    """Test prompt with no technologies returns empty"""
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("Looking for a sales manager")
    
    assert result["technologies"] == {}


if __name__ == "__main__":
    import sys
    
    tests = [
        test_basic_extraction,
        test_difficulty_ratings,
        test_alternatives,
        test_multiple_technologies,
        test_empty_prompt,
        test_no_tech_prompt
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
