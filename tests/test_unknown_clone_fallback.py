import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.ml_tech_extractor import MLTechExtractor  # noqa: E402
from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402


def test_unknown_clone_simple():
    ext = SimpleTechExtractor()
    out = ext.extract_technologies("perplexity copy")
    techs = out["technologies"]
    # Should infer generic stack
    assert set(techs.keys()) >= {"react", "node", "postgres"}
    # No experience_mentioned field when not specified
    for t in techs.values():
        assert "experience_mentioned" not in t


def test_unknown_clone_ml():
    ext = MLTechExtractor()
    out = ext.extract_technologies("perplexity clone")
    techs = out["technologies"]
    assert set(techs.keys()) >= {"react", "node", "postgres"}
    for t in techs.values():
        assert "experience_mentioned" not in t


if __name__ == "__main__":
    test_unknown_clone_simple()
    test_unknown_clone_ml()
    print("✓ unknown clone fallback tests passed")
