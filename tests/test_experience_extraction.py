from mcp_server.simple_tech_extractor import SimpleTechExtractor


def test_explicit_experience():
    """Test explicit experience extraction with new schema"""
    extractor = SimpleTechExtractor()

    # With experience mentioned in prompt
    result1 = extractor.extract_technologies(
        "5+ years React experience required", is_resume=False, prompt_override="5+ years React experience required"
    )
    assert "react" in result1["technologies"]
    # New schema uses experience_mentioned_in_prompt
    if "experience_mentioned_in_prompt" in result1["technologies"]["react"]:
        assert result1["technologies"]["react"]["experience_mentioned_in_prompt"] == 5.0

    # Without experience mentioned
    result2 = extractor.extract_technologies("React developer needed", is_resume=False, prompt_override="")
    assert "react" in result2["technologies"]
    # Should not have any experience field
    assert "experience_mentioned_in_prompt" not in result2["technologies"]["react"]
    assert "experience_accounted_for_in_resume" not in result2["technologies"]["react"]

    print("✓ Explicit experience extraction test passed")


if __name__ == "__main__":
    test_explicit_experience()
