import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402


def test_experience_mentioned_with_years():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "Seeking React and Node engineers with 5 years experience",
        is_resume=False,
        prompt_override="5 years experience",
    )
    # Should have experience_mentioned as number
    assert result["technologies"]["react"]["experience_mentioned"] == 5
    assert result["technologies"]["node"]["experience_mentioned"] == 5


def test_experience_mentioned_with_seniority():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies(
        "Senior React developer", is_resume=False, prompt_override="Senior React developer"
    )
    # Seniority no longer maps to years; should NOT include experience_mentioned field
    assert "experience_mentioned" not in result["technologies"]["react"]


def test_experience_mentioned_years_and_propagation():
    extractor = SimpleTechExtractor()
    prompt = "We need a React developer. 5+ years experience required."
    out = extractor.extract_technologies(prompt, is_resume=False, prompt_override=prompt)
    react = out["technologies"]["react"]
    # Expect experience_mentioned as range string due to '+' qualifier
    assert "experience_mentioned" in react
    assert react["experience_mentioned"] == ">= 5 years"
    # Alternatives should inherit
    for alt in react["alternatives"].values():
        assert "experience_mentioned" in alt
        assert alt["experience_mentioned"] == ">= 5 years"


def test_experience_mentioned_priority_resume():
    extractor = SimpleTechExtractor()
    resume_text = "Senior engineer with 4 years React experience"
    prompt_text = "Junior developer position"
    out = extractor.extract_technologies(resume_text, is_resume=True, prompt_override=prompt_text)
    react = out["technologies"]["react"]
    # Resume years should take priority over junior prompt
    assert react.get("experience_mentioned") == 4


def test_no_experience_mentioned_no_field():
    extractor = SimpleTechExtractor()
    result = extractor.extract_technologies("React developer needed", is_resume=False, prompt_override="")
    # Should not have experience_mentioned field when not mentioned
    assert "experience_mentioned" not in result["technologies"]["react"]


if __name__ == "__main__":
    test_experience_mentioned_with_years()
    test_experience_mentioned_with_seniority()
    test_experience_mentioned_years_and_propagation()
    test_experience_mentioned_priority_resume()
    test_no_experience_mentioned_no_field()
    print("✓ experience_mentioned tests passed")
