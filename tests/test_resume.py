#!/usr/bin/env python3
"""
Test resume skill extraction
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.simple_tech_extractor import SimpleTechExtractor


def test_resume_extraction():
    """Test resume skill extraction with new experience fields"""
    extractor = SimpleTechExtractor()
    
    resume = """
    John Doe
    Senior Full-Stack Developer
    
    EXPERIENCE:
    - 5 years React development at Tech Corp
    - 3 years Node.js backend development
    - PostgreSQL database management
    - Docker containerization
    
    SKILLS:
    React, Node.js, PostgreSQL, Docker, TypeScript
    """
    
    result = extractor.extract_technologies(resume, is_resume=True)
    
    print("Resume extraction test:")
    print(f"Detected: {list(result['technologies'].keys())}")
    
    # Check React
    assert "react" in result["technologies"]
    react_info = result["technologies"]["react"]
    # New schema: should have experience_accounted_for_in_resume
    assert "experience_accounted_for_in_resume" in react_info
    assert react_info["experience_accounted_for_in_resume"] == 5.0
    # Should have GitHub placeholder
    assert "experience_validated_via_github" in react_info
    assert react_info["experience_validated_via_github"] is None
    
    # Check Node.js
    assert "node" in result["technologies"]
    node_info = result["technologies"]["node"]
    assert "experience_accounted_for_in_resume" in node_info
    assert node_info["experience_accounted_for_in_resume"] == 3.0
    
    # Check PostgreSQL (no years mentioned)
    assert "postgres" in result["technologies"]
    postgres_info = result["technologies"]["postgres"]
    # No explicit years, so should not have experience field
    assert "experience_accounted_for_in_resume" not in postgres_info
    
    print("✓ Resume extraction test passed")


def test_job_description_extraction():
    """Test job description with prompt experience"""
    extractor = SimpleTechExtractor()
    
    job_desc = "Looking for 5+ years React and Node.js experience"
    
    result = extractor.extract_technologies(job_desc, is_resume=False, prompt_override=job_desc)
    
    print("\nJob description extraction test:")
    print(f"Detected: {list(result['technologies'].keys())}")
    
    # Check React
    assert "react" in result["technologies"]
    react_info = result["technologies"]["react"]
    # Should have experience_mentioned_in_prompt
    assert "experience_mentioned_in_prompt" in react_info
    assert react_info["experience_mentioned_in_prompt"] == 5.0
    # Should NOT have resume experience field
    assert "experience_accounted_for_in_resume" not in react_info
    
    print("✓ Job description extraction test passed")


def test_resume_without_experience():
    """Test resume without explicit years"""
    extractor = SimpleTechExtractor()
    
    resume = """
    Skills: React, Vue, MongoDB, Kubernetes
    """
    
    result = extractor.extract_technologies(resume, is_resume=True)
    
    print("\nResume without experience test:")
    print(f"Detected: {list(result['technologies'].keys())}")
    
    for tech, info in result["technologies"].items():
        # Should NOT have experience fields (no explicit years)
        assert "experience_accounted_for_in_resume" not in info
        assert "experience_mentioned_in_prompt" not in info
        # Should have GitHub placeholder
        assert "experience_validated_via_github" in info
    
    print("✓ Resume without experience test passed")


if __name__ == "__main__":
    test_resume_extraction()
    test_job_description_extraction()
    test_resume_without_experience()
    print("\n✅ All resume tests passed!")
