#!/usr/bin/env python3
"""
Test resume file parsing and new experience fields
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.simple_tech_extractor import SimpleTechExtractor
from mcp_server.resume_parser import parse_resume_file
import tempfile


def test_new_experience_fields():
    """Test new experience field structure"""
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
    
    prompt = "Looking for 7 years overall experience with React and Node.js"
    
    result = extractor.extract_technologies(resume, is_resume=True, prompt_override=prompt)
    
    print("Test 1: Resume + Prompt with overall experience")
    print(f"Detected: {list(result['technologies'].keys())}")
    
    # Check React
    if "react" in result["technologies"]:
        react_info = result["technologies"]["react"]
        print(f"\nReact fields: {list(react_info.keys())}")
        
        # Should have experience from resume
        if "experience_accounted_for_in_resume" in react_info:
            print(f"  experience_accounted_for_in_resume: {react_info['experience_accounted_for_in_resume']}")
            assert react_info["experience_accounted_for_in_resume"] == 5.0
        
        # Should have experience from prompt (overall fallback)
        if "experience_mentioned_in_prompt" in react_info:
            print(f"  experience_mentioned_in_prompt: {react_info['experience_mentioned_in_prompt']}")
            assert react_info["experience_mentioned_in_prompt"] == 7.0
        
        # Should have GitHub placeholder
        assert "experience_validated_via_github" in react_info
        assert react_info["experience_validated_via_github"] is None
        print(f"  experience_validated_via_github: {react_info['experience_validated_via_github']}")
    
    # Check Node.js
    if "node" in result["technologies"]:
        node_info = result["technologies"]["node"]
        if "experience_accounted_for_in_resume" in node_info:
            assert node_info["experience_accounted_for_in_resume"] == 3.0
            print(f"\nNode.js experience_accounted_for_in_resume: {node_info['experience_accounted_for_in_resume']}")
    
    print("\n✓ Test 1 passed")


def test_tech_specific_prompt_experience():
    """Test tech-specific experience in prompt overrides overall"""
    extractor = SimpleTechExtractor()
    
    resume = """
    Skills: React, Vue, Node.js
    Experience: 2 years with React
    """
    
    prompt = "Need 5 years overall experience and 8 years React specifically"
    
    result = extractor.extract_technologies(resume, is_resume=True, prompt_override=prompt)
    
    print("\n\nTest 2: Tech-specific experience in prompt")
    
    if "react" in result["technologies"]:
        react_info = result["technologies"]["react"]
        # Should use tech-specific (8) not overall (5)
        if "experience_mentioned_in_prompt" in react_info:
            print(f"React experience_mentioned_in_prompt: {react_info['experience_mentioned_in_prompt']}")
            assert react_info["experience_mentioned_in_prompt"] == 8.0
        
        if "experience_accounted_for_in_resume" in react_info:
            print(f"React experience_accounted_for_in_resume: {react_info['experience_accounted_for_in_resume']}")
            assert react_info["experience_accounted_for_in_resume"] == 2.0
    
    if "vue" in result["technologies"]:
        vue_info = result["technologies"]["vue"]
        # Should use overall (5) since no tech-specific
        if "experience_mentioned_in_prompt" in vue_info:
            print(f"Vue experience_mentioned_in_prompt: {vue_info['experience_mentioned_in_prompt']}")
            assert vue_info["experience_mentioned_in_prompt"] == 5.0
    
    print("✓ Test 2 passed")


def test_resume_only_no_prompt():
    """Test resume without prompt context"""
    extractor = SimpleTechExtractor()
    
    resume = """
    Skills: Python, Django, PostgreSQL
    - 4 years Python development
    - 3 years Django framework
    """
    
    result = extractor.extract_technologies(resume, is_resume=True, prompt_override="")
    
    print("\n\nTest 3: Resume only (no prompt)")
    print(f"Detected: {list(result['technologies'].keys())}")
    
    if "python_django" in result["technologies"]:
        django_info = result["technologies"]["python_django"]
        
        # Should have resume experience
        if "experience_accounted_for_in_resume" in django_info:
            print(f"Django experience_accounted_for_in_resume: {django_info['experience_accounted_for_in_resume']}")
            assert django_info["experience_accounted_for_in_resume"] == 3.0
        
        # Should NOT have prompt experience
        assert "experience_mentioned_in_prompt" not in django_info
        
        # Should have GitHub placeholder
        assert "experience_validated_via_github" in django_info
    
    print("✓ Test 3 passed")


def test_txt_file_parsing():
    """Test parsing .txt resume file"""
    print("\n\nTest 4: Parsing .txt file")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
        f.write("""
        John Smith
        Software Engineer
        
        Experience: 6 years with JavaScript, React, and Node.js
        
        Skills: React, Node.js, MongoDB, Docker
        """)
        temp_file = f.name
    
    try:
        text = parse_resume_file(temp_file)
        print(f"Extracted text length: {len(text)} chars")
        print(f"Contains 'React': {'react' in text.lower()}")
        print(f"Contains 'Node': {'node' in text.lower()}")
        
        assert 'react' in text.lower()
        assert 'node' in text.lower()
        assert '6 years' in text.lower()
        
        print("✓ Test 4 passed")
    finally:
        os.unlink(temp_file)


if __name__ == "__main__":
    test_new_experience_fields()
    test_tech_specific_prompt_experience()
    test_resume_only_no_prompt()
    test_txt_file_parsing()
    print("\n\n✅ All resume file tests passed!")
