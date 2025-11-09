#!/usr/bin/env python3
"""Final comprehensive test of simplified schema."""

import json
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_schema_structure():
    """Test the final schema structure matches requirements."""
    scorer = SoftwareComplexityScorer()
    
    prompts = [
        "build a twitter clone with react",
        "create an e-commerce platform with nextjs and postgres",
        "need to develop a real-time chat app"
    ]
    
    for prompt in prompts:
        print("=" * 80)
        print(f"Testing: {prompt}")
        print("=" * 80)
        
        result = scorer.analyze_text(prompt)
        simplified = scorer.get_simplified_output()
        
        # Check top-level fields
        assert "estimated_no_of_lines" in simplified, "Missing estimated_no_of_lines"
        assert isinstance(simplified["estimated_no_of_lines"], str), "LOC should be string (human readable)"
        assert "," in simplified["estimated_no_of_lines"] or int(simplified["estimated_no_of_lines"]) < 1000, "LOC should be formatted with commas"
        
        assert "human_lines_per_second" in simplified, "Missing human_lines_per_second"
        assert isinstance(simplified["human_lines_per_second"], float), "human_lines_per_second should be float"
        
        assert "ai_lines_per_second" in simplified, "Missing ai_lines_per_second"
        assert isinstance(simplified["ai_lines_per_second"], float), "ai_lines_per_second should be float"
        
        assert "human_to_ai_ratio_min" in simplified, "Missing human_to_ai_ratio_min"
        assert isinstance(simplified["human_to_ai_ratio_min"], float), "human_to_ai_ratio_min should be float"
        
        assert "estimated_ai_time" in simplified, "Missing estimated_ai_time"
        assert isinstance(simplified["estimated_ai_time"], str), "estimated_ai_time should be string with readable format"
        
        assert "estimated_human_time_min" in simplified, "Missing estimated_human_time_min"
        assert isinstance(simplified["estimated_human_time_min"], str), "estimated_human_time_min should be string"
        
        assert "estimated_human_time_average" in simplified, "Missing estimated_human_time_average"
        assert isinstance(simplified["estimated_human_time_average"], str), "estimated_human_time_average should be string"
        
        assert "technologies" in simplified, "Missing technologies"
        assert isinstance(simplified["technologies"], dict), "technologies should be dict"
        
        # Check technology fields
        techs = simplified["technologies"]
        if techs:
            sample_tech = next(iter(techs.values()))
            
            # Required fields
            assert "estimated_lines_of_code" in sample_tech, "Missing estimated_lines_of_code in tech"
            assert isinstance(sample_tech["estimated_lines_of_code"], str), "Tech LOC should be string"
            
            assert "estimated_human_time_min" in sample_tech, "Missing estimated_human_time_min in tech"
            assert isinstance(sample_tech["estimated_human_time_min"], float), "Tech time_min should be float"
            
            assert "estimated_human_time_min_readable" in sample_tech, "Missing readable time_min in tech"
            
            assert "estimated_human_time_average" in sample_tech, "Missing estimated_human_time_average in tech"
            assert isinstance(sample_tech["estimated_human_time_average"], float), "Tech time_avg should be float"
            
            assert "estimated_human_time_average_readable" in sample_tech, "Missing readable time_avg in tech"
            
            assert "mentioned_in_prompt" in sample_tech, "Missing mentioned_in_prompt in tech"
            assert isinstance(sample_tech["mentioned_in_prompt"], bool), "mentioned_in_prompt should be boolean"
            
            assert "recommended_in_standard_system_design" in sample_tech, "Missing recommended_in_standard_system_design in tech"
            assert isinstance(sample_tech["recommended_in_standard_system_design"], bool), "recommended should be boolean"
            
            assert "contribution_to_lines_of_code" in sample_tech, "Missing contribution_to_lines_of_code in tech"
            assert isinstance(sample_tech["contribution_to_lines_of_code"], float), "contribution should be float"
            
            assert "complexity_score" in sample_tech, "Missing complexity_score in tech"
            assert isinstance(sample_tech["complexity_score"], int), "complexity_score should be int"
        
        print(f"\n✅ Schema structure validated")
        print(f"   LOC: {simplified['estimated_no_of_lines']}")
        print(f"   Technologies: {len(techs)}")
        
        # Check mentioned technologies
        mentioned = [k for k, v in techs.items() if v.get('mentioned_in_prompt')]
        if mentioned:
            print(f"   Mentioned in prompt: {', '.join(mentioned)}")
        
        print()

if __name__ == "__main__":
    test_schema_structure()
    print("\n" + "=" * 80)
    print("✅ ALL TESTS PASSED - Schema matches requirements exactly!")
    print("=" * 80)
