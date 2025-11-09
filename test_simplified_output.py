#!/usr/bin/env python3
"""Test simplified output schema and software detection improvements."""

import json
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_software_detection():
    """Test that 'develop a twitter clone' is correctly identified as software."""
    scorer = SoftwareComplexityScorer()
    
    test_cases = [
        "need to develop a twitter clone",
        "build a twitter clone",
        "create an instagram clone",
        "implement a youtube clone",
        "I need someone to look after my dad"  # Should fail
    ]
    
    print("=" * 80)
    print("Testing Software Detection")
    print("=" * 80)
    
    for prompt in test_cases:
        result = scorer.analyze_text(prompt)
        is_error = "error" in result
        
        print(f"\nPrompt: {prompt}")
        if is_error:
            print(f"  ❌ Not detected as software: {result.get('error')}")
            print(f"  Software probability: {result.get('software_probability')}")
        else:
            print(f"  ✅ Detected as software")
            print(f"  Technologies: {len(result.get('technologies', {}))} categories")
            print(f"  LOC: {result.get('predicted_lines_of_code', 0):,}")
    
    print("\n")

def test_simplified_schema():
    """Test the simplified output schema."""
    scorer = SoftwareComplexityScorer()
    
    prompt = "need to develop a twitter clone"
    
    print("=" * 80)
    print("Testing Simplified Output Schema")
    print("=" * 80)
    print(f"\nPrompt: {prompt}\n")
    
    # First analyze
    full_result = scorer.analyze_text(prompt)
    
    if "error" in full_result:
        print(f"❌ Failed: {full_result['error']}")
        return
    
    # Get simplified output
    simplified = scorer.get_simplified_output()
    
    if "error" in simplified:
        print(f"❌ Failed: {simplified['error']}")
        return
    
    # Verify required fields
    required_fields = [
        "estimated_no_of_lines",
        "human_lines_per_second",
        "ai_lines_per_second",
        "human_to_ai_ratio_min",
        "estimated_ai_time",
        "estimated_human_time_min",
        "estimated_human_time_average",
        "technologies"
    ]
    
    missing = [f for f in required_fields if f not in simplified]
    if missing:
        print(f"❌ Missing fields: {missing}")
        return
    
    print("✅ All required fields present\n")
    
    # Display summary
    print(f"Estimated LOC: {simplified['estimated_no_of_lines']}")
    print(f"AI lines/sec: {simplified['ai_lines_per_second']}")
    print(f"Human lines/sec: {simplified['human_lines_per_second']}")
    print(f"Human/AI ratio (min): {simplified['human_to_ai_ratio_min']}")
    print(f"\nAI time: {simplified['estimated_ai_time']}")
    print(f"Human time (min): {simplified['estimated_human_time_min']}")
    print(f"Human time (avg): {simplified['estimated_human_time_average']}")
    
    # Show sample technologies
    techs = simplified.get('technologies', {})
    print(f"\nTechnologies detected: {len(techs)}")
    
    mentioned = [t for t, v in techs.items() if v.get('mentioned_in_prompt')]
    recommended = [t for t, v in techs.items() if v.get('recommended_in_standard_system_design')]
    
    print(f"  Mentioned in prompt: {len(mentioned)}")
    print(f"  Recommended by system design: {len(recommended)}")
    
    # Show first 5 technologies
    print("\nSample technology breakdown:")
    for i, (tech, data) in enumerate(list(techs.items())[:5]):
        print(f"\n  {tech}:")
        print(f"    LOC: {data['estimated_lines_of_code']}")
        print(f"    Human time (min): {data['estimated_human_time_min']} hours ({data['estimated_human_time_min_readable']})")
        print(f"    Human time (avg): {data['estimated_human_time_average']} hours ({data['estimated_human_time_average_readable']})")
        print(f"    Mentioned: {data['mentioned_in_prompt']}")
        print(f"    Recommended: {data['recommended_in_standard_system_design']}")
        print(f"    Contribution: {data['contribution_to_lines_of_code']}%")
        print(f"    Complexity: {data['complexity_score']}")
        if 'default_cli_code' in data:
            print(f"    CLI: {data['default_cli_code']}")
    
    # Save to file for inspection
    with open('test_simplified_output.json', 'w', encoding='utf-8') as f:
        json.dump(simplified, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Full output saved to: test_simplified_output.json")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_software_detection()
    test_simplified_schema()
