#!/usr/bin/env python3
"""Test the formatted detailed string feature."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
import json

def test_formatted_output():
    """Test the formatted_detailed_string in the simplified output."""
    scorer = SoftwareComplexityScorer()
    
    test_cases = [
        "I want someone to create a resume parser for pdf and docs using aws lambda",
        "build a twitter clone with react and postgres",
        "create an e-commerce platform with nextjs, stripe payments, and authentication"
    ]
    
    print("=" * 80)
    print("Testing formatted_detailed_string Feature")
    print("=" * 80)
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"Test Case {i}: {prompt}")
        print(f"{'='*80}\n")
        
        result = scorer.analyze_text(prompt)
        simplified = scorer.get_simplified_output()
        
        # Print the formatted string
        print(simplified['formatted_detailed_string'])
        print()
        
        # Save to file
        filename = f'output_example_{i}.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(simplified['formatted_detailed_string'])
        print(f"✅ Saved to: {filename}")
        
        # Also save JSON
        json_filename = f'output_example_{i}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(simplified, f, indent=2, ensure_ascii=False)
        print(f"✅ Saved JSON to: {json_filename}")
        print()

def show_usage():
    """Show how to use the formatted output in code."""
    print("\n" + "=" * 80)
    print("USAGE EXAMPLES")
    print("=" * 80)
    print("""
# Basic usage - Get formatted response
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

scorer = SoftwareComplexityScorer()
result = scorer.analyze_text("build a twitter clone")
simplified = scorer.get_simplified_output()

# Get the formatted detailed string
formatted_response = simplified['formatted_detailed_string']
print(formatted_response)  # Human-readable analysis

# Or get individual fields
print(f"LOC: {simplified['estimated_no_of_lines']}")
print(f"AI Time: {simplified['estimated_ai_time']}")
print(f"Technologies: {list(simplified['technologies'].keys())}")

# Save to file
with open('analysis.txt', 'w', encoding='utf-8') as f:
    f.write(formatted_response)

# Use in API response
api_response = {
    "status": "success",
    "analysis": simplified,
    "formatted_text": formatted_response
}
""")

if __name__ == "__main__":
    test_formatted_output()
    show_usage()
