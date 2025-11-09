#!/usr/bin/env python3
"""Demo: Using formatted_detailed_string for resume parser project."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

# Initialize
scorer = SoftwareComplexityScorer()

# Your original request
prompt = "I want someone to create a resume parser for pdf and docs using aws lambda"

# Analyze
result = scorer.analyze_text(prompt)
simplified = scorer.get_simplified_output()

# Method 1: Print the formatted string directly
print("="*80)
print("METHOD 1: Direct Output")
print("="*80)
print(simplified['formatted_detailed_string'])

# Method 2: Save to file
print("\n" + "="*80)
print("METHOD 2: Save to File")
print("="*80)
with open('resume_parser_analysis.txt', 'w', encoding='utf-8') as f:
    f.write(simplified['formatted_detailed_string'])
print("✅ Saved to: resume_parser_analysis.txt")

# Method 3: Use in API/JSON response
print("\n" + "="*80)
print("METHOD 3: API Response Format")
print("="*80)
import json
api_response = {
    "status": "success",
    "request": prompt,
    "analysis": {
        "loc": simplified['estimated_no_of_lines'],
        "ai_time": simplified['estimated_ai_time'],
        "human_time": simplified['estimated_human_time_average'],
        "technologies": list(simplified['technologies'].keys())
    },
    "formatted_summary": simplified['formatted_detailed_string']
}
print(json.dumps(api_response, indent=2))

# Method 4: Access individual components
print("\n" + "="*80)
print("METHOD 4: Individual Field Access")
print("="*80)
print(f"Lines of Code: {simplified['estimated_no_of_lines']}")
print(f"AI Development Time: {simplified['estimated_ai_time']}")
print(f"Human Development Time: {simplified['estimated_human_time_average']}")
print(f"Speed Improvement: {round(1/simplified['human_to_ai_ratio_min'])}x faster")
print(f"Technologies Detected: {len(simplified['technologies'])}")
for tech_name, tech_data in simplified['technologies'].items():
    print(f"  - {tech_name}: {tech_data['contribution_to_lines_of_code']}% of code")
