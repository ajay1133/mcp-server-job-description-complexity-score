#!/usr/bin/env python3
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
import json

s = SoftwareComplexityScorer()
result = s.analyze_text('build an e-commerce platform with react and postgres')
simplified = s.get_simplified_output()

with open('final_output_example.json', 'w', encoding='utf-8') as f:
    json.dump(simplified, f, indent=2, ensure_ascii=False)

print('Saved to final_output_example.json')
print(f"\nSummary:")
print(f"  LOC: {simplified['estimated_no_of_lines']}")
print(f"  Technologies: {len(simplified['technologies'])}")
print(f"  AI time: {simplified['estimated_ai_time']}")
print(f"  Human time (avg): {simplified['estimated_human_time_average']}")
