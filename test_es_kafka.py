#!/usr/bin/env python3
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

s = SoftwareComplexityScorer()
result = s.analyze_text('build a search and analytics platform using elasticsearch and kafka for event streaming')
analysis = result.get('per_technology_analysis', [])
complexity = result.get('per_technology_complexity', {})

print('Technologies found:', len(analysis))
print()
print(f"{'Technology':<20} {'Human %':>8} {'AI %':>8} {'Difficulty'}")
print('-'*60)

for t in sorted(analysis, key=lambda x: x.get('time_spent', {}).get('human_percent', 0), reverse=True)[:10]:
    tech = t.get('technology', '')
    diff = complexity.get(tech, {}).get('difficulty', 0)
    human_pct = t.get('time_spent', {}).get('human_percent', 0)
    ai_pct = t.get('time_spent', {}).get('ai_percent', 0)
    print(f"{tech:<20} {human_pct:>7.2f}% {ai_pct:>7.2f}% {diff:.1f}")
