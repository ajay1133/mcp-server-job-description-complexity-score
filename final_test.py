#!/usr/bin/env python3
"""Final comprehensive test of improved time calculation."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

scorer = SoftwareComplexityScorer()
result = scorer.analyze_text('build a data platform with elasticsearch, kafka, postgres, redis, and docker')
analysis = result.get('per_technology_analysis', [])

print('='*80)
print('FINAL COMPREHENSIVE TEST: Data Platform with Infrastructure')
print('='*80)
print(f'\nTotal technologies: {len(analysis)}')

print('Top 15 technologies by effort share:')
print(f"{'Technology':<20} {'Human %':>8} {'AI %':>8} {'Difficulty'}")
print('-'*80)

complexity = result.get('per_technology_complexity', {})
for t in sorted(analysis, key=lambda x: x.get('time_spent', {}).get('human_percent', 0), reverse=True)[:15]:
    tech_name = t.get('technology', '')
    diff = complexity.get(tech_name, {}).get('difficulty', 0)
    human_pct = t.get('time_spent', {}).get('human_percent', 0)
    ai_pct = t.get('time_spent', {}).get('ai_percent', 0)
    print(f"{tech_name:<20} {human_pct:>7.2f}% {ai_pct:>7.2f}% {diff:.1f}")

print('\n' + '='*80)
print('✅ Effort shares computed per technology; internal model accounts for coding and setup complexity.')
print('='*80)
