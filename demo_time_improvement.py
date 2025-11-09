#!/usr/bin/env python3
"""Demonstrate time calculation improvement with max(LOC, difficulty)."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def show_improvement():
    """Show dramatic improvement for technologies with small LOC but high difficulty."""
    scorer = SoftwareComplexityScorer()
    
    result = scorer.analyze_text("build a data platform with elasticsearch, kafka, and postgres")
    analysis = result.get('per_technology_analysis', [])
    complexity = result.get('per_technology_complexity', {})
    
    print("=" * 90)
    print("TIME CALCULATION IMPROVEMENT: max(LOC-based, Difficulty-based)")
    print("=" * 90)
    print()
    print(f"{'Technology':<18} {'Diff':<6} {'Human %':>10} {'AI %':>10}")
    print("-" * 60)
    
    examples = []
    for tech_data in sorted(analysis, key=lambda x: x.get('time_spent', {}).get('human_percent', 0), reverse=True)[:10]:
        tech = tech_data.get('technology', '')
        difficulty = complexity.get(tech, {}).get('difficulty', 0)
        human_pct = tech_data.get('time_spent', {}).get('human_percent', 0)
        ai_pct = tech_data.get('time_spent', {}).get('ai_percent', 0)
        print(f"{tech:<18} {difficulty:<6.1f} {human_pct:>9.2f}% {ai_pct:>9.2f}%")
    
    print()
    print("NOTE:")
    print("=" * 90)
    print("Time improvement demo now shows percentage effort shares instead of raw overhead hours.")
    print("Underlying model still uses max(LOC-based, difficulty-based) internally.")
    print()

if __name__ == "__main__":
    show_improvement()
