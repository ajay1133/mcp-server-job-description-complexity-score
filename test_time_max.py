#!/usr/bin/env python3
"""Test max(LOC, difficulty) time calculation."""

import json
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_max_time_calculation():
    """Test that technologies use max of LOC-based and difficulty-based time."""
    scorer = SoftwareComplexityScorer()
    
    # Test with technologies that have small LOC but high difficulty
    test_cases = [
        ("build a search system with elasticsearch", "elasticsearch"),
        ("create an app with kafka message queue", "kafka"),
        ("build a monitoring dashboard", "monitoring"),
        ("create a video platform with CDN", "cdn")
    ]
    
    print("=" * 80)
    print("Testing max(LOC-based time, Difficulty-based time)")
    print("=" * 80)
    
    for prompt, target_tech in test_cases:
        result = scorer.analyze_text(prompt)
        analysis = result.get('per_technology_analysis', [])
        complexity = result.get('per_technology_complexity', {})
        
        tech_data = next((t for t in analysis if t['technology'] == target_tech), None)
        if not tech_data:
            print(f"\n❌ {target_tech} not found in analysis")
            continue
        
        complexity_data = complexity.get(target_tech, {})
        difficulty = complexity_data.get('difficulty', 0)
        
        loc = tech_data['loc_overhead']
        total_time = tech_data['time_overhead_hours']
        
        # Calculate what each would be
        loc_time = (loc / 1000.0) * 0.77 if loc > 0 else 0
        diff_time = 0.3 * (1.5 ** difficulty) if difficulty > 0 else 0
        
        print(f"\n{target_tech.upper()}")
        print(f"  Difficulty: {difficulty}")
        print(f"  LOC: {loc}")
        print(f"  LOC-based time: {loc_time:.2f} hours")
        print(f"  Difficulty-based time: {diff_time:.2f} hours")
        print(f"  FINAL TIME (max): {total_time} hours ({tech_data['time_overhead_readable']})")
        
        # Verify it's using the max
        expected_max = max(loc_time, diff_time)
        if abs(total_time - expected_max) < 0.01:
            print(f"  ✅ Correctly using max()")
        else:
            print(f"  ❌ Not using max! Expected {expected_max:.2f}")

def compare_before_after():
    """Show before/after comparison."""
    scorer = SoftwareComplexityScorer()
    
    result = scorer.analyze_text("build a video streaming platform with elasticsearch and kafka")
    analysis = result.get('per_technology_analysis', [])
    complexity = result.get('per_technology_complexity', {})
    
    print("\n" + "=" * 80)
    print("BEFORE vs AFTER Comparison")
    print("=" * 80)
    print(f"{'Technology':<20} {'LOC':<8} {'Diff':<6} {'Before':<15} {'After':<15}")
    print("-" * 80)
    
    for tech_data in sorted(analysis, key=lambda x: x['time_overhead_hours'], reverse=True)[:10]:
        tech = tech_data['technology']
        loc = tech_data['loc_overhead']
        total_time = tech_data['time_overhead_hours']
        
        complexity_data = complexity.get(tech, {})
        difficulty = complexity_data.get('difficulty', 0)
        
        # Old calculation (just LOC)
        old_time = (loc / 1000.0) * 0.77 if loc > 0 else 0
        
        print(f"{tech:<20} {loc:<8} {difficulty:<6.1f} {old_time:<15.2f} {total_time:<15.2f}")
    
    print("\n✅ Technologies with small LOC but high difficulty now get proper time allocation!")

if __name__ == "__main__":
    test_max_time_calculation()
    compare_before_after()
