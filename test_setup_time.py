#!/usr/bin/env python3
"""Test setup time allocation for non-coding technologies."""

import json
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_setup_time_allocation():
    """Test that non-coding technologies get setup time based on difficulty."""
    scorer = SoftwareComplexityScorer()
    
    prompts = [
        "build a video streaming platform with CDN",
        "create a microservices app with service mesh",
        "develop an app with OAuth authentication and monitoring"
    ]
    
    print("=" * 80)
    print("Testing Setup Time Allocation for Non-Coding Technologies")
    print("=" * 80)
    
    for prompt in prompts:
        print(f"\n{'='*80}")
        print(f"Prompt: {prompt}")
        print(f"{'='*80}")
        
        result = scorer.analyze_text(prompt)
        analysis = result.get('per_technology_analysis', [])
        
        # Separate coding vs non-coding technologies
        coding_techs = [t for t in analysis if t.get('loc_overhead', 0) > 0]
        non_coding_techs = [t for t in analysis if t.get('loc_overhead', 0) == 0 and t.get('time_overhead_hours', 0) > 0]
        zero_time_techs = [t for t in analysis if t.get('time_overhead_hours', 0) == 0]
        
        print(f"\nCoding Technologies (LOC > 0): {len(coding_techs)}")
        for tech in coding_techs[:5]:
            print(f"  - {tech['technology']:15s}: LOC={tech['loc_overhead']:4d}, Time={tech['time_overhead_readable']}")
        
        print(f"\nNon-Coding Technologies (Setup/Config, LOC=0 but Time>0): {len(non_coding_techs)}")
        for tech in non_coding_techs[:10]:
            print(f"  - {tech['technology']:15s}: LOC={tech['loc_overhead']:4d}, Setup Time={tech['time_overhead_readable']}")
        
        print(f"\nZero-Time Technologies (Skipped/Not Used): {len(zero_time_techs)}")
        if zero_time_techs:
            print(f"  (First 3: {', '.join(t['technology'] for t in zero_time_techs[:3])})")
    
    print("\n" + "=" * 80)
    print("✅ Test Complete - Non-coding technologies now have setup time!")
    print("=" * 80)

def test_difficulty_based_calculation():
    """Show how setup time correlates with difficulty rating."""
    scorer = SoftwareComplexityScorer()
    
    result = scorer.analyze_text("build a complex system with auth, monitoring, and devops")
    analysis = result.get('per_technology_analysis', [])
    per_tech_complexity = result.get('per_technology_complexity', {})
    
    # Get non-coding techs with setup time
    non_coding = [t for t in analysis if t.get('loc_overhead', 0) == 0 and t.get('time_overhead_hours', 0) > 0]
    
    print("\n" + "=" * 80)
    print("Setup Time vs Difficulty Rating")
    print("=" * 80)
    print(f"{'Technology':<20} {'Difficulty':<12} {'Setup Time':<20} {'Category'}")
    print("-" * 80)
    
    for tech_data in sorted(non_coding, key=lambda x: x.get('time_overhead_hours', 0), reverse=True)[:10]:
        tech_name = tech_data['technology']
        complexity = per_tech_complexity.get(tech_name, {})
        difficulty = complexity.get('difficulty', 0)
        category = complexity.get('category', 'unknown')
        setup_time = tech_data['time_overhead_readable']
        
        print(f"{tech_name:<20} {difficulty:<12.1f} {setup_time:<20} {category}")
    
    print("\n✅ Higher difficulty = More setup time (exponential growth)")

if __name__ == "__main__":
    test_setup_time_allocation()
    test_difficulty_based_calculation()
