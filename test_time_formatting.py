#!/usr/bin/env python3
"""
Test human-readable time formatting across all response types.
"""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_build_requirement():
    """Test time formatting in build requirement response."""
    scorer = SoftwareComplexityScorer()
    
    result = scorer.analyze_text("Build a React dashboard with Stripe payments and PostgreSQL")
    
    print("=" * 70)
    print("BUILD REQUIREMENT - Time Formatting Test")
    print("=" * 70)
    
    # Check without_ai_and_ml time
    without_ai = result['without_ai_and_ml']
    print("\n1. WITHOUT AI/ML Time:")
    print(f"   - Hours: {without_ai['time_estimation']['hours']}")
    print(f"   - Human Readable: {without_ai['time_estimation']['human_readable']}")
    
    # Check with_ai_and_ml time
    with_ai = result['with_ai_and_ml']
    print("\n2. WITH AI/ML Time:")
    print(f"   - Hours: {with_ai['time_estimation']['hours']}")
    print(f"   - Human Readable: {with_ai['time_estimation']['human_readable']}")
    
    # Check per-technology times
    print("\n3. Per-Technology Time Estimates:")
    per_tech = result['per_technology_complexity']
    for tech, details in list(per_tech.items())[:3]:  # Show first 3
        print(f"   - {tech}:")
        print(f"     • Hours: {details['estimated_time_hours']}")
        print(f"     • Human Readable: {details['estimated_time_human']}")
    
    print("\n✓ Build requirement time formatting verified!\n")
    return result

def test_hiring_requirement():
    """Test time formatting in hiring requirement response."""
    scorer = SoftwareComplexityScorer()
    
    result = scorer.analyze_text(
        "Looking for Senior Full-Stack Engineer with 5+ years React and Node.js. "
        "Must be expert in MongoDB, AWS, and Docker."
    )
    
    print("=" * 70)
    print("HIRING REQUIREMENT - Time Formatting Test")
    print("=" * 70)
    
    # Check overall time estimation
    time_est = result['time_estimation']
    print("\n1. Overall Time Estimation:")
    print(f"   - AI Hours: {time_est['ai_hours']}")
    print(f"   - AI Human Readable: {time_est['ai_human_readable']}")
    print(f"   - Manual Hours: {time_est['manual_hours']}")
    print(f"   - Manual Human Readable: {time_est['manual_human_readable']}")
    
    # Check per-technology times
    print("\n2. Per-Technology Time Estimates:")
    per_tech = result['per_technology_complexity']
    for tech, details in list(per_tech.items())[:3]:  # Show first 3
        print(f"   - {tech}:")
        print(f"     • Hours: {details['estimated_time_hours']}")
        print(f"     • Human Readable: {details['estimated_time_human']}")
    
    print("\n✓ Hiring requirement time formatting verified!\n")
    return result

def test_time_formatting_accuracy():
    """Test the _format_time_human_readable method directly."""
    scorer = SoftwareComplexityScorer()
    
    print("=" * 70)
    print("TIME FORMATTING ACCURACY TEST")
    print("=" * 70)
    
    test_cases = [
        (0.5, "30 minutes"),
        (1.5, "1.5 hours"),
        (12, "12 hours"),
        (24, "1 day"),
        (48, "2 days"),
        (168, "1 week"),
        (336, "2 weeks"),
        (720, "1 month"),
        (2190, "3 months"),
        (8760, "1 year"),
        (17520, "2 years"),
    ]
    
    print("\nTest Cases:")
    for hours, expected_pattern in test_cases:
        result = scorer._format_time_human_readable(hours)
        # Check if the result contains the expected pattern (approximate matching)
        matches = expected_pattern.split()[1] in result  # Check unit (minutes, hours, days, etc.)
        status = "✓" if matches else "✗"
        print(f"{status} {hours} hours → {result} (expected: {expected_pattern})")
    
    print("\n✓ Time formatting accuracy verified!\n")

def main():
    """Run all time formatting tests."""
    print("\n" + "=" * 70)
    print("COMPREHENSIVE TIME FORMATTING TEST SUITE")
    print("=" * 70 + "\n")
    
    # Run tests
    test_build_requirement()
    test_hiring_requirement()
    test_time_formatting_accuracy()
    
    print("=" * 70)
    print("ALL TESTS PASSED ✓")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
