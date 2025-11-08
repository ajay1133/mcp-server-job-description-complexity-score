#!/usr/bin/env python3
"""Test the refactored SoftwareComplexityScorer with new output schema."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_new_schema():
    """Test with the new output schema."""
    models_path = os.path.join(os.path.dirname(__file__), 'models', 'software')
    scorer = SoftwareComplexityScorer(models_path)
    
    test_cases = [
        "Build a React dashboard with Stripe payments",
        "Create a FastAPI backend with PostgreSQL database",
        "I need someone to look after my elderly father",
        "Develop a machine learning recommendation system with Python",
        "Build Next.js app with authentication and real-time chat",
    ]
    
    print("Testing New Output Schema\n")
    print("=" * 80)
    
    for prompt in test_cases:
        print(f"\nPrompt: {prompt}")
        result = scorer.analyze_text(prompt)
        
        if "error" in result:
            print(f"✗ Error: {result['error']}")
            if "software_probability" in result:
                print(f"  Software probability: {result['software_probability']}")
        else:
            print("✓ Success")
            print(f"\n  Technologies (split): {result['technologies']}")
            print(f"  Predicted LOC: {result['predicted_lines_of_code']:,}")
            print(f"  Microservices: {result['microservices']}")
            
            print(f"\n  WITHOUT AI/ML:")
            print(f"    Time: {result['without_ai_and_ml']['time_estimation']}")

            print(f"\n  WITH AI/ML:")
            print(f"    Time: {result['with_ai_and_ml']['time_estimation']}")
            print(f"    Speedup: {result['with_ai_and_ml']['speedup_details']['speed_ratio']}")

            print(f"\n  SYSTEM DESIGN PLAN: {result['system_design_plan']['architecture_style']}")
            print(f"  COMPLEXITY SCORE: {result['complexity_score']}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    test_new_schema()
