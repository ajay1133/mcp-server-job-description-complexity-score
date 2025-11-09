"""Test script for SoftwareComplexityScorer with trained models."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer


def test_software_prompts():
    """Test with software prompts using real trained models."""
    models_path = os.path.join(os.path.dirname(__file__), 'models', 'software')
    print(f"Loading models from: {models_path}\n")
    
    scorer = SoftwareComplexityScorer(models_path)
    
    test_cases = [
        "Build a React dashboard with Stripe payments",
        "Create a FastAPI backend with PostgreSQL database",
        "I need someone to look after my elderly father",
        "Build authentication system with JWT",
        "Need plumber to fix sink"
    ]
    
    print("Testing SoftwareComplexityScorer with Trained Models\n")
    print("=" * 70)
    
    for prompt in test_cases:
        print(f"\nPrompt: {prompt}")
        result = scorer.analyze_text(prompt)

        # New schema returns either {error, software_probability} OR the structured metrics
        if 'error' in result:
            print(f"✗ Not software (prob={result.get('software_probability', 'n/a')})")
            print(f"  Error: {result['error']}")
        else:
            without_ai = result['without_ai_and_ml']
            with_ai = result['with_ai_and_ml']
            print("✓ Software detected")
            print(f"  Technologies split: {result['technologies']}")
            print(f"  Microservices: {result['microservices']}")
            print(f"  Manual time (h): {without_ai['time_estimation']}")
            print(f"  AI-assisted time (h): {with_ai['time_estimation']}")
            if 'data_flow' in result:
                print(f"  Data flow: {result['data_flow']}")
            if 'complexity_score' in result:
                print(f"  Complexity score: {result['complexity_score']}")
    
    print("\n" + "=" * 70)
    print("\nTEST COMPLETE - Using real trained models!")
    print(f"Models loaded from: {models_path}")


if __name__ == "__main__":
    test_software_prompts()
