#!/usr/bin/env python3
"""Test technology detection specifically."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_technology_detection():
    """Test with prompts that explicitly mention technologies."""
    models_path = os.path.join(os.path.dirname(__file__), 'models', 'software')
    scorer = SoftwareComplexityScorer(models_path)
    
    test_cases = [
        "Build React dashboard with authentication",
        "Create FastAPI REST API with PostgreSQL",
        "Develop Next.js app with Stripe integration",
        "Build Node.js backend with MongoDB database",
        "Create Django web app with MySQL",
        "Implement Vue.js frontend with authentication",
        "Build Flask microservice with Redis caching",
        "Create Angular app with Material UI",
        "Develop Python FastAPI with machine learning model",
        "Build Rails application with PostgreSQL database",
    ]
    
    print("Technology Detection Test\n")
    print("=" * 70)
    
    for prompt in test_cases:
        result = scorer.analyze_text(prompt)
        if result.get("ok"):
            print(f"\nPrompt: {prompt}")
            print(f"  Technologies: {result['technologies']}")
            print(f"  Complexity: {result['complexity_score']}")
            print(f"  LOC: {result['predicted_loc']}")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    test_technology_detection()
