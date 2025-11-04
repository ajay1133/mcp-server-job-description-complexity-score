#!/usr/bin/env python3
"""
Demo script showcasing the complexity scorer with time estimation
"""

from mcp_server.complexity_scorer import ComplexityScorer
import json

def print_analysis(title: str, requirement: str):
    """Print a formatted analysis result"""
    print(f"\n{'='*70}")
    print(f"📋 {title}")
    print(f"{'='*70}")
    print(f"Requirement: {requirement}")
    print(f"{'-'*70}")
    
    scorer = ComplexityScorer()
    result = scorer.analyze_text(requirement)
    
    print(f"Complexity Score: {result['complexity_score']}")
    print(f"Task Size: {result['task_size']}")
    print(f"Difficulty: {result['difficulty_rating']}")
    print("\n⏱️  TIME ESTIMATE:")
    time_est = result['estimated_completion_time']
    print(f"  Best Estimate: {time_est['best_estimate']}")
    print(f"  Range: {time_est['time_range']}")
    print(f"  (Detailed: {time_est['hours']} hours / {time_est['days']} days / {time_est['weeks']} weeks)")
    print("\n📊 Top Complexity Factors:")
    if result['detected_factors']:
        sorted_factors = sorted(
            result['detected_factors'].items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )[:3]
        for i, (factor, data) in enumerate(sorted_factors, 1):
            print(f"  {i}. {factor.replace('_', ' ').title()} (weight: {data['weight']})")
    else:
        print("  No significant complexity factors detected")
    
    print(f"\n💡 Summary: {result['summary']}")

def main():
    print("\n" + "="*70)
    print("🚀 MCP COMPLEXITY SCORER - TIME ESTIMATION DEMO")
    print("="*70)
    print("Baseline: Replit Agent 3 = 100 points = 8 hours")
    print("Assumes: Developer skilled in using AI coding agents")
    
    # Demo various complexity levels
    test_cases = [
        (
            "Quick Task",
            "Create a simple contact form with HTML and CSS"
        ),
        (
            "Moderate Task",
            "Build a REST API with FastAPI, PostgreSQL database, and JWT authentication"
        ),
        (
            "Complex Task",
            "Develop a full-stack e-commerce platform with React frontend, Django backend, payment integration, and user authentication"
        ),
        (
            "Very Complex Task",
            "Build a real-time collaborative document editor with WebSocket support, MongoDB, user permissions, and deployment on AWS"
        ),
        (
            "Expert-Level Task",
            "Design and implement an enterprise-grade microservices architecture with Kubernetes orchestration, machine learning recommendation engine, comprehensive testing suite, CI/CD pipeline, and scalable cloud deployment"
        )
    ]
    
    for title, requirement in test_cases:
        print_analysis(title, requirement)
    
    print(f"\n{'='*70}")
    print("✅ Demo Complete!")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
