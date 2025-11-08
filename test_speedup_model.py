#!/usr/bin/env python3
"""Test realistic AI speedup calculations with various project types."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_speedup_variations():
    scorer = SoftwareComplexityScorer()
    
    test_cases = [
        {
            "name": "Simple CRUD (Easy Tech)",
            "prompt": "Build a simple Flask REST API with basic CRUD operations for a todo list using SQLite"
        },
        {
            "name": "Twitter Clone (Medium Tech)",
            "prompt": "Build a Twitter clone with React frontend, Node.js backend, PostgreSQL database, Redis cache, and CDN for media"
        },
        {
            "name": "YouTube Clone (Hard Tech + Large)",
            "prompt": "Build a YouTube-like video platform with video processing, transcoding, CDN delivery, recommendation engine, and real-time comments"
        },
        {
            "name": "Kubernetes ML System (Expert Tech)",
            "prompt": "Build a distributed machine learning pipeline on Kubernetes with Kafka message queues, Cassandra database, custom TensorFlow models, and Elasticsearch for logging"
        },
        {
            "name": "E-commerce Platform (Medium-Large)",
            "prompt": "Build an e-commerce platform like Shopify with product catalog, shopping cart, payment processing, inventory management, order tracking, and admin dashboard"
        }
    ]
    
    print("=" * 100)
    print("REALISTIC AI SPEEDUP MODEL TEST")
    print("=" * 100)
    print()
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. {test['name']}")
        print(f"   Prompt: {test['prompt'][:80]}...")
        print()
        
        result = scorer.analyze_text(test['prompt'])
        
        if 'error' in result:
            print(f"   ✗ Error: {result['error']}")
            print()
            continue
        
        manual_hours = result['without_ai_and_ml']['time_estimation']
        ai_hours = result['with_ai_and_ml']['time_estimation']
        speedup_factor = ai_hours / manual_hours
        time_saved_pct = (1 - speedup_factor) * 100
        
        print(f"   Manual time: {manual_hours:.1f} hours ({manual_hours/8:.1f} days)")
        print(f"   AI-assisted: {ai_hours:.1f} hours ({ai_hours/8:.1f} days)")
        print(f"   Speedup: {speedup_factor*100:.0f}% of manual ({time_saved_pct:.0f}% time saved)")
        print(f"   Technologies: {len(result['without_ai_and_ml']['technologies'])} detected")
        print(f"   Microservices: {len(result['without_ai_and_ml']['microservices'])}")
        print(f"   Complexity Score: {result['complexity_score']:.0f}/200")
        
        # Show time explanation
        if 'time_estimation_explanation' in result:
            print()
            print("   Explanation:")
            for line in result['time_estimation_explanation'].split('\n'):
                if line.strip():
                    print(f"     {line}")
        
        print()
        print("-" * 100)
        print()

if __name__ == "__main__":
    test_speedup_variations()
