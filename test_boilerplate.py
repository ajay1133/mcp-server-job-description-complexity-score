#!/usr/bin/env python3
"""Test boilerplate LOC deduction feature."""

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

def test_boilerplate_deduction():
    scorer = SoftwareComplexityScorer()
    
    # Test Django project (large boilerplate: 1,200 LOC)
    print("=" * 60)
    print("Django REST API Test")
    print("=" * 60)
    result = scorer.analyze_text('Build a Django REST API with PostgreSQL and Redis caching')
    print(f"Technologies: {result['technologies']}")
    print(f"Predicted LOC: {result['predicted_lines_of_code']:,}")
    print(f"Boilerplate deducted: {result['without_ai_and_ml']['time_estimation']['boilerplate_loc_deducted']}")
    print(f"Human coding LOC: {result['without_ai_and_ml']['time_estimation']['human_coding_loc']:,.0f}")
    print(f"Manual time (min): {result['without_ai_and_ml']['time_estimation']['human_readable_min']}")
    print(f"Manual time (avg): {result['without_ai_and_ml']['time_estimation']['human_readable_avg']}")
    print(f"Manual time (max): {result['without_ai_and_ml']['time_estimation']['human_readable_max']}")
    print(f"AI time: {result['with_ai_and_ml']['time_estimation']['human_readable']}")
    print()
    
    # Test Spring Boot project (very large boilerplate: 2,500 LOC)
    print("=" * 60)
    print("Spring Boot Microservices Test")
    print("=" * 60)
    result = scorer.analyze_text('Build Spring Boot microservices with MySQL database')
    print(f"Technologies: {result['technologies']}")
    print(f"Predicted LOC: {result['predicted_lines_of_code']:,}")
    print(f"Boilerplate deducted: {result['without_ai_and_ml']['time_estimation']['boilerplate_loc_deducted']}")
    print(f"Human coding LOC: {result['without_ai_and_ml']['time_estimation']['human_coding_loc']:,.0f}")
    print(f"Manual time (min): {result['without_ai_and_ml']['time_estimation']['human_readable_min']}")
    print(f"Manual time (avg): {result['without_ai_and_ml']['time_estimation']['human_readable_avg']}")
    print(f"Manual time (max): {result['without_ai_and_ml']['time_estimation']['human_readable_max']}")
    print(f"AI time: {result['with_ai_and_ml']['time_estimation']['human_readable']}")
    print()
    
    # Test Angular project (moderate boilerplate: 500 LOC)
    print("=" * 60)
    print("Angular Dashboard Test")
    print("=" * 60)
    result = scorer.analyze_text('Build an Angular dashboard with charts and authentication')
    print(f"Technologies: {result['technologies']}")
    print(f"Predicted LOC: {result['predicted_lines_of_code']:,}")
    print(f"Boilerplate deducted: {result['without_ai_and_ml']['time_estimation']['boilerplate_loc_deducted']}")
    print(f"Human coding LOC: {result['without_ai_and_ml']['time_estimation']['human_coding_loc']:,.0f}")
    print(f"Manual time (min): {result['without_ai_and_ml']['time_estimation']['human_readable_min']}")
    print(f"Manual time (avg): {result['without_ai_and_ml']['time_estimation']['human_readable_avg']}")
    print(f"Manual time (max): {result['without_ai_and_ml']['time_estimation']['human_readable_max']}")
    print(f"AI time: {result['with_ai_and_ml']['time_estimation']['human_readable']}")
    print()

if __name__ == "__main__":
    test_boilerplate_deduction()
