#!/usr/bin/env python3
"""Validation test for AWS serverless technology detection.

Ensures prompts mentioning API Gateway, Lambda, and DynamoDB surface
normalized technologies in nested schema with correct mention flags.
Run directly: python test_serverless_detection.py
"""
import json
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

PROMPT = "deploy apis on aws serverless (api gateway, lambda, dynamodb)"


def main():
    scorer = SoftwareComplexityScorer()
    result = scorer.analyze_text(PROMPT)
    techs = result.get("technologies", {})
    backend = techs.get("backend", {})
    database = techs.get("database", {})

    # Assertions: presence
    assert "aws_lambda" in backend, f"aws_lambda missing. Backend keys: {list(backend.keys())}"
    assert "api_gateway" in backend, f"api_gateway missing. Backend keys: {list(backend.keys())}"
    assert "dynamodb" in database, f"dynamodb missing. Database keys: {list(database.keys())}"

    # Assertions: mention flags
    assert backend["api_gateway"]["is_mentioned_in_prompt"], "api_gateway should be marked mentioned"
    assert database["dynamodb"]["is_mentioned_in_prompt"], "dynamodb should be marked mentioned"
    # Lambda was explicitly in parentheses; ensure marked (if false, detection heuristic needs tweak)
    assert backend["aws_lambda"]["is_mentioned_in_prompt"], "aws_lambda should be marked mentioned"

    summary = {
        "prompt": PROMPT,
        "backend_keys": list(backend.keys()),
        "database_keys": list(database.keys()),
        "aws_lambda": backend.get("aws_lambda"),
        "api_gateway": backend.get("api_gateway"),
        "dynamodb": database.get("dynamodb"),
        "complexity_score": result.get("complexity_score"),
    }
    print(json.dumps(summary, indent=2))
    print("AWS serverless detection test passed.")


if __name__ == "__main__":
    main()
