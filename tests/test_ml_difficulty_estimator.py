#!/usr/bin/env python3
"""Tests for ML-based difficulty estimator."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_server.ml_difficulty_estimator import MLDifficultyEstimator, get_difficulty_estimator  # noqa: E402


def test_estimate_known_tech():
    """Test estimating difficulty for a well-known technology."""
    estimator = MLDifficultyEstimator()

    # React - should have real metrics (if API available)
    result = estimator.estimate_difficulty("react")

    assert "difficulty" in result
    assert 1.0 <= result["difficulty"] <= 10.0
    assert "confidence" in result
    assert "source" in result
    assert "explanation" in result


def test_estimate_new_tech():
    """Test estimating difficulty for a brand new technology."""
    estimator = MLDifficultyEstimator()

    # Bun - newer runtime
    result = estimator.estimate_difficulty("bun")

    assert "difficulty" in result
    assert 1.0 <= result["difficulty"] <= 10.0
    assert "features" in result


def test_estimate_unknown_tech():
    """Test fallback for completely unknown tech."""
    estimator = MLDifficultyEstimator()

    # Non-existent tech
    result = estimator.estimate_difficulty("quantum-framework-xyz-2026")

    assert result["difficulty"] == 5.0  # Baseline fallback
    assert result["confidence"] == 0.3  # Low confidence
    assert result["source"] == "fallback"


def test_feature_extraction():
    """Test feature extraction from metrics."""
    estimator = MLDifficultyEstimator()

    metrics = {
        "github_stars": 50000,
        "github_forks": 8000,
        "github_open_issues": 200,
        "github_created_at": "2015-01-01T00:00:00Z",
        "npm_downloads": 5000000,
        "stackoverflow_questions": 15000,
    }

    features = estimator._extract_features("react", metrics)

    assert "github_stars_log" in features
    assert "years_in_market" in features
    assert "npm_downloads_log" in features
    assert features["years_in_market"] > 9  # Should be ~10 years


def test_heuristic_prediction():
    """Test heuristic-based difficulty prediction."""
    estimator = MLDifficultyEstimator()

    # High-quality mature tech (lots of stars, old, well-maintained)
    features_easy = {
        "github_stars_log": 11.0,  # ~60k stars
        "years_in_market": 8.0,
        "maintenance_score": 10.0,
        "stackoverflow_questions_log": 10.0,
    }

    difficulty_easy = estimator._predict_with_heuristic(features_easy)
    assert difficulty_easy < 5.5  # Should be easier

    # New tech with few resources
    features_hard = {
        "github_stars_log": 5.0,  # ~150 stars
        "years_in_market": 0.5,
        "maintenance_score": 2.0,
        "stackoverflow_questions_log": 4.0,
    }

    difficulty_hard = estimator._predict_with_heuristic(features_hard)
    assert difficulty_hard > 5.0  # Should be harder


def test_singleton():
    """Test singleton pattern."""
    estimator1 = get_difficulty_estimator()
    estimator2 = get_difficulty_estimator()
    assert estimator1 is estimator2


def test_caching():
    """Test that results are cached."""
    estimator = MLDifficultyEstimator()

    # First call (fetches)
    result1 = estimator.estimate_difficulty("svelte")

    # Second call (should use cache)
    result2 = estimator.estimate_difficulty("svelte")

    # Should be identical
    assert result1["difficulty"] == result2["difficulty"]
    assert result1["timestamp"] == result2["timestamp"]
