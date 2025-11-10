#!/usr/bin/env python3
"""Train ML models for technology extraction and analysis.

This script trains the following models:
1. Technology Extractor (NER): Identifies tech mentions in text
2. Difficulty Scorer: Predicts difficulty rating for technologies
3. Experience Extractor: Validates experience extraction
4. Alternatives Recommender: Suggests alternative technologies

Models are saved to the `models/` directory with version tracking.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

MODELS_DIR = Path("models")
REGISTRY_FILE = MODELS_DIR / "registry.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train technology extraction models")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode (minimal data, for CI)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["difficulty", "alternatives", "experience", "all"],
        default=["all"],
        help="Which models to train",
    )
    args = parser.parse_args()

    # Check for skip flag
    if os.getenv("SKIP_MODEL_TRAIN") == "1":
        print("[train] SKIPPED via SKIP_MODEL_TRAIN=1")
        return 0

    print("[train] Starting model training pipeline...")
    start_time = time.time()

    # Create models directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Determine which models to train
    models_to_train = args.models
    if "all" in models_to_train:
        models_to_train = ["difficulty", "experience", "alternatives"]

    trained_models = {}
    errors = []

    # Train each model
    if "difficulty" in models_to_train:
        try:
            print("\n[train] Training difficulty scoring model...")
            difficulty_model = train_difficulty_model(quick=args.quick)
            difficulty_model.save(MODELS_DIR / "difficulty")
            trained_models["difficulty"] = "models/difficulty"
            print("[train] ✓ Difficulty model saved")
        except Exception as e:
            print(f"[train] ✗ Difficulty model failed: {e}")
            errors.append(("difficulty", str(e)))

    if "experience" in models_to_train:
        try:
            print("\n[train] Training experience validation model...")
            experience_model = train_experience_model(quick=args.quick)
            experience_model.save(MODELS_DIR / "experience")
            trained_models["experience"] = "models/experience"
            print("[train] ✓ Experience model saved")
        except Exception as e:
            print(f"[train] ✗ Experience model failed: {e}")
            errors.append(("experience", str(e)))

    if "alternatives" in models_to_train:
        try:
            print("\n[train] Training alternatives recommendation model...")
            alternatives_model = train_alternatives_model(quick=args.quick)
            alternatives_model.save(MODELS_DIR / "alternatives")
            trained_models["alternatives"] = "models/alternatives"
            print("[train] ✓ Alternatives model saved")
        except Exception as e:
            print(f"[train] ✗ Alternatives model failed: {e}")
            errors.append(("alternatives", str(e)))

    # Save model registry
    elapsed = time.time() - start_time
    registry = {
        "version": "1.0.0",
        "timestamp": int(time.time()),
        "elapsed_seconds": round(elapsed, 2),
        "models": trained_models,
        "errors": errors,
        "quick_mode": args.quick,
    }

    with REGISTRY_FILE.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2)

    print(f"\n[train] Training completed in {elapsed:.1f}s")
    print(f"[train] Registry saved to {REGISTRY_FILE}")
    print(f"[train] Trained {len(trained_models)} models successfully")

    if errors:
        print(f"[train] WARNING: {len(errors)} models failed to train")
        for model_name, error in errors:
            print(f"  - {model_name}: {error}")

    return 0


def train_difficulty_model(quick: bool = False):
    """Train difficulty scoring model."""
    from training_data.sample_data import DIFFICULTY_TRAINING_DATA
    from mcp_server.ml_models.difficulty_scorer import DifficultyTrainer

    trainer = DifficultyTrainer()

    # Use sample data (in production, load from larger dataset)
    training_data = DIFFICULTY_TRAINING_DATA

    if quick:
        # Use minimal data for fast CI
        training_data = training_data[:3]

    # For minimal training, just create model with baseline
    if len(training_data) < 4:
        print("[train] Not enough data for full training, using baseline")
        from mcp_server.ml_models.difficulty_scorer import DifficultyModel
        return DifficultyModel()

    model = trainer.train(training_data, model_type="gradient_boosting")
    return model


def train_experience_model(quick: bool = False):
    """Train experience extraction validation model."""
    from training_data.sample_data import EXPERIENCE_TRAINING_DATA
    from mcp_server.ml_models.experience_extractor import ExperienceTrainer

    trainer = ExperienceTrainer()
    training_data = EXPERIENCE_TRAINING_DATA

    if quick:
        training_data = training_data[:3]

    # Need minimum data for classifier
    if len(training_data) < 4:
        print("[train] Not enough data for full training, using regex only")
        from mcp_server.ml_models.experience_extractor import ExperienceModel
        return ExperienceModel()

    model = trainer.train(training_data)
    return model


def train_alternatives_model(quick: bool = False):
    """Train alternatives recommendation model."""
    from training_data.sample_data import (
        TECHNOLOGY_CORPUS,
        SIMILARITY_PAIRS,
    )
    from mcp_server.ml_models.alternatives_recommender import AlternativesTrainer

    corpus = TECHNOLOGY_CORPUS
    pairs = SIMILARITY_PAIRS

    if quick:
        corpus = corpus[:5]
        pairs = pairs[:5]
        # Use smaller embedding dimension for small datasets
        embedding_dim = 16
    else:
        embedding_dim = 64

    trainer = AlternativesTrainer(embedding_dim=embedding_dim)
    model = trainer.train(corpus, pairs)
    return model


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
