"""Difficulty scoring model using gradient boosting or neural network.

Learns difficulty ratings from historical job postings, interview feedback,
and community data (GitHub stars, Stack Overflow questions, learning curves).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class DifficultyModel:
    """Predict difficulty score (1-10) for technologies using ML."""

    def __init__(self, model_path: Path | None = None):
        """Initialize difficulty scoring model.

        Args:
            model_path: Path to saved model. If None, uses heuristics.
        """
        self.model_path = model_path
        self.model = None
        self.feature_extractor = None

        # Hardcoded baseline (will be replaced by ML)
        self.baseline_difficulty = {
            "react": 5.2,
            "vue": 4.8,
            "angular": 6.5,
            "nextjs": 5.5,
            "svelte": 4.5,
            "typescript": 5.8,
            "node": 5.0,
            "python": 4.0,
            "fastapi": 4.5,
            "flask": 4.2,
            "django": 5.8,
            "golang": 6.2,
            "java": 6.0,
            "spring": 7.0,
            "ruby": 5.0,
            "rails": 5.5,
            "postgres": 5.5,
            "mysql": 5.0,
            "mongodb": 4.8,
            "redis": 4.5,
            "dynamodb": 5.8,
            "cassandra": 7.2,
            "docker": 5.5,
            "kubernetes": 7.5,
            "aws": 6.5,
            "lambda": 5.8,
            "kafka": 7.0,
            "rabbitmq": 6.2,
            "elasticsearch": 6.5,
        }

        if model_path and model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        """Load trained difficulty model."""
        try:
            import joblib

            model_file = self.model_path / "difficulty_model.pkl"
            if model_file.exists():
                self.model = joblib.load(model_file)

            config_file = self.model_path / "difficulty_config.json"
            if config_file.exists():
                with open(config_file, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.baseline_difficulty = config.get("baseline_difficulty", self.baseline_difficulty)
        except ImportError:
            print("[DifficultyModel] joblib not available, using baseline")
        except Exception as e:
            print(f"[DifficultyModel] Error loading model: {e}, using baseline")

    def predict(self, tech_name: str, context: dict[str, Any] | None = None) -> float:
        """Predict difficulty score for a technology.

        Args:
            tech_name: Technology name
            context: Optional context with features like:
                - years_in_market: How long the tech has existed
                - github_stars: Popularity metric
                - stackoverflow_questions: Community size
                - learning_resources: Availability of tutorials
                - api_complexity: API surface area
                - ecosystem_size: Number of packages/plugins

        Returns:
            Difficulty score (1.0 - 10.0)
        """
        if self.model is not None:
            return self._predict_with_model(tech_name, context or {})
        else:
            return self._predict_with_baseline(tech_name)

    def _predict_with_model(self, tech_name: str, context: dict[str, Any]) -> float:
        """Predict using trained ML model."""
        # Extract features
        features = self._extract_features(tech_name, context)

        # Predict
        prediction = self.model.predict([features])[0]

        # Clip to valid range
        return float(np.clip(prediction, 1.0, 10.0))

    def _extract_features(self, tech_name: str, context: dict[str, Any]) -> np.ndarray:
        """Extract feature vector for difficulty prediction.

        Features:
        - Tech name embedding (one-hot or learned)
        - Years in market
        - GitHub stars (log scale)
        - Stack Overflow questions (log scale)
        - Learning curve indicators
        - API complexity metrics
        - Ecosystem maturity
        """
        features = []

        # Baseline difficulty as a feature
        baseline = self.baseline_difficulty.get(tech_name.lower(), 5.0)
        features.append(baseline)

        # Context features
        features.append(context.get("years_in_market", 5.0))
        features.append(np.log1p(context.get("github_stars", 1000)))
        features.append(np.log1p(context.get("stackoverflow_questions", 500)))
        features.append(context.get("learning_resources", 5.0))
        features.append(context.get("api_complexity", 5.0))
        features.append(context.get("ecosystem_size", 5.0))

        return np.array(features, dtype=np.float32)

    def _predict_with_baseline(self, tech_name: str) -> float:
        """Fallback to baseline difficulty scores."""
        return self.baseline_difficulty.get(tech_name.lower(), 5.0)

    def batch_predict(self, technologies: list[tuple[str, dict[str, Any]]]) -> dict[str, float]:
        """Predict difficulty for multiple technologies.

        Args:
            technologies: List of (tech_name, context) tuples

        Returns:
            Dict mapping tech_name -> difficulty_score
        """
        results = {}
        for tech_name, context in technologies:
            results[tech_name] = self.predict(tech_name, context)
        return results

    def save(self, output_path: Path) -> None:
        """Save model to disk."""
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            import joblib

            joblib.dump(self.model, output_path / "difficulty_model.pkl")

        # Save config
        config = {
            "baseline_difficulty": self.baseline_difficulty,
            "version": "1.0.0",
            "features": [
                "baseline",
                "years_in_market",
                "log_github_stars",
                "log_stackoverflow_questions",
                "learning_resources",
                "api_complexity",
                "ecosystem_size",
            ],
        }
        with open(output_path / "difficulty_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


class DifficultyTrainer:
    """Train difficulty scoring model from labeled data."""

    def __init__(self):
        self.model = None

    def train(
        self,
        training_data: list[dict[str, Any]],
        model_type: str = "gradient_boosting",
    ) -> DifficultyModel:
        """Train difficulty model.

        Args:
            training_data: List of training examples:
                [
                    {
                        "tech_name": "react",
                        "context": {...},
                        "difficulty": 5.2
                    }
                ]
            model_type: "gradient_boosting", "random_forest", or "neural_network"

        Returns:
            Trained DifficultyModel
        """
        from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, mean_squared_error
        from sklearn.model_selection import train_test_split

        # Prepare features and targets
        X = []
        y = []
        for example in training_data:
            tech_name = example["tech_name"]
            context = example.get("context", {})
            difficulty = example["difficulty"]

            # Use baseline model's feature extractor
            baseline_model = DifficultyModel()
            features = baseline_model._extract_features(tech_name, context)
            X.append(features)
            y.append(difficulty)

        X = np.array(X)
        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        if model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.model.fit(X_train, y_train)

        # Evaluate
        train_preds = self.model.predict(X_train)
        test_preds = self.model.predict(X_test)

        train_mae = mean_absolute_error(y_train, train_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
        train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_preds))

        print(f"[DifficultyTrainer] Train MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}")
        print(f"[DifficultyTrainer] Test MAE: {test_mae:.3f}, RMSE: {test_rmse:.3f}")

        # Create DifficultyModel wrapper
        difficulty_model = DifficultyModel()
        difficulty_model.model = self.model

        return difficulty_model
