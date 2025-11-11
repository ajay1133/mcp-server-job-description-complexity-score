"""Experience extraction model using regex + ML validation.

Extracts years of experience mentioned for technologies, validates
with ML to handle implicit mentions and ambiguous cases.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


class ExperienceModel:
    """Extract experience requirements from text."""

    def __init__(self, model_path: Path | None = None):
        """Initialize experience extraction model.

        Args:
            model_path: Path to saved model. If None, uses regex only.
        """
        self.model_path = model_path
        self.validator = None

        if model_path and model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        """Load ML validator for experience extraction."""
        try:
            import joblib

            model_file = self.model_path / "experience_validator.pkl"
            if model_file.exists():
                self.validator = joblib.load(model_file)
        except Exception as e:
            print(f"[ExperienceModel] Error loading validator: {e}")

    def extract_tech_experience(self, text: str, tech_name: str) -> float | None:
        """Extract experience years for a specific technology.

        Args:
            text: Input text
            tech_name: Technology to search for

        Returns:
            Years of experience or None if not found
        """
        text_lower = text.lower()
        tech_lower = tech_name.lower()

        # Regex patterns for tech-specific experience
        patterns = [
            # "5+ years React", "5 years of React"
            rf"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?{re.escape(tech_lower)}",
            # "React 5+ years", "React experience: 5 years"
            rf"{re.escape(tech_lower)}.*?(\d+)\+?\s*(?:years?|yrs?)",
            # "5+ years in React"
            rf"(\d+)\+?\s*(?:years?|yrs?)\s+(?:in|with)\s+{re.escape(tech_lower)}",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = float(match.group(1))
                # Validate with ML if available
                if self.validator:
                    confidence = self._validate_extraction(text, tech_name, years)
                    if confidence > 0.5:
                        return years
                else:
                    return years

        return None

    def extract_overall_experience(self, text: str) -> float | None:
        """Extract overall experience from text.

        Args:
            text: Input text

        Returns:
            Years of overall experience or None
        """
        text_lower = text.lower()

        patterns = [
            r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:overall\s+)?experience",
            r"(?:overall\s+)?experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)",
            r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:in\s+)?(?:the\s+)?(?:field|industry)",
            r"(\d+)\+?\s*(?:years?|yrs?)\s+(?:professional\s+)?(?:work\s+)?experience",
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))

        return None

    def _validate_extraction(self, text: str, tech_name: str, years: float) -> float:
        """Validate extraction using ML classifier.

        Args:
            text: Original text
            tech_name: Technology name
            years: Extracted years

        Returns:
            Confidence score (0-1)
        """
        if self.validator is None:
            return 1.0

        # Extract features for validation
        features = self._extract_validation_features(text, tech_name, years)

        # Predict confidence
        try:
            confidence = self.validator.predict_proba([features])[0][1]
            return float(confidence)
        except Exception:
            return 1.0  # Fallback

    def _extract_validation_features(self, text: str, tech_name: str, years: float) -> list[float]:
        """Extract features for ML validation.

        Features:
        - Years value (normalized)
        - Distance from tech mention to years mention
        - Context words (senior, junior, lead, etc.)
        - Sentence structure indicators
        """
        features = []

        # Normalized years
        features.append(years / 10.0)

        # Context analysis
        text_lower = text.lower()
        context_keywords = {
            "senior": 1.0,
            "lead": 1.2,
            "principal": 1.5,
            "junior": -0.5,
            "entry": -0.8,
            "mid": 0.0,
        }

        context_score = 0.0
        for keyword, weight in context_keywords.items():
            if keyword in text_lower:
                context_score += weight
        features.append(context_score)

        # Tech mention proximity
        tech_pos = text_lower.find(tech_name.lower())
        years_pattern = rf"\b{int(years)}\+?\s*(?:years?|yrs?)\b"
        years_match = re.search(years_pattern, text_lower)

        if tech_pos >= 0 and years_match:
            distance = abs(years_match.start() - tech_pos)
            # Normalize distance (closer = higher feature value)
            features.append(1.0 / (1.0 + distance / 100.0))
        else:
            features.append(0.0)

        return features

    def batch_extract(self, text: str, tech_names: list[str]) -> dict[str, float | None]:
        """Extract experience for multiple technologies.

        Args:
            text: Input text
            tech_names: List of technology names

        Returns:
            Dict mapping tech_name -> experience_years (or None)
        """
        results = {}
        for tech_name in tech_names:
            results[tech_name] = self.extract_tech_experience(text, tech_name)
        return results

    def save(self, output_path: Path) -> None:
        """Save model to disk."""
        output_path.mkdir(parents=True, exist_ok=True)

        if self.validator is not None:
            import joblib

            joblib.dump(self.validator, output_path / "experience_validator.pkl")

        # Save config
        config = {
            "version": "1.0.0",
            "patterns": [
                "X+ years [of] tech",
                "tech X+ years",
                "X+ years in/with tech",
            ],
        }
        with open(output_path / "experience_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


class ExperienceTrainer:
    """Train ML validator for experience extraction."""

    def __init__(self):
        self.validator = None

    def train(self, training_data: list[dict[str, Any]]) -> ExperienceModel:
        """Train experience validation model.

        Args:
            training_data: List of training examples:
                [
                    {
                        "text": "5 years React experience",
                        "tech_name": "react",
                        "years": 5.0,
                        "is_valid": True
                    }
                ]

        Returns:
            Trained ExperienceModel
        """
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split

        # Prepare features
        X = []
        y = []
        experience_model = ExperienceModel()

        for example in training_data:
            features = experience_model._extract_validation_features(
                example["text"], example["tech_name"], example["years"]
            )
            X.append(features)
            y.append(1 if example["is_valid"] else 0)

        X = np.array(X)
        y = np.array(y)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train validator
        self.validator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.validator.fit(X_train, y_train)

        # Evaluate
        train_preds = self.validator.predict(X_train)
        test_preds = self.validator.predict(X_test)

        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        precision = precision_score(y_test, test_preds)
        recall = recall_score(y_test, test_preds)

        print(f"[ExperienceTrainer] Train Accuracy: {train_acc:.3f}")
        print(f"[ExperienceTrainer] Test Accuracy: {test_acc:.3f}")
        print(f"[ExperienceTrainer] Precision: {precision:.3f}, Recall: {recall:.3f}")

        # Create ExperienceModel wrapper
        exp_model = ExperienceModel()
        exp_model.validator = self.validator

        return exp_model


# Import numpy here since it's used in trainer
try:
    import numpy as np
except ImportError:
    np = None
