"""ML-based difficulty estimator using external metrics.

This module predicts difficulty for NEW technologies without retraining by:
1. Fetching real-world metrics (GitHub stars, StackOverflow questions, npm downloads)
2. Using a pre-trained model that generalizes from features (not tech names)
3. Providing intelligent heuristics when models/APIs unavailable

No retraining needed when new techs emerge!
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Optional imports
try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class MLDifficultyEstimator:
    """Estimate difficulty for ANY technology using external metrics + ML."""

    def __init__(self, cache_dir: Path | None = None, github_token: Optional[str] = None):
        """Initialize estimator.

        Args:
            cache_dir: Directory for caching API results
            github_token: GitHub API token (optional, for higher rate limits)
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache" / "difficulty"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.github_token = github_token

        # Load pre-trained feature-based model (if available)
        self.model = self._load_pretrained_model()

        # Heuristic weights (learned from historical data)
        self.feature_weights = {
            "github_stars_log": -0.15,  # More stars = more docs = slightly easier
            "stackoverflow_questions_log": 0.08,  # More questions = complexity
            "years_in_market": -0.12,  # Older = more mature = easier
            "github_contributors_log": -0.05,  # More contributors = better maintained
            "npm_downloads_log": -0.10,  # Popular = more resources
            "github_issues_ratio": 0.20,  # High open issues = harder to use
            "doc_quality_score": -0.25,  # Better docs = easier
            "api_surface_complexity": 0.30,  # Larger API = harder
        }

        # Baseline difficulty (when no data available)
        self.baseline = 5.0

    def _load_pretrained_model(self):
        """Load pre-trained ML model (feature-based, not tech-name-based)."""
        model_path = Path(__file__).parent.parent / "models" / "difficulty" / "feature_model.pkl"
        if model_path.exists():
            try:
                import joblib

                return joblib.load(model_path)
            except (ImportError, Exception) as e:
                print(f"[MLDifficultyEstimator] Could not load model: {e}")
        return None

    def estimate_difficulty(self, tech_name: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Estimate difficulty for any technology.

        Args:
            tech_name: Technology name (e.g., "bun", "htmx", "astro")
            force_refresh: Skip cache and fetch fresh data

        Returns:
            {
                "difficulty": 5.3,
                "confidence": 0.85,
                "source": "ml_model" | "heuristic" | "fallback",
                "features": {...},
                "explanation": "..."
            }
        """
        # 1. Check cache
        if not force_refresh:
            cached = self._load_from_cache(tech_name)
            if cached:
                return cached

        # 2. Fetch external metrics
        metrics = self._fetch_metrics(tech_name)

        # 3. Extract features
        features = self._extract_features(tech_name, metrics)

        # 4. Predict difficulty
        if self.model is not None:
            # Use ML model
            difficulty = self._predict_with_model(features)
            source = "ml_model"
            confidence = 0.85
        elif metrics.get("has_data"):
            # Use heuristic formula
            difficulty = self._predict_with_heuristic(features)
            source = "heuristic"
            confidence = 0.70
        else:
            # Fallback
            difficulty = self.baseline
            source = "fallback"
            confidence = 0.30

        # Clip to valid range
        difficulty = max(1.0, min(10.0, difficulty))

        result = {
            "difficulty": round(difficulty, 1),
            "confidence": confidence,
            "source": source,
            "features": features,
            "explanation": self._generate_explanation(tech_name, features, difficulty),
            "timestamp": datetime.now().isoformat(),
        }

        # 5. Cache result
        self._save_to_cache(tech_name, result)

        return result

    def _fetch_metrics(self, tech_name: str) -> Dict[str, Any]:
        """Fetch metrics from external APIs."""
        metrics = {"has_data": False}

        if not REQUESTS_AVAILABLE:
            return metrics

        # Try GitHub
        github_data = self._fetch_github_metrics(tech_name)
        if github_data:
            metrics.update(github_data)
            metrics["has_data"] = True

        # Try npm (for JS libraries)
        npm_data = self._fetch_npm_metrics(tech_name)
        if npm_data:
            metrics.update(npm_data)
            metrics["has_data"] = True

        # Try StackOverflow
        so_data = self._fetch_stackoverflow_metrics(tech_name)
        if so_data:
            metrics.update(so_data)
            metrics["has_data"] = True

        return metrics

    def _fetch_github_metrics(self, tech_name: str) -> Optional[Dict[str, Any]]:
        """Fetch GitHub metrics (stars, contributors, issues)."""
        # Common repo name patterns
        repo_patterns = [
            f"{tech_name}/{tech_name}",  # e.g., sveltejs/svelte
            f"{tech_name}js/{tech_name}",  # e.g., reactjs/react
            f"facebook/{tech_name}",  # Facebook projects
            f"vercel/{tech_name}",  # Vercel projects
            f"{tech_name}/{tech_name}.js",  # .js suffix
        ]

        headers = {}
        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        for repo_name in repo_patterns:
            try:
                response = requests.get(
                    f"https://api.github.com/repos/{repo_name}",
                    headers=headers,
                    timeout=5,
                )
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "github_stars": data.get("stargazers_count", 0),
                        "github_forks": data.get("forks_count", 0),
                        "github_open_issues": data.get("open_issues_count", 0),
                        "github_watchers": data.get("watchers_count", 0),
                        "github_created_at": data.get("created_at"),
                        "github_updated_at": data.get("updated_at"),
                    }
                elif response.status_code == 403:
                    # Rate limit
                    print(f"[GitHub] Rate limit hit for {tech_name}")
                    break
            except Exception as e:
                print(f"[GitHub] Error fetching {repo_name}: {e}")
                continue

        return None

    def _fetch_npm_metrics(self, tech_name: str) -> Optional[Dict[str, Any]]:
        """Fetch npm download stats."""
        try:
            # Try direct package name
            response = requests.get(
                f"https://api.npmjs.org/downloads/point/last-month/{tech_name}",
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                return {"npm_downloads": data.get("downloads", 0)}
        except Exception as e:
            print(f"[npm] Error fetching {tech_name}: {e}")

        return None

    def _fetch_stackoverflow_metrics(self, tech_name: str) -> Optional[Dict[str, Any]]:
        """Fetch StackOverflow question count."""
        try:
            response = requests.get(
                "https://api.stackexchange.com/2.3/tags",
                params={"order": "desc", "sort": "popular", "inname": tech_name, "site": "stackoverflow"},
                timeout=5,
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("items"):
                    tag = data["items"][0]
                    return {"stackoverflow_questions": tag.get("count", 0)}
        except Exception as e:
            print(f"[StackOverflow] Error fetching {tech_name}: {e}")

        return None

    def _extract_features(self, tech_name: str, metrics: Dict[str, Any]) -> Dict[str, float]:
        """Extract normalized features for prediction."""
        features = {}

        # GitHub features (log-scaled)
        github_stars = metrics.get("github_stars", 1000)
        features["github_stars_log"] = float(np.log1p(github_stars))

        github_forks = metrics.get("github_forks", 100)
        features["github_forks_log"] = float(np.log1p(github_forks))

        github_open_issues = metrics.get("github_open_issues", 50)
        features["github_issues_ratio"] = float(github_open_issues / max(github_stars, 1) * 1000)

        # Age of project
        created_at = metrics.get("github_created_at")
        if created_at:
            created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
            years_old = (datetime.now(created_date.tzinfo) - created_date).days / 365.25
            features["years_in_market"] = float(min(years_old, 20))
        else:
            features["years_in_market"] = 5.0  # Default

        # npm popularity
        npm_downloads = metrics.get("npm_downloads", 10000)
        features["npm_downloads_log"] = float(np.log1p(npm_downloads))

        # StackOverflow community size
        so_questions = metrics.get("stackoverflow_questions", 500)
        features["stackoverflow_questions_log"] = float(np.log1p(so_questions))

        # Derived features
        # High stars + low issues = well-maintained = easier
        features["maintenance_score"] = float(features["github_stars_log"] - features["github_issues_ratio"])

        # Popular + old = mature = easier
        features["maturity_score"] = float(features["years_in_market"] * features["github_stars_log"] / 10)

        return features

    def _predict_with_model(self, features: Dict[str, float]) -> float:
        """Predict using ML model (feature-based, works for any tech)."""
        # Convert to array (order matters!)
        feature_order = [
            "github_stars_log",
            "github_forks_log",
            "github_issues_ratio",
            "years_in_market",
            "npm_downloads_log",
            "stackoverflow_questions_log",
            "maintenance_score",
            "maturity_score",
        ]
        X = np.array([[features.get(f, 0.0) for f in feature_order]], dtype=np.float32)

        prediction = self.model.predict(X)[0]
        return float(prediction)

    def _predict_with_heuristic(self, features: Dict[str, float]) -> float:
        """Predict using weighted heuristic (when model unavailable)."""
        difficulty = self.baseline

        # Apply feature weights
        for feature, weight in self.feature_weights.items():
            if feature in features:
                difficulty += weight * (features[feature] / 10.0)  # Normalize

        # Additional heuristics
        github_stars = np.exp(features.get("github_stars_log", 0)) - 1

        # Very popular (>50k stars) = mature = easier
        if github_stars > 50000:
            difficulty -= 0.5

        # Very new (<1 year) = less stable = harder
        if features.get("years_in_market", 5) < 1:
            difficulty += 0.8

        # High maintenance score = easier
        if features.get("maintenance_score", 0) > 8:
            difficulty -= 0.4

        return difficulty

    def _generate_explanation(self, tech_name: str, features: Dict[str, float], difficulty: float) -> str:
        """Generate human-readable explanation."""
        github_stars = int(np.exp(features.get("github_stars_log", 0)) - 1)
        years_old = features.get("years_in_market", 5)
        so_questions = int(np.exp(features.get("stackoverflow_questions_log", 0)) - 1)

        explanation_parts = [f"{tech_name.title()} difficulty: {difficulty:.1f}/10"]

        if github_stars > 50000:
            explanation_parts.append(f"Very popular ({github_stars:,} stars) with extensive resources")
        elif github_stars > 10000:
            explanation_parts.append(f"Popular ({github_stars:,} stars) with good community support")
        elif github_stars < 1000:
            explanation_parts.append(f"Limited community ({github_stars:,} stars), fewer resources")

        if years_old > 5:
            explanation_parts.append(f"Mature technology ({years_old:.1f} years)")
        elif years_old < 2:
            explanation_parts.append(f"Relatively new ({years_old:.1f} years), evolving rapidly")

        if so_questions > 10000:
            explanation_parts.append(f"Large StackOverflow community ({so_questions:,} questions)")
        elif so_questions < 500:
            explanation_parts.append(f"Small community ({so_questions} SO questions)")

        return ". ".join(explanation_parts)

    def _load_from_cache(self, tech_name: str) -> Optional[Dict[str, Any]]:
        """Load cached difficulty estimate."""
        cache_file = self.cache_dir / f"{tech_name.lower()}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    # Check if cache is fresh (24 hours)
                    timestamp = datetime.fromisoformat(cached.get("timestamp", "2000-01-01"))
                    if (datetime.now() - timestamp).total_seconds() < 86400:
                        return cached
            except (json.JSONDecodeError, ValueError):
                pass
        return None

    def _save_to_cache(self, tech_name: str, result: Dict[str, Any]) -> None:
        """Save difficulty estimate to cache."""
        cache_file = self.cache_dir / f"{tech_name.lower()}.json"
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)


# Singleton instance
_estimator_instance: Optional[MLDifficultyEstimator] = None


def get_difficulty_estimator() -> MLDifficultyEstimator:
    """Get singleton difficulty estimator."""
    global _estimator_instance
    if _estimator_instance is None:
        _estimator_instance = MLDifficultyEstimator()
    return _estimator_instance
