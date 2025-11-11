"""Dynamic technology registry with external data integration.

This module provides a self-updating technology database that can:
1. Fetch trending technologies from external sources
2. Estimate difficulty from community metrics (GitHub stars, StackOverflow questions)
3. Find similar technologies using semantic search
4. Cache results to avoid rate limits
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# Optional imports for external data
try:
    import requests  # noqa: F401

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


class TechRegistry:
    """Dynamic registry that learns about new technologies."""

    def __init__(self, cache_dir: Path | None = None, cache_ttl_hours: int = 24):
        """Initialize tech registry.

        Args:
            cache_dir: Directory for caching external data
            cache_ttl_hours: How long to cache external data
        """
        self.cache_dir = cache_dir or Path(__file__).parent.parent / "cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)

        # Load baseline tech database (for offline fallback)
        self.baseline_db = self._load_baseline_db()

        # Load cached external data
        self.external_cache = self._load_cache("external_techs.json")

    def _load_baseline_db(self) -> Dict[str, Any]:
        """Load baseline technology database (embedded fallback)."""
        return {
            "react": {
                "difficulty": 5.2,
                "category": "frontend",
                "keywords": ["react", "react.js", "reactjs"],
            },
            "vue": {
                "difficulty": 4.8,
                "category": "frontend",
                "keywords": ["vue", "vue.js", "vuejs"],
            },
            "svelte": {
                "difficulty": 4.5,
                "category": "frontend",
                "keywords": ["svelte", "sveltekit"],
            },
            "node": {
                "difficulty": 5.0,
                "category": "backend",
                "keywords": ["node", "node.js", "nodejs"],
            },
            "deno": {
                "difficulty": 5.5,
                "category": "backend",
                "keywords": ["deno"],
            },
            "bun": {
                "difficulty": 5.3,
                "category": "backend",
                "keywords": ["bun", "bun.js"],
            },
            "postgres": {
                "difficulty": 5.5,
                "category": "database",
                "keywords": ["postgres", "postgresql"],
            },
            "docker": {
                "difficulty": 5.5,
                "category": "infrastructure",
                "keywords": ["docker"],
            },
            # Add more as needed...
        }

    def _load_cache(self, filename: str) -> Dict[str, Any]:
        """Load cached data if not expired."""
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    cache_time = datetime.fromisoformat(cache.get("timestamp", "2000-01-01"))
                    if datetime.now() - cache_time < self.cache_ttl:
                        return cache.get("data", {})
            except (json.JSONDecodeError, ValueError):
                pass
        return {}

    def _save_cache(self, filename: str, data: Dict[str, Any]) -> None:
        """Save data to cache."""
        cache_path = self.cache_dir / filename
        cache = {"timestamp": datetime.now().isoformat(), "data": data}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)

    def get_tech_info(self, tech_name: str) -> Optional[Dict[str, Any]]:
        """Get info for a technology (with external enrichment).

        Args:
            tech_name: Technology name (e.g., "react", "bun", "htmx")

        Returns:
            Tech info dict with difficulty, category, keywords, etc.
        """
        tech_lower = tech_name.lower()

        # 1. Check baseline database
        if tech_lower in self.baseline_db:
            return self.baseline_db[tech_lower]

        # 2. Check external cache
        if tech_lower in self.external_cache:
            return self.external_cache[tech_lower]

        # 3. Try to fetch from external sources
        if REQUESTS_AVAILABLE:
            external_info = self._fetch_tech_info(tech_lower)
            if external_info:
                # Cache it
                self.external_cache[tech_lower] = external_info
                self._save_cache("external_techs.json", self.external_cache)
                return external_info

        # 4. Return best guess with low confidence
        return {
            "difficulty": 5.0,  # Default mid-range
            "category": self.infer_category_by_name(tech_lower),
            "keywords": [tech_lower],
            "confidence": 0.3,  # Low confidence for unknown tech
            "source": "fallback",
        }

    def _fetch_tech_info(self, tech_name: str) -> Optional[Dict[str, Any]]:
        """Fetch technology info from external sources using ML estimator.

        Uses MLDifficultyEstimator to predict difficulty from:
        - GitHub API (stars, contributors, activity)
        - StackOverflow API (questions, tags)
        - npm/PyPI stats
        """
        try:
            # Use ML-based difficulty estimator
            from mcp_server.ml_difficulty_estimator import get_difficulty_estimator

            estimator = get_difficulty_estimator()
            result = estimator.estimate_difficulty(tech_name)

            if result["confidence"] > 0.5:
                # Good confidence - use it
                return {
                    "difficulty": result["difficulty"],
                    "category": self._infer_category_from_features(result["features"]),
                    "keywords": [tech_name.lower()],
                    "confidence": result["confidence"],
                    "source": result["source"],
                    "explanation": result["explanation"],
                }

            return None

        except Exception as e:
            print(f"[TechRegistry] Error fetching {tech_name}: {e}")
            return None

    def _infer_category_from_features(self, features: Dict[str, Any]) -> str:
        """Infer category from metrics (heuristic).

        If no reliable features, fall back to name-based inference.
        """
        name = (features or {}).get("name") if isinstance(features, dict) else None
        if isinstance(name, str) and name:
            return self.infer_category_by_name(name)
        return "other"

    def infer_category_by_name(self, tech_name: str) -> str:
        """Heuristically infer category from a technology name.

        Keeps categories broad and stable to work with alternatives search.
        """
        name = (tech_name or "").lower()

        mapping = [
            (
                "observability",
                ["grafana", "prometheus", "datadog", "splunk", "newrelic", "new relic", "kibana", "logstash"],
            ),
            ("infrastructure", ["terraform", "pulumi", "packer", "vagrant", "ansible", "puppet", "chef", "salt"]),
            ("cicd", ["jenkins", "gitlab ci", "github actions", "travis", "circleci", "teamcity", "bamboo"]),
            ("orchestration", ["kubernetes", "k8s", "helm", "istio", "linkerd", "envoy"]),
            ("webserver", ["nginx", "apache", "httpd", "caddy"]),
            (
                "database",
                [
                    "postgres",
                    "postgresql",
                    "mysql",
                    "mariadb",
                    "sqlite",
                    "mongodb",
                    "cassandra",
                    "neo4j",
                    "influxdb",
                    "timescaledb",
                    "dynamodb",
                ],
            ),
            ("cache", ["redis", "memcached"]),
            ("search", ["elasticsearch", "solr", "meilisearch", "algolia"]),
            ("frontend", ["react", "vue", "angular", "svelte", "nextjs", "vite", "nuxt", "ember", "backbone"]),
            (
                "backend",
                [
                    "flask",
                    "django",
                    "fastapi",
                    "express",
                    "nestjs",
                    "spring",
                    "spring boot",
                    "rails",
                    "laravel",
                    "symfony",
                    "golang",
                    "rust",
                    "java",
                    "dotnet",
                ],
            ),
            ("devops", ["docker", "containerd", "podman", "kubernetes", "k8s"]),
            ("cloud", ["aws", "gcp", "azure", "google cloud", "amazon web services", "microsoft azure"]),
        ]

        for category, keywords in mapping:
            for kw in keywords:
                if kw in name:
                    return category

        return "other"

    def search_similar_techs(self, tech_name: str, top_k: int = 5) -> List[str]:
        """Find similar technologies (for alternatives).

        Args:
            tech_name: Technology to find alternatives for
            top_k: Number of alternatives to return

        Returns:
            List of similar technology names
        """
        # Simple keyword-based similarity for now
        # In production: use embeddings or external APIs
        tech_lower = tech_name.lower()
        candidates = []

        tech_info = self.get_tech_info(tech_lower)
        if not tech_info:
            return []

        target_category = tech_info.get("category", "other")

        # Find techs in same category
        all_techs = list(self.baseline_db.keys()) + list(self.external_cache.keys())
        for candidate in all_techs:
            if candidate == tech_lower:
                continue

            candidate_info = self.get_tech_info(candidate)
            if candidate_info and candidate_info.get("category") == target_category:
                candidates.append(candidate)

        return candidates[:top_k]

    def add_custom_tech(
        self,
        tech_name: str,
        difficulty: float,
        category: str,
        keywords: Optional[List[str]] = None,
    ) -> None:
        """Manually add a custom technology to the registry.

        Args:
            tech_name: Technology name
            difficulty: Difficulty score (1-10)
            category: Category (frontend, backend, database, etc.)
            keywords: List of keyword variations
        """
        tech_lower = tech_name.lower()
        self.baseline_db[tech_lower] = {
            "difficulty": difficulty,
            "category": category,
            "keywords": keywords or [tech_lower],
            "source": "custom",
        }

    def get_all_keywords(self) -> Dict[str, List[str]]:
        """Get all known keywords for pattern matching.

        Returns:
            Dict mapping tech_id -> list of keywords
        """
        keywords = {}
        all_techs = list(self.baseline_db.keys()) + list(self.external_cache.keys())

        for tech_name in all_techs:
            tech_info = self.get_tech_info(tech_name)
            if tech_info and "keywords" in tech_info:
                keywords[tech_name] = tech_info["keywords"]
            else:
                keywords[tech_name] = [tech_name]

        return keywords


# Singleton instance
_registry_instance: Optional[TechRegistry] = None


def get_tech_registry() -> TechRegistry:
    """Get singleton tech registry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = TechRegistry()
    return _registry_instance
