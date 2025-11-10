"""Alternatives recommendation model using embeddings and similarity.

Suggests alternative technologies based on:
- Semantic similarity (use case overlap)
- Skill transferability (learning curve)
- Industry adoption patterns
- Ecosystem compatibility
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class AlternativesModel:
    """Recommend alternative technologies using ML embeddings."""

    def __init__(self, model_path: Path | None = None):
        """Initialize alternatives recommender.

        Args:
            model_path: Path to saved embeddings/model
        """
        self.model_path = model_path
        self.embeddings = {}
        self.tech_graph = {}

        # Hardcoded alternatives (fallback)
        self.baseline_alternatives = {
            "react": ["vue", "angular", "svelte", "nextjs"],
            "vue": ["react", "angular", "svelte"],
            "angular": ["react", "vue", "svelte"],
            "nextjs": ["react", "gatsby", "nuxt"],
            "svelte": ["react", "vue", "angular"],
            "typescript": ["javascript", "flow"],
            "node": ["python_fastapi", "golang", "java_spring"],
            "python_fastapi": ["flask", "django", "node"],
            "flask": ["fastapi", "django", "node"],
            "django": ["flask", "fastapi", "ruby_rails"],
            "golang": ["node", "python_fastapi", "java_spring"],
            "java_spring": ["golang", "django", "node"],
            "ruby_rails": ["django", "node", "php"],
            "postgres": ["mysql", "mariadb", "mongodb"],
            "mysql": ["postgres", "mariadb", "mongodb"],
            "mongodb": ["postgres", "mysql", "dynamodb"],
            "redis": ["memcached", "elasticsearch"],
            "dynamodb": ["mongodb", "cassandra", "postgres"],
            "cassandra": ["dynamodb", "mongodb", "postgres"],
            "docker": ["kubernetes", "podman"],
            "kubernetes": ["docker", "nomad", "ecs"],
            "aws": ["gcp", "azure", "digitalocean"],
            "lambda": ["azure_functions", "google_cloud_functions"],
            "kafka": ["rabbitmq", "sqs", "redis"],
            "rabbitmq": ["kafka", "sqs", "redis"],
            "elasticsearch": ["solr", "algolia", "meilisearch"],
        }

        if model_path and model_path.exists():
            self._load_model()

    def _load_model(self) -> None:
        """Load embeddings and similarity model."""
        try:
            embeddings_file = self.model_path / "tech_embeddings.npy"
            metadata_file = self.model_path / "tech_metadata.json"

            if embeddings_file.exists() and metadata_file.exists():
                # Load embeddings
                embeddings_array = np.load(embeddings_file)

                # Load metadata
                with open(metadata_file, "r", encoding="utf-8") as f:
                    metadata = json.load(f)

                # Rebuild embeddings dict
                for idx, tech_name in enumerate(metadata["tech_names"]):
                    self.embeddings[tech_name] = embeddings_array[idx]

                # Load tech graph if available
                graph_file = self.model_path / "tech_graph.json"
                if graph_file.exists():
                    with open(graph_file, "r", encoding="utf-8") as f:
                        self.tech_graph = json.load(f)

        except Exception as e:
            print(f"[AlternativesModel] Error loading model: {e}, using baseline")

    def recommend(
        self, tech_name: str, top_k: int = 4, filters: dict[str, Any] | None = None
    ) -> list[tuple[str, float]]:
        """Recommend alternative technologies.

        Args:
            tech_name: Source technology
            top_k: Number of alternatives to return
            filters: Optional filters:
                - same_category: Only suggest same category
                - max_difficulty_diff: Max difficulty difference
                - exclude: List of techs to exclude

        Returns:
            List of (tech_name, similarity_score) tuples
        """
        if self.embeddings:
            return self._recommend_with_embeddings(tech_name, top_k, filters)
        else:
            return self._recommend_with_baseline(tech_name, top_k)

    def _recommend_with_embeddings(
        self, tech_name: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[tuple[str, float]]:
        """Recommend using learned embeddings."""
        if tech_name not in self.embeddings:
            return self._recommend_with_baseline(tech_name, top_k)

        source_embedding = self.embeddings[tech_name]
        similarities = []

        for candidate_tech, candidate_embedding in self.embeddings.items():
            if candidate_tech == tech_name:
                continue

            # Apply filters
            if filters:
                if filters.get("exclude") and candidate_tech in filters["exclude"]:
                    continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(
                source_embedding, candidate_embedding
            )
            similarities.append((candidate_tech, float(similarity)))

        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _recommend_with_baseline(
        self, tech_name: str, top_k: int
    ) -> list[tuple[str, float]]:
        """Fallback to hardcoded alternatives."""
        alternatives = self.baseline_alternatives.get(tech_name.lower(), [])

        # Return with dummy similarity scores
        return [(alt, 0.8 - i * 0.1) for i, alt in enumerate(alternatives[:top_k])]

    def _cosine_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def batch_recommend(
        self, tech_names: list[str], top_k: int = 4
    ) -> dict[str, list[tuple[str, float]]]:
        """Recommend alternatives for multiple technologies.

        Args:
            tech_names: List of technology names
            top_k: Number of alternatives per tech

        Returns:
            Dict mapping tech_name -> list of (alternative, score)
        """
        results = {}
        for tech_name in tech_names:
            results[tech_name] = self.recommend(tech_name, top_k)
        return results

    def save(self, output_path: Path) -> None:
        """Save embeddings and model to disk."""
        output_path.mkdir(parents=True, exist_ok=True)

        if self.embeddings:
            # Save embeddings
            tech_names = list(self.embeddings.keys())
            embeddings_array = np.array(
                [self.embeddings[tech] for tech in tech_names]
            )
            np.save(output_path / "tech_embeddings.npy", embeddings_array)

            # Save metadata
            metadata = {"tech_names": tech_names, "embedding_dim": embeddings_array.shape[1]}
            with open(output_path / "tech_metadata.json", "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)

            # Save tech graph
            if self.tech_graph:
                with open(output_path / "tech_graph.json", "w", encoding="utf-8") as f:
                    json.dump(self.tech_graph, f, indent=2)

        # Save baseline alternatives
        config = {
            "baseline_alternatives": self.baseline_alternatives,
            "version": "1.0.0",
        }
        with open(output_path / "alternatives_config.json", "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)


class AlternativesTrainer:
    """Train embeddings for technology alternatives."""

    def __init__(self, embedding_dim: int = 64):
        """Initialize trainer.

        Args:
            embedding_dim: Dimension of learned embeddings
        """
        self.embedding_dim = embedding_dim
        self.embeddings = {}

    def train(
        self,
        technology_corpus: list[dict[str, Any]],
        similarity_pairs: list[tuple[str, str, float]],
    ) -> AlternativesModel:
        """Train embeddings from technology descriptions and similarity data.

        Args:
            technology_corpus: List of tech descriptions:
                [
                    {
                        "name": "react",
                        "description": "JavaScript library for building UIs",
                        "use_cases": ["spa", "web apps"],
                        "category": "frontend"
                    }
                ]
            similarity_pairs: List of (tech1, tech2, similarity_score) tuples
                where similarity_score is 0-1 (1 = very similar)

        Returns:
            Trained AlternativesModel
        """
        print(f"[AlternativesTrainer] Training embeddings with {len(technology_corpus)} techs")

        # Simple approach: Use TF-IDF + dimensionality reduction
        # In production, use Word2Vec, BERT, or graph embeddings

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        # Prepare text corpus
        tech_names = []
        documents = []

        for tech in technology_corpus:
            tech_names.append(tech["name"])
            # Combine description and use cases
            text = tech.get("description", "")
            text += " " + " ".join(tech.get("use_cases", []))
            text += " " + tech.get("category", "")
            documents.append(text)

        # Compute TF-IDF
        vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Reduce dimensionality
        svd = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        embeddings_array = svd.fit_transform(tfidf_matrix)

        # Normalize embeddings
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        embeddings_array = embeddings_array / (norms + 1e-8)

        # Store embeddings
        for tech_name, embedding in zip(tech_names, embeddings_array):
            self.embeddings[tech_name] = embedding

        print(f"[AlternativesTrainer] Generated {len(self.embeddings)} embeddings")

        # Evaluate on similarity pairs
        if similarity_pairs:
            self._evaluate_embeddings(similarity_pairs)

        # Create AlternativesModel
        model = AlternativesModel()
        model.embeddings = self.embeddings

        return model

    def _evaluate_embeddings(
        self, similarity_pairs: list[tuple[str, str, float]]
    ) -> None:
        """Evaluate embeddings on known similarity pairs."""
        errors = []

        for tech1, tech2, true_similarity in similarity_pairs:
            if tech1 in self.embeddings and tech2 in self.embeddings:
                pred_similarity = self._cosine_similarity(
                    self.embeddings[tech1], self.embeddings[tech2]
                )
                error = abs(pred_similarity - true_similarity)
                errors.append(error)

        if errors:
            mae = np.mean(errors)
            print(f"[AlternativesTrainer] Similarity MAE: {mae:.3f}")

    def _cosine_similarity(
        self, vec1: np.ndarray, vec2: np.ndarray
    ) -> float:
        """Compute cosine similarity."""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
