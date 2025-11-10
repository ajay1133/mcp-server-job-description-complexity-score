"""Technology extraction model using transformer-based NER.

This model identifies technology mentions in job descriptions and resumes.
Uses a fine-tuned transformer (DistilBERT/RoBERTa) for token classification.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np


class TechExtractorModel:
    """Extract technology mentions from text using ML-based NER."""

    def __init__(self, model_path: Path | None = None):
        """Initialize the tech extractor model.

        Args:
            model_path: Path to saved model weights. If None, uses built-in patterns.
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.tech_categories = {
            "frontend": ["react", "vue", "angular", "nextjs", "svelte", "typescript"],
            "backend": [
                "node",
                "python",
                "fastapi",
                "flask",
                "django",
                "golang",
                "java",
                "spring",
                "ruby",
                "rails",
            ],
            "database": [
                "postgres",
                "mysql",
                "mongodb",
                "redis",
                "dynamodb",
                "cassandra",
            ],
            "infrastructure": ["docker", "kubernetes", "aws", "lambda"],
            "messaging": ["kafka", "rabbitmq"],
            "search": ["elasticsearch"],
        }

        if model_path and model_path.exists():
            self._load_model()
        else:
            # Use pattern-based fallback until model is trained
            self._initialize_patterns()

    def _initialize_patterns(self) -> None:
        """Initialize regex patterns for technology detection (fallback)."""
        self.tech_patterns = {
            "react": r"\b(?:react|react\.js|reactjs)\b",
            "vue": r"\b(?:vue|vue\.js|vuejs)\b",
            "angular": r"\bangular\b",
            "nextjs": r"\b(?:next|next\.js|nextjs)\b",
            "svelte": r"\b(?:svelte|sveltekit)\b",
            "typescript": r"\b(?:typescript|ts)\b",
            "node": r"\b(?:node|node\.js|nodejs|express|expressjs)\b",
            "python": r"\bpython\b",
            "fastapi": r"\b(?:fastapi|fast api)\b",
            "flask": r"\bflask\b",
            "django": r"\bdjango\b",
            "golang": r"\b(?:golang|go lang|go)\b",
            "java": r"\bjava\b",
            "spring": r"\b(?:spring|spring boot|springboot)\b",
            "ruby": r"\bruby\b",
            "rails": r"\b(?:ruby on rails|rails|ror)\b",
            "postgres": r"\b(?:postgres|postgresql)\b",
            "mysql": r"\bmysql\b",
            "mongodb": r"\b(?:mongodb|mongo)\b",
            "redis": r"\bredis\b",
            "dynamodb": r"\bdynamodb\b",
            "cassandra": r"\bcassandra\b",
            "docker": r"\bdocker\b",
            "kubernetes": r"\b(?:kubernetes|k8s)\b",
            "aws": r"\b(?:aws|amazon web services)\b",
            "lambda": r"\b(?:lambda|aws lambda)\b",
            "kafka": r"\bkafka\b",
            "rabbitmq": r"\b(?:rabbitmq|rabbit mq)\b",
            "elasticsearch": r"\b(?:elasticsearch|elastic search)\b",
        }

    def _load_model(self) -> None:
        """Load trained transformer model."""
        try:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
            )

            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.model = AutoModelForTokenClassification.from_pretrained(str(self.model_path))
            self.model.eval()

            # Load label mapping
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                    self.id2label = config.get("id2label", {})
                    self.label2id = config.get("label2id", {})
        except ImportError:
            print("[TechExtractorModel] transformers/torch not available, using patterns")
            self._initialize_patterns()
        except Exception as e:
            print(f"[TechExtractorModel] Error loading model: {e}, using patterns")
            self._initialize_patterns()

    def extract(self, text: str) -> list[dict[str, Any]]:
        """Extract technologies from text.

        Args:
            text: Input text (job description or resume)

        Returns:
            List of extracted technologies with metadata:
            [
                {
                    "name": "react",
                    "category": "frontend",
                    "span": (start, end),
                    "confidence": 0.95
                }
            ]
        """
        if self.model is not None:
            return self._extract_with_model(text)
        else:
            return self._extract_with_patterns(text)

    def _extract_with_model(self, text: str) -> list[dict[str, Any]]:
        """Extract using transformer model."""
        import torch

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )

        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            probabilities = torch.softmax(outputs.logits, dim=-1)

        # Decode predictions to spans
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        predictions = predictions[0].cpu().numpy()
        probabilities = probabilities[0].cpu().numpy()

        technologies = []
        current_tech = None
        current_span_start = None

        for idx, (token, pred_id, probs) in enumerate(zip(tokens, predictions, probabilities)):
            label = self.id2label.get(str(pred_id), "O")
            confidence = float(np.max(probs))

            if label.startswith("B-"):
                # Begin new entity
                if current_tech:
                    technologies.append(current_tech)

                tech_name = label[2:].lower()
                category = self._infer_category(tech_name)
                current_tech = {
                    "name": tech_name,
                    "category": category,
                    "span": (idx, idx + 1),
                    "confidence": confidence,
                }
                current_span_start = idx

            elif label.startswith("I-") and current_tech:
                # Continue current entity
                current_tech["span"] = (current_span_start, idx + 1)
                current_tech["confidence"] = (current_tech["confidence"] + confidence) / 2

            elif label == "O" and current_tech:
                # End current entity
                technologies.append(current_tech)
                current_tech = None
                current_span_start = None

        if current_tech:
            technologies.append(current_tech)

        return technologies

    def _extract_with_patterns(self, text: str) -> list[dict[str, Any]]:
        """Extract using regex patterns (fallback)."""
        text_lower = text.lower()
        technologies = []

        for tech_name, pattern in self.tech_patterns.items():
            matches = list(re.finditer(pattern, text_lower, re.IGNORECASE))
            if matches:
                category = self._infer_category(tech_name)
                for match in matches:
                    technologies.append(
                        {
                            "name": tech_name,
                            "category": category,
                            "span": match.span(),
                            "confidence": 0.85,  # Pattern-based confidence
                        }
                    )

        # Deduplicate by name (keep highest confidence)
        unique_techs = {}
        for tech in technologies:
            name = tech["name"]
            if name not in unique_techs or tech["confidence"] > unique_techs[name]["confidence"]:
                unique_techs[name] = tech

        return list(unique_techs.values())

    def _infer_category(self, tech_name: str) -> str:
        """Infer category from tech name."""
        for category, techs in self.tech_categories.items():
            if tech_name in techs or any(tech_name.startswith(t) for t in techs):
                return category
        return "other"

    def save(self, output_path: Path) -> None:
        """Save model to disk."""
        output_path.mkdir(parents=True, exist_ok=True)

        if self.model is not None:
            self.model.save_pretrained(str(output_path))
            self.tokenizer.save_pretrained(str(output_path))

            # Save additional config
            config = {
                "id2label": self.id2label,
                "label2id": self.label2id,
                "tech_categories": self.tech_categories,
            }
            with open(output_path / "config.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
        else:
            # Save patterns
            config = {
                "patterns": self.tech_patterns,
                "tech_categories": self.tech_categories,
            }
            with open(output_path / "patterns.json", "w", encoding="utf-8") as f:
                json.dump(config, f, indent=2)
