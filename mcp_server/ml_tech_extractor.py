"""ML-powered technology extractor.

This module provides an interface that matches the simple_tech_extractor API
but uses trained ML models for:
- Technology extraction (NER or pattern-based fallback)
- Difficulty prediction (ML or baseline scores)
- Experience extraction (regex + ML validation)
- Alternative recommendations (embeddings or baseline)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

# Import ML models
from mcp_server.ml_models import (
    AlternativesModel,
    DifficultyModel,
    ExperienceModel,
    TechExtractorModel,
)

MODELS_DIR = Path(__file__).parent.parent / "models"


class MLTechExtractor:
    """ML-powered technology extractor with fallback to baseline."""

    def __init__(self, models_dir: Path | None = None):
        """Initialize ML tech extractor.

        Args:
            models_dir: Path to models directory. If None, uses default.
        """
        self.models_dir = models_dir or MODELS_DIR

        # Load models (graceful fallback if not trained)
        self.tech_extractor = self._load_tech_extractor()
        self.difficulty_model = self._load_difficulty_model()
        self.experience_model = self._load_experience_model()
        self.alternatives_model = self._load_alternatives_model()

        print(f"[MLTechExtractor] Initialized with models from {self.models_dir}")

    def _load_tech_extractor(self) -> TechExtractorModel:
        """Load technology extraction model."""
        model_path = self.models_dir / "tech_extractor"
        if model_path.exists():
            return TechExtractorModel(model_path)
        else:
            # Fallback to pattern-based
            return TechExtractorModel()

    def _load_difficulty_model(self) -> DifficultyModel:
        """Load difficulty scoring model."""
        model_path = self.models_dir / "difficulty"
        if model_path.exists():
            return DifficultyModel(model_path)
        else:
            return DifficultyModel()

    def _load_experience_model(self) -> ExperienceModel:
        """Load experience extraction model."""
        model_path = self.models_dir / "experience"
        if model_path.exists():
            return ExperienceModel(model_path)
        else:
            return ExperienceModel()

    def _load_alternatives_model(self) -> AlternativesModel:
        """Load alternatives recommendation model."""
        model_path = self.models_dir / "alternatives"
        if model_path.exists():
            return AlternativesModel(model_path)
        else:
            return AlternativesModel()

    def extract_technologies(
        self,
        text: str,
        is_resume: bool = False,
        prompt_override: str = "",
    ) -> Dict[str, Any]:
        """Extract technologies from text using ML models.

        This method matches the API of SimpleTechExtractor for backward compatibility.

        Args:
            text: Job description, prompt, or resume text
            is_resume: If True, treats input as resume
            prompt_override: Additional prompt context to merge with resume

        Returns:
            {
                "technologies": {
                    "react": {
                        "difficulty": 5.2,
                        "category": "frontend",
                        "alternatives": {"vue": {"difficulty": 4.8}, ...},
                        "experience_mentioned_in_prompt": 5.0,
                        "experience_accounted_for_in_resume": 3.0,
                        "experience_validated_via_github": None
                    }
                }
            }
        """
        # Combine text sources
        combined_text = text
        if prompt_override:
            combined_text = f"{text}\n\n{prompt_override}"

        # Step 1: Extract technologies using NER or patterns
        extracted_techs = self.tech_extractor.extract(combined_text)

        # Extract overall experience from prompt if provided
        overall_exp_from_prompt = None
        if prompt_override:
            overall_exp_from_prompt = self.experience_model.extract_overall_experience(prompt_override)

        # Step 2: Build technology objects
        technologies = {}

        for tech in extracted_techs:
            tech_name = tech["name"]
            category = tech["category"]

            # Predict difficulty using ML
            difficulty = self.difficulty_model.predict(tech_name)

            # Get alternatives using embeddings
            alternatives_list = self.alternatives_model.recommend(tech_name, top_k=4)

            # Build alternatives dict with difficulty scores
            alternatives = {}
            for alt_name, similarity in alternatives_list:
                alt_difficulty = self.difficulty_model.predict(alt_name)
                alternatives[alt_name] = {"difficulty": alt_difficulty}

            # Build tech entry
            tech_entry = {
                "difficulty": difficulty,
                "category": category,
                "alternatives": alternatives,
                "experience_validated_via_github": None,  # Placeholder
            }

            # Extract experience from prompt (if provided)
            if prompt_override:
                # Check tech-specific experience first
                prompt_exp = self.experience_model.extract_tech_experience(prompt_override, tech_name)
                # Fall back to overall experience if tech-specific not found
                if prompt_exp is None and overall_exp_from_prompt is not None:
                    prompt_exp = overall_exp_from_prompt

                if prompt_exp is not None:
                    tech_entry["experience_mentioned_in_prompt"] = prompt_exp

            # Extract experience from resume (if is_resume=True)
            if is_resume:
                resume_exp = self.experience_model.extract_tech_experience(text, tech_name)
                if resume_exp is not None:
                    tech_entry["experience_accounted_for_in_resume"] = resume_exp

            technologies[tech_name] = tech_entry

        return {"technologies": technologies}


# Factory function for backward compatibility
def create_tech_extractor(use_ml: bool = True) -> MLTechExtractor:
    """Create a technology extractor.

    Args:
        use_ml: If True, use ML models. If False, fall back to patterns.

    Returns:
        MLTechExtractor instance
    """
    if not use_ml:
        # If ML is disabled, just use pattern-based models
        from mcp_server.simple_tech_extractor import SimpleTechExtractor

        return SimpleTechExtractor()

    return MLTechExtractor()
