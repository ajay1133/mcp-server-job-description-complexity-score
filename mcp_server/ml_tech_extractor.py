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
from mcp_server.ml_models import AlternativesModel, DifficultyModel, ExperienceModel, TechExtractorModel

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

        # Role-based defaults for fail-safe behavior (when no explicit techs found)
        # Mirrors SimpleTechExtractor.role_defaults to keep behavior consistent
        self.role_defaults = {
            "fullstack": ["react", "node", "postgres", "docker"],
            "full stack": ["react", "node", "postgres", "docker"],
            "full-stack": ["react", "node", "postgres", "docker"],
            "frontend": ["react", "typescript"],
            "front-end": ["react", "typescript"],
            "backend": ["node", "python_fastapi", "postgres"],
            "back-end": ["node", "python_fastapi", "postgres"],
            "devops": ["docker", "kubernetes", "aws"],
            "site reliability": ["docker", "kubernetes", "aws"],
            "sre": ["docker", "kubernetes", "aws"],
            "data engineer": ["python_django", "postgres", "redis"],
        }

        # Minimal category mapping for known technologies
        self.category_map = {
            "react": "frontend",
            "vue": "frontend",
            "angular": "frontend",
            "nextjs": "frontend",
            "svelte": "frontend",
            "typescript": "frontend",
            "node": "backend",
            "python_fastapi": "backend",
            "flask": "backend",
            "python_django": "backend",
            "golang": "backend",
            "java_spring": "backend",
            "ruby_rails": "backend",
            "postgres": "database",
            "mysql": "database",
            "mongodb": "database",
            "redis": "cache",
            "dynamodb": "database",
            "cassandra": "database",
            "docker": "infrastructure",
            "kubernetes": "infrastructure",
            "aws": "cloud",
            "kafka": "messaging",
            "rabbitmq": "messaging",
            "elasticsearch": "search",
        }

        # Role keywords for semantic matching (maps role category → representative keywords)
        self.role_keywords = {
            "fullstack": ["fullstack", "full stack", "full-stack", "polyglot", "generalist", "product engineer"],
            "frontend": ["frontend", "front-end", "front end", "ui", "ux", "web developer"],
            "backend": ["backend", "back-end", "back end", "server", "api"],
            "devops": ["devops", "site reliability", "sre", "platform engineer", "infrastructure", "cloud engineer"],
            "data": ["data engineer", "data scientist", "ml engineer", "analytics"],
        }

        # Project descriptions mapped to representative tech stacks (same as SimpleTechExtractor)
        self.project_defaults = {
            "amazon": ["react", "node", "postgres", "redis", "aws", "docker", "elasticsearch"],
            "ebay": ["react", "node", "postgres", "redis", "aws", "docker"],
            "shopify": ["react", "node", "postgres", "redis", "docker"],
            "twitter": ["react", "node", "postgres", "redis", "kafka", "docker"],
            "facebook": ["react", "node", "postgres", "redis", "docker", "kafka"],
            "instagram": ["react", "node", "postgres", "redis", "docker"],
            "netflix": ["react", "node", "postgres", "redis", "kafka", "aws", "docker"],
            "youtube": ["react", "node", "postgres", "redis", "kafka", "docker"],
            "spotify": ["react", "node", "postgres", "redis", "kafka", "docker"],
            "slack": ["react", "node", "postgres", "redis", "docker"],
            "uber": ["react", "node", "postgres", "redis", "docker", "kafka", "aws"],
            "airbnb": ["react", "node", "postgres", "redis", "elasticsearch", "docker"],
            "medium": ["react", "node", "postgres", "redis", "docker"],
            "reddit": ["react", "node", "postgres", "redis", "docker"],
        }

        print(f"[MLTechExtractor] Initialized with models from {self.models_dir}")

    def _extract_seniority_level(self, text: str) -> str | None:
        """Extract seniority level from text (senior/mid/junior)."""
        text_lower = text.lower()
        if any(k in text_lower for k in ["principal", "staff", "lead", "senior"]):
            return "senior"
        if any(k in text_lower for k in ["mid", "intermediate", "mid-level", "mid level"]):
            return "mid"
        if any(k in text_lower for k in ["junior", "entry", "associate", "entry-level", "entry level"]):
            return "junior"
        return None

    def _seniority_to_experience_string(self, seniority: str) -> str:
        """Convert seniority level to experience string format."""
        seniority_map = {
            "senior": ">= 5 years",
            "mid": ">= 3 years",
            "junior": "<= 2 years",
        }
        return seniority_map.get(seniority, seniority)

    def _get_experience_mentioned_value(
        self,
        years_resume: float | None,
        years_prompt: float | None,
        seniority_resume: str | None,
        seniority_prompt: str | None,
        overall_years_resume: float | None,
        overall_years_prompt: float | None,
    ) -> str | float | None:
        """Get experience_mentioned value based on priority rules.

        Returns number for explicit years, string for seniority-derived ranges, or None
        """
        if years_resume is not None:
            return int(years_resume) if years_resume == int(years_resume) else years_resume
        if seniority_resume:
            return self._seniority_to_experience_string(seniority_resume)
        if overall_years_resume is not None:
            return (
                int(overall_years_resume) if overall_years_resume == int(overall_years_resume) else overall_years_resume
            )
        if years_prompt is not None:
            return int(years_prompt) if years_prompt == int(years_prompt) else years_prompt
        if seniority_prompt:
            return self._seniority_to_experience_string(seniority_prompt)
        if overall_years_prompt is not None:
            return (
                int(overall_years_prompt) if overall_years_prompt == int(overall_years_prompt) else overall_years_prompt
            )
        return None

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

    def _find_best_role_match(self, text_lower: str) -> str | None:
        """Use semantic matching to find the best role category from text.

        Args:
            text_lower: Lowercased text to search

        Returns:
            Best matching role category key (e.g., 'fullstack', 'frontend') or None
        """
        best_role = None
        best_score = 0

        for role_category, keywords in self.role_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > best_score:
                best_score = score
                best_role = role_category

        return best_role if best_score > 0 else None

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

        # Extract resume/prompt seniority and overall resume years for later propagation
        overall_exp_from_resume = self.experience_model.extract_overall_experience(text) if is_resume else None
        seniority_from_prompt = self._extract_seniority_level(prompt_override) if prompt_override else None
        seniority_from_resume = self._extract_seniority_level(text) if is_resume else None

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
            }

            # Add experience_validated_via_github only if resume is provided
            if is_resume:
                tech_entry["experience_validated_via_github"] = None

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

            # Calculate experience_mentioned value
            years_resume = self.experience_model.extract_tech_experience(text, tech_name) if is_resume else None
            years_prompt = tech_entry.get("experience_mentioned_in_prompt")
            exp_mentioned = self._get_experience_mentioned_value(
                years_resume,
                years_prompt,
                seniority_from_resume,
                seniority_from_prompt,
                overall_exp_from_resume,
                overall_exp_from_prompt,
            )
            if exp_mentioned is not None:
                tech_entry["experience_mentioned"] = exp_mentioned

            # Propagate global estimate to alternatives
            alt_global_exp = self._get_experience_mentioned_value(
                None,
                None,
                seniority_from_resume,
                seniority_from_prompt,
                overall_exp_from_resume,
                overall_exp_from_prompt,
            )
            if alt_global_exp is not None:
                for alt_id in tech_entry["alternatives"]:
                    tech_entry["alternatives"][alt_id]["experience_mentioned"] = alt_global_exp

            technologies[tech_name] = tech_entry

        # Fail-safe role defaults: if nothing detected, infer from generic role phrases
        if not technologies:
            text_lower = combined_text.lower()

            # First try exact phrase matching
            matched_role = None
            for phrase, tech_list in self.role_defaults.items():
                if phrase in text_lower:
                    matched_role = phrase
                    break

            # If no exact match, use semantic matching
            if not matched_role:
                best_role_category = self._find_best_role_match(text_lower)
                if best_role_category:
                    # Map category back to a default key
                    role_map = {
                        "fullstack": "full stack",
                        "frontend": "frontend",
                        "backend": "backend",
                        "devops": "devops",
                        "data": "data engineer",
                    }
                    matched_role = role_map.get(best_role_category)

            # Inject technologies for matched role
            if matched_role and matched_role in self.role_defaults:
                tech_list = self.role_defaults[matched_role]
                for tech_name in tech_list:
                    # Predict difficulty
                    difficulty = self.difficulty_model.predict(tech_name)
                    # Alternatives
                    alternatives_list = self.alternatives_model.recommend(tech_name, top_k=4)
                    alternatives = {
                        alt: {"difficulty": self.difficulty_model.predict(alt)} for alt, _ in alternatives_list
                    }
                    # Category
                    category = self.category_map.get(tech_name, "other")
                    # Build entry
                    tech_entry = {
                        "difficulty": difficulty,
                        "category": category,
                        "alternatives": alternatives,
                    }

                    # Add experience_validated_via_github only if resume is provided
                    if is_resume:
                        tech_entry["experience_validated_via_github"] = None

                    # Apply global experience estimate
                    exp_mentioned = self._get_experience_mentioned_value(
                        None,
                        None,
                        seniority_from_resume,
                        seniority_from_prompt,
                        overall_exp_from_resume,
                        overall_exp_from_prompt,
                    )
                    if exp_mentioned is not None:
                        tech_entry["experience_mentioned"] = exp_mentioned
                        for alt_id in tech_entry["alternatives"]:
                            tech_entry["alternatives"][alt_id]["experience_mentioned"] = exp_mentioned

                    technologies[tech_name] = tech_entry

        # If still nothing, check for project descriptions (e.g., "amazon clone")
        if not technologies:
            text_lower = combined_text.lower()
            for project_name, tech_list in self.project_defaults.items():
                # Match patterns: "X clone", "X copy", "like X", etc.
                patterns = [
                    f"{project_name} clone",
                    f"{project_name} copy",
                    f"clone of {project_name}",
                    f"copy of {project_name}",
                    f"like {project_name}",
                    f"{project_name} like",  # reverse order
                    f"similar to {project_name}",
                    f"build {project_name}",
                    f"{project_name} site",
                    f"{project_name}-like",
                    f"{project_name} for",  # "uber for X"
                    f"build a {project_name}",
                ]
                if any(pattern in text_lower for pattern in patterns):
                    for tech_name in tech_list:
                        # Predict difficulty
                        difficulty = self.difficulty_model.predict(tech_name)
                        # Alternatives
                        alternatives_list = self.alternatives_model.recommend(tech_name, top_k=4)
                        alternatives = {
                            alt: {"difficulty": self.difficulty_model.predict(alt)} for alt, _ in alternatives_list
                        }
                        # Category
                        category = self.category_map.get(tech_name, "other")
                        # Build entry
                        tech_entry = {
                            "difficulty": difficulty,
                            "category": category,
                            "alternatives": alternatives,
                        }

                        # Add experience_validated_via_github only if resume is provided
                        if is_resume:
                            tech_entry["experience_validated_via_github"] = None

                        # Apply global experience estimate
                        exp_mentioned = self._get_experience_mentioned_value(
                            None,
                            None,
                            seniority_from_resume,
                            seniority_from_prompt,
                            overall_exp_from_resume,
                            overall_exp_from_prompt,
                        )
                        if exp_mentioned is not None:
                            tech_entry["experience_mentioned"] = exp_mentioned
                            for alt_id in tech_entry["alternatives"]:
                                tech_entry["alternatives"][alt_id]["experience_mentioned"] = exp_mentioned

                        technologies[tech_name] = tech_entry
                    # Apply first matching project defaults only
                    break

        # Generic clone/copy fallback for unknown projects (non-hardcoded brands)
        if not technologies:
            import re

            clone_like_patterns = [r"\bclone\b", r"\bcopy\b", r"\blike\b", r"\bsimilar to\b"]
            if any(re.search(p, text_lower) for p in clone_like_patterns):
                # Generic modern web baseline
                generic_stack = ["react", "node", "postgres", "redis", "docker"]
                # Optional enrichment
                if any(k in text_lower for k in ["search", "semantic", "index"]):
                    generic_stack.append("elasticsearch")
                if any(k in text_lower for k in ["cloud", "aws", "azure", "gcp"]):
                    generic_stack.append("aws")
                if any(k in text_lower for k in ["serverless", "lambda"]):
                    generic_stack.append("aws_lambda")

                for tech_name in generic_stack:
                    difficulty = self.difficulty_model.predict(tech_name)
                    alternatives_list = self.alternatives_model.recommend(tech_name, top_k=4)
                    alternatives = {
                        alt: {"difficulty": self.difficulty_model.predict(alt)} for alt, _ in alternatives_list
                    }
                    category = self.category_map.get(tech_name, "other")
                    tech_entry = {
                        "difficulty": difficulty,
                        "category": category,
                        "alternatives": alternatives,
                    }

                    # Add experience_validated_via_github only if resume is provided
                    if is_resume:
                        tech_entry["experience_validated_via_github"] = None

                    technologies[tech_name] = tech_entry

                # Apply global experience estimate if present
                exp_mentioned = self._get_experience_mentioned_value(
                    None,
                    None,
                    seniority_from_resume,
                    seniority_from_prompt,
                    overall_exp_from_resume,
                    overall_exp_from_prompt,
                )
                if exp_mentioned is not None:
                    for tech_name in technologies:
                        technologies[tech_name]["experience_mentioned"] = exp_mentioned
                        for alt_id in technologies[tech_name]["alternatives"]:
                            technologies[tech_name]["alternatives"][alt_id]["experience_mentioned"] = exp_mentioned

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
