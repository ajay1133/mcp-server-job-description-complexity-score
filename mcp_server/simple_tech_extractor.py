#!/usr/bin/env python3
"""
Simplified Technology Extractor
Extracts required technologies from job prompts and provides alternatives with difficulty ratings.
"""

from typing import Dict, Any


class SimpleTechExtractor:
    """Extracts technologies from text and provides difficulty ratings and alternatives."""

    def __init__(self):
        # Technology database with difficulty (1-10), experience required (years), and alternatives
        self.tech_db = {
            # Frontend
            "react": {
                "difficulty": 5.2,
                "experience_required": 2.5,
                "category": "frontend",
                "alternatives": ["vue", "angular", "svelte", "nextjs"]
            },
            "vue": {
                "difficulty": 4.8,
                "experience_required": 2.0,
                "category": "frontend",
                "alternatives": ["react", "angular", "svelte"]
            },
            "angular": {
                "difficulty": 6.5,
                "experience_required": 3.0,
                "category": "frontend",
                "alternatives": ["react", "vue", "svelte"]
            },
            "nextjs": {
                "difficulty": 5.5,
                "experience_required": 2.8,
                "category": "frontend",
                "alternatives": ["react", "gatsby", "nuxt"]
            },
            "svelte": {
                "difficulty": 4.5,
                "experience_required": 1.8,
                "category": "frontend",
                "alternatives": ["react", "vue", "angular"]
            },
            "typescript": {
                "difficulty": 5.8,
                "experience_required": 2.2,
                "category": "frontend",
                "alternatives": ["javascript", "flow"]
            },

            # Backend
            "node": {
                "difficulty": 5.0,
                "experience_required": 2.5,
                "category": "backend",
                "alternatives": ["python_fastapi", "golang", "java_spring"]
            },
            "python_fastapi": {
                "difficulty": 4.5,
                "experience_required": 2.0,
                "category": "backend",
                "alternatives": ["flask", "python_django", "node"]
            },
            "flask": {
                "difficulty": 4.2,
                "experience_required": 1.8,
                "category": "backend",
                "alternatives": ["python_fastapi", "python_django", "node"]
            },
            "python_django": {
                "difficulty": 5.8,
                "experience_required": 2.8,
                "category": "backend",
                "alternatives": ["flask", "python_fastapi", "ruby_rails"]
            },
            "golang": {
                "difficulty": 6.2,
                "experience_required": 3.0,
                "category": "backend",
                "alternatives": ["node", "python_fastapi", "java_spring"]
            },
            "java_spring": {
                "difficulty": 7.0,
                "experience_required": 3.5,
                "category": "backend",
                "alternatives": ["golang", "python_django", "node"]
            },
            "ruby_rails": {
                "difficulty": 5.5,
                "experience_required": 2.8,
                "category": "backend",
                "alternatives": ["python_django", "node", "php"]
            },

            # Database
            "postgres": {
                "difficulty": 5.5,
                "experience_required": 2.5,
                "category": "database",
                "alternatives": ["mysql", "mariadb", "mongodb"]
            },
            "mysql": {
                "difficulty": 5.0,
                "experience_required": 2.3,
                "category": "database",
                "alternatives": ["postgres", "mariadb", "mongodb"]
            },
            "mongodb": {
                "difficulty": 4.8,
                "experience_required": 2.0,
                "category": "database",
                "alternatives": ["postgres", "mysql", "dynamodb"]
            },
            "redis": {
                "difficulty": 4.5,
                "experience_required": 1.8,
                "category": "cache",
                "alternatives": ["memcached", "elasticsearch"]
            },
            "dynamodb": {
                "difficulty": 5.8,
                "experience_required": 2.5,
                "category": "database",
                "alternatives": ["mongodb", "cassandra", "postgres"]
            },
            "cassandra": {
                "difficulty": 7.2,
                "experience_required": 3.5,
                "category": "database",
                "alternatives": ["dynamodb", "mongodb", "postgres"]
            },

            # Infrastructure
            "docker": {
                "difficulty": 5.5,
                "experience_required": 2.0,
                "category": "infrastructure",
                "alternatives": ["kubernetes", "podman"]
            },
            "kubernetes": {
                "difficulty": 7.5,
                "experience_required": 3.5,
                "category": "infrastructure",
                "alternatives": ["docker", "nomad", "ecs"]
            },
            "aws": {
                "difficulty": 6.5,
                "experience_required": 3.0,
                "category": "cloud",
                "alternatives": ["gcp", "azure", "digitalocean"]
            },
            "aws_lambda": {
                "difficulty": 5.8,
                "experience_required": 2.5,
                "category": "serverless",
                "alternatives": ["azure_functions", "google_cloud_functions"]
            },

            # Message Queue
            "kafka": {
                "difficulty": 7.0,
                "experience_required": 3.2,
                "category": "messaging",
                "alternatives": ["rabbitmq", "sqs", "redis"]
            },
            "rabbitmq": {
                "difficulty": 6.2,
                "experience_required": 2.8,
                "category": "messaging",
                "alternatives": ["kafka", "sqs", "redis"]
            },

            # Search
            "elasticsearch": {
                "difficulty": 6.5,
                "experience_required": 2.8,
                "category": "search",
                "alternatives": ["solr", "algolia", "meilisearch"]
            }
        }

        # Keywords to detect technologies
        self.tech_keywords = {
            "react": ["react", "react.js", "reactjs"],
            "vue": ["vue", "vue.js", "vuejs"],
            "angular": ["angular"],
            "nextjs": ["nextjs", "next.js", "next js"],
            "svelte": ["svelte", "sveltekit"],
            "typescript": ["typescript", "ts"],
            "node": ["node", "node.js", "nodejs", "express", "expressjs"],
            "python_fastapi": ["fastapi", "fast api"],
            "flask": ["flask"],
            "python_django": ["django"],
            "golang": ["golang", " go ", "go language"],
            "java_spring": ["spring", "spring boot", "springboot"],
            "ruby_rails": ["ruby on rails", "rails", "ror"],
            "postgres": ["postgres", "postgresql"],
            "mysql": ["mysql"],
            "mongodb": ["mongodb", "mongo"],
            "redis": ["redis"],
            "dynamodb": ["dynamodb"],
            "cassandra": ["cassandra"],
            "docker": ["docker"],
            "kubernetes": ["kubernetes", "k8s"],
            "aws": ["aws", "amazon web services"],
            "aws_lambda": ["lambda", "aws lambda"],
            "kafka": ["kafka"],
            "rabbitmq": ["rabbitmq", "rabbit mq"],
            "elasticsearch": ["elasticsearch", "elastic search"]
        }

    def _extract_experience(self, text: str, tech_name: str) -> float | None:
        """Extract explicit experience mentions for a technology (e.g., '5+ years React')."""
        import re
        text_lower = text.lower()

        # Pattern: "X+ years [of] <tech>" or "<tech> X+ years"
        patterns = [
            rf'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?{tech_name}',
            rf'{tech_name}.*?(\d+)\+?\s*(?:years?|yrs?)',
            rf'(\d+)\+?\s*(?:years?|yrs?).*?{tech_name}'
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))

        return None

    def _extract_overall_experience(self, text: str) -> float | None:
        """Extract overall experience from prompt (e.g., '5 years experience', '3+ years')."""
        import re
        text_lower = text.lower()

        # Patterns for overall experience
        patterns = [
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:overall\s+)?experience',
            r'(?:overall\s+)?experience[:\s]+(\d+)\+?\s*(?:years?|yrs?)',
            r'(\d+)\+?\s*(?:years?|yrs?)\s+(?:in\s+)?(?:the\s+)?(?:field|industry)'
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1))

        return None

    def extract_technologies(
        self,
        text: str,
        is_resume: bool = False,
        prompt_override: str = ""
    ) -> Dict[str, Any]:
        """
        Extract technologies from text and return with difficulty and alternatives.

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
                        "alternatives": {...},
                        "experience_mentioned_in_prompt": 5.0,  # if found in prompt
                        "experience_accounted_for_in_resume": 3.0,  # if found in resume
                        "experience_validated_via_github": null  # placeholder for future
                    }
                }
            }
        """
        # Combine text sources if both provided
        combined_text = text
        if prompt_override:
            combined_text = f"{text}\n\n{prompt_override}"

        text_lower = combined_text.lower()
        detected = {}

        # Extract overall experience from prompt if provided
        overall_exp_from_prompt = None
        if prompt_override:
            overall_exp_from_prompt = self._extract_overall_experience(prompt_override)

        # Detect technologies mentioned in text
        for tech_id, keywords in self.tech_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tech_info = self.tech_db.get(tech_id, {})

                # Build alternatives dict (without experience_required)
                alternatives = {}
                for alt_id in tech_info.get("alternatives", []):
                    alt_info = self.tech_db.get(alt_id, {})
                    if alt_info:
                        alternatives[alt_id] = {
                            "difficulty": alt_info.get("difficulty", 5.0)
                        }

                tech_entry = {
                    "difficulty": tech_info.get("difficulty", 5.0),
                    "category": tech_info.get("category", "other"),
                    "alternatives": alternatives,
                    "experience_validated_via_github": None  # Placeholder for future implementation
                }

                # Extract experience from prompt (if provided)
                prompt_exp = None
                if prompt_override:
                    # Check tech-specific experience first
                    prompt_exp = self._extract_experience(prompt_override, tech_id)
                    # Fall back to overall experience if tech-specific not found
                    if prompt_exp is None and overall_exp_from_prompt is not None:
                        prompt_exp = overall_exp_from_prompt

                if prompt_exp is not None:
                    tech_entry["experience_mentioned_in_prompt"] = prompt_exp

                # Extract experience from resume text (if is_resume=True)
                if is_resume:
                    resume_exp = self._extract_experience(text, tech_id)
                    if resume_exp is not None:
                        tech_entry["experience_accounted_for_in_resume"] = resume_exp

                detected[tech_id] = tech_entry

        return {"technologies": detected}


def main():
    """Test the extractor with sample prompts."""
    extractor = SimpleTechExtractor()

    test_prompts = [
        "Senior Full-Stack Engineer with 5+ years React and Node.js experience",
        "Backend developer with FastAPI, PostgreSQL, and Redis",
        "DevOps engineer with 3 years Kubernetes, Docker, and AWS",
        "Looking for React, TypeScript, and MongoDB expert"
    ]

    for prompt in test_prompts:
        print(f"\nPrompt: {prompt}")
        result = extractor.extract_technologies(prompt)
        print(f"Detected: {list(result['technologies'].keys())}")
        for tech, info in result['technologies'].items():
            exp_str = f", exp={info['experience_required']}y" if 'experience_required' in info else ""
            print(f"  {tech}: difficulty={info['difficulty']}/10{exp_str}, alternatives={len(info['alternatives'])}")


if __name__ == "__main__":
    main()
