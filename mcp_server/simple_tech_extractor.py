#!/usr/bin/env python3
"""
Simplified Technology Extractor
Extracts required technologies from job prompts and provides alternatives with difficulty ratings.
"""

from typing import Any, Dict

from mcp_server.tech_registry import TechRegistry


class SimpleTechExtractor:
    """Extracts technologies from text and provides difficulty ratings and alternatives."""

    def __init__(self):
        # Initialize tech registry for discovering unknown technologies
        self.tech_registry = TechRegistry()

        # Technology database with difficulty (1-10), experience required (years), and alternatives
        self.tech_db = {
            # Frontend
            "react": {
                "difficulty": 5.2,
                "experience_required": 2.5,
                "category": "frontend",
                "alternatives": ["vue", "angular", "svelte", "nextjs"],
            },
            "vue": {
                "difficulty": 4.8,
                "experience_required": 2.0,
                "category": "frontend",
                "alternatives": ["react", "angular", "svelte"],
            },
            "angular": {
                "difficulty": 6.5,
                "experience_required": 3.0,
                "category": "frontend",
                "alternatives": ["react", "vue", "svelte"],
            },
            "nextjs": {
                "difficulty": 5.5,
                "experience_required": 2.8,
                "category": "frontend",
                "alternatives": ["react", "gatsby", "nuxt"],
            },
            "svelte": {
                "difficulty": 4.5,
                "experience_required": 1.8,
                "category": "frontend",
                "alternatives": ["react", "vue", "angular"],
            },
            "typescript": {
                "difficulty": 5.8,
                "experience_required": 2.2,
                "category": "frontend",
                "alternatives": ["javascript", "flow"],
            },
            # Backend / Languages
            "python": {
                "difficulty": 4.5,
                "experience_required": 2.0,
                "category": "backend",
                "alternatives": ["java_spring", "golang", "node"],
            },
            "dotnet": {
                "difficulty": 6.5,
                "experience_required": 3.0,
                "category": "backend",
                "alternatives": ["java_spring", "golang", "node"],
            },
            "node": {
                "difficulty": 5.0,
                "experience_required": 2.5,
                "category": "backend",
                "alternatives": ["python_fastapi", "golang", "java_spring"],
            },
            "python_fastapi": {
                "difficulty": 4.5,
                "experience_required": 2.0,
                "category": "backend",
                "alternatives": ["flask", "python_django", "node"],
            },
            "flask": {
                "difficulty": 4.2,
                "experience_required": 1.8,
                "category": "backend",
                "alternatives": ["python_fastapi", "python_django", "node"],
            },
            "python_django": {
                "difficulty": 5.8,
                "experience_required": 2.8,
                "category": "backend",
                "alternatives": ["flask", "python_fastapi", "ruby_rails"],
            },
            "golang": {
                "difficulty": 6.2,
                "experience_required": 3.0,
                "category": "backend",
                "alternatives": ["node", "python_fastapi", "java_spring"],
            },
            "java_spring": {
                "difficulty": 7.0,
                "experience_required": 3.5,
                "category": "backend",
                "alternatives": ["golang", "python_django", "node"],
            },
            "ruby_rails": {
                "difficulty": 5.5,
                "experience_required": 2.8,
                "category": "backend",
                "alternatives": ["python_django", "node", "php"],
            },
            # Database
            "postgres": {
                "difficulty": 5.5,
                "experience_required": 2.5,
                "category": "database",
                "alternatives": ["mysql", "mariadb", "mongodb"],
            },
            "mysql": {
                "difficulty": 5.0,
                "experience_required": 2.3,
                "category": "database",
                "alternatives": ["postgres", "mariadb", "mongodb"],
            },
            "mongodb": {
                "difficulty": 4.8,
                "experience_required": 2.0,
                "category": "database",
                "alternatives": ["postgres", "mysql", "dynamodb"],
            },
            "redis": {
                "difficulty": 4.5,
                "experience_required": 1.8,
                "category": "cache",
                "alternatives": ["memcached", "elasticsearch"],
            },
            "dynamodb": {
                "difficulty": 5.8,
                "experience_required": 2.5,
                "category": "database",
                "alternatives": ["mongodb", "cassandra", "postgres"],
            },
            "cassandra": {
                "difficulty": 7.2,
                "experience_required": 3.5,
                "category": "database",
                "alternatives": ["dynamodb", "mongodb", "postgres"],
            },
            # Infrastructure
            "docker": {
                "difficulty": 5.5,
                "experience_required": 2.0,
                "category": "devops",
                "alternatives": ["podman", "containerd"],
            },
            "kubernetes": {
                "difficulty": 7.5,
                "experience_required": 3.5,
                "category": "devops",
                "alternatives": ["docker_swarm", "nomad"],
            },
            "aws": {
                "difficulty": 6.8,
                "experience_required": 3.2,
                "category": "cloud",
                "alternatives": ["azure", "gcp"],
            },
            "gcp": {
                "difficulty": 6.7,
                "experience_required": 3.0,
                "category": "cloud",
                "alternatives": ["aws", "azure"],
            },
            "azure": {
                "difficulty": 6.8,
                "experience_required": 3.1,
                "category": "cloud",
                "alternatives": ["aws", "gcp"],
            },
            "aws_lambda": {
                "difficulty": 5.8,
                "experience_required": 2.5,
                "category": "serverless",
                "alternatives": ["azure_functions", "google_cloud_functions"],
            },
            # Message Queue
            "kafka": {
                "difficulty": 7.0,
                "experience_required": 3.2,
                "category": "messaging",
                "alternatives": ["rabbitmq", "sqs", "redis"],
            },
            "rabbitmq": {
                "difficulty": 6.2,
                "experience_required": 2.8,
                "category": "messaging",
                "alternatives": ["kafka", "sqs", "redis"],
            },
            # Search
            "elasticsearch": {
                "difficulty": 6.5,
                "experience_required": 2.8,
                "category": "search",
                "alternatives": ["solr", "algolia", "meilisearch"],
            },
        }

        # Keywords to detect technologies
        self.tech_keywords = {
            "react": ["react", "react.js", "reactjs"],
            "vue": ["vue", "vue.js", "vuejs"],
            "angular": ["angular"],
            "nextjs": ["nextjs", "next.js", "next js"],
            "svelte": ["svelte", "sveltekit"],
            "typescript": ["typescript", "ts"],
            "python": ["python", " py "],
            "dotnet": [".net", "dotnet", "dot net", ".net core", "asp.net"],
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
            "gcp": ["gcp", "google cloud", "google cloud platform"],
            "azure": ["azure", "microsoft azure"],
            "aws_lambda": ["lambda", "aws lambda"],
            "kafka": ["kafka"],
            "rabbitmq": ["rabbitmq", "rabbit mq"],
            "elasticsearch": ["elasticsearch", "elastic search"],
        }

        # Generic role phrases mapped to representative baseline stacks
        # Used as a fallback when no explicit technologies are found
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

        # Project descriptions mapped to representative tech stacks
        # Used when users describe a project to build (e.g., "clone of X")
        self.project_defaults = {
            # E-commerce platforms
            "amazon": ["react", "node", "postgres", "redis", "aws", "docker", "elasticsearch"],
            "ebay": ["react", "node", "postgres", "redis", "aws", "docker"],
            "shopify": ["react", "node", "postgres", "redis", "docker"],
            "etsy": ["react", "node", "postgres", "redis", "docker"],
            # Social media
            "twitter": ["react", "node", "postgres", "redis", "kafka", "docker"],
            "facebook": ["react", "node", "postgres", "redis", "docker", "kafka"],
            "instagram": ["react", "node", "postgres", "redis", "docker"],
            "linkedin": ["react", "node", "postgres", "redis", "elasticsearch", "docker"],
            # Streaming/media
            "netflix": ["react", "node", "postgres", "redis", "kafka", "aws", "docker"],
            "youtube": ["react", "node", "postgres", "redis", "kafka", "docker"],
            "spotify": ["react", "node", "postgres", "redis", "kafka", "docker"],
            # Productivity
            "slack": ["react", "node", "postgres", "redis", "docker"],
            "discord": ["react", "node", "postgres", "redis", "docker"],
            "notion": ["react", "node", "postgres", "redis", "docker"],
            "trello": ["react", "node", "postgres", "redis", "docker"],
            # Food delivery
            "uber eats": ["react", "node", "postgres", "redis", "docker", "aws"],
            "doordash": ["react", "node", "postgres", "redis", "docker"],
            "grubhub": ["react", "node", "postgres", "redis", "docker"],
            # Ride sharing
            "uber": ["react", "node", "postgres", "redis", "docker", "kafka", "aws"],
            "lyft": ["react", "node", "postgres", "redis", "docker", "aws"],
            # Booking/travel
            "airbnb": ["react", "node", "postgres", "redis", "elasticsearch", "docker"],
            "booking": ["react", "node", "postgres", "redis", "docker"],
            # Content platforms
            "medium": ["react", "node", "postgres", "redis", "docker"],
            "reddit": ["react", "node", "postgres", "redis", "docker"],
            "quora": ["react", "node", "postgres", "elasticsearch", "docker"],
        }

    def _extract_experience(self, text: str, tech_name: str) -> float | None:
        """Extract explicit experience mentions for a technology (e.g., '5+ years React', '2 years in react').

        Only matches if years and tech are close together (within ~20 chars) to avoid false positives.
        """
        import re

        text_lower = text.lower()

        # Pattern order matters! Most specific first to avoid greedy matches
        # Using word boundaries and limiting distance to avoid false matches
        patterns = [
            # Most specific: "N years in/of/with <tech>" (handles "2 years in react")
            rf'(\d+)(\+)?\s*(?:years?|yrs?)\s+(?:in|of|with)\s+{re.escape(tech_name)}\b',
            # "N years <tech>" (no preposition)
            rf'(\d+)(\+)?\s*(?:years?|yrs?)\s+{re.escape(tech_name)}\b',
            # "<tech> N years" (reverse order, but limited to nearby context)
            rf'{re.escape(tech_name)}\b\s+(?:\w+\s+){{0,3}}(\d+)(\+)?\s*(?:years?|yrs?)',
            # "at least N years <tech>"
            rf'(?:at\s+least|min(?:imum)?)\s+(\d+)\s*(?:years?|yrs?)\s+(?:\w+\s+){{0,2}}{re.escape(tech_name)}\b',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = float(match.group(1))
                plus = bool(match.group(2)) if match.lastindex >= 2 else False
                plus = plus or ('at least' in pattern or 'min' in pattern)
                return -years if plus else years

        return None

    def _extract_overall_experience(self, text: str) -> float | None:
        """Extract overall experience from prompt (e.g., '5 years experience', '3+ years')."""
        import re

        text_lower = text.lower()

        # Patterns for overall experience (capture optional '+')
        patterns = [
            r'(\d+)(\+)?\s*(?:years?|yrs?)\s+(?:of\s+)?(?:overall\s+)?experience',
            r'(?:overall\s+)?experience[:\s]+(\d+)(\+)?\s*(?:years?|yrs?)',
            r'(\d+)(\+)?\s*(?:years?|yrs?)\s+(?:in\s+)?(?:the\s+)?(?:field|industry)',
            r'with\s+(\d+)(\+)?\s*(?:years?|yrs?)',  # "with 8+ years"
            r'(?:at\s+least|min(?:imum)?\s+)(\d+)\s*(?:years?|yrs?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                years = float(match.group(1))
                plus = bool(match.group(2)) if match.lastindex and match.lastindex >= 2 else False
                return -years if plus else years

        return None

    def _extract_seniority_level(self, text: str) -> str | None:
        """Extract seniority level from text (e.g., 'senior', 'junior', 'mid')."""
        text_lower = text.lower()
        # Priority: senior > mid > junior if multiple appear, choose strongest
        if any(k in text_lower for k in ["principal", "staff", "lead", "senior"]):
            return "senior"
        if any(k in text_lower for k in ["mid", "intermediate", "mid-level", "mid level"]):
            return "mid"
        if any(k in text_lower for k in ["junior", "entry", "associate", "entry-level", "entry level"]):
            return "junior"
        return None

    def _seniority_to_experience_string(self, seniority: str) -> str:
        """Deprecated: no longer converts seniority to years. Returns input unchanged."""
        return seniority

    def _format_year_value(self, val: float | None) -> str | float | None:
        """Format numeric years value.

        Negative values indicate a minimum (e.g., -5 => '>= 5 years').
        Positive values return as int when whole, else float.
        """
        if val is None:
            return None
        if val < 0:
            years = abs(val)
            years = int(years) if years == int(years) else years
            return f">= {years} years"
        return int(val) if val == int(val) else val

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

        Priority (tech-specific ALWAYS wins over overall):
        1) years from resume (tech-specific)
        2) years from prompt (tech-specific)
        3) overall years from resume
        4) overall years from prompt

        Seniority terms are ignored for experience_mentioned (no hard-coded mapping).
        """
        # Tech-specific experience always takes priority
        if years_resume is not None:
            return self._format_year_value(years_resume)
        if years_prompt is not None:
            return self._format_year_value(years_prompt)
        # Fall back to overall only if no tech-specific experience
        if overall_years_resume is not None:
            return self._format_year_value(overall_years_resume)
        if overall_years_prompt is not None:
            return self._format_year_value(overall_years_prompt)
        return None

    def extract_technologies(self, text: str, is_resume: bool = False, prompt_override: str = "") -> Dict[str, Any]:
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

        # Extract overall experience and seniority from resume/prompt
        if is_resume:
            # When is_resume=True, text is the resume
            overall_exp_from_resume = self._extract_overall_experience(text)
            seniority_from_resume = self._extract_seniority_level(text)
            # prompt_override is the job prompt in this case
            overall_exp_from_prompt = self._extract_overall_experience(prompt_override) if prompt_override else None
            seniority_from_prompt = self._extract_seniority_level(prompt_override) if prompt_override else None
        else:
            # When is_resume=False, text is the job prompt
            overall_exp_from_resume = None
            seniority_from_resume = None
            overall_exp_from_prompt = self._extract_overall_experience(text)
            seniority_from_prompt = self._extract_seniority_level(text)

        # Detect technologies mentioned in text
        for tech_id, keywords in self.tech_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tech_info = self.tech_db.get(tech_id, {})

                # Build alternatives dict (without experience_required)
                alternatives = {}
                for alt_id in tech_info.get("alternatives", []):
                    alt_info = self.tech_db.get(alt_id, {})
                    if alt_info:
                        alternatives[alt_id] = {"difficulty": alt_info.get("difficulty", 5.0)}

                tech_entry = {
                    "difficulty": tech_info.get("difficulty", 5.0),
                    "category": tech_info.get("category", "other"),
                    "alternatives": alternatives,
                }

                # Add experience_validated_via_github only if resume is provided
                if is_resume:
                    tech_entry["experience_validated_via_github"] = None

                # Extract experience from prompt (if provided)
                prompt_exp = None
                if is_resume and prompt_override:
                    # When is_resume=True, prompt_override contains the job prompt
                    prompt_exp = self._extract_experience(prompt_override, tech_id)
                    if prompt_exp is None and overall_exp_from_prompt is not None:
                        prompt_exp = overall_exp_from_prompt
                elif not is_resume:
                    # When is_resume=False, text IS the prompt - extract from it
                    prompt_exp = self._extract_experience(text, tech_id)
                    if prompt_exp is None and overall_exp_from_prompt is not None:
                        prompt_exp = overall_exp_from_prompt

                if prompt_exp is not None:
                    # Keep prompt-specific field numeric for compatibility
                    tech_entry["experience_mentioned_in_prompt"] = abs(prompt_exp)

                # Extract experience from resume text (if is_resume=True)
                if is_resume:
                    resume_exp = self._extract_experience(text, tech_id)
                    if resume_exp is not None:
                        # Keep resume-specific field numeric for compatibility
                        tech_entry["experience_accounted_for_in_resume"] = abs(resume_exp)

                # Calculate experience_mentioned value (tech-specific takes priority over overall)
                years_resume = self._extract_experience(text, tech_id) if is_resume else None
                years_prompt = prompt_exp
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

                # Propagate SAME experience to alternatives (use tech-specific if available, otherwise overall)
                # This ensures "2 years React" gives "2 years" to Vue/Angular, not the overall "9+ years"
                if exp_mentioned is not None:
                    for alt_id in tech_entry["alternatives"]:
                        tech_entry["alternatives"][alt_id]["experience_mentioned"] = exp_mentioned

                detected[tech_id] = tech_entry

        # Always try to discover unknown technologies using TechRegistry (not just when detected is empty)
        # This allows ServiceNow, Twilio, CI/CD, etc. to be found alongside React/Node
        import re

        # Look for common tech name patterns (lowercase to match)
        # Expanded patterns to include more technologies
        potential_techs = re.findall(
            r'\b(?:grafana|prometheus|jenkins|terraform|ansible|datadog|splunk|newrelic|new\s*relic|'
            r'elasticsearch|kibana|logstash|nginx|apache|redis|memcached|'
            r'mongodb|mysql|cassandra|neo4j|influxdb|timescaledb|dynamodb|'
            r'kubernetes|k8s|helm|istio|linkerd|envoy|'
            r'jenkins|gitlab|circleci|travis|github\s*actions?|'
            r'terraform|ansible|puppet|chef|salt|packer|vagrant|'
            r'vue|angular|svelte|ember|backbone|'
            r'flask|django|fastapi|spring|hibernate|'
            r'ruby|rails|laravel|symfony|express|nestjs|'
            r'servicenow|twilio|sendgrid|stripe|'
            r'python|\.net|dotnet|csharp|c#|'
            r'ci/cd|cicd|continuous\s+integration|continuous\s+delivery)\b',
            text_lower,
        )

        for potential_tech in set(potential_techs):  # use set to dedupe
            # Skip if already detected
            tech_id = potential_tech.replace('/', '').replace(' ', '_')  # normalize name
            if tech_id in detected or potential_tech in detected:
                continue

            # Try to get info from tech registry
            tech_info = self.tech_registry.get_tech_info(potential_tech)
            if tech_info:
                # Found a technology!
                tech_id = potential_tech

                # Get similar technologies as alternatives
                try:
                    similar_names = self.tech_registry.search_similar_techs(tech_id, top_k=3)
                    alternatives = {}
                    for sim_name in similar_names:
                        sim_info = self.tech_registry.get_tech_info(sim_name)
                        if sim_info and sim_name != tech_id:
                            alternatives[sim_name] = {"difficulty": sim_info.get("difficulty", 5.0)}
                except Exception:
                    # Fallback if similarity search fails
                    alternatives = {}

                raw_cat = tech_info.get("category", "other")
                refined_cat = raw_cat if raw_cat != "other" else self.tech_registry.infer_category_by_name(tech_id)
                if refined_cat == "other":
                    refined_cat = "uncategorized"
                tech_entry = {
                    "difficulty": tech_info.get("difficulty", 5.0),
                    "category": refined_cat,
                    "alternatives": alternatives,
                }

                # Add experience_validated_via_github only if resume is provided
                if is_resume:
                    tech_entry["experience_validated_via_github"] = None

                # Calculate experience_mentioned value (tech-specific takes priority over overall)
                years_resume = self._extract_experience(text, tech_id) if is_resume else None
                years_prompt = self._extract_experience(text, tech_id) if not is_resume else None
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
                    # Propagate same experience to alternatives
                    for alt_id in tech_entry["alternatives"]:
                        tech_entry["alternatives"][alt_id]["experience_mentioned"] = exp_mentioned

                detected[tech_id] = tech_entry

        # If nothing detected, fall back to role defaults based on generic phrases
        if not detected:
            for role_phrase, tech_list in self.role_defaults.items():
                if role_phrase in text_lower:
                    for tech_id in tech_list:
                        tech_info = self.tech_db.get(tech_id, {})
                        if not tech_info:
                            continue
                        # Build alternatives
                        alternatives = {}
                        for alt_id in tech_info.get("alternatives", []):
                            alt_info = self.tech_db.get(alt_id, {})
                            if alt_info:
                                alternatives[alt_id] = {"difficulty": alt_info.get("difficulty", 5.0)}

                        tech_entry = {
                            "difficulty": tech_info.get("difficulty", 5.0),
                            "category": tech_info.get("category", "other"),
                            "alternatives": alternatives,
                        }

                        # Add experience_validated_via_github only if resume is provided
                        if is_resume:
                            tech_entry["experience_validated_via_github"] = None

                        # Apply global estimate to tech and its alternatives
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

                        detected[tech_id] = tech_entry
                    # Apply first matching role defaults only
                    break

        # If still nothing, check for project descriptions (e.g., "amazon clone", "netflix copy")
        if not detected:
            for project_name, tech_list in self.project_defaults.items():
                # Match patterns: "X clone", "X copy", "like X", "similar to X", "build X"
                # Also check reverse: "twitter like" → "twitter-like"
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
                    for tech_id in tech_list:
                        tech_info = self.tech_db.get(tech_id, {})
                        if not tech_info:
                            continue
                        # Build alternatives
                        alternatives = {}
                        for alt_id in tech_info.get("alternatives", []):
                            alt_info = self.tech_db.get(alt_id, {})
                            if alt_info:
                                alternatives[alt_id] = {"difficulty": alt_info.get("difficulty", 5.0)}

                        tech_entry = {
                            "difficulty": tech_info.get("difficulty", 5.0),
                            "category": tech_info.get("category", "other"),
                            "alternatives": alternatives,
                        }

                        # Add experience_validated_via_github only if resume is provided
                        if is_resume:
                            tech_entry["experience_validated_via_github"] = None

                        # Apply global estimate to tech and its alternatives
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

                        detected[tech_id] = tech_entry
                    # Apply first matching project defaults only
                    break

        # Generic clone/copy fallback for unknown projects (non-hardcoded brands)
        if not detected:
            # Detect clone/copy/like patterns generically
            import re

            clone_like_patterns = [
                r"\bclone\b",
                r"\bcopy\b",
                r"\blike\b",
                r"\bsimilar to\b",
            ]
            if any(re.search(p, text_lower) for p in clone_like_patterns):
                # Start from a generic modern web baseline
                generic_stack = ["react", "node", "postgres", "redis", "docker"]
                # Optional enrichments based on keywords
                if any(k in text_lower for k in ["search", "semantic", "index"]):
                    generic_stack.append("elasticsearch")
                if any(k in text_lower for k in ["cloud", "aws", "azure", "gcp"]):
                    generic_stack.append("aws")
                if any(k in text_lower for k in ["serverless", "lambda"]):
                    generic_stack.append("aws_lambda")

                for tech_id in generic_stack:
                    tech_info = self.tech_db.get(tech_id, {})
                    if not tech_info:
                        continue
                    # Build alternatives
                    alternatives = {}
                    for alt_id in tech_info.get("alternatives", []):
                        alt_info = self.tech_db.get(alt_id, {})
                        if alt_info:
                            alternatives[alt_id] = {"difficulty": alt_info.get("difficulty", 5.0)}

                    tech_entry = {
                        "difficulty": tech_info.get("difficulty", 5.0),
                        "category": tech_info.get("category", "other"),
                        "alternatives": alternatives,
                    }

                    # Add experience_validated_via_github only if resume is provided
                    if is_resume:
                        tech_entry["experience_validated_via_github"] = None

                    detected[tech_id] = tech_entry

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
                    for tech_id in detected:
                        detected[tech_id]["experience_mentioned"] = exp_mentioned
                        for alt_id in detected[tech_id]["alternatives"]:
                            detected[tech_id]["alternatives"][alt_id]["experience_mentioned"] = exp_mentioned

        return {"technologies": detected}


def main():
    """Test the extractor with sample prompts."""
    extractor = SimpleTechExtractor()

    test_prompts = [
        "Senior Full-Stack Engineer with 5+ years React and Node.js experience",
        "Backend developer with FastAPI, PostgreSQL, and Redis",
        "DevOps engineer with 3 years Kubernetes, Docker, and AWS",
        "Looking for React, TypeScript, and MongoDB expert",
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
