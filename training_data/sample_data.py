"""Sample training data for technology extraction models.

This file contains synthetic and curated training examples for:
- Technology extraction (NER)
- Difficulty scoring
- Experience extraction validation
- Technology similarity/alternatives

In production, augment with:
- Real job postings from APIs
- Stack Overflow survey data
- GitHub technology trends
- Developer surveys and interviews
"""

from __future__ import annotations

# Technology corpus for embeddings training
TECHNOLOGY_CORPUS = [
    {
        "name": "react",
        "description": "JavaScript library for building user interfaces with components",
        "use_cases": ["spa", "web apps", "mobile apps", "ui"],
        "category": "frontend",
        "years_in_market": 10,
        "github_stars": 220000,
        "stackoverflow_questions": 450000,
    },
    {
        "name": "vue",
        "description": "Progressive JavaScript framework for building web interfaces",
        "use_cases": ["spa", "web apps", "ui", "progressive enhancement"],
        "category": "frontend",
        "years_in_market": 9,
        "github_stars": 200000,
        "stackoverflow_questions": 100000,
    },
    {
        "name": "angular",
        "description": "TypeScript-based web application framework by Google",
        "use_cases": ["spa", "enterprise apps", "web apps"],
        "category": "frontend",
        "years_in_market": 13,
        "github_stars": 90000,
        "stackoverflow_questions": 300000,
    },
    {
        "name": "node",
        "description": "JavaScript runtime for building server-side applications",
        "use_cases": ["api", "backend", "microservices", "real-time"],
        "category": "backend",
        "years_in_market": 14,
        "github_stars": 100000,
        "stackoverflow_questions": 400000,
    },
    {
        "name": "python",
        "description": "High-level programming language for web, data, and automation",
        "use_cases": ["backend", "data science", "scripting", "ml"],
        "category": "backend",
        "years_in_market": 30,
        "github_stars": 50000,
        "stackoverflow_questions": 2000000,
    },
    {
        "name": "fastapi",
        "description": "Modern Python web framework for building APIs",
        "use_cases": ["api", "backend", "microservices"],
        "category": "backend",
        "years_in_market": 5,
        "github_stars": 70000,
        "stackoverflow_questions": 15000,
    },
    {
        "name": "django",
        "description": "High-level Python web framework for rapid development",
        "use_cases": ["web apps", "backend", "cms", "admin"],
        "category": "backend",
        "years_in_market": 18,
        "github_stars": 75000,
        "stackoverflow_questions": 350000,
    },
    {
        "name": "postgres",
        "description": "Advanced open-source relational database",
        "use_cases": ["database", "sql", "data storage", "transactions"],
        "category": "database",
        "years_in_market": 30,
        "github_stars": 13000,
        "stackoverflow_questions": 200000,
    },
    {
        "name": "mongodb",
        "description": "NoSQL document database for flexible data storage",
        "use_cases": ["database", "nosql", "json storage", "scalability"],
        "category": "database",
        "years_in_market": 15,
        "github_stars": 25000,
        "stackoverflow_questions": 150000,
    },
    {
        "name": "docker",
        "description": "Platform for containerizing and deploying applications",
        "use_cases": ["containerization", "deployment", "devops"],
        "category": "infrastructure",
        "years_in_market": 11,
        "github_stars": 65000,
        "stackoverflow_questions": 180000,
    },
    {
        "name": "kubernetes",
        "description": "Container orchestration platform for managing deployments",
        "use_cases": ["orchestration", "scalability", "devops", "cloud"],
        "category": "infrastructure",
        "years_in_market": 9,
        "github_stars": 105000,
        "stackoverflow_questions": 60000,
    },
]

# Difficulty training data
DIFFICULTY_TRAINING_DATA = [
    {
        "tech_name": "react",
        "context": {
            "years_in_market": 10,
            "github_stars": 220000,
            "stackoverflow_questions": 450000,
            "learning_resources": 9.0,
            "api_complexity": 5.0,
            "ecosystem_size": 9.5,
        },
        "difficulty": 5.2,
    },
    {
        "tech_name": "vue",
        "context": {
            "years_in_market": 9,
            "github_stars": 200000,
            "stackoverflow_questions": 100000,
            "learning_resources": 8.5,
            "api_complexity": 4.5,
            "ecosystem_size": 8.0,
        },
        "difficulty": 4.8,
    },
    {
        "tech_name": "angular",
        "context": {
            "years_in_market": 13,
            "github_stars": 90000,
            "stackoverflow_questions": 300000,
            "learning_resources": 8.0,
            "api_complexity": 7.5,
            "ecosystem_size": 8.5,
        },
        "difficulty": 6.5,
    },
    {
        "tech_name": "python",
        "context": {
            "years_in_market": 30,
            "github_stars": 50000,
            "stackoverflow_questions": 2000000,
            "learning_resources": 10.0,
            "api_complexity": 3.0,
            "ecosystem_size": 10.0,
        },
        "difficulty": 4.0,
    },
    {
        "tech_name": "kubernetes",
        "context": {
            "years_in_market": 9,
            "github_stars": 105000,
            "stackoverflow_questions": 60000,
            "learning_resources": 7.0,
            "api_complexity": 9.0,
            "ecosystem_size": 8.5,
        },
        "difficulty": 7.5,
    },
    {
        "tech_name": "docker",
        "context": {
            "years_in_market": 11,
            "github_stars": 65000,
            "stackoverflow_questions": 180000,
            "learning_resources": 8.5,
            "api_complexity": 5.0,
            "ecosystem_size": 9.0,
        },
        "difficulty": 5.5,
    },
]

# Experience extraction validation data
EXPERIENCE_TRAINING_DATA = [
    {
        "text": "5+ years of React experience required",
        "tech_name": "react",
        "years": 5.0,
        "is_valid": True,
    },
    {
        "text": "React development with 3 years minimum",
        "tech_name": "react",
        "years": 3.0,
        "is_valid": True,
    },
    {
        "text": "Looking for someone with React skills, 7 years overall experience",
        "tech_name": "react",
        "years": 7.0,
        "is_valid": False,  # 7 years is overall, not React-specific
    },
    {
        "text": "Senior developer, 10 years in Python",
        "tech_name": "python",
        "years": 10.0,
        "is_valid": True,
    },
    {
        "text": "2+ years Docker and Kubernetes",
        "tech_name": "docker",
        "years": 2.0,
        "is_valid": True,
    },
]

# Technology similarity pairs (for alternatives training)
SIMILARITY_PAIRS = [
    ("react", "vue", 0.85),  # Very similar (both frontend frameworks)
    ("react", "angular", 0.75),  # Similar but different approaches
    ("react", "node", 0.40),  # Different layers but often used together
    ("react", "postgres", 0.15),  # Different domains
    ("vue", "angular", 0.80),  # Similar frameworks
    ("node", "python", 0.60),  # Both backend, similar use cases
    ("node", "fastapi", 0.70),  # Both API frameworks
    ("python", "fastapi", 0.85),  # FastAPI is Python framework
    ("postgres", "mongodb", 0.50),  # Both databases but different paradigms
    ("postgres", "mysql", 0.90),  # Very similar (both SQL)
    ("docker", "kubernetes", 0.75),  # Related containerization tools
    ("django", "flask", 0.70),  # Both Python web frameworks
    ("django", "rails", 0.60),  # Similar MVC frameworks, different languages
]


def generate_synthetic_job_postings(count: int = 100) -> list[str]:
    """Generate synthetic job postings for testing.

    Args:
        count: Number of postings to generate

    Returns:
        List of job posting texts
    """
    import random

    templates = [
        "Looking for a {seniority} {role} with {years}+ years of {tech1} and {tech2} experience.",
        "We need a {role} proficient in {tech1}, {tech2}, and {tech3}. {years} years required.",
        "{seniority} {role} position. Must have {years}+ years with {tech1} and strong {tech2} skills.",
        "Seeking {role} with expertise in {tech1} ({years}+ years) and {tech2}.",
    ]

    seniorities = ["Junior", "Mid-level", "Senior", "Lead", "Principal"]
    roles = ["Engineer", "Developer", "Architect", "Consultant"]
    tech_names = [tech["name"] for tech in TECHNOLOGY_CORPUS]

    postings = []
    for _ in range(count):
        template = random.choice(templates)
        posting = template.format(
            seniority=random.choice(seniorities),
            role=random.choice(roles),
            years=random.randint(2, 8),
            tech1=random.choice(tech_names),
            tech2=random.choice(tech_names),
            tech3=random.choice(tech_names),
        )
        postings.append(posting)

    return postings
