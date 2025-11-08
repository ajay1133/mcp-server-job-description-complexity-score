#!/usr/bin/env python3
"""Generate training data by collecting software requirements from the web.

Sources:
1. GitHub issue templates and real issues (software examples)
2. Job posting sites (mix of software and non-software)
3. Tech documentation examples
4. Stack Overflow questions (software examples)

This script uses web scraping and public APIs to collect diverse examples,
then estimates LOC and hours based on heuristics.
"""

import json
import os
import re
import time
from typing import List, Dict, Any
import random


def estimate_loc_and_hours(text: str, technologies: List[str]) -> tuple[int, float]:
    """Heuristic estimation of LOC and hours based on text and technologies."""
    text_lower = text.lower()
    
    # Base estimate
    base_loc = 500
    base_hours = 25.0
    
    # Complexity indicators
    complexity_keywords = {
        'simple': 0.5, 'basic': 0.6, 'simple app': 0.5,
        'full stack': 2.0, 'microservice': 1.8, 'platform': 2.5,
        'real-time': 1.6, 'scalable': 1.8, 'distributed': 2.2,
        'ml': 2.0, 'ai': 2.0, 'machine learning': 2.2,
        'dashboard': 1.2, 'admin panel': 1.3, 'cms': 1.5,
        'api': 1.0, 'rest api': 1.1, 'graphql': 1.3,
        'authentication': 1.2, 'auth': 1.2, 'payment': 1.4,
        'deploy': 1.1, 'ci/cd': 1.2, 'kubernetes': 1.5,
    }
    
    multiplier = 1.0
    for keyword, mult in complexity_keywords.items():
        if keyword in text_lower:
            multiplier = max(multiplier, mult)
    
    # Technology count impact
    tech_multiplier = 1.0 + (len(technologies) * 0.2)
    
    # Feature count (rough estimate from verbs)
    features = len(re.findall(r'\b(create|build|add|implement|integrate|deploy|support)\b', text_lower))
    feature_multiplier = 1.0 + (features * 0.15)
    
    final_loc = int(base_loc * multiplier * tech_multiplier * feature_multiplier)
    final_hours = base_hours * multiplier * tech_multiplier * feature_multiplier
    
    # Apply reasonable bounds
    final_loc = max(100, min(final_loc, 50000))
    final_hours = max(5.0, min(final_hours, 2000.0))
    
    return final_loc, final_hours


def detect_technologies(text: str) -> List[str]:
    """Extract technology tags from text."""
    text_lower = text.lower()
    technologies = []
    
    tech_patterns = {
    'react': r'\breact\b|\breact native\b',
    'nextjs': r'\bnext\.?js\b|\bnext\s+js\b',
    'vue': r'\bvue\b|\bvue\.?js\b',
    'angular': r'\bangular\b',
        'svelte': r'\bsvelte\b',
    'node': r'\bnode\.?js\b|\bexpress\b|\bbackend.*javascript\b',
        'python_fastapi': r'\bfastapi\b',
        'python_django': r'\bdjango\b',
        'flask': r'\bflask\b',
    'rails': r'\bruby on rails\b|\brails\b|\bruby\b',
    'postgres': r'\bpostgres\b|\bpostgresql\b',
        'mysql': r'\bmysql\b',
    'mongodb': r'\bmongo\b|\bmongodb\b',
        'redis': r'\bredis\b',
    'auth': r'\bauth\b|\bauthentication\b|\blogin\b|\bjwt\b|\boauth\b|\bsign[- ]?in\b|\bsign[- ]?up\b',
        'payments': r'\bpayment\b|\bstripe\b|\bpaypal\b',
    'devops': r'\bdevops\b|\bci/cd\b|\bdocker\b|\bkubernetes\b|\bk8s\b|\bdeploy\b|\bdeployment\b',
        'aws': r'\baws\b|\bamazon web services\b',
        'azure': r'\bazure\b',
        'gcp': r'\bgcp\b|\bgoogle cloud\b',
    'ai_llm': r'\bopenai\b|\bgpt\b|\bllm\b|\blarge language model\b|\bchatbot\b',
    'ml': r'\bmachine learning\b|\bml model\b|\bml\b|\bai model\b|\brecommendation\b',
    'websocket': r'\bwebsocket\b|\breal-?time.*chat\b',
    'realtime': r'\breal-time\b|\breal time\b|\blive update\b',
    'testing': r'\btesting\b|\bunit test\b|\be2e\b|\bcypress\b|\bjest\b',
    }
    
    for tech, pattern in tech_patterns.items():
        if re.search(pattern, text_lower):
            technologies.append(tech)
    
    # Add implicit technologies based on context
    if not technologies:
        # If mentions API, likely need node or fastapi
        if re.search(r'\bapi\b|\brest\b|\bgraphql\b', text_lower):
            technologies.append('node')
        # If mentions dashboard/admin, likely need react
        if re.search(r'\bdashboard\b|\badmin panel\b|\bfrontend\b', text_lower):
            technologies.append('react')
        # If mentions database without specific one, default to postgres
        if re.search(r'\bdatabase\b|\bdb\b', text_lower) and not any(db in text_lower for db in ['mongo', 'mysql', 'postgres', 'redis']):
            technologies.append('postgres')
    
    return list(set(technologies))


# Sample software requirements from common patterns
SOFTWARE_EXAMPLES = [
    "Build a React dashboard with user authentication and role-based access control",
    "Create a REST API using FastAPI that integrates with PostgreSQL database",
        # Focused explicit tech combinations to strengthen labels
        "Develop Next.js SaaS with Stripe payments and Google OAuth",
        "Build Next.js storefront with Stripe checkout and user authentication",
        "Create Next.js dashboard with JWT auth and role-based access",
        "Develop Django web app with MySQL database and Celery background tasks",
        "Build Django REST API with MySQL and JWT authentication",
        "Create FastAPI microservice with MongoDB and Redis cache",
        "Develop FastAPI backend with MongoDB and JWT-based auth",
        "Build Vue.js SPA with OAuth login and protected routes",
        "Create Vue.js frontend integrated with REST API and JWT",
        "Develop Angular portal with role-based access and JWT authentication",
        "Build Angular app with OAuth2 login and protected APIs",
        "Create Node.js Express API with MongoDB and JWT authentication",
        "Build Node.js backend with MongoDB, Stripe payments and webhooks",
        "Develop Rails app with PostgreSQL, Devise auth and Stripe billing",
        "Create React app with Auth0 authentication and Stripe subscriptions",
        "Build React + Next.js hybrid app with payments and login",
        "Implement Django + Channels real-time chat with Redis",
        "Create FastAPI + WebSocket real-time notifications with Redis",
        "Build Next.js + Prisma + PostgreSQL full-stack app with auth",
        "Develop Remix/Next.js app with Stripe payments and webhook processing",
        "Create Flask API with SQLAlchemy (MySQL) and JWT auth",
        "Build Flask + MongoDB API with JWT and role-based permissions",
        "Develop React Native mobile app with Firebase Auth and Firestore",
        "Create Angular + NestJS API with PostgreSQL and JWT",
        "Build Vue.js + Nuxt app with Auth and Stripe payments",
        "Develop Next.js B2B SaaS with organizations, RBAC and billing",
        "Create e-commerce with Next.js, Stripe, and PostgreSQL inventory",
        "Build booking platform using Django, MySQL, and Stripe",
        "Develop learning portal using React, FastAPI, and MongoDB",
        "Create community forum with Node.js, MongoDB, and JWT auth",
        "Build analytics dashboard with React, Next.js API routes, and Postgres",
        "Develop CMS with Django, MySQL, authentication and image uploads",
        "Create REST API in FastAPI with MongoDB and rate limiting",
        "Build social app with Vue.js frontend, Node.js API, and JWT",
    "Implement real-time chat application using WebSocket and Redis pub/sub",
    "Develop a Next.js e-commerce site with Stripe payment integration",
    "Build a microservices architecture with Docker and Kubernetes deployment",
    "Create a mobile app with React Native for iOS and Android",
    "Implement CI/CD pipeline using GitHub Actions and deploy to AWS",
    "Build an admin dashboard with data visualization using Django and Chart.js",
    "Create a GraphQL API with Node.js and MongoDB backend",
    "Develop a machine learning model API using FastAPI and scikit-learn",
    "Build a video streaming platform with adaptive bitrate using HLS",
    "Create a serverless application using AWS Lambda and DynamoDB",
    "Implement OAuth2 authentication with social login providers",
    "Build a content management system with Next.js and headless CMS",
    "Create a real-time collaboration tool like Google Docs using Yjs",
    "Develop a recommendation engine using collaborative filtering",
    "Build a progressive web app with offline support using service workers",
    "Create a blockchain-based application with Ethereum smart contracts",
    "Implement a search engine using Elasticsearch and Redis caching",
    "Build a data pipeline with Apache Kafka and Spark for analytics",
    "Create a chatbot using OpenAI GPT API and LangChain",
    "Develop a monitoring dashboard with Prometheus and Grafana",
    "Build a PDF generation service using headless Chrome",
    "Create an email marketing platform with SendGrid integration",
    "Implement a file upload system with S3 and image processing",
    "Build a booking system with calendar integration and Stripe",
    "Create a multi-tenant SaaS application with row-level security",
    "Develop a REST API with rate limiting and API key authentication",
    "Build a social media analytics dashboard with Twitter API",
    "Create a webhook receiver and event processor with queue system",
    "Implement a full-text search with fuzzy matching using PostgreSQL",
    "Build a video conferencing app using WebRTC and signaling server",
    "Create a notification system with push notifications and email",
    "Develop a inventory management system with barcode scanning",
    "Build a workflow automation tool like Zapier with integrations",
    "Create a code review tool with GitHub integration",
    "Implement a feature flag system with A/B testing capabilities",
    "Build a URL shortener service with analytics tracking",
    "Create a screenshot API using Puppeteer and Docker",
    "Develop a document collaboration platform with real-time sync",
    "Build a cryptocurrency trading bot with exchange API integration",
    "Create a web scraper with proxy rotation and anti-bot detection",
    "Implement a cache layer with Redis and cache invalidation strategy",
    "Build a log aggregation system with ELK stack",
    "Create a user onboarding flow with email verification",
    "Develop a backup and restore system for database",
    "Build a testing framework with automated UI testing",
    "Create a deployment tool with blue-green deployment strategy",
    "Implement a multi-language support system with i18n",
    "Build a image compression and optimization service",
    # Additional explicit technology examples
    "Build Vue.js single page application with authentication using JWT",
    "Create Angular dashboard with Material UI and TypeScript",
    "Develop Node.js REST API with Express and MySQL database",
    "Build Python Django web application with PostgreSQL backend",
    "Create Flask microservice with Redis caching and MongoDB",
    "Implement React Native mobile app with Firebase backend",
    "Build Next.js blog platform with Markdown support",
    "Create Ruby on Rails e-commerce site with PostgreSQL",
    "Develop FastAPI service with SQLAlchemy ORM and PostgreSQL",
    "Build Svelte application with real-time WebSocket updates",
    "Create GraphQL server using Apollo and PostgreSQL",
    "Implement serverless functions on AWS Lambda with Python",
    "Build Azure Functions app with C# and CosmosDB",
    "Create GCP Cloud Functions with Node.js and Firestore",
    "Develop microservices with Docker containers on Kubernetes",
    "Build CI/CD pipeline with Jenkins and Docker deployment",
    "Create monitoring system with Prometheus, Grafana, and alerts",
    "Implement machine learning API with TensorFlow and FastAPI",
    "Build data analytics platform with Apache Spark and Kafka",
    "Create real-time messaging with WebSocket and Redis pub/sub",
    "Develop payment processing with Stripe API integration",
    "Build OAuth authentication with Google and Facebook login",
    "Create file storage service with AWS S3 and CloudFront CDN",
    "Implement search functionality with Elasticsearch indexing",
    "Build email service with SendGrid API and templates",
    "Create notification system with Firebase Cloud Messaging",
    "Develop task queue system with Celery and Redis",
    "Build content delivery with Cloudflare and edge caching",
    "Create video processing pipeline with FFmpeg and S3",
    "Implement API gateway with rate limiting and authentication",
    "Build admin panel with React, Material-UI, and REST API",
    "Create e-commerce cart with Redux state management",
    "Develop booking system with Stripe payments and calendar",
    "Build CMS with Next.js, Contentful, and TypeScript",
    "Create social network with Neo4j graph database",
    "Implement chat application with Socket.io and MongoDB",
    "Build analytics dashboard with D3.js visualizations",
    "Create testing suite with Jest, Cypress, and CI integration",
    "Develop API documentation with Swagger and OpenAPI",
    "Build authentication service with JWT and refresh tokens",
    "Create image processing API with Sharp and S3 storage",
    "Implement logging system with Winston and Elasticsearch",
    "Build rate limiter middleware with Redis counters",
    "Create webhook dispatcher with queue and retry logic",
    "Develop PDF generator with Puppeteer and template engine",
    "Build multi-tenant app with PostgreSQL row-level security",
    "Create scraping service with Playwright and proxy pool",
    "Implement feature flags with LaunchDarkly integration",
    "Build A/B testing framework with analytics tracking",
]

NON_SOFTWARE_EXAMPLES = [
    "Need someone to look after my elderly father who needs daily assistance",
    "Looking for a caregiver to help with meal preparation and medication reminders",
    "Need a plumber to fix leaking kitchen sink and replace faucet",
    "Electrician needed to install ceiling fan and fix outlet",
    "Carpenter wanted to build custom bookshelf and repair deck",
    "Looking for wedding planner to coordinate venue and vendors",
    "Need photographer for wedding ceremony and reception",
    "Party planner needed for birthday celebration with 50 guests",
    "Need babysitter for two kids ages 5 and 7 on weekends",
    "Looking for nanny for newborn, must have experience with infants",
    "Housekeeper needed for weekly cleaning and laundry service",
    "Need someone to organize and declutter entire house",
    "Personal trainer wanted for weight loss and fitness program",
    "Massage therapist needed for weekly deep tissue massage",
    "Tutor needed for high school math and physics",
    "Piano teacher wanted for beginner student",
    "Dog walker needed for daily walks with golden retriever",
    "Pet sitter wanted for cat while on vacation",
    "Landscaper needed to design and plant garden",
    "Need someone to mow lawn and trim hedges weekly",
    "Hair stylist wanted for wedding day styling",
    "Makeup artist needed for special event",
    "Personal chef wanted for meal prep service",
    "Caterer needed for corporate event with 100 people",
    "Driver needed for airport pickup and drop-off",
    "Moving help needed to pack and transport furniture",
    "Handyman wanted to fix door hinge and patch drywall",
    "Painter needed to repaint living room and bedroom",
    "Need mechanic to diagnose car engine problem",
    "Auto detailer wanted for interior and exterior cleaning",
    "Lawyer needed for contract review and legal advice",
    "Accountant wanted to prepare tax returns",
    "Real estate agent needed to sell house",
    "Interior designer wanted to decorate living room",
    "Event coordinator needed for corporate conference",
    "Translator wanted for Spanish to English documents",
    "Voice actor needed for commercial recording",
    "Seamstress wanted to alter wedding dress",
    "Furniture assembly help needed for IKEA items",
    "Need someone to help organize garage sale",
    # Additional diverse non-software examples
    "Doctor needed for annual physical exam and health checkup",
    "Dentist wanted for teeth cleaning and cavity filling",
    "Physical therapist needed for back pain treatment",
    "Chiropractor wanted for spine alignment sessions",
    "Nutritionist needed for meal planning and diet consultation",
    "Life coach wanted to help with career transition",
    "Financial advisor needed for retirement planning",
    "Tax consultant wanted for small business accounting",
    "Immigration lawyer needed for visa application help",
    "Divorce attorney wanted for legal representation",
    "Architect needed to design house renovation plans",
    "Structural engineer wanted for building inspection",
    "HVAC technician needed to repair air conditioning",
    "Roofer wanted to fix leaking roof and replace shingles",
    "Flooring installer needed for hardwood installation",
    "Tile setter wanted for bathroom remodel",
    "Drywall contractor needed to finish basement walls",
    "Window washer wanted for high-rise building cleaning",
    "Pest control specialist needed for termite treatment",
    "Arborist wanted to remove dead tree from yard",
    "Pool cleaner needed for weekly maintenance service",
    "HVAC installer wanted for new furnace installation",
    "Locksmith needed to rekey house locks",
    "Appraiser wanted for home valuation",
    "Home inspector needed before house purchase",
    "Veterinarian needed for dog's annual checkup",
    "Groomer wanted for poodle haircut and bath",
    "Horse trainer needed for riding lessons",
    "Farrier wanted to shoe horses",
    "Dance instructor needed for ballroom dance lessons",
    "Yoga teacher wanted for private sessions",
    "Pilates instructor needed for group classes",
    "Swim coach wanted for competitive training",
    "Tennis instructor needed for beginner lessons",
    "Golf pro wanted for swing improvement coaching",
    "Personal assistant needed for scheduling and errands",
    "Virtual assistant wanted for email management",
    "Bookkeeper needed for small business finances",
    "Notary public wanted for document signing",
    "Process server needed to deliver legal documents",
]


def generate_software_entry(text: str) -> Dict[str, Any]:
    """Generate a complete training entry for a software requirement."""
    technologies = detect_technologies(text)
    loc, hours = estimate_loc_and_hours(text, technologies)
    
    # Compute complexity score heuristically
    complexity_score = 50 + (len(technologies) * 12) + (loc / 100) + (hours / 2)
    complexity_score = max(10, min(int(complexity_score), 200))
    
    return {
        "text": text,
        "is_software": True,
        "technologies": technologies,
        "loc": loc,
        "hours": round(hours, 1),
        "complexity_score": complexity_score
    }


def generate_non_software_entry(text: str) -> Dict[str, Any]:
    """Generate a training entry for a non-software requirement."""
    return {
        "text": text,
        "is_software": False
    }


def generate_variations(base_text: str, count: int = 3) -> List[str]:
    """Generate variations of a base requirement to increase dataset diversity."""
    variations = [base_text]
    
    # Simple variations
    prefixes = [
        "I need ",
        "Looking for someone to ",
        "Need help with ",
        "Can you ",
        "Please help me ",
    ]
    
    suffixes = [
        " ASAP",
        " for my startup",
        " with modern tech stack",
        " using best practices",
        " that scales",
    ]
    
    for _ in range(count - 1):
        variation = base_text
        if random.random() > 0.5:
            variation = random.choice(prefixes) + variation.lower()
        if random.random() > 0.7:
            variation = variation + random.choice(suffixes)
        variations.append(variation)
    
    return variations


def main():
    output_file = "data/software_training_data.jsonl"
    os.makedirs("data", exist_ok=True)
    
    print("=" * 70)
    print("Automated Training Data Generator")
    print("=" * 70)
    print(f"\nGenerating training data from templates and patterns...")
    print(f"Output: {output_file}\n")
    
    entries = []
    
    # Generate software examples with variations
    print(f"Generating software examples...")
    for base_text in SOFTWARE_EXAMPLES:
        variations = generate_variations(base_text, count=2)
        for text in variations:
            entry = generate_software_entry(text)
            entries.append(entry)
    
    print(f"Generated {len(entries)} software examples")
    
    # Generate non-software examples
    print(f"Generating non-software examples...")
    non_software_count = 0
    for base_text in NON_SOFTWARE_EXAMPLES:
        variations = generate_variations(base_text, count=2)
        for text in variations:
            entry = generate_non_software_entry(text)
            entries.append(entry)
            non_software_count += 1
    
    print(f"Generated {non_software_count} non-software examples")
    
    # Shuffle to mix software and non-software
    random.shuffle(entries)
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Statistics
    software_entries = [e for e in entries if e.get('is_software')]
    
    print(f"\n{'=' * 70}")
    print(f"Generation Complete!")
    print(f"{'=' * 70}")
    print(f"Total entries: {len(entries)}")
    print(f"Software: {len(software_entries)}")
    print(f"Non-software: {len(entries) - len(software_entries)}")
    print(f"\nSoftware statistics:")
    if software_entries:
        locs = [e['loc'] for e in software_entries]
        hours = [e['hours'] for e in software_entries]
        scores = [e['complexity_score'] for e in software_entries]
        print(f"  LOC range: {min(locs)} - {max(locs)}")
        print(f"  Hours range: {min(hours):.1f} - {max(hours):.1f}")
        print(f"  Complexity range: {min(scores)} - {max(scores)}")
    
    print(f"\nSaved to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Review and edit {output_file} if needed")
    print(f"2. Add more real-world examples using: python curate_training_data.py")
    print(f"3. Train models: python train_software_models.py --data {output_file} --out models/software")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
