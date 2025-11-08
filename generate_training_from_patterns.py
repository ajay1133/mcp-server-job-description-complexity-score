#!/usr/bin/env python3
"""
Generate training data from system design patterns.

This script uses the system_design_patterns.json knowledge base to generate
high-quality training examples for the ML models. It creates realistic prompts
with accurate technology labels based on industry best practices.
"""

import json
import random
import os
from typing import List, Dict, Any
from pathlib import Path


def load_system_patterns() -> Dict[str, Any]:
    """Load system design patterns from config."""
    config_path = Path(__file__).parent / 'config' / 'system_design_patterns.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_tech_to_training_format(tech: str) -> str:
    """Normalize technology names to training data format."""
    mapping = {
        # Keep these as-is - already in training format
        "react": "react",
        "vue": "vue",
        "angular": "angular",
        "nextjs": "nextjs",
        "node": "node",
        "python_fastapi": "python_fastapi",
        "python_django": "python_django",
        "flask": "flask",
        "golang": "golang",
        "postgres": "postgres",
        "mysql": "mysql",
        "mongodb": "mongodb",
        "redis": "redis",
        "cassandra": "cassandra",
        "elasticsearch": "elasticsearch",
        "kafka": "kafka",
        "rabbitmq": "rabbitmq",
        "docker": "docker",
        "stripe": "stripe",
        
        # Map to existing training labels
        "typescript": "typescript",
        "redux": "redux",
        "ruby_rails": "rails",
        "java_spring": "java",
        "memcached": "cache",
        "s3": "s3",
        "cdn": "cdn",
        "websocket": "websocket",
        "nginx": "nginx",
        "kubernetes": "devops",
        "monitoring": "monitoring",
        "prometheus": "monitoring",
        "grafana": "monitoring",
        "datadog": "monitoring",
        
        # Mobile
        "react_native": "mobile",
        "swift": "mobile",
        "kotlin": "mobile",
        
        # Auth
        "jwt": "auth",
        "oauth": "auth",
        "auth": "auth",
        
        # Payments
        "paypal": "payments",
        "payments": "payments",
        
        # Maps
        "google_maps": "maps",
        "mapbox": "maps",
        
        # ML
        "tensorflow": "ml",
        "pytorch": "ml",
        "ml": "ml",
        
        # Video/Image
        "ffmpeg": "video_processing",
        "video_processing": "video_processing",
        "pillow": "image_processing",
        "imagemagick": "image_processing",
        
        # Real-time
        "realtime": "realtime",
        "xmpp": "realtime",
        
        # Storage
        "blob_store": "storage",
        
        # Search
        "search": "search",
        
        # Serverless
        "lambda": "serverless",
        "serverless": "serverless",
        
        # Message queue
        "sqs": "message_queue",
        "message_queue": "message_queue",
        
        # Misc
        "php": "php",
        "electron": "electron",
        "streaming": "streaming",
        "hls": "streaming",
        "dash": "streaming",
    }
    
    tech_lower = tech.lower().replace("-", "_").replace(" ", "_")
    return mapping.get(tech_lower, None)


def generate_prompts_for_pattern(pattern_name: str, pattern_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate multiple training examples for a pattern."""
    examples = []
    
    # Get core info
    description = pattern_data.get('description', '')
    keywords = pattern_data.get('keywords', [])
    microservices = pattern_data.get('microservices', [])
    technologies = pattern_data.get('technologies', {})
    
    # Collect all unique technologies
    all_techs = set()
    for category, tech_list in technologies.items():
        for tech in tech_list:
            normalized = normalize_tech_to_training_format(tech)
            if normalized:
                all_techs.add(normalized)
    
    # Remove duplicates and sort
    tech_list = sorted(list(all_techs))
    
    # Estimate complexity based on pattern
    # High complexity patterns: video, ride-hailing, social media
    if any(k in pattern_name for k in ['youtube', 'netflix', 'uber', 'twitter', 'instagram', 'tiktok']):
        base_hours = random.randint(600, 1200)
        complexity_score = random.randint(100, 150)
        loc = base_hours * 20  # Approximate LOC
    # Medium complexity: messaging, e-commerce
    elif any(k in pattern_name for k in ['whatsapp', 'slack', 'shopify', 'airbnb']):
        base_hours = random.randint(300, 600)
        complexity_score = random.randint(80, 110)
        loc = base_hours * 18
    else:
        base_hours = random.randint(200, 400)
        complexity_score = random.randint(60, 90)
        loc = base_hours * 15
    
    # Generate varied prompts for this pattern
    
    # Template 1: Direct "Build X" style
    prompt_templates = [
        f"Build a {keywords[0]} application",
        f"Create a {keywords[0]} platform",
        f"Develop a {keywords[0]} system like {pattern_name.split('_')[0].title()}",
        f"I need a {keywords[0]} app similar to {pattern_name.split('_')[0].title()}",
        f"Build a full-stack {keywords[0]} application with all features",
    ]
    
    # Template 2: Feature-focused
    if len(keywords) > 1:
        prompt_templates.extend([
            f"Build an app for {keywords[1]} with {keywords[0]} features",
            f"Create a platform that handles {keywords[1]} and {keywords[2] if len(keywords) > 2 else keywords[0]}",
        ])
    
    # Template 3: Microservice mentions
    if len(microservices) >= 3:
        sample_services = random.sample(microservices, min(3, len(microservices)))
        service_desc = ", ".join([s.replace('-service', '').replace('-', ' ') for s in sample_services])
        prompt_templates.append(
            f"Build a {keywords[0]} system with {service_desc} functionality"
        )
    
    # Template 4: Tech stack mentions (occasionally include specific tech)
    if random.random() > 0.5 and 'react' in tech_list:
        prompt_templates.append(
            f"Build a React-based {keywords[0]} application with backend services"
        )
    
    # Generate examples (2-4 per pattern)
    num_examples = random.randint(2, 4)
    for i in range(min(num_examples, len(prompt_templates))):
        prompt = prompt_templates[i]
        
        # Add some variation to hours/complexity
        variation = random.uniform(0.85, 1.15)
        example = {
            "text": prompt,
            "is_software": True,
            "technologies": tech_list,
            "loc": int(loc * variation),
            "hours": round(base_hours * variation, 1),
            "complexity_score": int(complexity_score * variation),
            "source": f"system_design_pattern:{pattern_name}",
            "microservices": microservices
        }
        examples.append(example)
    
    return examples


def generate_negative_examples() -> List[Dict[str, Any]]:
    """Generate negative examples (non-software)."""
    negative_prompts = [
        "Need a plumber to fix kitchen sink",
        "Looking for experienced nanny for 2 children",
        "Hire carpenter for custom furniture",
        "Need electrician to install new outlets",
        "Looking for personal trainer with nutrition expertise",
        "Hire photographer for wedding event",
        "Need landscaper for garden maintenance",
        "Looking for tutor for high school math",
        "Hire cleaner for weekly house cleaning",
        "Need mechanic to repair car transmission",
        "Looking for chef for private dinner party",
        "Hire accountant for tax preparation",
        "Need lawyer for business contract review",
        "Looking for real estate agent to sell house",
        "Hire interior designer for home renovation",
    ]
    
    examples = []
    for prompt in negative_prompts:
        examples.append({
            "text": prompt,
            "is_software": False
        })
    
    return examples


def generate_simple_software_examples() -> List[Dict[str, Any]]:
    """Generate simple software examples (low complexity)."""
    simple_examples = [
        {
            "text": "Build a simple todo list app with React",
            "is_software": True,
            "technologies": ["react"],
            "loc": 500,
            "hours": 25.0,
            "complexity_score": 45
        },
        {
            "text": "Create a blog with markdown support",
            "is_software": True,
            "technologies": ["nextjs", "markdown"],
            "loc": 800,
            "hours": 40.0,
            "complexity_score": 55
        },
        {
            "text": "Build a REST API with FastAPI and Postgres",
            "is_software": True,
            "technologies": ["python_fastapi", "postgres"],
            "loc": 1200,
            "hours": 60.0,
            "complexity_score": 70
        },
        {
            "text": "Create a portfolio website with contact form",
            "is_software": True,
            "technologies": ["react", "node"],
            "loc": 600,
            "hours": 30.0,
            "complexity_score": 50
        },
        {
            "text": "Build an authentication system with JWT",
            "is_software": True,
            "technologies": ["auth", "jwt", "node", "postgres"],
            "loc": 1500,
            "hours": 75.0,
            "complexity_score": 80
        },
    ]
    
    return simple_examples


def generate_training_dataset(output_path: str = None):
    """Generate comprehensive training dataset from patterns."""
    if output_path is None:
        output_path = Path(__file__).parent / 'data' / 'training_from_patterns.jsonl'
    
    print("Loading system design patterns...")
    patterns = load_system_patterns()
    
    all_examples = []
    
    # Generate examples from each application pattern
    print("\nGenerating examples from application patterns...")
    for pattern_name, pattern_data in patterns.get('application_patterns', {}).items():
        print(f"  - Processing {pattern_name}...")
        examples = generate_prompts_for_pattern(pattern_name, pattern_data)
        all_examples.extend(examples)
        print(f"    Generated {len(examples)} examples")
    
    # Add negative examples
    print("\nAdding negative examples (non-software)...")
    negative_examples = generate_negative_examples()
    all_examples.extend(negative_examples)
    print(f"  Added {len(negative_examples)} negative examples")
    
    # Add simple software examples
    print("\nAdding simple software examples...")
    simple_examples = generate_simple_software_examples()
    all_examples.extend(simple_examples)
    print(f"  Added {len(simple_examples)} simple examples")
    
    # Shuffle to mix positive and negative examples
    random.shuffle(all_examples)
    
    # Write to file
    print(f"\nWriting {len(all_examples)} examples to {output_path}...")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in all_examples:
            f.write(json.dumps(example, ensure_ascii=False) + '\n')
    
    # Print statistics
    print("\n" + "="*70)
    print("Dataset Statistics:")
    print("="*70)
    
    software_examples = [e for e in all_examples if e.get('is_software', False)]
    non_software_examples = [e for e in all_examples if not e.get('is_software', False)]
    
    print(f"Total examples: {len(all_examples)}")
    print(f"  - Software examples: {len(software_examples)}")
    print(f"  - Non-software examples: {len(non_software_examples)}")
    
    if software_examples:
        # Technology distribution
        tech_counts = {}
        for example in software_examples:
            for tech in example.get('technologies', []):
                tech_counts[tech] = tech_counts.get(tech, 0) + 1
        
        print(f"\nTop 10 technologies:")
        for tech, count in sorted(tech_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  - {tech}: {count} examples")
        
        # Complexity distribution
        avg_hours = sum(e.get('hours', 0) for e in software_examples) / len(software_examples)
        avg_loc = sum(e.get('loc', 0) for e in software_examples) / len(software_examples)
        avg_complexity = sum(e.get('complexity_score', 0) for e in software_examples) / len(software_examples)
        
        print(f"\nComplexity Averages:")
        print(f"  - Hours: {avg_hours:.1f}")
        print(f"  - LOC: {avg_loc:.0f}")
        print(f"  - Complexity Score: {avg_complexity:.1f}")
    
    print(f"\nDataset saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Review generated examples in the output file")
    print("  2. Merge with existing training data if desired:")
    print(f"     python merge_training_data.py --sources {output_path} data/software_training_data.jsonl --out data/merged_training_data.jsonl")
    print("  3. Train models with new dataset:")
    print("     python train_software_models.py --data data/merged_training_data.jsonl --out models/software")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate training data from system design patterns')
    parser.add_argument('--output', '-o', 
                       help='Output JSONL file path',
                       default='data/training_from_patterns.jsonl')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Generate dataset
    output_path = generate_training_dataset(args.output)
    
    print("\n✅ Done!")


if __name__ == '__main__':
    main()
