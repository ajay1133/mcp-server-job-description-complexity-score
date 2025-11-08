#!/usr/bin/env python3
"""Bootstrap training data from real-world sources without hardcoded keywords.

Approach:
1. Scrape diverse text from GitHub, Stack Overflow, Reddit, job boards
2. Use a small seed set for initial binary classification (software vs non-software)
3. Active learning: model identifies uncertain examples for manual labeling
4. Iteratively improve with human-in-the-loop feedback
5. Extract technologies, LOC, time from actual project data (GitHub repos)

This avoids hardcoded keywords while building a robust training set.
"""

import json
import os
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import requests
from urllib.parse import quote_plus
import time
import random


@dataclass
class TrainingExample:
    """A training example without hardcoded assumptions."""
    text: str
    is_software: Optional[bool] = None  # None = unlabeled
    technologies: Optional[List[str]] = None
    loc: Optional[int] = None
    hours: Optional[float] = None
    complexity_score: Optional[int] = None
    source: str = "unknown"
    confidence: float = 0.0  # For active learning
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dict, excluding None and metadata fields."""
        result = {}
        if self.text:
            result['text'] = self.text
        if self.is_software is not None:
            result['is_software'] = self.is_software
        if self.technologies:
            result['technologies'] = self.technologies
        if self.loc is not None:
            result['loc'] = self.loc
        if self.hours is not None:
            result['hours'] = self.hours
        if self.complexity_score is not None:
            result['complexity_score'] = self.complexity_score
        return result


class BootstrapDataCollector:
    """Collect diverse data from real sources without keyword filtering."""
    
    def __init__(self):
        self.examples = []
        
    def collect_from_github_repos(self, limit: int = 50) -> List[TrainingExample]:
        """
        Collect project descriptions from GitHub repos.
        Uses GitHub's trending/popular repos for diversity.
        """
        examples = []
        
        # Get diverse repos from different languages and topics
        topics = ['web', 'mobile', 'machine-learning', 'data-science', 'devops', 
                  'database', 'api', 'cli', 'desktop', 'game']
        
        for topic in topics[:5]:  # Limit API calls
            print(f"Fetching GitHub repos for topic: {topic}")
            try:
                # Note: In real implementation, use GitHub API token for higher rate limits
                url = f"https://api.github.com/search/repositories"
                params = {
                    'q': f'topic:{topic} stars:>100',
                    'sort': 'stars',
                    'per_page': 10
                }
                
                # Add delay to respect rate limits
                time.sleep(1)
                
                # In production, you'd make the actual API call here
                # response = requests.get(url, params=params)
                # For now, we'll create placeholder structure
                
                example = TrainingExample(
                    text=f"Project description for {topic}-related repository",
                    source=f"github_topic_{topic}",
                    confidence=0.0
                )
                examples.append(example)
                
            except Exception as e:
                print(f"Error fetching GitHub data: {e}")
                
        return examples
    
    def collect_from_reddit(self, limit: int = 50) -> List[TrainingExample]:
        """
        Collect diverse posts from subreddits without filtering.
        Mix of software and non-software subs.
        """
        examples = []
        
        # Mix of tech and non-tech subreddits for diversity
        subreddits = [
            'webdev', 'programming', 'MachineLearning',  # Tech
            'cooking', 'fitness', 'HomeImprovement',     # Non-tech
            'freelance', 'entrepreneur', 'smallbusiness'  # Mixed
        ]
        
        for sub in subreddits:
            print(f"Collecting posts from r/{sub}")
            # In production: use Reddit API (PRAW)
            # For now: placeholder
            example = TrainingExample(
                text=f"Sample post from r/{sub}",
                source=f"reddit_{sub}",
                confidence=0.0
            )
            examples.append(example)
            
        return examples
    
    def collect_from_stackoverflow(self, limit: int = 50) -> List[TrainingExample]:
        """
        Collect question titles from Stack Overflow.
        These are naturally software-related but diverse in complexity.
        """
        examples = []
        
        print("Fetching Stack Overflow questions")
        try:
            # Stack Exchange API (no auth needed for basic queries)
            url = "https://api.stackexchange.com/2.3/questions"
            params = {
                'order': 'desc',
                'sort': 'votes',
                'site': 'stackoverflow',
                'pagesize': limit,
                'filter': 'withbody'  # Get question body too
            }
            
            # Add delay for rate limiting
            time.sleep(1)
            
            # In production, make actual API call
            # response = requests.get(url, params=params)
            
            # Placeholder
            example = TrainingExample(
                text="Sample Stack Overflow question",
                source="stackoverflow",
                confidence=0.0
            )
            examples.append(example)
            
        except Exception as e:
            print(f"Error fetching Stack Overflow data: {e}")
            
        return examples
    
    def collect_from_job_postings(self, limit: int = 100) -> List[TrainingExample]:
        """
        Collect job postings from public APIs (Indeed, RemoteOK, etc.).
        Natural mix of software and non-software jobs.
        """
        examples = []
        
        print("Fetching job postings")
        # In production: use Indeed API, RemoteOK API, etc.
        # These provide diverse real-world job descriptions
        
        # Placeholder for demonstration
        example = TrainingExample(
            text="Sample job posting description",
            source="job_posting",
            confidence=0.0
        )
        examples.append(example)
        
        return examples
    
    def create_minimal_seed_set(self) -> List[TrainingExample]:
        """
        Create a minimal seed set with VERY clear examples.
        Only ~20-30 examples needed to bootstrap.
        """
        seed_examples = []
        
        # Clear software examples (10)
        software_seeds = [
            "Build a web application with user login",
            "Create an API that processes data",
            "Develop mobile app for tracking tasks",
            "Write Python script to analyze logs",
            "Implement database schema for storing products",
            "Create dashboard to visualize metrics",
            "Build Chrome extension for bookmarking",
            "Develop CLI tool for file conversion",
            "Write automation script for testing",
            "Create website with contact form"
        ]
        
        # Clear non-software examples (10)
        non_software_seeds = [
            "Clean my house every week",
            "Fix broken kitchen sink",
            "Paint bedroom walls",
            "Walk my dog twice daily",
            "Tutor my child in math",
            "Cook meals for family",
            "Drive me to airport",
            "Mow lawn and trim hedges",
            "Plan wedding ceremony",
            "Take care of elderly parent"
        ]
        
        for text in software_seeds:
            seed_examples.append(TrainingExample(
                text=text,
                is_software=True,
                source="seed_set",
                confidence=1.0
            ))
        
        for text in non_software_seeds:
            seed_examples.append(TrainingExample(
                text=text,
                is_software=False,
                source="seed_set",
                confidence=1.0
            ))
        
        return seed_examples


class ActiveLearningLabeler:
    """Interactive labeling for uncertain examples."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        if model_path and os.path.exists(model_path):
            # Load existing model if available
            import joblib
            self.model = joblib.load(model_path)
    
    def predict_with_uncertainty(self, examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Predict is_software with confidence scores.
        Examples with low confidence (0.4-0.6) need manual review.
        """
        if not self.model:
            # No model yet, return all as uncertain
            for ex in examples:
                ex.confidence = 0.5
            return examples
        
        # In production: use model to predict probabilities
        # For now: placeholder
        for ex in examples:
            ex.confidence = random.random()  # Simulate confidence
            if ex.confidence > 0.6:
                ex.is_software = True
            elif ex.confidence < 0.4:
                ex.is_software = False
            # else: leave as None (uncertain, needs labeling)
        
        return examples
    
    def interactive_labeling(self, uncertain_examples: List[TrainingExample]) -> List[TrainingExample]:
        """
        Present uncertain examples to user for labeling.
        """
        labeled = []
        
        print("\n" + "=" * 70)
        print("ACTIVE LEARNING: Label Uncertain Examples")
        print("=" * 70)
        print("Please review examples where the model is uncertain.\n")
        
        for i, ex in enumerate(uncertain_examples, 1):
            if ex.is_software is not None:
                continue  # Already labeled
            
            print(f"\n[{i}/{len(uncertain_examples)}] Confidence: {ex.confidence:.2f}")
            print(f"Text: {ex.text[:200]}...")
            print("\nIs this a SOFTWARE development task?")
            print("  1) Yes (software)")
            print("  2) No (not software)")
            print("  3) Skip")
            
            choice = input("Your choice (1/2/3): ").strip()
            
            if choice == '1':
                ex.is_software = True
                ex.confidence = 1.0
                labeled.append(ex)
            elif choice == '2':
                ex.is_software = False
                ex.confidence = 1.0
                labeled.append(ex)
            else:
                print("Skipped")
        
        return labeled


class GitHubRepoAnalyzer:
    """Analyze actual GitHub repositories to extract LOC, tech stack, etc."""
    
    def analyze_repo(self, repo_url: str) -> Dict[str, Any]:
        """
        Analyze a repo to extract real metrics.
        Returns: technologies used, LOC, estimated hours (from commits).
        """
        # In production:
        # 1. Clone repo (or use GitHub API)
        # 2. Detect languages (GitHub API provides this)
        # 3. Count LOC using tools like cloc
        # 4. Analyze package.json, requirements.txt, etc. for dependencies
        # 5. Estimate hours from commit history
        
        return {
            'technologies': [],
            'loc': 0,
            'estimated_hours': 0.0,
            'languages': {}
        }


def bootstrap_approach():
    """
    Main bootstrap workflow:
    1. Create tiny seed set (20 examples)
    2. Collect diverse unlabeled data from web
    3. Train initial model on seed
    4. Use active learning to label uncertain cases
    5. Retrain and repeat
    """
    
    print("\n" + "=" * 70)
    print("BOOTSTRAP TRAINING DATA - NO HARDCODED KEYWORDS")
    print("=" * 70)
    print("\nThis approach:")
    print("  ✓ Uses minimal seed examples (20)")
    print("  ✓ Collects diverse real-world data")
    print("  ✓ Active learning for uncertain cases")
    print("  ✓ Analyzes real GitHub repos for metrics")
    print("  ✓ No hardcoded keywords or templates")
    print("=" * 70 + "\n")
    
    collector = BootstrapDataCollector()
    
    # Step 1: Create minimal seed set
    print("Step 1: Creating minimal seed set...")
    seed_examples = collector.create_minimal_seed_set()
    print(f"Created {len(seed_examples)} seed examples\n")
    
    # Step 2: Collect diverse unlabeled data
    print("Step 2: Collecting diverse data from web sources...")
    unlabeled = []
    
    # In production, you'd actually call these:
    # unlabeled.extend(collector.collect_from_github_repos(50))
    # unlabeled.extend(collector.collect_from_reddit(50))
    # unlabeled.extend(collector.collect_from_stackoverflow(50))
    # unlabeled.extend(collector.collect_from_job_postings(100))
    
    print(f"Collected {len(unlabeled)} unlabeled examples (placeholder)\n")
    
    # Step 3: Train initial model on seed
    print("Step 3: Training initial model on seed set...")
    print("(In production: train sklearn model on 20 seed examples)")
    
    # Step 4: Active learning
    print("\nStep 4: Active learning loop...")
    print("(In production: predict on unlabeled, ask user to label uncertain ones)")
    
    # Step 5: Save bootstrapped dataset
    output_file = "data/bootstrapped_training_data.jsonl"
    os.makedirs("data", exist_ok=True)
    
    all_examples = seed_examples + unlabeled
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for ex in all_examples:
            if ex.is_software is not None:  # Only save labeled examples
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + '\n')
    
    print(f"\n{'=' * 70}")
    print(f"Bootstrap Complete!")
    print(f"{'=' * 70}")
    print(f"Saved seed set to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Implement actual API calls to data sources")
    print(f"2. Train initial model: python train_software_models.py")
    print(f"3. Run active learning: python active_learning_loop.py")
    print(f"4. Analyze GitHub repos for real LOC/tech data")
    print(f"5. Iteratively improve with more labeling")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    bootstrap_approach()
