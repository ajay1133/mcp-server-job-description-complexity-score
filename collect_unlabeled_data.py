#!/usr/bin/env python3
"""Collect diverse UNLABELED text samples without keyword filtering.

This script intentionally avoids any hardcoded keyword heuristics. It produces a
pool of mixed software/non-software snippets for the active learning loop.

Data sources (placeholder implementations):
  - GitHub repository descriptions (public API)
  - Stack Overflow question titles/bodies
  - Reddit mixed subreddits (webdev + non-tech subs)
  - Generic job posting snippets

In production you should replace the placeholder fetch_* functions with real API
calls (and respect rate limits + auth tokens).

Output: data/unlabeled_pool.jsonl
Each line: { "text": "..." }

Usage:
  python collect_unlabeled_data.py --limit 200
"""

import os
import json
import argparse
from typing import List, Dict
import random

# ---------- Placeholder fetchers (to be replaced with real API calls) ---------- #

def fetch_github_repo_descriptions(limit: int) -> List[str]:
    samples = [
        "A modern web framework focusing on developer experience and performance",
        "High performance asynchronous networking library",
        "Cross-platform GUI toolkit for building desktop applications",
        "Declarative data visualization library",
        "Static site generator supporting MDX and image optimization",
    ]
    return random.sample(samples, min(limit, len(samples)))


def fetch_stackoverflow_questions(limit: int) -> List[str]:
    samples = [
        "How to optimize database queries in PostgreSQL?",
        "React component not re-rendering after state update",
        "FastAPI dependency injection for database sessions",
        "Best way to handle JWT refresh tokens in Node.js?",
        "Improving build times in large monorepo project",
    ]
    return random.sample(samples, min(limit, len(samples)))


def fetch_reddit_mixed(limit: int) -> List[str]:
    tech = [
        "Show HN: Minimal real-time chat built with WebSocket and Redis",
        "Deploying microservices with zero-downtime rolling updates",
        "Implementing vector search over embeddings for documents",
    ]
    non_tech = [
        "Need advice on caring for indoor tropical plants",
        "Weekly meal prep ideas for busy professionals",
        "Tips for organizing a small apartment effectively",
        "Best way to recover after an intense workout session",
    ]
    pool = tech + non_tech
    random.shuffle(pool)
    return pool[:limit]


def fetch_job_snippets(limit: int) -> List[str]:
    samples = [
        "Senior backend engineer responsible for designing scalable APIs",
        "Frontend developer building component libraries and design systems",
        "Caretaker needed for elderly assistance and companionship",
        "Housekeeper required for weekly cleaning and laundry",
        "Looking for fitness trainer to build custom workout plan",
    ]
    return random.sample(samples, min(limit, len(samples)))

# ----------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description='Collect unlabeled mixed-domain text for active learning.')
    ap.add_argument('--limit', type=int, default=200, help='Approximate total number of unlabeled samples')
    ap.add_argument('--out', default='data/unlabeled_pool.jsonl', help='Output JSONL file path')
    args = ap.parse_args()

    target_each = max(5, args.limit // 4)

    github_texts = fetch_github_repo_descriptions(target_each)
    so_texts = fetch_stackoverflow_questions(target_each)
    reddit_texts = fetch_reddit_mixed(target_each)
    job_texts = fetch_job_snippets(target_each)

    combined: List[str] = github_texts + so_texts + reddit_texts + job_texts
    random.shuffle(combined)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for t in combined:
            f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n')

    print(f'Collected {len(combined)} unlabeled samples -> {args.out}')
    print('Next: python active_learning_loop.py')


if __name__ == '__main__':
    main()
