#!/usr/bin/env python3
"""Active learning sampler for hiring vs build classifier.

Surfaces examples where the model is uncertain (probability between 0.35-0.65)
for manual labeling. Helps efficiently grow the training dataset by focusing
on the decision boundary where the model needs more guidance.

Usage (PowerShell):
  # Sample uncertain examples from unlabeled pool
  python active_learning_hiring.py --unlabeled data/unlabeled_prompts.txt --model models/software/hiring_build_classifier.joblib --limit 50 --out data/uncertain_samples.jsonl

  # After manual labeling, merge back into training data
  # Edit data/uncertain_samples.jsonl to add "label": 0 or 1
  # Then: cat data/hiring_build_training_data.jsonl data/uncertain_samples.jsonl > data/merged.jsonl
"""
import argparse
import os
import sys
import json
import numpy as np
from typing import List, Tuple

# Add project root
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mcp_server.hiring_classifier import HiringBuildClassifier


def load_unlabeled_texts(path: str) -> List[str]:
    """Load unlabeled texts from a file (one per line or JSONL with 'text' field)."""
    texts = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Try JSON first
            try:
                obj = json.loads(line)
                texts.append(obj.get('text', ''))
            except json.JSONDecodeError:
                # Plain text
                texts.append(line)
    return [t for t in texts if t]


def sample_uncertain(clf: HiringBuildClassifier, texts: List[str],
                     lower: float = 0.35, upper: float = 0.65,
                     limit: int = 50) -> List[Tuple[str, float]]:
    """Return texts with predicted probability in [lower, upper] range."""
    uncertain = []
    for t in texts:
        p = clf.predict_proba(t)
        if lower <= p <= upper:
            uncertain.append((t, p))
    
    # Sort by proximity to 0.5 (most uncertain first)
    uncertain.sort(key=lambda x: abs(x[1] - 0.5))
    return uncertain[:limit]


def main():
    parser = argparse.ArgumentParser(description="Active learning sampler for hiring classifier")
    parser.add_argument('--unlabeled', type=str, required=True,
                        help='Path to unlabeled text file (one prompt per line or JSONL with "text" field)')
    parser.add_argument('--model', type=str,
                        default=os.path.join('models', 'software', 'hiring_build_classifier.joblib'),
                        help='Path to trained model')
    parser.add_argument('--lower', type=float, default=0.35,
                        help='Lower probability bound for uncertainty')
    parser.add_argument('--upper', type=float, default=0.65,
                        help='Upper probability bound for uncertainty')
    parser.add_argument('--limit', type=int, default=50,
                        help='Maximum number of uncertain samples to return')
    parser.add_argument('--out', type=str, default='data/uncertain_samples.jsonl',
                        help='Output JSONL file with uncertain samples (no labels yet)')
    args = parser.parse_args()

    if not os.path.isfile(args.unlabeled):
        raise SystemExit(f"Unlabeled file not found: {args.unlabeled}")
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train it first with train_hiring_classifier.py.")

    clf = HiringBuildClassifier(model_path=args.model)
    texts = load_unlabeled_texts(args.unlabeled)
    print(f"Loaded {len(texts)} unlabeled examples from {args.unlabeled}")

    uncertain = sample_uncertain(clf, texts, args.lower, args.upper, args.limit)
    print(f"Found {len(uncertain)} uncertain examples (probability in [{args.lower}, {args.upper}])")

    if not uncertain:
        print("No uncertain samples found. Model is confident on all examples.")
        return

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for text, proba in uncertain:
            obj = {
                'text': text,
                'model_proba': round(proba, 3),
                'label': None,  # to be filled manually
                'note': 'label: 1 for hiring, 0 for build'
            }
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    
    print(f"\nWrote {len(uncertain)} uncertain samples to {args.out}")
    print("Next steps:")
    print("  1. Open the file and manually add 'label': 0 or 1 for each example")
    print("  2. Remove the 'note' field (optional)")
    print("  3. Merge labeled examples back into training data:")
    print(f"     type data\\hiring_build_training_data.jsonl {args.out} > data\\merged.jsonl")
    print("  4. Retrain: $env:HIRING_BUILD_DATA='data\\merged.jsonl'; python train_hiring_classifier.py")


if __name__ == '__main__':
    main()
