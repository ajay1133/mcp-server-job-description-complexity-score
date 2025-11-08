#!/usr/bin/env python3
"""Tune optimal decision threshold for hiring vs build classifier.

Finds the threshold that minimizes a weighted misclassification cost.
Allows you to specify different penalties for false positives vs false negatives.

Usage (PowerShell):
  # Default: FP cost=1, FN cost=2 (false negatives are twice as bad)
  python tune_hiring_threshold.py --data data/hiring_build_training_data.jsonl

  # Custom costs: FP very expensive (build wrongly flagged as hiring disrupts flow)
  python tune_hiring_threshold.py --data data/hiring_build_training_data.jsonl --cost-fp 3.0 --cost-fn 1.0

  # Output updated config for scorer integration
  python tune_hiring_threshold.py --data data/hiring_build_training_data.jsonl --write-config
"""
import argparse
import os
import sys
import json
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mcp_server.hiring_classifier import load_dataset, HiringBuildClassifier


def tune_threshold(y_true, y_proba, cost_fp=1.0, cost_fn=2.0, granularity=100):
    """Grid search over thresholds to minimize cost.
    
    Returns dict with optimal threshold and diagnostic info.
    """
    thresholds = np.linspace(0, 1, granularity)
    results = []
    
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        cm = confusion_matrix(y_true, preds)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            # Edge case: all predictions same class
            if len(np.unique(preds)) == 1:
                if preds[0] == 1:
                    tn, fp, fn, tp = 0, sum(y_true == 0), 0, sum(y_true == 1)
                else:
                    tn, fp, fn, tp = sum(y_true == 0), 0, sum(y_true == 1), 0
            else:
                continue
        
        cost = cost_fp * fp + cost_fn * fn
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results.append({
            'threshold': t,
            'cost': cost,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    # Find minimum cost
    min_idx = min(range(len(results)), key=lambda i: results[i]['cost'])
    return results[min_idx], results


def main():
    parser = argparse.ArgumentParser(description="Tune decision threshold for hiring classifier")
    parser.add_argument('--data', type=str, default=os.getenv('HIRING_BUILD_DATA'),
                        help='Path to labeled JSONL dataset')
    parser.add_argument('--model', type=str,
                        default=os.path.join('models', 'software', 'hiring_build_classifier.joblib'),
                        help='Path to trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction for test set')
    parser.add_argument('--cost-fp', type=float, default=1.0,
                        help='Cost of false positive (build → hiring)')
    parser.add_argument('--cost-fn', type=float, default=2.0,
                        help='Cost of false negative (hiring → build)')
    parser.add_argument('--granularity', type=int, default=100,
                        help='Number of threshold steps to evaluate')
    parser.add_argument('--write-config', action='store_true',
                        help='Write recommended threshold to config/hiring_threshold.json')
    parser.add_argument('--output-dir', type=str, default='logs',
                        help='Directory for detailed results')
    args = parser.parse_args()

    if not args.data or not os.path.isfile(args.data):
        raise SystemExit(f"Dataset not found: {args.data}")
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found: {args.model}")

    texts, labels = load_dataset(args.data)
    print(f"Loaded {len(texts)} examples")
    
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=args.test_size,
                                            random_state=42, stratify=labels)
    print(f"Test set: {len(X_test)} examples\n")

    clf = HiringBuildClassifier(model_path=args.model)
    y_proba = np.array([clf.predict_proba(t) for t in X_test])

    print(f"Tuning threshold (FP cost={args.cost_fp}, FN cost={args.cost_fn})...")
    optimal, all_results = tune_threshold(y_test, y_proba, args.cost_fp, args.cost_fn, args.granularity)

    print("\n=== Optimal Threshold ===")
    print(f"Threshold: {optimal['threshold']:.3f}")
    print(f"Total Cost: {optimal['cost']:.1f}")
    print(f"  TP: {optimal['tp']}, TN: {optimal['tn']}, FP: {optimal['fp']}, FN: {optimal['fn']}")
    print(f"  Precision: {optimal['precision']:.3f}")
    print(f"  Recall: {optimal['recall']:.3f}")
    print(f"  F1: {optimal['f1']:.3f}")

    # Compare to default 0.5
    default_res = [r for r in all_results if abs(r['threshold'] - 0.5) < 0.01]
    if default_res:
        default = default_res[0]
        print(f"\nDefault threshold (0.5) cost: {default['cost']:.1f}")
        print(f"  Improvement: {default['cost'] - optimal['cost']:.1f} cost units")

    # Save detailed results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, 'hiring_threshold_tuning.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'optimal': optimal,
            'cost_fp': args.cost_fp,
            'cost_fn': args.cost_fn,
            'all_thresholds': all_results
        }, f, indent=2)
    print(f"\nSaved detailed results to {results_path}")

    # Write config if requested
    if args.write_config:
        config_dir = 'config'
        os.makedirs(config_dir, exist_ok=True)
        config_path = os.path.join(config_dir, 'hiring_threshold.json')
        config = {
            'threshold_hiring': optimal['threshold'],
            'threshold_build': 1.0 - optimal['threshold'],
            'note': f"Optimized for FP cost={args.cost_fp}, FN cost={args.cost_fn}",
            'date_tuned': '2025-11-08'
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        print(f"Wrote config to {config_path}")
        print("\nTo use this threshold in the scorer, update _predict_is_hiring method:")
        print(f"  if proba >= {optimal['threshold']:.3f}: return True, ...")


if __name__ == '__main__':
    main()
