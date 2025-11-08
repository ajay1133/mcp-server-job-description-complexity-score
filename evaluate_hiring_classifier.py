#!/usr/bin/env python3
"""Evaluate hiring vs build classifier against heuristic baseline.

Computes precision, recall, F1, ROC AUC for the trained model and heuristic,
plots PR curve, and recommends optimal threshold for production use.

Usage (PowerShell):
  $env:HIRING_BUILD_DATA="data/hiring_build_training_data.jsonl"
  python evaluate_hiring_classifier.py --test-size 0.2

Or specify paths:
  python evaluate_hiring_classifier.py --data data/hiring_build_training_data.jsonl --model models/software/hiring_build_classifier.joblib --test-size 0.3
"""
import argparse
import os
import sys
import json
import numpy as np
from sklearn.metrics import (
    classification_report, roc_auc_score, precision_recall_curve,
    average_precision_score, confusion_matrix, roc_curve
)
from sklearn.model_selection import train_test_split

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from mcp_server.hiring_classifier import load_dataset
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer


def evaluate_heuristic(texts, labels):
    """Evaluate the existing heuristic baseline."""
    from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
    # Use static method directly
    preds = [int(SoftwareComplexityScorer._is_hiring_requirement(t)) for t in texts]
    return preds


def evaluate_model(clf, texts):
    """Evaluate trained classifier probabilities."""
    probs = [clf.predict_proba(t) for t in texts]
    return np.array(probs)


def plot_curves(y_true, y_proba_model, y_proba_heuristic=None, output_dir='logs'):
    """Plot PR and ROC curves."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots.")
        return

    os.makedirs(output_dir, exist_ok=True)

    # PR curve
    precision_m, recall_m, thresholds_m = precision_recall_curve(y_true, y_proba_model)
    ap_m = average_precision_score(y_true, y_proba_model)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall_m, precision_m, label=f'Model (AP={ap_m:.3f})', linewidth=2)
    
    if y_proba_heuristic is not None:
        precision_h, recall_h, _ = precision_recall_curve(y_true, y_proba_heuristic)
        ap_h = average_precision_score(y_true, y_proba_heuristic)
        plt.plot(recall_h, precision_h, label=f'Heuristic (AP={ap_h:.3f})', linestyle='--', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    pr_path = os.path.join(output_dir, 'hiring_classifier_pr_curve.png')
    plt.savefig(pr_path, dpi=150)
    print(f"Saved PR curve to {pr_path}")
    plt.close()

    # ROC curve
    fpr_m, tpr_m, _ = roc_curve(y_true, y_proba_model)
    auc_m = roc_auc_score(y_true, y_proba_model)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_m, tpr_m, label=f'Model (AUC={auc_m:.3f})', linewidth=2)
    
    if y_proba_heuristic is not None:
        fpr_h, tpr_h, _ = roc_curve(y_true, y_proba_heuristic)
        auc_h = roc_auc_score(y_true, y_proba_heuristic)
        plt.plot(fpr_h, tpr_h, label=f'Heuristic (AUC={auc_h:.3f})', linestyle='--', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(alpha=0.3)
    roc_path = os.path.join(output_dir, 'hiring_classifier_roc_curve.png')
    plt.savefig(roc_path, dpi=150)
    print(f"Saved ROC curve to {roc_path}")
    plt.close()


def recommend_threshold(y_true, y_proba, cost_fp=1.0, cost_fn=2.0):
    """Recommend optimal threshold minimizing misclassification cost.
    
    cost_fp: cost of false positive (build classified as hiring)
    cost_fn: cost of false negative (hiring classified as build)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Append dummy threshold for last point
    thresholds = np.append(thresholds, 1.0)
    
    costs = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        cost = cost_fp * fp + cost_fn * fn
        costs.append(cost)
    
    min_idx = np.argmin(costs)
    optimal_t = thresholds[min_idx]
    min_cost = costs[min_idx]
    
    # Also report F1-optimal threshold
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
    f1_max_idx = np.argmax(f1_scores)
    f1_optimal_t = thresholds[f1_max_idx]
    
    return {
        'cost_optimal': optimal_t,
        'cost_optimal_cost': min_cost,
        'f1_optimal': f1_optimal_t,
        'f1_optimal_score': f1_scores[f1_max_idx]
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.getenv('HIRING_BUILD_DATA'),
                        help='Path to labeled JSONL dataset')
    parser.add_argument('--model', type=str,
                        default=os.path.join('models', 'software', 'hiring_build_classifier.joblib'),
                        help='Path to trained model bundle')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of data to use for test set')
    parser.add_argument('--cost-fp', type=float, default=1.0,
                        help='Cost of false positive (build → hiring)')
    parser.add_argument('--cost-fn', type=float, default=2.0,
                        help='Cost of false negative (hiring → build)')
    parser.add_argument('--output-dir', type=str, default='logs',
                        help='Directory for plots and reports')
    args = parser.parse_args()

    if not args.data or not os.path.isfile(args.data):
        raise SystemExit(f"Dataset not found: {args.data}. Set HIRING_BUILD_DATA or use --data.")
    if not os.path.isfile(args.model):
        raise SystemExit(f"Model not found: {args.model}. Train it first with train_hiring_classifier.py.")

    texts, labels = load_dataset(args.data)
    print(f"Loaded {len(texts)} examples from {args.data}")
    print(f"Class distribution: {sum(labels)} hiring, {len(labels)-sum(labels)} build\n")

    # Split
    _, X_test, _, y_test = train_test_split(texts, labels, test_size=args.test_size,
                                            random_state=42, stratify=labels)
    print(f"Test set: {len(X_test)} examples\n")

    # Load model
    from mcp_server.hiring_classifier import HiringBuildClassifier
    clf = HiringBuildClassifier(model_path=args.model)

    # Evaluate heuristic
    print("=== Heuristic Baseline ===")
    y_pred_heuristic = evaluate_heuristic(X_test, y_test)
    print(classification_report(y_test, y_pred_heuristic, target_names=['build', 'hiring'], digits=3))
    
    # Heuristic doesn't produce probabilities; use binary predictions as proxy
    y_proba_heuristic = np.array(y_pred_heuristic, dtype=float)

    # Evaluate model
    print("\n=== Trained Model (threshold=0.5) ===")
    y_proba_model = evaluate_model(clf, X_test)
    y_pred_model = (y_proba_model >= 0.5).astype(int)
    print(classification_report(y_test, y_pred_model, target_names=['build', 'hiring'], digits=3))
    
    try:
        auc = roc_auc_score(y_test, y_proba_model)
        ap = average_precision_score(y_test, y_proba_model)
        print(f"\nROC AUC: {auc:.3f}")
        print(f"Average Precision: {ap:.3f}")
    except Exception as e:
        print(f"Could not compute AUC/AP: {e}")

    # Threshold tuning
    print("\n=== Threshold Tuning ===")
    rec = recommend_threshold(y_test, y_proba_model, args.cost_fp, args.cost_fn)
    print(f"Cost-optimal threshold (FP cost={args.cost_fp}, FN cost={args.cost_fn}): {rec['cost_optimal']:.3f}")
    print(f"  → Minimized cost: {rec['cost_optimal_cost']:.1f}")
    print(f"F1-optimal threshold: {rec['f1_optimal']:.3f}")
    print(f"  → F1 score: {rec['f1_optimal_score']:.3f}")

    # Evaluate at recommended threshold
    print(f"\n=== Model at cost-optimal threshold={rec['cost_optimal']:.3f} ===")
    y_pred_tuned = (y_proba_model >= rec['cost_optimal']).astype(int)
    print(classification_report(y_test, y_pred_tuned, target_names=['build', 'hiring'], digits=3))

    # Plot curves
    plot_curves(y_test, y_proba_model, y_proba_heuristic, args.output_dir)

    # Save recommendations
    report = {
        'dataset': args.data,
        'test_size': args.test_size,
        'n_test': len(y_test),
        'heuristic_accuracy': float(np.mean(np.array(y_pred_heuristic) == np.array(y_test))),
        'model_accuracy_default': float(np.mean(y_pred_model == np.array(y_test))),
        'model_auc': float(roc_auc_score(y_test, y_proba_model)),
        'model_ap': float(average_precision_score(y_test, y_proba_model)),
        'recommended_thresholds': {
            'cost_optimal': float(rec['cost_optimal']),
            'f1_optimal': float(rec['f1_optimal'])
        }
    }
    report_path = os.path.join(args.output_dir, 'hiring_classifier_evaluation.json')
    os.makedirs(args.output_dir, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to {report_path}")


if __name__ == '__main__':
    main()
