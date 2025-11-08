# Hiring vs Build Classifier: Complete Workflow

This document demonstrates the end-to-end workflow for training, evaluating, and improving the hiring vs build binary classifier.

## Quick Start

### 1. Initial Training (Minimal Dataset)

Start with the example dataset (20 examples):

```powershell
python train_hiring_classifier.py --data data/hiring_build_training_data.example.jsonl --out models/software/hiring_build_classifier.joblib
```

**Output:**
- Warning about small dataset size (20 < 50 recommended)
- Classification report on test set
- ROC AUC score
- Model saved to `models/software/hiring_build_classifier.joblib`

### 2. Active Learning (Grow Dataset Efficiently)

Find uncertain examples where the model needs guidance:

```powershell
python active_learning_hiring.py --unlabeled data/unlabeled_prompts.txt --model models/software/hiring_build_classifier.joblib --limit 50 --out data/uncertain_samples.jsonl
```

**Output:**
- JSONL file with uncertain examples (probability 0.35–0.65)
- Each entry has `"model_proba"` and `"label": null`

**Manual step:** Edit `data/uncertain_samples.jsonl` and add labels:
- `"label": 1` for hiring/job descriptions
- `"label": 0` for build/implementation requirements

### 3. Merge and Retrain

Combine labeled uncertain samples with existing training data:

```powershell
# Merge files
type data\hiring_build_training_data.jsonl data\uncertain_samples.jsonl > data\hiring_build_merged.jsonl

# Retrain
python train_hiring_classifier.py --data data/hiring_build_merged.jsonl --out models/software/hiring_build_classifier.joblib
```

### 4. Evaluate Performance

Compare model vs heuristic baseline:

```powershell
python evaluate_hiring_classifier.py --data data/hiring_build_merged.jsonl --test-size 0.2 --output-dir logs
```

**Output:**
- Precision/recall/F1 for model and heuristic
- ROC AUC and Average Precision scores
- `logs/hiring_classifier_pr_curve.png` (Precision-Recall curve)
- `logs/hiring_classifier_roc_curve.png` (ROC curve)
- `logs/hiring_classifier_evaluation.json` (metrics summary)
- Recommended thresholds (cost-optimal and F1-optimal)

### 5. Tune Decision Threshold

Optimize for your production cost function:

```powershell
# Example: False negatives (hiring → build) are 2x worse than false positives
python tune_hiring_threshold.py --data data/hiring_build_merged.jsonl --cost-fp 1.0 --cost-fn 2.0 --write-config
```

**Output:**
- Optimal threshold minimizing weighted cost
- Confusion matrix at optimal threshold
- Comparison to default threshold (0.5)
- `config/hiring_threshold.json` (if --write-config used)
- `logs/hiring_threshold_tuning.json` (detailed results)

**Integration:** Update `_predict_is_hiring` in `software_complexity_scorer.py`:

```python
if proba >= 0.72:  # use your tuned threshold instead of 0.65
    return True, proba, "model"
if proba <= 0.28:  # use 1 - threshold instead of 0.35
    return False, proba, "model"
```

## Iterative Improvement Cycle

```
┌─────────────────────────────────────────────────┐
│ 1. Train initial model (~100 examples)         │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 2. Evaluate: precision/recall/AUC/AP            │
│    - Compare vs heuristic baseline              │
│    - Identify weak areas (confusion matrix)     │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 3. Active learning: label uncertain examples   │
│    - Model surfaces probabilities 0.35–0.65     │
│    - Manual annotation (fastest ROI)            │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 4. Merge labeled data and retrain              │
│    - Incremental dataset growth                 │
│    - Re-evaluate performance                    │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│ 5. Tune threshold for production cost function │
│    - Minimize misclassification cost            │
│    - Deploy updated thresholds                  │
└──────────────────┬──────────────────────────────┘
                   │
                   │ Repeat until targets met ──────┐
                   └────────────────────────────────┘
```

## Target Metrics

Recommended performance targets for production:

| Metric | Minimum | Good | Excellent |
|--------|---------|------|-----------|
| ROC AUC | 0.85 | 0.90 | 0.95+ |
| Average Precision | 0.80 | 0.88 | 0.95+ |
| Precision (hiring) | 0.85 | 0.90 | 0.95+ |
| Recall (hiring) | 0.80 | 0.90 | 0.95+ |
| F1 Score | 0.82 | 0.90 | 0.95+ |

**Dataset size recommendations:**
- Minimum: 100 examples (50 hiring, 50 build)
- Good: 500 examples (balanced)
- Production: 1,000+ examples (balanced, diverse edge cases)

## Cost Function Guidance

Choose `cost_fp` and `cost_fn` based on downstream impact:

### Scenario 1: False negatives are worse (default)
**Use case:** Missing a hiring prompt causes wrong schema (no skills score, includes time estimates)
```powershell
--cost-fp 1.0 --cost-fn 2.0
```

### Scenario 2: False positives are worse
**Use case:** Build requirements wrongly flagged as hiring disrupt workflow
```powershell
--cost-fp 3.0 --cost-fn 1.0
```

### Scenario 3: Balanced
**Use case:** Both errors equally problematic
```powershell
--cost-fp 1.0 --cost-fn 1.0
```

## Example Results

From the demo run with 20 examples:

```
Test set: 4 examples

=== Heuristic Baseline ===
              precision    recall  f1-score   support
       build       1.00      0.50      0.67         2
      hiring       0.67      1.00      0.80         2

=== Trained Model (threshold=0.5) ===
              precision    recall  f1-score   support
       build       1.00      0.50      0.67         2
      hiring       0.67      1.00      0.80         2

ROC AUC: 0.750

=== Active Learning ===
Found 10 uncertain examples (probability 0.35–0.65):
- "Need a full-stack developer who knows Django..." (0.496)
- "Refactor monolith to microservices..." (0.487)
- "Migrate database from MySQL to PostgreSQL" (0.487)
- ...
```

## Troubleshooting

**Q: Model performance is same as heuristic**
- Need more training data (< 100 examples likely insufficient)
- Add diverse examples covering edge cases
- Check for label noise (review annotations)

**Q: Active learning finds no uncertain examples**
- Model is confident (good!) or overfit (bad)
- Lower/widen probability bounds: `--lower 0.25 --upper 0.75`
- Check if unlabeled pool is diverse enough

**Q: High AUC but poor precision/recall at default threshold**
- Threshold needs tuning for your cost function
- Use `tune_hiring_threshold.py` to find optimal cutoff

**Q: Precision high but recall low**
- Model is conservative (many false negatives)
- Lower hiring threshold (e.g., 0.60 instead of 0.65)
- Add more positive examples to training data

**Q: Recall high but precision low**
- Model is aggressive (many false positives)
- Raise hiring threshold (e.g., 0.75 instead of 0.65)
- Add more negative examples or hard negatives

## Files Created

```
models/software/hiring_build_classifier.joblib     # Trained model bundle
data/uncertain_samples.jsonl                        # Uncertain examples for labeling
data/hiring_build_merged.jsonl                      # Expanded training data
config/hiring_threshold.json                        # Tuned thresholds
logs/hiring_classifier_pr_curve.png                 # Precision-Recall curve
logs/hiring_classifier_roc_curve.png                # ROC curve
logs/hiring_classifier_evaluation.json              # Metrics summary
logs/hiring_threshold_tuning.json                   # Threshold tuning results
```

## Next Steps

1. **Collect production examples:** Log real prompts with human labels
2. **Monitor drift:** Track model confidence distribution over time
3. **A/B test thresholds:** Compare cost-optimal vs F1-optimal in production
4. **Error analysis:** Review false positives/negatives to identify patterns
5. **Expand ontology:** Add multi-class labels if needed (e.g., hiring_senior, hiring_junior, build_simple, build_complex)
