# Hiring vs Build Classifier Implementation Summary

## Overview

Successfully implemented an optional binary classifier to distinguish hiring/job descriptions from build/implementation requirements, enabling clean schema separation without manual phrase detection.

## Components Created

### Core Classifier Module
**File:** `mcp_server/hiring_classifier.py`
- `HiringBuildClassifier` dataclass: Model loader with safe probability prediction
- `load_dataset()`: JSONL parser for training data
- `train_and_save()`: Training pipeline (TF-IDF + LogisticRegression)
- Fallback to neutral probability (0.5) on errors

### Training Script
**File:** `train_hiring_classifier.py`
- CLI to train classifier from JSONL dataset
- Splits data, trains, reports metrics (precision/recall/F1/AUC)
- Saves model bundle to `models/software/hiring_build_classifier.joblib`
- Warning for small datasets (< 50 examples)

### Evaluation Script
**File:** `evaluate_hiring_classifier.py`
- Compares model vs heuristic baseline
- Plots PR curve and ROC curve (saves to logs/)
- Recommends optimal thresholds (cost-based and F1-based)
- Saves evaluation report JSON

### Active Learning Sampler
**File:** `active_learning_hiring.py`
- Surfaces uncertain examples (probability 0.35–0.65)
- Enables efficient dataset growth by focusing on decision boundary
- Outputs JSONL with model probabilities for manual labeling

### Threshold Tuning Utility
**File:** `tune_hiring_threshold.py`
- Grid search over thresholds to minimize misclassification cost
- Configurable costs for false positives vs false negatives
- Saves detailed results and optional config file
- Provides integration guidance for scorer

### Integration in Scorer
**File:** `mcp_server/software_complexity_scorer.py`
- Optional import of `HiringBuildClassifier` (no hard dependency)
- `_predict_is_hiring()`: Traced method with confidence thresholds
  - `>= 0.65`: hiring (high confidence)
  - `<= 0.35`: build (high confidence)
  - Between: falls back to heuristic
- Detection metadata injected into response: `hiring_detection: {source, proba}`
- Maintains backward compatibility (works without classifier)

### Dataset and Documentation
**Files:**
- `data/hiring_build_training_data.example.jsonl`: 20 labeled examples (10 hiring, 10 build)
- `data/unlabeled_prompts.txt`: 32 prompts for active learning demo
- `HIRING_CLASSIFIER_WORKFLOW.md`: Complete end-to-end workflow guide
- `README.md`: Updated with training instructions and workflow overview

### Unit Test
**File:** `tests/test_hiring_classifier_integration.py`
- Trains tiny model in tmpdir
- Asserts scorer correctly uses classifier
- Verifies hiring and build response schemas

## Architecture Decisions

### 1. Optional Dependency Pattern
```python
try:
    from .hiring_classifier import HiringBuildClassifier
except Exception:
    HiringBuildClassifier = None
```
Scorer degrades gracefully if classifier not trained or module import fails.

### 2. Confidence Thresholds with Heuristic Fallback
```python
if proba >= 0.65:
    return True, proba, "model"
if proba <= 0.35:
    return False, proba, "model"
# Low confidence → use existing heuristic
return self._is_hiring_requirement(t), proba, "model_low_conf+heuristic"
```
Combines ML confidence with rule-based safety net.

### 3. Tracing and Transparency
- `_predict_is_hiring` decorated with `@traced`
- Response includes `hiring_detection` metadata for debugging
- Per-request logs capture classifier probability and source

### 4. Cost-Based Threshold Tuning
Allows customization for production use cases:
- Healthcare/HR: FN cost > FP cost (missing hiring prompt = wrong schema)
- Developer tools: FP cost > FN cost (false alarm disrupts workflow)

## Workflow Summary

```
1. Train initial model (100+ examples)
   ↓
2. Evaluate vs heuristic baseline
   ↓
3. Active learning: surface uncertain examples
   ↓
4. Manual labeling (highest ROI)
   ↓
5. Merge and retrain
   ↓
6. Tune threshold for production cost function
   ↓
7. Deploy with updated thresholds
   ↓
8. Monitor and iterate
```

## Example Usage

### Training
```powershell
$env:HIRING_BUILD_DATA = "data/hiring_build_training_data.jsonl"
python train_hiring_classifier.py
```

### Evaluation
```powershell
python evaluate_hiring_classifier.py --test-size 0.2
# Output: logs/hiring_classifier_{pr,roc}_curve.png, evaluation.json
```

### Active Learning
```powershell
python active_learning_hiring.py --unlabeled data/unlabeled_prompts.txt --limit 50
# Manual step: edit data/uncertain_samples.jsonl to add labels
type data\hiring_build_training_data.jsonl data\uncertain_samples.jsonl > data\merged.jsonl
python train_hiring_classifier.py --data data/merged.jsonl
```

### Threshold Tuning
```powershell
python tune_hiring_threshold.py --cost-fp 1.0 --cost-fn 2.0 --write-config
# Update _predict_is_hiring thresholds in scorer
```

## Testing Results

### Unit Tests
- ✅ `test_hiring_classifier_integration.py`: Trains model, asserts integration
- ✅ `test_software_scorer.py`: Existing tests still pass
- ✅ `test_new_schema.py`: Schema validation passes

### CLI Tests
```powershell
python run_requirements_cli.py --text "Looking for senior React engineer 5+ years"
```
**Output:**
```json
{
  "technologies": {"frontend": ["react"], ...},
  "skills_complexity_score": 139.14,
  "is_hiring_requirement": true,
  "hiring_detection": {
    "source": "heuristic",  // or "model" when classifier present
    "proba": 0.0            // or actual probability
  }
}
```

### Demo Run
With 20-example dataset:
- ROC AUC: 0.750
- Found 10 uncertain samples (probabilities 0.46–0.56)
- Active learning ready for manual labeling

## Performance Targets

| Metric | Minimum | Production |
|--------|---------|------------|
| Dataset size | 100 | 500–1,000 |
| ROC AUC | 0.85 | 0.95+ |
| Precision | 0.85 | 0.95+ |
| Recall | 0.80 | 0.95+ |

## Key Features

1. **No Breaking Changes**: Existing heuristic remains as fallback
2. **Transparent**: Detection source and probability in response
3. **Tunable**: Cost-based threshold optimization
4. **Efficient Growth**: Active learning focuses on decision boundary
5. **Observable**: Full tracing in per-request logs
6. **Documented**: Complete workflow guide and troubleshooting

## Next Steps for Production

1. **Data Collection**: Curate 500–1,000 balanced examples
   - Use production logs with human labels
   - Include edge cases (ambiguous prompts, mixed intent)
   
2. **Evaluation**: Target AUC > 0.95, precision/recall > 0.90
   - Compare to heuristic baseline
   - Identify systematic errors
   
3. **Threshold Tuning**: Optimize for production cost function
   - Measure downstream impact of FP vs FN
   - A/B test different thresholds
   
4. **Monitoring**: Track confidence distribution over time
   - Alert on drift or confidence drop
   - Log predictions for periodic re-evaluation
   
5. **Iteration**: Monthly active learning cycles
   - Review false positives/negatives
   - Add hard negatives to training set
   - Re-tune thresholds based on production metrics

## Files Modified/Created

### New Files (9)
1. `mcp_server/hiring_classifier.py`
2. `train_hiring_classifier.py`
3. `evaluate_hiring_classifier.py`
4. `active_learning_hiring.py`
5. `tune_hiring_threshold.py`
6. `data/hiring_build_training_data.example.jsonl`
7. `data/unlabeled_prompts.txt`
8. `tests/test_hiring_classifier_integration.py`
9. `HIRING_CLASSIFIER_WORKFLOW.md` (this file)

### Modified Files (2)
1. `mcp_server/software_complexity_scorer.py`: Integration with optional classifier
2. `README.md`: Added training instructions and workflow section

## Conclusion

The hiring vs build classifier provides a clean, ML-based solution for schema separation with:
- **Robustness**: Heuristic fallback ensures no regressions
- **Efficiency**: Active learning minimizes labeling effort
- **Tunability**: Cost-based thresholds optimize for business impact
- **Observability**: Full tracing and evaluation tools

With 500+ labeled examples, this approach will significantly outperform rule-based heuristics while maintaining the safety net for edge cases.
