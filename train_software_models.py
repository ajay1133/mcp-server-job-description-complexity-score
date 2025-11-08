"""Training script for software-only complexity models.

Models trained:
1. software_classifier (binary: software vs other requirement)
2. tech_multilabel_classifier (multi-label technology tags)
3. loc_regressor (estimate lines of code)
4. time_regressor (estimate manual coding hours)
5. Optional score_regressor (direct complexity score)

Dataset format (JSONL recommended): each line is an object
{
  "text": "Build a React dashboard with user auth and payments",
  "is_software": true,
  "technologies": ["react", "auth", "payments", "node"],
  "loc": 1800,
  "hours": 120,
  "complexity_score": 105  # optional
}

For non-software examples:
{
  "text": "Need someone to look after my elderly father", "is_software": false }

Instructions:
1. Curate balanced dataset (aim for >= 500 software, >= 500 non-software examples initially).
2. Expand technology tag ontology as needed; keep tags lowercase, snake/kebab style.
3. Run: python train_software_models.py --data data/software_training_data.jsonl --out models/software
4. Integrate via SoftwareComplexityScorer.
"""

import os
import json
import argparse
from typing import List, Dict, Any, Tuple

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import GradientBoostingRegressor


def load_dataset(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            # normalize
            if 'text' in obj and isinstance(obj['text'], str):
                obj['text'] = obj['text'].strip()
                if not obj['text']:
                    continue
            data.append(obj)
    return data


def merge_datasets(paths: List[str]) -> List[Dict[str, Any]]:
    """Merge multiple JSONL datasets, de-duplicate by normalized text."""
    seen = set()
    merged: List[Dict[str, Any]] = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: data file not found: {p}")
            continue
        for ex in load_dataset(p):
            key = ex.get('text', '')
            k = key.lower()
            if not k or k in seen:
                continue
            seen.add(k)
            merged.append(ex)
    return merged


def main():
    ap = argparse.ArgumentParser(description='Train software-only complexity models (no keyword heuristics).')
    ap.add_argument('--data', required=True, nargs='+', help='One or more JSONL training data files')
    ap.add_argument('--out', required=True, help='Output model directory')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    examples = merge_datasets(args.data)
    if not examples:
        raise SystemExit('No training data found.')

    # Split into task-specific subsets to avoid relying on missing fields
    # A) Software classifier requires labeled is_software
    clf_texts = [e['text'] for e in examples if e.get('is_software') is not None]
    clf_labels = np.array([1 if e.get('is_software') else 0 for e in examples if e.get('is_software') is not None])

    # B) Tech classifier requires technologies present (non-empty list)
    tech_texts = [e['text'] for e in examples if isinstance(e.get('technologies'), list) and len(e.get('technologies')) > 0]
    tech_lists = [e['technologies'] for e in examples if isinstance(e.get('technologies'), list) and len(e.get('technologies')) > 0]

    # C) LOC/time regressors require numeric loc/hours
    reg_texts_loc = [e['text'] for e in examples if isinstance(e.get('loc'), (int, float))]
    reg_vals_loc = np.array([float(e['loc']) for e in examples if isinstance(e.get('loc'), (int, float))], dtype=float)

    reg_texts_time = [e['text'] for e in examples if isinstance(e.get('hours'), (int, float)) or isinstance(e.get('loc'), (int, float))]
    # Backfill hours from LOC if missing: hours ≈ max(1, loc/25)
    reg_vals_time = np.array([
        float(e['hours']) if isinstance(e.get('hours'), (int, float)) else max(1.0, float(e.get('loc', 200)) / 25.0)
        for e in examples if isinstance(e.get('hours'), (int, float)) or isinstance(e.get('loc'), (int, float))
    ], dtype=float)

    # D) Optional score regressor if any complexity_score provided
    score_texts = [e['text'] for e in examples if isinstance(e.get('complexity_score'), (int, float))]
    score_vals = np.array([float(e['complexity_score']) for e in examples if isinstance(e.get('complexity_score'), (int, float))], dtype=float)

    # Build a unified vocabulary over all texts used by any model
    corpus_texts = list({*clf_texts, *tech_texts, *reg_texts_loc, *reg_texts_time, *score_texts})
    if not corpus_texts:
        raise SystemExit('No usable texts found for training. Ensure your data contains labeled examples.')

    # Vectorizer: adapt to dataset size
    min_df = 2 if len(corpus_texts) > 200 else 1
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=min_df, max_df=0.95)
    X_all = vectorizer.fit_transform(corpus_texts)
    # Map each subset to vectors via a small helper
    def vec(texts: List[str]):
        if not texts:
            return None
        return vectorizer.transform(texts)

    # 1) Software classifier
    if len(clf_texts) < 20 or len(set(clf_labels)) < 2:
        raise SystemExit('Insufficient labeled data for software classifier. Need at least 20 examples with both classes.')
    X_clf = vec(clf_texts)
    software_clf = LogisticRegression(max_iter=2000)
    software_clf.fit(X_clf, clf_labels)

    # 2) Tech multi-label classifier
    if not tech_texts:
        raise SystemExit('No technology labels present in dataset. Provide at least some examples with technologies.')
    mlb = MultiLabelBinarizer()
    Y_tech = mlb.fit_transform(tech_lists)
    X_tech = vec(tech_texts)
    tech_clf = OneVsRestClassifier(LogisticRegression(max_iter=1500))
    tech_clf.fit(X_tech, Y_tech)
    tech_clf.classes_ = mlb.classes_  # attach for downstream use

    # 3) LOC regressor
    if not reg_texts_loc:
        raise SystemExit('No LOC values present in dataset to train loc_regressor.')
    X_loc = vec(reg_texts_loc)
    loc_reg = GradientBoostingRegressor()
    loc_reg.fit(X_loc, reg_vals_loc)

    # 4) Time regressor
    if not reg_texts_time:
        raise SystemExit('No time values present (or derivable from LOC) to train time_regressor.')
    X_time = vec(reg_texts_time)
    time_reg = GradientBoostingRegressor()
    time_reg.fit(X_time, reg_vals_time)

    # 5) Optional complexity score regressor
    score_reg = None
    if len(score_texts) >= 20:
        X_score = vec(score_texts)
        score_reg = GradientBoostingRegressor()
        score_reg.fit(X_score, score_vals)

    # Persist
    joblib.dump(vectorizer, os.path.join(args.out, 'tfidf_vectorizer.joblib'))
    joblib.dump(software_clf, os.path.join(args.out, 'software_classifier.joblib'))
    joblib.dump(tech_clf, os.path.join(args.out, 'tech_multilabel_classifier.joblib'))
    joblib.dump(loc_reg, os.path.join(args.out, 'loc_regressor.joblib'))
    joblib.dump(time_reg, os.path.join(args.out, 'time_regressor.joblib'))
    if score_reg is not None:
        joblib.dump(score_reg, os.path.join(args.out, 'score_regressor.joblib'))

    # Save label ontology for reference
    with open(os.path.join(args.out, 'technology_labels.json'), 'w', encoding='utf-8') as f:
        json.dump(list(mlb.classes_), f, indent=2)

    # Report brief training summary
    print('Training complete. Models saved to', args.out)
    print('Summary:')
    print(f'  Software clf samples: {len(clf_texts)}')
    print(f'  Tech clf samples:     {len(tech_texts)} labels: {list(mlb.classes_)}')
    print(f'  LOC reg samples:      {len(reg_texts_loc)}')
    print(f'  Time reg samples:     {len(reg_texts_time)}')
    if score_reg is not None:
        print(f'  Score reg samples:    {len(score_texts)}')


if __name__ == '__main__':
    main()
