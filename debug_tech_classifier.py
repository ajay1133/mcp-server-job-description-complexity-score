#!/usr/bin/env python3
"""Debug technology classifier predictions."""

import sys
import os
import joblib
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

models_path = os.path.join(os.path.dirname(__file__), 'models', 'software')

# Load models
vectorizer = joblib.load(os.path.join(models_path, 'tfidf_vectorizer.joblib'))
tech_classifier = joblib.load(os.path.join(models_path, 'tech_multilabel_classifier.joblib'))
with open(os.path.join(models_path, 'technology_labels.json'), 'r') as f:
    tech_labels = json.load(f)

test_texts = [
    "Build React dashboard with authentication",
    "Create FastAPI REST API with PostgreSQL",
    "Develop Node.js backend with MongoDB",
]

print("Technology Classifier Debug\n")
print("=" * 70)

for text in test_texts:
    X = vectorizer.transform([text])
    predictions = tech_classifier.predict(X)[0]
    probabilities = tech_classifier.predict_proba(X)[0]
    
    print(f"\nText: {text}")
    print(f"Predictions: {predictions}")
    print(f"Sum: {predictions.sum()}")
    
    # Show top probabilities
    top_indices = probabilities.argsort()[-5:][::-1]
    print("Top 5 probabilities:")
    for idx in top_indices:
        print(f"  {tech_labels[idx]}: {probabilities[idx]:.3f}")
    
    predicted_techs = [tech_labels[i] for i, pred in enumerate(predictions) if pred == 1]
    print(f"Predicted technologies: {predicted_techs}")

print("\n" + "=" * 70)
