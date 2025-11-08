#!/usr/bin/env python3
"""
Train AI models for system design and technology criticality prediction.
"""

import json
import joblib
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

def load_jsonl(filepath):
    """Load JSONL file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                data.append(json.loads(line))
    return data

def train_system_design_classifier():
    """Train architecture pattern classifier."""
    print("\n=== Training System Design Classifier ===")
    
    # Load training data
    data = load_jsonl(DATA_DIR / "system_design_training.jsonl")
    print(f"Loaded {len(data)} training samples")
    
    # Extract features and labels
    requirements = [d['requirement'] for d in data]
    labels = [d['architecture'] for d in data]
    
    # Vectorize requirements
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
    X = vectorizer.fit_transform(requirements)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        min_samples_leaf=2,
        max_features='sqrt'
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save models
    joblib.dump(clf, MODELS_DIR / "system_design_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "system_design_vectorizer.pkl")
    print(f"\nSaved classifier to {MODELS_DIR / 'system_design_classifier.pkl'}")
    print(f"Saved vectorizer to {MODELS_DIR / 'system_design_vectorizer.pkl'}")
    
    # Show feature importance
    feature_names = vectorizer.get_feature_names_out()
    importances = clf.feature_importances_
    top_features = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:15]
    print("\nTop 15 Important Features:")
    for feat, imp in top_features:
        print(f"  {feat}: {imp:.4f}")
    
    return clf, vectorizer

def train_tech_criticality_classifier():
    """Train technology criticality classifier."""
    print("\n=== Training Technology Criticality Classifier ===")
    
    # Load training data
    data = load_jsonl(DATA_DIR / "tech_criticality_training.jsonl")
    print(f"Loaded {len(data)} training samples")
    
    # Extract features and labels
    # Combine requirement + tech as feature
    features = [f"{d['requirement']} {d['tech']}" for d in data]
    labels = [d['criticality'] for d in data]
    
    # Vectorize features
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 3))
    X = vectorizer.fit_transform(features)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        min_samples_leaf=2,
        max_features='sqrt'
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save models
    joblib.dump(clf, MODELS_DIR / "tech_criticality_classifier.pkl")
    joblib.dump(vectorizer, MODELS_DIR / "tech_criticality_vectorizer.pkl")
    print(f"\nSaved classifier to {MODELS_DIR / 'tech_criticality_classifier.pkl'}")
    print(f"Saved vectorizer to {MODELS_DIR / 'tech_criticality_vectorizer.pkl'}")
    
    # Create LOC overhead mapping (tech -> avg_loc_overhead)
    loc_map = {}
    for d in data:
        tech = d['tech']
        loc = d['loc_overhead']
        if tech not in loc_map:
            loc_map[tech] = []
        loc_map[tech].append(loc)
    
    # Average LOC per tech
    avg_loc_map = {tech: sum(locs) / len(locs) for tech, locs in loc_map.items()}
    
    # Save LOC mapping
    with open(MODELS_DIR / "tech_loc_overhead_map.json", 'w') as f:
        json.dump(avg_loc_map, f, indent=2)
    print(f"Saved LOC overhead mapping to {MODELS_DIR / 'tech_loc_overhead_map.json'}")
    
    # Show sample LOC overheads
    print("\nSample LOC Overheads:")
    for tech, avg_loc in sorted(avg_loc_map.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {tech}: {avg_loc:.0f} LOC")
    
    return clf, vectorizer, avg_loc_map

def test_predictions():
    """Test predictions with sample inputs."""
    print("\n=== Testing Predictions ===")
    
    # Load models
    system_design_clf = joblib.load(MODELS_DIR / "system_design_classifier.pkl")
    system_design_vec = joblib.load(MODELS_DIR / "system_design_vectorizer.pkl")
    tech_criticality_clf = joblib.load(MODELS_DIR / "tech_criticality_classifier.pkl")
    tech_criticality_vec = joblib.load(MODELS_DIR / "tech_criticality_vectorizer.pkl")
    
    with open(MODELS_DIR / "tech_loc_overhead_map.json", 'r') as f:
        loc_map = json.load(f)
    
    # Test system design
    test_requirements = [
        "Build a simple blog",
        "Create a microservices-based food delivery platform",
        "Develop a serverless image processing API",
        "Build a real-time stock trading system"
    ]
    
    print("\nSystem Design Predictions:")
    for req in test_requirements:
        X = system_design_vec.transform([req])
        pred = system_design_clf.predict(X)[0]
        proba = system_design_clf.predict_proba(X)[0]
        classes = system_design_clf.classes_
        confidence = dict(zip(classes, proba))
        print(f"\n  Requirement: {req}")
        print(f"  Predicted: {pred}")
        print(f"  Confidence: {confidence[pred]:.2%}")
    
    # Test tech criticality
    test_cases = [
        ("Build a simple blog", "react"),
        ("Build a simple blog", "elasticsearch"),
        ("Create a video streaming service", "ffmpeg"),
        ("Create a video streaming service", "redis"),
    ]
    
    print("\n\nTechnology Criticality Predictions:")
    for req, tech in test_cases:
        feature = f"{req} {tech}"
        X = tech_criticality_vec.transform([feature])
        pred = tech_criticality_clf.predict(X)[0]
        proba = tech_criticality_clf.predict_proba(X)[0]
        classes = tech_criticality_clf.classes_
        confidence = dict(zip(classes, proba))
        avg_loc = loc_map.get(tech, 0)
        print(f"\n  Requirement: {req}")
        print(f"  Technology: {tech}")
        print(f"  Criticality: {pred} (confidence: {confidence[pred]:.2%})")
        print(f"  LOC Overhead: ~{avg_loc:.0f} lines")

if __name__ == "__main__":
    print("Training Software Design Models")
    print("=" * 50)
    
    # Train both models
    train_system_design_classifier()
    train_tech_criticality_classifier()
    
    # Test predictions
    test_predictions()
    
    print("\n" + "=" * 50)
    print("Training complete!")
