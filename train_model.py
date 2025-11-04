#!/usr/bin/env python3
"""
Train machine learning models for complexity scoring

This script:
1. Loads training data
2. Creates TF-IDF features from text
3. Trains Random Forest models for score and time prediction
4. Saves trained models to disk
"""

import os
import sys
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Add parent directory to path to import training_data
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training_data import get_training_data, get_test_cases


def prepare_data():
    """Load and prepare training data"""
    print("Loading training data...")
    data = get_training_data()
    
    # Extract features and targets
    texts = [item['requirement'] for item in data]
    scores = [item['complexity_score'] for item in data]
    hours = [item['estimated_hours'] for item in data]
    
    print(f"Loaded {len(texts)} training examples")
    print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
    print(f"Hours range: {min(hours):.1f} - {max(hours):.1f}")
    
    return texts, scores, hours


def train_models(texts, scores, hours):
    """Train TF-IDF vectorizer and prediction models"""
    
    print("\n" + "="*70)
    print("TRAINING MODELS")
    print("="*70)
    
    # Create TF-IDF vectorizer
    print("\n1. Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 3),  # Unigrams, bigrams, and trigrams
        min_df=1,
        max_df=0.95,
        lowercase=True,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(texts)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Split data
    X_train, X_test, score_train, score_test, hours_train, hours_test = train_test_split(
        X, scores, hours, test_size=0.2, random_state=42
    )
    
    print(f"\n2. Training/Test split:")
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Test samples: {X_test.shape[0]}")
    
    # Train complexity score model
    print("\n3. Training Complexity Score model...")
    score_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=3,
        random_state=42
    )
    score_model.fit(X_train, score_train)
    
    # Evaluate score model
    score_pred_train = score_model.predict(X_train)
    score_pred_test = score_model.predict(X_test)
    
    print(f"   Training MAE: {mean_absolute_error(score_train, score_pred_train):.2f}")
    print(f"   Test MAE: {mean_absolute_error(score_test, score_pred_test):.2f}")
    print(f"   Test R²: {r2_score(score_test, score_pred_test):.3f}")
    
    # Train time estimation model
    print("\n4. Training Time Estimation model...")
    time_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        min_samples_split=3,
        random_state=42
    )
    time_model.fit(X_train, hours_train)
    
    # Evaluate time model
    time_pred_train = time_model.predict(X_train)
    time_pred_test = time_model.predict(X_test)
    
    print(f"   Training MAE: {mean_absolute_error(hours_train, time_pred_train):.2f} hours")
    print(f"   Test MAE: {mean_absolute_error(hours_test, time_pred_test):.2f} hours")
    print(f"   Test R²: {r2_score(hours_test, time_pred_test):.3f}")
    
    # Show some predictions vs actual
    print("\n5. Sample predictions on test set:")
    print(f"   {'Actual Score':<15} {'Predicted':<15} {'Actual Hours':<15} {'Predicted':<15}")
    print(f"   {'-'*60}")
    for i in range(min(5, len(score_test))):
        print(f"   {score_test[i]:<15.1f} {score_pred_test[i]:<15.1f} {hours_test[i]:<15.1f} {time_pred_test[i]:<15.1f}")
    
    return vectorizer, score_model, time_model


def save_models(vectorizer, score_model, time_model):
    """Save trained models to disk"""
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    # Create models directory
    models_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # Save models
    vectorizer_path = os.path.join(models_dir, 'tfidf_vectorizer.joblib')
    score_model_path = os.path.join(models_dir, 'score_model.joblib')
    time_model_path = os.path.join(models_dir, 'time_model.joblib')
    
    joblib.dump(vectorizer, vectorizer_path)
    joblib.dump(score_model, score_model_path)
    joblib.dump(time_model, time_model_path)
    
    print(f"\n✓ Models saved to: {models_dir}")
    print(f"  - {os.path.basename(vectorizer_path)}")
    print(f"  - {os.path.basename(score_model_path)}")
    print(f"  - {os.path.basename(time_model_path)}")


def validate_on_test_cases(vectorizer, score_model, time_model):
    """Validate models on predefined test cases"""
    print("\n" + "="*70)
    print("VALIDATION ON TEST CASES")
    print("="*70)
    
    test_cases = get_test_cases()
    
    for i, test_case in enumerate(test_cases, 1):
        text = test_case['requirement']
        expected_score_range = test_case['expected_score_range']
        expected_hours_range = test_case['expected_hours_range']
        
        # Predict
        X = vectorizer.transform([text])
        pred_score = score_model.predict(X)[0]
        pred_hours = time_model.predict(X)[0]
        
        # Check if in range
        score_ok = expected_score_range[0] <= pred_score <= expected_score_range[1]
        hours_ok = expected_hours_range[0] <= pred_hours <= expected_hours_range[1]
        
        status = "✓" if (score_ok and hours_ok) else "✗"
        
        print(f"\n{status} Test Case {i}:")
        print(f"  Requirement: {text[:60]}...")
        print(f"  Predicted Score: {pred_score:.1f} (expected: {expected_score_range[0]}-{expected_score_range[1]})")
        print(f"  Predicted Hours: {pred_hours:.1f} (expected: {expected_hours_range[0]}-{expected_hours_range[1]})")


def main():
    print("="*70)
    print("MCP COMPLEXITY SCORER - MODEL TRAINING")
    print("="*70)
    
    # Prepare data
    texts, scores, hours = prepare_data()
    
    # Train models
    vectorizer, score_model, time_model = train_models(texts, scores, hours)
    
    # Save models
    save_models(vectorizer, score_model, time_model)
    
    # Validate on test cases
    validate_on_test_cases(vectorizer, score_model, time_model)
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Run 'python test_scoring.py' to test the ML models")
    print("  2. Run 'python demo_time_estimation.py' to see examples")
    print("  3. Run 'python mcp_server/server.py' to start the MCP server")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
