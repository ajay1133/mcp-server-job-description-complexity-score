#!/usr/bin/env python3
"""Active learning loop for iterative model improvement.

Workflow:
1. Train model on current labeled data
2. Predict on unlabeled pool
3. Select most uncertain examples (confidence 0.3-0.7)
4. Present to user for labeling
5. Add to training set and retrain
6. Repeat until performance satisfies

This minimizes human labeling effort by focusing on uncertain cases.
"""

import json
import os
import sys
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import joblib


class ActiveLearner:
    """Active learning system for efficient labeling."""
    
    def __init__(self, labeled_file: str, unlabeled_file: str):
        self.labeled_file = labeled_file
        self.unlabeled_file = unlabeled_file
        self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        self.model = LogisticRegression(max_iter=1000)
        self.labeled_data = []
        self.unlabeled_data = []
        
        self._load_data()
    
    def _load_data(self):
        """Load labeled and unlabeled datasets."""
        # Load labeled
        if os.path.exists(self.labeled_file):
            with open(self.labeled_file, 'r', encoding='utf-8') as f:
                self.labeled_data = [json.loads(line) for line in f]
        
        # Load unlabeled
        if os.path.exists(self.unlabeled_file):
            with open(self.unlabeled_file, 'r', encoding='utf-8') as f:
                self.unlabeled_data = [json.loads(line) for line in f]
    
    def train_model(self) -> float:
        """Train binary classifier on current labeled data."""
        if len(self.labeled_data) < 10:
            print("Not enough labeled data to train (need at least 10)")
            return 0.0
        
        texts = [ex['text'] for ex in self.labeled_data]
        labels = [ex['is_software'] for ex in self.labeled_data]
        
        # Vectorize
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Train
        self.model.fit(X, y)
        
        # Cross-validation score
        scores = cross_val_score(self.model, X, y, cv=min(5, len(self.labeled_data)))
        mean_score = scores.mean()
        
        print(f"Model trained on {len(self.labeled_data)} examples")
        print(f"Cross-validation accuracy: {mean_score:.3f}")
        
        return mean_score
    
    def select_uncertain_examples(self, n: int = 20) -> List[Tuple[int, Dict, float]]:
        """
        Select n most uncertain examples from unlabeled pool.
        Returns: [(index, example, confidence), ...]
        """
        if not self.unlabeled_data:
            return []
        
        texts = [ex['text'] for ex in self.unlabeled_data]
        X = self.vectorizer.transform(texts)
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(X)
        
        # Calculate uncertainty (entropy or distance from 0.5)
        # Examples near 0.5 probability are most uncertain
        uncertainties = []
        for i, probs in enumerate(probabilities):
            # probs = [prob_false, prob_true]
            confidence = abs(probs[1] - 0.5)  # Distance from 0.5
            uncertainty = 0.5 - confidence     # Lower = more uncertain
            uncertainties.append((i, self.unlabeled_data[i], probs[1], uncertainty))
        
        # Sort by uncertainty (most uncertain first)
        uncertainties.sort(key=lambda x: x[3])
        
        # Return top n most uncertain
        return [(idx, ex, conf) for idx, ex, conf, _ in uncertainties[:n]]
    
    def interactive_labeling(self, uncertain_examples: List[Tuple[int, Dict, float]]) -> List[Dict]:
        """Present examples to user for labeling."""
        newly_labeled = []
        
        print("\n" + "=" * 70)
        print("ACTIVE LEARNING: Label Uncertain Examples")
        print("=" * 70)
        print("The model is uncertain about these examples.")
        print("Your labels will help improve accuracy.\n")
        
        for i, (idx, ex, confidence) in enumerate(uncertain_examples, 1):
            print(f"\n[{i}/{len(uncertain_examples)}]")
            print(f"Model prediction: {'SOFTWARE' if confidence > 0.5 else 'NON-SOFTWARE'}")
            print(f"Confidence: {confidence:.2f} (uncertain!)")
            print(f"\nText: {ex['text'][:300]}")
            if len(ex['text']) > 300:
                print("...")
            
            print("\nIs this a SOFTWARE development task?")
            print("  y) Yes (software)")
            print("  n) No (not software)")
            print("  s) Skip")
            print("  q) Quit labeling")
            
            choice = input("\nYour choice (y/n/s/q): ").strip().lower()
            
            if choice == 'q':
                print("Quitting labeling session")
                break
            elif choice == 'y':
                ex['is_software'] = True
                newly_labeled.append((idx, ex))
                print("✓ Labeled as SOFTWARE")
            elif choice == 'n':
                ex['is_software'] = False
                newly_labeled.append((idx, ex))
                print("✓ Labeled as NON-SOFTWARE")
            else:
                print("⊘ Skipped")
        
        return newly_labeled
    
    def update_datasets(self, newly_labeled: List[Tuple[int, Dict]]):
        """Move newly labeled examples from unlabeled to labeled."""
        # Sort indices in reverse to remove from back first
        indices_to_remove = sorted([idx for idx, _ in newly_labeled], reverse=True)
        
        # Add to labeled
        for _, ex in newly_labeled:
            self.labeled_data.append(ex)
        
        # Remove from unlabeled
        for idx in indices_to_remove:
            del self.unlabeled_data[idx]
        
        # Save updated datasets
        self._save_data()
    
    def _save_data(self):
        """Save labeled and unlabeled datasets."""
        with open(self.labeled_file, 'w', encoding='utf-8') as f:
            for ex in self.labeled_data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
        
        with open(self.unlabeled_file, 'w', encoding='utf-8') as f:
            for ex in self.unlabeled_data:
                f.write(json.dumps(ex, ensure_ascii=False) + '\n')
    
    def run_active_learning_loop(self, iterations: int = 5, examples_per_iteration: int = 20):
        """
        Main active learning loop.
        Each iteration: train -> select uncertain -> label -> update
        """
        print("\n" + "=" * 70)
        print("ACTIVE LEARNING LOOP")
        print("=" * 70)
        print(f"Starting with {len(self.labeled_data)} labeled examples")
        print(f"Unlabeled pool: {len(self.unlabeled_data)} examples")
        print(f"Will run {iterations} iterations, {examples_per_iteration} examples each\n")
        
        for iteration in range(1, iterations + 1):
            print(f"\n{'='*70}")
            print(f"ITERATION {iteration}/{iterations}")
            print(f"{'='*70}\n")
            
            # Step 1: Train model
            print("Step 1: Training model...")
            accuracy = self.train_model()
            
            if accuracy > 0.95:
                print(f"\n✓ Achieved {accuracy:.1%} accuracy - stopping early!")
                break
            
            # Step 2: Select uncertain examples
            print("\nStep 2: Selecting uncertain examples...")
            uncertain = self.select_uncertain_examples(examples_per_iteration)
            
            if not uncertain:
                print("No more unlabeled data!")
                break
            
            print(f"Selected {len(uncertain)} uncertain examples")
            
            # Step 3: Interactive labeling
            print("\nStep 3: Interactive labeling...")
            newly_labeled = self.interactive_labeling(uncertain)
            
            if not newly_labeled:
                print("No examples labeled - stopping")
                break
            
            # Step 4: Update datasets
            print(f"\nStep 4: Updating datasets...")
            self.update_datasets(newly_labeled)
            print(f"Added {len(newly_labeled)} new labeled examples")
            print(f"Total labeled: {len(self.labeled_data)}")
            print(f"Remaining unlabeled: {len(self.unlabeled_data)}")
        
        # Final model
        print("\n" + "=" * 70)
        print("FINAL MODEL")
        print("=" * 70)
        final_accuracy = self.train_model()
        
        # Save final model
        model_dir = "models/active_learning"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'vectorizer.joblib'))
        joblib.dump(self.model, os.path.join(model_dir, 'classifier.joblib'))
        
        print(f"\nFinal model saved to {model_dir}")
        print(f"Final accuracy: {final_accuracy:.1%}")
        print(f"Total labeled examples: {len(self.labeled_data)}")
        print("=" * 70 + "\n")


def main():
    labeled_file = "data/labeled_training_data.jsonl"
    unlabeled_file = "data/unlabeled_pool.jsonl"
    
    if not os.path.exists(labeled_file):
        print(f"Error: {labeled_file} not found")
        print("\nCreate it with minimal seed examples:")
        print("  python bootstrap_training_data.py")
        sys.exit(1)
    
    if not os.path.exists(unlabeled_file):
        print(f"Warning: {unlabeled_file} not found")
        print("Will only use existing labeled data")
    
    learner = ActiveLearner(labeled_file, unlabeled_file)
    learner.run_active_learning_loop(iterations=5, examples_per_iteration=20)


if __name__ == "__main__":
    main()
