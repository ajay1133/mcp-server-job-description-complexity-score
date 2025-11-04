"""
ML-based Complexity Scorer for programming requirements
Uses trained machine learning models instead of keyword matching
"""

import os
import re
from typing import Dict, List
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np


class ComplexityScorer:
    def __init__(self, model_dir: str = None):
        """
        Initialize the ML-based complexity scorer
        
        Args:
            model_dir: Directory containing trained models. If None, uses default location.
        """
        self.replit_agent_3_baseline = 100
        
        if model_dir is None:
            # Default to models directory in the same folder as this file
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        self.model_dir = model_dir
        self.models_loaded = False
        
        # Try to load pre-trained models
        try:
            self._load_models()
            self.models_loaded = True
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not load pre-trained models: {e}")
            print("Please run train_model.py to train the models first.")
            self.models_loaded = False
        
        # Define complexity factor categories for interpretation
        self.complexity_factors = {
            'basic_web': ['html', 'css', 'landing page', 'webpage', 'static site'],
            'database': ['database', 'postgresql', 'mysql', 'mongodb', 'sqlite', 'orm', 'schema', 'nosql', 'dynamodb'],
            'api_integration': ['rest api', 'graphql', 'webhook', 'api', 'endpoints', 'oauth', 'stripe', 'payment'],
            'frontend': ['react', 'vue', 'angular', 'frontend', 'javascript', 'typescript', 'next.js', 'svelte'],
            'backend': ['backend', 'flask', 'django', 'fastapi', 'node.js', 'nodejs', 'express', 'microservice', 'lambda'],
            'real_time': ['websocket', 'real-time', 'streaming', 'socket.io', 'sse', 'mqtt', 'webrtc'],
            'ai_ml': ['machine learning', 'neural network', 'openai', 'ai', 'ml', 'recommendation', 'nlp'],
            'deployment': ['deployment', 'ci/cd', 'docker', 'kubernetes', 'aws', 'azure', 'cloud', 'heroku'],
            'security': ['security', 'encryption', 'jwt', 'authentication', 'authorization', 'password', 'pki'],
            'testing': ['testing', 'unit test', 'integration test', 'e2e', 'pytest', 'jest', 'coverage'],
            'scalability': ['scalable', 'load balancing', 'caching', 'redis', 'kafka', 'message queue', 'distributed']
        }
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        score_model_path = os.path.join(self.model_dir, 'score_model.joblib')
        time_model_path = os.path.join(self.model_dir, 'time_model.joblib')
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.score_model = joblib.load(score_model_path)
        self.time_model = joblib.load(time_model_path)
    
    def analyze_text(self, text: str) -> Dict:
        """
        Analyze text using ML models to predict complexity and time
        
        Args:
            text: The requirement or job description to analyze
            
        Returns:
            Dictionary containing complexity score, time estimate, and analysis
        """
        if not self.models_loaded:
            raise RuntimeError(
                "Models not loaded. Please run 'python train_model.py' to train the models first."
            )
        
        # Vectorize the input text
        text_vector = self.vectorizer.transform([text])
        
        # Predict complexity score
        predicted_score = self.score_model.predict(text_vector)[0]
        predicted_score = max(10, predicted_score)  # Minimum score of 10
        
        # Predict estimated hours
        predicted_hours = self.time_model.predict(text_vector)[0]
        predicted_hours = max(0.5, predicted_hours)  # Minimum 30 minutes
        
        # Detect complexity factors present in the text
        detected_factors = self._detect_factors(text)
        
        # Estimate task size based on score and factors
        task_size = self._estimate_task_size(predicted_score, len(detected_factors))
        
        # Get difficulty rating
        difficulty_rating = self._get_difficulty_rating(predicted_score)
        
        # Calculate time estimate details
        time_estimate = self._format_time_estimate(predicted_hours, task_size)
        
        # Generate summary
        summary = self._generate_summary(predicted_score, detected_factors, time_estimate)
        
        return {
            'complexity_score': round(predicted_score, 2),
            'baseline_reference': self.replit_agent_3_baseline,
            'detected_factors': detected_factors,
            'task_size': task_size,
            'difficulty_rating': difficulty_rating,
            'estimated_completion_time': time_estimate,
            'summary': summary,
            'model_type': 'machine_learning'
        }
    
    def _detect_factors(self, text: str) -> Dict:
        """Detect which complexity factors are present in the text"""
        text_lower = text.lower()
        detected = {}
        
        for factor_name, keywords in self.complexity_factors.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Calculate a relevance score based on matches
                detected[factor_name] = {
                    'matches': matches,
                    'relevance': min(matches / len(keywords), 1.0)
                }
        
        return detected
    
    def _estimate_task_size(self, score: float, num_factors: int) -> str:
        """Estimate task size category based on score and factors"""
        if score >= 170:
            return 'expert'
        elif score >= 140:
            return 'very_complex'
        elif score >= 90:
            return 'complex'
        elif score >= 50:
            return 'moderate'
        else:
            return 'simple'
    
    def _get_difficulty_rating(self, score: float) -> str:
        """Get human-readable difficulty rating"""
        if score < 50:
            return "Much easier than Replit Agent 3 capabilities"
        elif score < 80:
            return "Easier than Replit Agent 3 capabilities"
        elif score < 120:
            return "Similar to Replit Agent 3 capabilities"
        elif score < 150:
            return "More challenging than Replit Agent 3 capabilities"
        else:
            return "Significantly more challenging than Replit Agent 3 capabilities"
    
    def _format_time_estimate(self, hours: float, task_size: str) -> Dict:
        """Format time estimate into various units"""
        days = hours / 8  # 8-hour workday
        weeks = days / 5  # 5-day work week
        
        # Determine best time unit for display
        if hours < 1:
            time_range = f"{int(hours * 60)}-{int(hours * 60 * 1.3)} minutes"
            best_estimate = f"{int(hours * 60)} minutes"
        elif hours < 8:
            time_range = f"{hours:.1f}-{hours * 1.3:.1f} hours"
            best_estimate = f"{hours:.1f} hours"
        elif days < 5:
            time_range = f"{days:.1f}-{days * 1.3:.1f} days"
            best_estimate = f"{days:.1f} days"
        else:
            time_range = f"{weeks:.1f}-{weeks * 1.3:.1f} weeks"
            best_estimate = f"{weeks:.1f} weeks"
        
        return {
            'hours': round(hours, 2),
            'days': round(days, 2),
            'weeks': round(weeks, 2),
            'best_estimate': best_estimate,
            'time_range': time_range,
            'assumptions': 'Assumes developer skilled in using AI coding agents like Replit'
        }
    
    def _generate_summary(self, score: float, factors: Dict, time_estimate: Dict) -> str:
        """Generate a summary of the analysis"""
        if not factors:
            summary = f"Complexity score: {score:.2f}. Low complexity task with minimal technical requirements. "
        else:
            # Get top 3 factors by relevance
            top_factors = sorted(
                factors.items(),
                key=lambda x: x[1]['relevance'],
                reverse=True
            )[:3]
            
            factor_names = [f.replace('_', ' ') for f, _ in top_factors]
            summary = f"Complexity score: {score:.2f}. Primary complexity factors: {', '.join(factor_names)}. "
        
        summary += f"Estimated completion time: {time_estimate['best_estimate']}."
        return summary

