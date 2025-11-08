"""
ML-based Complexity Scorer for programming requirements
Uses trained machine learning models instead of keyword matching
"""

import os
import re
import json
from typing import Dict, List, Optional, Tuple
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

# Load prototype dataset for category matching (few-shot, no hardcoded rules)
try:
    from category_training_data import get_category_training_data
except Exception:
    get_category_training_data = None


class ComplexityScorer:
    """
    DEPRECATED: This scorer is being phased out in favor of SoftwareComplexityScorer.
    
    This class handles both software and non-software jobs using online search heuristics
    and profession categorization. For software-only complexity scoring with model-based
    classification and no hardcoded keyword matching, use:
    
        from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
    
    See SOFTWARE_SCORER.md for migration guide.
    """
    
    def __init__(self, model_dir: str = None, complexity_factors: Dict[str, "List[str]"] = None):
        """
        Initialize the ML-based complexity scorer
        
        Args:
            model_dir: Directory containing trained models. If None, uses default location.
            complexity_factors: Optional mapping of factor name -> list of keywords. If not provided,
                values are loaded from config/complexity_factors.json (env-overridable) with a safe fallback.
        """
        self.replit_agent_3_baseline = 100
        
        if model_dir is None:
            # Default to models directory in the same folder as this file
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        
        self.model_dir = model_dir
        self.models_loaded = False
        self.category_models_loaded = False
        
        # Initialize prototype-based matcher (nearest neighbor on TF-IDF)
        self._init_category_prototypes()
        
        # Try to load pre-trained models
        try:
            self._load_models()
            self.models_loaded = True
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not load pre-trained models: {e}")
            print("Please run train_model.py to train the models first.")
            self.models_loaded = False
        
        # Try to load category classification models (optional, falls back if missing)
        try:
            self._load_category_models()
            self.category_models_loaded = True
        except (FileNotFoundError, Exception) as e:
            print(f"Info: Category models not loaded: {e}")
            self.category_models_loaded = False
        
        # Load complexity factor categories (config-driven with a safe fallback)
        self.complexity_factors = complexity_factors if complexity_factors is not None else self._load_complexity_factors()
    
    def _load_models(self):
        """Load pre-trained models from disk"""
        vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.joblib')
        score_model_path = os.path.join(self.model_dir, 'score_model.joblib')
        time_model_path = os.path.join(self.model_dir, 'time_model.joblib')
        
        self.vectorizer = joblib.load(vectorizer_path)
        self.score_model = joblib.load(score_model_path)
        self.time_model = joblib.load(time_model_path)

    def _init_category_prototypes(self):
        """Prepare a lightweight nearest-neighbor classifier over labeled prototypes."""
        self.proto_vectorizer = None
        self.proto_matrix = None
        self.proto_labels = []  # List of tuples (category, subcategory)
        self.proto_texts = []
        
        if get_category_training_data is None:
            return
        try:
            data = get_category_training_data()
            texts = [d["text"] for d in data]
            labels = [(d["category"], d["subcategory"]) for d in data]
            if not texts:
                return
            # Use character and word n-grams to be robust to phrasing
            self.proto_vectorizer = TfidfVectorizer(ngram_range=(1,2), analyzer='word', min_df=1)
            self.proto_matrix = self.proto_vectorizer.fit_transform(texts)
            self.proto_labels = labels
            self.proto_texts = texts
        except Exception as e:
            print(f"Prototype init failed: {e}")

    def _classify_job_category_nn(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Nearest-neighbor category classification using TF-IDF cosine similarity."""
        if self.proto_vectorizer is None or self.proto_matrix is None:
            return None
        try:
            vec = self.proto_vectorizer.transform([text])
            sims = cosine_similarity(vec, self.proto_matrix)[0]
            idx = int(np.argmax(sims))
            score = float(sims[idx])
            # Require a minimal similarity to avoid random matches
            if score < 0.15:
                return None
            cat, sub = self.proto_labels[idx]
            return (cat, sub, 'nn_classification')
        except Exception as e:
            print(f"NN classification error: {e}")
            return None

    def _load_category_models(self):
        """Load ML models for job category and subcategory classification (optional)."""
        category_model_path = os.path.join(self.model_dir, 'category_classifier.joblib')
        subcategory_model_path = os.path.join(self.model_dir, 'subcategory_classifier.joblib')
        
        # Pipelines include their own TfidfVectorizer
        self.category_classifier = joblib.load(category_model_path)
        self.subcategory_classifier = joblib.load(subcategory_model_path)

    def _classify_job_category_ml(self, text: str) -> Optional[Tuple[str, str, str]]:
        """Classify job category and subcategory using ML models if available."""
        if not self.category_models_loaded:
            return None
        try:
            pred_cat = self.category_classifier.predict([text])[0]
            pred_sub = self.subcategory_classifier.predict([text])[0]
            return (pred_cat, pred_sub, 'ml_classification')
        except Exception as e:
            print(f"Category ML classification error: {e}")
            return None

    def _default_complexity_factors(self) -> Dict[str, List[str]]:
        """Built-in fallback mapping used when no config file is found.
        Kept for backwards compatibility and to ensure sensible defaults.
        """
        return {
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

    def _load_complexity_factors(self) -> Dict[str, List[str]]:
        """Load complexity factors from JSON config or fall back to defaults.

        Lookup order:
        1) Environment variable MCP_COMPLEXITY_FACTORS (path to JSON)
        2) repo_root/config/complexity_factors.json
        3) Built-in defaults
        """
        # 1) Env var override
        env_path = os.environ.get('MCP_COMPLEXITY_FACTORS')
        candidate_paths: List[str] = []
        if env_path:
            candidate_paths.append(env_path)

        # 2) Default config under repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        default_cfg = os.path.join(repo_root, 'config', 'complexity_factors.json')
        candidate_paths.append(default_cfg)

        for path in candidate_paths:
            try:
                if path and os.path.isfile(path):
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    # Basic validation: dict of str -> list[str]
                    if isinstance(data, dict) and all(
                        isinstance(k, str) and isinstance(v, list) and all(isinstance(x, str) for x in v)
                        for k, v in data.items()
                    ):
                        return data
                    else:
                        print(f"Invalid complexity_factors schema in {path}; using defaults.")
            except Exception as e:
                print(f"Failed to load complexity_factors from {path}: {e}")

        # 3) Fallback
        return self._default_complexity_factors()
    
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
        
        # Predict baseline complexity score (will be adjusted later)
        predicted_score = self.score_model.predict(text_vector)[0]
        predicted_score = max(10, predicted_score)  # Minimum score of 10
        
        # Predict estimated hours (will be overridden by user-mentioned duration if present)
        predicted_hours = self.time_model.predict(text_vector)[0]
        predicted_hours = max(0.5, predicted_hours)  # Minimum 30 minutes
        
        # Extract duration requirement from text (e.g., "couple of days", "3 weeks")
        duration_info = self._extract_duration(text)
        
        # Detect complexity factors present in the text
        detected_factors = self._detect_factors(text)
        
        # Deduce job category and subcategory first (needed for time estimation)
        job_category, job_sub_category, category_lookup_method = self._deduce_job_categories(text, detected_factors)
        
        # Heuristic: treat hiring/job postings as long-term engagements (avoid tiny estimates)
        is_job_posting = self._is_job_posting(text)
        adjusted_predicted_hours = predicted_hours
        if is_job_posting and not duration_info['mentioned']:
            # Floor at 12 weeks (work weeks) for senior roles as a conservative minimum
            adjusted_predicted_hours = max(predicted_hours, 12 * 40.0)
        else:
            # Heuristic: creating and deploying an app typically requires at least a full sprint or two
            tl = text.lower()
            if (not duration_info['mentioned'] and
                ('deploy' in tl or 'deployment' in tl) and
                any(k in tl for k in ['create', 'build', 'develop']) and
                any(k in tl for k in ['app', 'application', 'web app', 'mobile app', 'website'])):
                adjusted_predicted_hours = max(adjusted_predicted_hours, 80.0)
        
        # Calculate time estimate details (user-mentioned durations take precedence here)
        # Do this BEFORE adjusting complexity score so we can incorporate duration/availability into the score
        # Use a temporary task_size based on baseline score; will recompute after adjustment
        temp_task_size = self._estimate_task_size(predicted_score, len(detected_factors))
        time_estimate = self._format_time_estimate(adjusted_predicted_hours, temp_task_size, duration_info, job_category)
        
        # Adjust complexity score based on profession availability and estimated time involved
        used_hours = time_estimate.get('hours', predicted_hours)
        availability_mult = self._get_availability_multiplier(job_category, job_sub_category)
        time_mult = self._get_time_multiplier(used_hours)
        adjusted_score = min(200.0, max(10.0, predicted_score * availability_mult * time_mult))
        
        # Recompute task size and difficulty based on adjusted score (deduce score last)
        task_size = self._estimate_task_size(adjusted_score, len(detected_factors))
        difficulty_rating = self._get_difficulty_rating(adjusted_score)
        
        # Generate summary with adjusted score
        summary = self._generate_summary(adjusted_score, detected_factors, time_estimate)
        
        return {
            'complexity_score': round(adjusted_score, 2),
            'baseline_reference': self.replit_agent_3_baseline,
            'detected_factors': detected_factors,
            'task_size': task_size,
            'difficulty_rating': difficulty_rating,
            'estimated_completion_time': time_estimate,
            'job_category': job_category,
            'job_sub_category': job_sub_category,
            'category_lookup_method': category_lookup_method,
            'summary': summary,
            'model_type': 'machine_learning'
        }

    def _is_job_posting(self, text: str) -> bool:
        """Detects if the prompt is a hiring/job description rather than a discrete task."""
        t = text.lower()
        cues = [
            'job description', 'we are seeking', 'what you\'ll do', "what we'll do", 'what we\'re looking for',
            'current & evolving tech stack', 'apply if you', 'immediate joiners', 'join within', 'role',
            'engineering lead', 'principal', 'staff engineer', 'head of engineering', 'head of technology'
        ]
        return any(c in t for c in cues)
    
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
    
    def _extract_duration(self, text: str) -> Dict:
        """
        Extract duration requirement from text (e.g., "couple of days", "3 weeks", "2 hours")
        
        Returns:
            Dict with 'hours', 'mentioned' (bool), 'phrase' (str), and 'is_deadline' (bool)
        """
        text_lower = text.lower()
        # Remove availability/start-date and schedule phrases so they don't get misinterpreted as duration
        availability_patterns = [
            r'(immediate\s+joiners?)',
            r'(join|start)\s+(within|in|by)\s+(?:max\s*)?\d+\s*(day|days|week|weeks|month|months)',
            r'notice\s+period\s*:\s*\d+\s*(day|days|week|weeks|month|months)',
            r'available\s+from\s+\d{1,2}(?:\.\d{1,2})?\s*(am|pm)\s*to\s*\d{1,2}(?:\.\d{1,2})?\s*(am|pm)',
            r'\b\d{1,2}(?:\.\d{1,2})?\s*(am|pm)\s*to\s*\d{1,2}(?:\.\d{1,2})?\s*(am|pm)\b',
            r'\bist\b'
        ]
        duration_text = text_lower
        for pat in availability_patterns:
            duration_text = re.sub(pat, ' ', duration_text)
        duration_info = {'hours': None, 'mentioned': False, 'phrase': None, 'is_deadline': False}
        
        # Check if this is a project deadline vs continuous care duration
        deadline_indicators = ['needs to be done', 'deadline', 'due in', 'complete in', 'finish in', 'deliver in']
        is_deadline = any(indicator in duration_text for indicator in deadline_indicators)
        
        # Common duration patterns
        patterns = [
            # Numeric patterns
            (r'(\d+)\s*(hour|hr|hrs)', lambda m: float(m.group(1))),
            (r'(\d+)\s*(day|days)', lambda m: float(m.group(1)) * (8 if is_deadline else 24)),  # 8h workdays for projects, 24h for care
            (r'(\d+)\s*(week|weeks|wk|wks)', lambda m: float(m.group(1)) * (40 if is_deadline else 168)),  # 40h work-week vs 168h full week
            (r'(\d+)\s*(month|months)', lambda m: float(m.group(1)) * (160 if is_deadline else 720)),  # ~160h work-month vs 720h full month
            
            # Word-based patterns
            (r'couple of (day|days)', lambda m: 2 * (8 if is_deadline else 24)),
            (r'few (day|days)', lambda m: 3 * (8 if is_deadline else 24)),
            (r'couple of (week|weeks)', lambda m: 2 * (40 if is_deadline else 168)),
            (r'few (week|weeks)', lambda m: 3 * (40 if is_deadline else 168)),
            (r'couple of (hour|hours)', lambda m: 2.0),
            (r'few (hour|hours)', lambda m: 3.0),

            # Relative/future time expressions
            (r'next\s*(day)', lambda m: 24.0 if not is_deadline else 8.0),
            (r'next\s*(week)', lambda m: 168.0 if not is_deadline else 40.0),
            (r'next\s*(month)', lambda m: 720.0 if not is_deadline else 160.0),
            (r'in\s*(a|one|1)\s*(day)', lambda m: 24.0 if not is_deadline else 8.0),
            (r'in\s*(a|one|1)\s*(week)', lambda m: 168.0 if not is_deadline else 40.0),
            (r'in\s*(a|one|1)\s*(month)', lambda m: 720.0 if not is_deadline else 160.0),
            (r'in\s*(\d+)\s*(days)', lambda m: float(m.group(1)) * (24.0 if not is_deadline else 8.0)),
            (r'in\s*(\d+)\s*(weeks)', lambda m: float(m.group(1)) * (168.0 if not is_deadline else 40.0)),
            (r'in\s*(\d+)\s*(months)', lambda m: float(m.group(1)) * (720.0 if not is_deadline else 160.0)),
            
            # Time-of-day patterns (before evening, by morning, etc.)
            (r'before\s+(evening|tonight)', lambda m: 12.0),  # Typical workday to evening
            (r'by\s+(evening|tonight)', lambda m: 12.0),
            (r'before\s+(morning|noon)', lambda m: 6.0),  # Early morning deadline
            (r'by\s+(morning|noon)', lambda m: 6.0),
            (r'by\s+end\s+of\s+(day|today)', lambda m: 8.0),
            (r'today', lambda m: 8.0),  # Generic "today" = rest of workday
            
            # Special cases
            (r'overnight', lambda m: 12.0),
            (r'all day', lambda m: 8.0),
            (r'full day', lambda m: 8.0),
            (r'half day', lambda m: 4.0),
            (r'weekend', lambda m: 48.0),  # 2 days * 24 hours (continuous)
        ]
        
        for pattern, converter in patterns:
            match = re.search(pattern, duration_text)
            if match:
                duration_info['hours'] = converter(match)
                duration_info['mentioned'] = True
                duration_info['phrase'] = match.group(0)
                duration_info['is_deadline'] = is_deadline
                break
        
        return duration_info
    
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
    
    def _format_time_estimate(self, hours: float, task_size: str, duration_info: Dict, job_category: str) -> Dict:
        """Format time estimate into various units, considering user-specified duration"""
        
        # Determine if this is a continuous care job (24/7) vs project work (8-hour days)
        continuous_care_categories = ['Caregiver', 'Nurse', 'Child Care Provider', 'Housekeeper']
        is_continuous_care = job_category in continuous_care_categories
        
        # If user specified a duration, use that instead of ML prediction
        if duration_info['mentioned'] and duration_info['hours']:
            hours = duration_info['hours']
            duration_source = f"based on specified duration: '{duration_info['phrase']}'"
        else:
            duration_source = "based on task complexity and typical completion times"
        
        # Calculate days and weeks based on job type
        if is_continuous_care and hours >= 24:
            # For continuous care, use 24-hour days
            actual_days = hours / 24
            actual_weeks = actual_days / 7
            days = hours / 8  # Still calculate standard workdays for reference
            weeks = days / 5
        else:
            # For project work, use 8-hour workdays
            days = hours / 8
            weeks = days / 5
            actual_days = days
            actual_weeks = weeks
        
        # Determine best time unit for display
        if hours < 1:
            time_range = f"{int(hours * 60)}-{int(hours * 60 * 1.3)} minutes"
            best_estimate = f"{int(hours * 60)} minutes"
        elif hours < 8:
            time_range = f"{hours:.1f}-{hours * 1.3:.1f} hours"
            best_estimate = f"{hours:.1f} hours"
        elif is_continuous_care and hours >= 24:
            # For continuous care
            if actual_days < 7:
                time_range = f"{actual_days:.1f}-{actual_days * 1.3:.1f} days (continuous care)"
                best_estimate = f"{actual_days:.1f} days (continuous care)"
            else:
                time_range = f"{actual_weeks:.1f}-{actual_weeks * 1.3:.1f} weeks (continuous care)"
                best_estimate = f"{actual_weeks:.1f} weeks (continuous care)"
        elif days < 5:
            time_range = f"{days:.1f}-{days * 1.3:.1f} days"
            best_estimate = f"{days:.1f} days"
        else:
            # If user specified a duration in months, present months for readability
            if duration_info.get('mentioned') and duration_info.get('phrase') and 'month' in duration_info.get('phrase'):
                if duration_info.get('is_deadline'):
                    work_months = hours / 160.0
                    time_range = f"{work_months:.1f}-{work_months * 1.3:.1f} work-months"
                    best_estimate = f"{work_months:.1f} work-months"
                else:
                    months = hours / 720.0
                    time_range = f"{months:.1f}-{months * 1.3:.1f} months"
                    best_estimate = f"{months:.1f} months"
            else:
                time_range = f"{weeks:.1f}-{weeks * 1.3:.1f} weeks"
                best_estimate = f"{weeks:.1f} weeks"
        
        result = {
            'hours': round(hours, 2),
            'days': round(days, 2),
            'weeks': round(weeks, 2),
            'best_estimate': best_estimate,
            'time_range': time_range,
            'assumptions': f'Time estimate {duration_source} for similar requirements'
        }
        
        if duration_info['mentioned']:
            result['user_specified_duration'] = duration_info['phrase']
        
        return result
    
    def _generate_summary(self, score: float, factors: Dict, time_estimate: Dict) -> str:
        """Generate a summary of the analysis"""
        # Determine task complexity description
        if score < 50:
            complexity_desc = "Low complexity task"
        elif score < 90:
            complexity_desc = "Moderate complexity task"
        elif score < 140:
            complexity_desc = "Complex task"
        else:
            complexity_desc = "High complexity task"
        
        if not factors:
            summary = f"Complexity score: {score:.2f}. {complexity_desc} with minimal technical requirements. "
        else:
            # Get top 3 factors by relevance
            top_factors = sorted(
                factors.items(),
                key=lambda x: x[1]['relevance'],
                reverse=True
            )[:3]
            
            factor_names = [f.replace('_', ' ') for f, _ in top_factors]
            summary = f"Complexity score: {score:.2f}. {complexity_desc}. Primary complexity factors: {', '.join(factor_names)}. "
        
        summary += f"Estimated completion time: {time_estimate['best_estimate']}."
        return summary

    def _get_availability_multiplier(self, job_category: str, job_sub_category: str) -> float:
        """Rough availability multiplier by profession scarcity (1.0 = typical)."""
        scarcity_map = {
            'Doctor': 1.20,
            'Specialist Doctor': 1.25,
            'Nurse': 1.10,
            'Caregiver': 1.05,
            'Child Care Provider': 1.05,
            'Housekeeper': 1.00,
            'Plumber': 1.05,
            'Electrician': 1.05,
            'Carpenter': 1.05,
            'Driver': 1.00,
            'Lawyer': 1.15,
            'Teacher': 1.00,
            'Accountant': 1.00,
            'Software Developer': 1.05,
            'Data Scientist': 1.10,
            'Event Planner': 1.10,
            'Wedding Planner': 1.15,
            'Photographer': 1.05,
            'Mechanic': 1.05,
            'Chef': 1.05,
            'Veterinarian': 1.15,
            'General Professional': 1.00,
        }
        # Prefer subcategory-specific if present
        if job_sub_category in scarcity_map:
            return scarcity_map[job_sub_category]
        return scarcity_map.get(job_category, 1.0)

    def _get_time_multiplier(self, hours: float) -> float:
        """Increase score modestly for longer engagements to reflect coordination/complexity."""
        try:
            h = float(hours)
        except Exception:
            return 1.0
        if h < 8:
            return 1.0
        elif h < 40:
            return 1.05
        elif h < 160:
            return 1.15
        elif h < 720:
            return 1.30
        else:
            return 1.40
    
    def _search_online_for_job_category(self, text: str, use_online: bool = False) -> Optional[Tuple[str, str, str]]:
        """
        Use extended pattern matching with profession database to determine job category
        This serves as a comprehensive fallback when primary categorization fails
        
        Args:
            text: The requirement text
            use_online: If True, attempt online AI search when extended database fails
            
        Returns:
            Tuple of (job_category, job_sub_category, lookup_method) or None if all methods fail
        """
        text_lower = text.lower()
        
        # Extended profession database for uncommon roles
        extended_professions = {
            # Medical & Health
            'veterinarian': ('Veterinarian', 'Animal Healthcare Professional'),
            'vet': ('Veterinarian', 'Animal Healthcare Professional'),
            'dentist': ('Dentist', 'General Dentist'),
            'dental': ('Dentist', 'General Dentist'),
            'therapist': ('Therapist', 'Licensed Therapist'),
            'psychologist': ('Psychologist', 'Clinical Psychologist'),
            'psychiatrist': ('Psychiatrist', 'Mental Health Specialist'),
            'pharmacist': ('Pharmacist', 'Licensed Pharmacist'),
            'paramedic': ('Paramedic', 'Emergency Medical Technician'),
            
            # Creative & Arts
            'photographer': ('Photographer', 'Professional Photographer'),
            'videographer': ('Videographer', 'Video Production Specialist'),
            'graphic designer': ('Graphic Designer', 'Visual Designer'),
            'designer': ('Designer', 'Creative Professional'),
            'artist': ('Artist', 'Creative Professional'),
            'musician': ('Musician', 'Music Professional'),
            'writer': ('Writer', 'Content Creator'),
            'editor': ('Editor', 'Content Editor'),
            
            # Culinary
            'chef': ('Chef', 'Culinary Professional'),
            'cook': ('Cook', 'Kitchen Professional'),
            'baker': ('Baker', 'Baking Specialist'),
            'bartender': ('Bartender', 'Beverage Service Professional'),
            'waiter': ('Server', 'Food Service Professional'),
            'waitress': ('Server', 'Food Service Professional'),
            
            # Trades (Extended)
            'mechanic': ('Mechanic', 'Automotive Technician'),
            'hvac': ('HVAC Technician', 'Climate Control Specialist'),
            'welder': ('Welder', 'Metal Fabrication Specialist'),
            'mason': ('Mason', 'Masonry Specialist'),
            'roofer': ('Roofer', 'Roofing Specialist'),
            'painter': ('Painter', 'Painting Contractor'),
            
            # Services
            'hairdresser': ('Hairdresser', 'Hair Styling Professional'),
            'barber': ('Barber', 'Hair Cutting Specialist'),
            'beautician': ('Beautician', 'Beauty Professional'),
            'massage therapist': ('Massage Therapist', 'Therapeutic Massage Specialist'),
            'personal trainer': ('Personal Trainer', 'Fitness Professional'),
            'trainer': ('Trainer', 'Fitness Professional'),
            
            # Real Estate & Construction
            'architect': ('Architect', 'Licensed Architect'),
            'engineer': ('Engineer', 'Engineering Professional'),
            'surveyor': ('Surveyor', 'Land Surveyor'),
            'contractor': ('Contractor', 'General Contractor'),
            
            # Business & Finance
            'consultant': ('Consultant', 'Business Consultant'),
            'analyst': ('Analyst', 'Business Analyst'),
            'banker': ('Banker', 'Banking Professional'),
            'broker': ('Broker', 'Financial Broker'),
            'realtor': ('Real Estate Agent', 'Property Sales Professional'),
            'real estate agent': ('Real Estate Agent', 'Property Sales Professional'),
            
            # Education & Childcare
            'professor': ('Professor', 'University Educator'),
            'lecturer': ('Lecturer', 'Academic Instructor'),
            'counselor': ('Counselor', 'Guidance Counselor'),
            
            # Security & Emergency
            'security guard': ('Security Guard', 'Security Professional'),
            'firefighter': ('Firefighter', 'Emergency Response Professional'),
            'police': ('Police Officer', 'Law Enforcement Professional'),
            
            # Logistics & Transportation
            'pilot': ('Pilot', 'Aviation Professional'),
            'flight attendant': ('Flight Attendant', 'Airline Service Professional'),
            'delivery': ('Delivery Driver', 'Logistics Professional'),
            'courier': ('Courier', 'Delivery Professional'),
            
            # Technology (Extended)
            'it support': ('IT Support Specialist', 'Technical Support Professional'),
            'system administrator': ('System Administrator', 'IT Infrastructure Professional'),
            'network engineer': ('Network Engineer', 'Network Infrastructure Specialist'),
            
            # Agriculture & Environment
            'farmer': ('Farmer', 'Agricultural Professional'),
            'gardener': ('Gardener', 'Landscape Professional'),
            'landscaper': ('Landscaper', 'Landscape Design Professional'),
            'florist': ('Florist', 'Floral Design Professional'),
            
            # Manufacturing & Production
            'factory worker': ('Factory Worker', 'Manufacturing Professional'),
            'machine operator': ('Machine Operator', 'Equipment Operator'),
            'quality inspector': ('Quality Inspector', 'Quality Assurance Professional'),
        }
        
        # Check extended database
        for keyword, (category, subcategory) in extended_professions.items():
            if keyword in text_lower:
                return (category, subcategory, 'extended_database')
        
        # If still not found and online search is enabled, try AI-powered search
        if use_online:
            try:
                online_result = self._perform_online_search(text)
                if online_result:
                    return online_result
            except Exception as e:
                print(f"Online search failed: {e}")
        
        return None
    
    def _perform_online_search(self, text: str) -> Optional[Tuple[str, str, str]]:
        """
        Perform intelligent keyword-based search to identify job category
        This validates pattern matching results and catches false positives
        
        Args:
            text: The requirement text
            
        Returns:
            Tuple of (job_category, job_sub_category, 'online_search') or None
        """
        try:
            text_lower = text.lower()
            # Context guards to reduce false positives for 'delivery/deliver' in software contexts
            tech_signals = [
                'software', 'developer', 'engineer', 'architect', 'architecture', 'system design',
                'react', 'next.js', 'react native', 'node', 'node.js', 'typescript', 'python', 'rails',
                'aws', 'kubernetes', 'docker', 'ci/cd', 'microservices', 'distributed systems', 'saas',
                'postgresql', 'redis', 'cloudwatch', 'sentry', 'github actions', 'infrastructure',
                'app', 'application', 'web app', 'mobile app', 'website', 'deploy', 'deployment'
            ]
            has_strong_tech_context = any(tok in text_lower for tok in tech_signals)
            # App/software build signals
            app_signals = ['app', 'application', 'web app', 'mobile app', 'website']
            build_verbs = ['create', 'build', 'develop', 'make']
            # Strong software role phrases
            software_role_phrases = [
                'full stack developer', 'software developer', 'software engineer', 'principal engineer',
                'principal developer', 'staff engineer', 'engineering lead', 'tech lead', 'head of engineering',
                'head of technology'
            ]
            
            # Priority 0: Strong software role phrases (override ambiguous terms like 'architect' or 'delivery')
            if any(p in text_lower for p in software_role_phrases):
                # Try to infer subcategory
                frontend_signals = ['react', 'next.js', 'frontend', 'ui']
                backend_signals = ['node', 'node.js', 'typescript', 'python', 'rails', 'backend', 'api']
                if any(s in text_lower for s in frontend_signals) and any(s in text_lower for s in backend_signals):
                    return ('Software Developer', 'Full Stack Developer', 'online_search')
                return ('Software Developer', 'General Software Developer', 'online_search')

            # Priority 0.5: Implicit software creation intent (create/build + deploy + app)
            if any(s in text_lower for s in app_signals) and any(v in text_lower for v in build_verbs) and (
                'deploy' in text_lower or 'deployment' in text_lower
            ):
                # Creating and deploying an app implies full-stack responsibilities
                return ('Software Developer', 'Full Stack Developer', 'online_search')

            # Priority 1: Direct profession mentions (most specific)
            direct_professions = {
                # Medical & Health
                'veterinarian': ('Veterinarian', 'Animal Healthcare Professional'),
                'vet ': ('Veterinarian', 'Animal Healthcare Professional'),
                'dentist': ('Dentist', 'General Dentist'),
                'dental': ('Dentist', 'General Dentist'),
                'therapist': ('Therapist', 'Licensed Therapist'),
                'psychologist': ('Psychologist', 'Clinical Psychologist'),
                'psychiatrist': ('Psychiatrist', 'Mental Health Specialist'),
                'pharmacist': ('Pharmacist', 'Licensed Pharmacist'),
                'paramedic': ('Paramedic', 'Emergency Medical Technician'),
                
                # Delivery & Logistics
                'courier': ('Delivery Service', 'Courier'),
                'ship package': ('Delivery Service', 'Package Delivery'),
                'deliver package': ('Delivery Service', 'Package Delivery'),
                'deliver parcel': ('Delivery Service', 'Package Delivery'),
                
                # Creative & Arts
                'photographer': ('Photographer', 'Professional Photographer'),
                'videographer': ('Videographer', 'Video Production Specialist'),
                'graphic designer': ('Graphic Designer', 'Visual Designer'),
                'artist': ('Artist', 'Creative Professional'),
                'musician': ('Musician', 'Music Professional'),
                'writer': ('Writer', 'Content Creator'),
                
                # Culinary
                'chef': ('Chef', 'Culinary Professional'),
                'cook': ('Cook', 'Kitchen Professional'),
                'baker': ('Baker', 'Baking Specialist'),
                'bartender': ('Bartender', 'Beverage Service Professional'),
                
                # Trades
                'mechanic': ('Mechanic', 'Automotive Technician'),
                'hvac': ('HVAC Technician', 'Climate Control Specialist'),
                'welder': ('Welder', 'Metal Fabrication Specialist'),
                'mason': ('Mason', 'Masonry Specialist'),
                'roofer': ('Roofer', 'Roofing Specialist'),
                'painter': ('Painter', 'Painting Contractor'),
                
                # Services
                'hairdresser': ('Hairdresser', 'Hair Styling Professional'),
                'barber': ('Barber', 'Hair Cutting Specialist'),
                'beautician': ('Beautician', 'Beauty Professional'),
                'massage therapist': ('Massage Therapist', 'Therapeutic Massage Specialist'),
                'personal trainer': ('Personal Trainer', 'Fitness Professional'),
                
                # Real Estate & Construction
                'architect': ('Architect', 'Licensed Architect'),
                'civil engineer': ('Civil Engineer', 'Engineering Professional'),
                'surveyor': ('Surveyor', 'Land Surveyor'),
                'contractor': ('Contractor', 'General Contractor'),
                
                # Business & Finance
                'consultant': ('Consultant', 'Business Consultant'),
                'analyst': ('Analyst', 'Business Analyst'),
                'banker': ('Banker', 'Banking Professional'),
                'broker': ('Broker', 'Financial Broker'),
                'realtor': ('Real Estate Agent', 'Property Sales Professional'),
                
                # Security & Emergency
                'security guard': ('Security Guard', 'Security Professional'),
                'firefighter': ('Firefighter', 'Emergency Response Professional'),
                'police': ('Police Officer', 'Law Enforcement Professional'),
            }
            
            # Check direct profession mentions first (most reliable)
            for keyword, (category, subcategory) in direct_professions.items():
                if keyword in text_lower:
                    # If keyword is delivery-like, ensure shipping context and not strong tech context
                    if category == 'Delivery Service':
                        shipping_words = ['package', 'parcel', 'courier', 'ship', 'shipping', 'pickup', 'drop', 'address', 'destination', 'delivery boy', 'doorstep']
                        if any(w in text_lower for w in shipping_words) and not has_strong_tech_context:
                            return (category, subcategory, 'online_search')
                        continue
                    # If keyword is building architect but we are in software context, map to Software Architect
                    if keyword == 'architect' and has_strong_tech_context:
                        return ('Software Developer', 'Software Architect', 'online_search')
                    return (category, subcategory, 'online_search')
            
            # Priority 1.5: Medical symptom/body-part context (before generic action words)
            medical_bodyparts_map = {
                'eye': ('Doctor', 'Ophthalmologist'), 'eyes': ('Doctor', 'Ophthalmologist'),
                'ear': ('Doctor', 'ENT Specialist'), 'ears': ('Doctor', 'ENT Specialist'),
                'nose': ('Doctor', 'ENT Specialist'), 'throat': ('Doctor', 'ENT Specialist'),
                'skin': ('Doctor', 'Dermatologist'), 'derma': ('Doctor', 'Dermatologist'),
                'heart': ('Doctor', 'Cardiologist'),
                'liver': ('Doctor', 'Gastroenterologist'), 'stomach': ('Doctor', 'Gastroenterologist'), 'abdomen': ('Doctor', 'Gastroenterologist'),
                'back': ('Doctor', 'Orthopedic Doctor'), 'knee': ('Doctor', 'Orthopedic Doctor'), 'shoulder': ('Doctor', 'Orthopedic Doctor'), 'leg': ('Doctor', 'Orthopedic Doctor'), 'arm': ('Doctor', 'Orthopedic Doctor'),
                'brain': ('Doctor', 'Neurologist'), 'neuro': ('Doctor', 'Neurologist'),
                'tooth': ('Dentist', 'General Dentist'), 'teeth': ('Dentist', 'General Dentist')
            }
            medical_symptom_tokens = [
                'problem', 'issue', 'pain', 'hurt', 'injury', 'infection', 'swelling', 'redness', 'bleeding',
                'itch', 'itching', 'irritation', 'blurry', 'vision', 'sore', 'burn', 'cut', 'wound', 'fever',
                'cough', 'cold', 'flu', 'vomit', 'nausea', 'diarrhea', 'rash'
            ]
            has_medical_context = any(tok in text_lower for tok in medical_symptom_tokens)
            bodypart_hit = None
            for bp in medical_bodyparts_map.keys():
                if bp in text_lower:
                    bodypart_hit = bp
                    break
            if has_medical_context or bodypart_hit:
                if bodypart_hit:
                    return (*medical_bodyparts_map[bodypart_hit], 'online_search')
                # Generic medical context → General Physician by default
                return ('Doctor', 'General Physician', 'online_search')
            
            # Priority 2: Action-based inference (less specific, check after direct mentions)
            action_professions = {
                # Delivery actions (BEFORE generic repair/fix)
                'deliver package': ('Delivery Service', 'Package Delivery'),
                'ship package': ('Delivery Service', 'Package Delivery'),
                'deliver parcel': ('Delivery Service', 'Package Delivery'),
                'shipping': ('Delivery Service', 'Shipping Service'),
                'courier service': ('Delivery Service', 'Courier Service'),
                
                # Household/cleaning (BEFORE generic help) - SPECIFIC PHRASES FIRST
                'clean my house': ('Housekeeper', 'Cleaning Specialist'),
                'clean my home': ('Housekeeper', 'Cleaning Specialist'),
                'clean the house': ('Housekeeper', 'Cleaning Specialist'),
                'clean the home': ('Housekeeper', 'Cleaning Specialist'),
                'clean clothes': ('Housekeeper', 'Laundry Service'),
                'wash clothes': ('Housekeeper', 'Laundry Service'),
                'do laundry': ('Housekeeper', 'Laundry Service'),
                'laundry': ('Housekeeper', 'Laundry Service'),
                'clean house': ('Housekeeper', 'Cleaning Specialist'),
                'clean home': ('Housekeeper', 'Cleaning Specialist'),
                'housekeeping': ('Housekeeper', 'Cleaning Specialist'),
                'housecleaning': ('Housekeeper', 'Cleaning Specialist'),
                
                # Event planning (SPECIFIC COMBINATIONS - check before generic "plan")
                'plan wedding': ('Event Planner', 'Wedding Planner'),
                'plan event': ('Event Planner', 'Event Coordinator'),
                'plan party': ('Event Planner', 'Event Coordinator'),
                'organize wedding': ('Event Planner', 'Wedding Planner'),
                'organize event': ('Event Planner', 'Event Coordinator'),
                'coordinate wedding': ('Event Planner', 'Wedding Planner'),
                'coordinate event': ('Event Planner', 'Event Coordinator'),
                
                # Technical actions
                'repair': ('Repair Technician', 'General Repair Specialist'),
                'fix': ('Repair Technician', 'Maintenance Specialist'),
                'install': ('Installation Technician', 'Installation Specialist'),
                
                # Creative actions
                'design': ('Designer', 'Design Professional'),
                'build': ('Builder', 'Construction Professional'),
                'create': ('Creator', 'Creative Professional'),
                
                # Management actions (AFTER specific event planning combinations)
                'manage': ('Manager', 'Management Professional'),
            }
            
            # Check action-based professions (contextual)
            for keyword, (category, subcategory) in action_professions.items():
                if keyword in text_lower:
                    # If action refers to making an app/website or mentions deploy, treat as software
                    if keyword in ['create', 'build', 'design'] and (
                        any(s in text_lower for s in app_signals) or 'deploy' in text_lower or 'deployment' in text_lower
                    ):
                        # Choose Full Stack if both build and deploy are implied; else general software developer
                        if any(v in text_lower for v in build_verbs) and ('deploy' in text_lower or 'deployment' in text_lower):
                            return ('Software Developer', 'Full Stack Developer', 'online_search')
                        return ('Software Developer', 'General Software Developer', 'online_search')
                    if category == 'Delivery Service':
                        if not has_strong_tech_context:
                            return (category, subcategory, 'online_search')
                        # else ignore ambiguous delivery in tech contexts
                    else:
                        return (category, subcategory, 'online_search')
            
            # Priority 3: Generic fallback (least specific, only if nothing else matches)
            generic_professions = {
                'help': ('General Assistant', 'Support Specialist'),
                'assist': ('Assistant', 'Support Specialist'),
                'service': ('Service Provider', 'General Service Professional'),
            }
            
            for keyword, (category, subcategory) in generic_professions.items():
                if keyword in text_lower:
                    # Do not map to generic help if we clearly have medical context
                    if has_medical_context:
                        return ('Doctor', 'General Physician', 'online_search')
                    return (category, subcategory, 'online_search')
            
        except Exception as e:
            print(f"Online search error: {e}")
        
        return None
    
    def _deduce_job_categories(self, text: str, detected_factors: Dict) -> tuple:
        """
        Deduce job category and sub-category from the requirement text
        Uses parallel validation: pattern matching + online search
        
        Args:
            text: The requirement text
            detected_factors: Dictionary of detected technical factors
            
        Returns:
            Tuple of (job_category, job_sub_category, lookup_method)
            lookup_method can be: 'primary_pattern', 'validated_online', 'online_search_override', 'extended_database', 'default_fallback'
        """
        text_lower = text.lower()
        lookup_method = 'primary_pattern'  # Default to primary pattern matching
        
        # ALWAYS perform online search in parallel for validation
        online_result = self._perform_online_search(text)
        online_category = online_result[0] if online_result else None
        online_subcategory = online_result[1] if online_result else None

        # Preferred path #1: Prototype nearest-neighbor classification (no hardcoded rules)
        nn_result = self._classify_job_category_nn(text)
        if nn_result:
            job_category, job_sub_category, lookup_method = nn_result
            if online_result:
                category_groups = {
                    'medical': ['Doctor', 'Nurse', 'Caregiver', 'Physician', 'Healthcare'],
                    'tech': ['Software Developer', 'Data Scientist', 'DevOps Engineer', 'Developer', 'Engineer', 'Programmer'],
                    'delivery': ['Delivery Service', 'Delivery Driver', 'Courier', 'Logistics'],
                    'household': ['Housekeeper', 'Cleaner', 'Cleaning', 'Laundry'],
                    'childcare': ['Child Care Provider', 'Nanny', 'Babysitter'],
                    'event': ['Event Planner', 'Wedding Planner', 'Coordinator'],
                    'trades': ['Plumber', 'Electrician', 'Carpenter', 'Mechanic', 'Technician'],
                    'transportation': ['Driver', 'Chauffeur'],
                    'legal': ['Lawyer', 'Attorney', 'Legal'],
                    'education': ['Teacher', 'Tutor', 'Instructor', 'Educator'],
                    'finance': ['Accountant', 'Bookkeeper', 'Financial']
                }
                def get_group(cat):
                    cat_lower = cat.lower()
                    for g, ks in category_groups.items():
                        if any(k.lower() in cat_lower for k in ks):
                            return g
                    return 'other'
                nn_group = get_group(job_category)
                online_group = get_group(online_category)
                # Check if we have strong medical context from the prompt
                medical_symptom_tokens = [
                    'problem', 'issue', 'pain', 'hurt', 'injury', 'infection', 'swelling', 'redness', 'bleeding',
                    'itch', 'itching', 'irritation', 'blurry', 'vision', 'sore', 'burn', 'cut', 'wound', 'fever',
                    'cough', 'cold', 'flu', 'vomit', 'nausea', 'diarrhea', 'rash'
                ]
                has_medical_context = any(tok in text_lower for tok in medical_symptom_tokens)
                medical_bodyparts = ['eye', 'eyes', 'ear', 'ears', 'nose', 'throat', 'skin', 'heart', 'liver', 'stomach', 'abdomen', 'back', 'knee', 'shoulder', 'leg', 'arm', 'brain', 'tooth', 'teeth']
                has_bodypart_mention = any(bp in text_lower for bp in medical_bodyparts)
                strong_medical_signal = has_medical_context and has_bodypart_mention
                # Prefer NN 'tech' over online 'delivery' when strong software context exists (to avoid 'deliver' ambiguity)
                tech_signals = [
                    'software', 'developer', 'engineer', 'architect', 'architecture', 'system design',
                    'react', 'next.js', 'react native', 'node', 'node.js', 'typescript', 'python', 'rails',
                    'aws', 'kubernetes', 'docker', 'ci/cd', 'microservices', 'distributed systems', 'saas',
                    'postgresql', 'redis', 'cloudwatch', 'sentry', 'github actions', 'infrastructure',
                    'app', 'application', 'web app', 'mobile app', 'website', 'deploy', 'deployment'
                ]
                has_strong_tech_context = any(tok in text_lower for tok in tech_signals)
                if nn_group != online_group:
                    # Prefer online medical specialist when strong medical signal exists
                    if online_group == 'medical' and strong_medical_signal:
                        print("Warning: Category mismatch detected (NN vs Online):")
                        print(f"   NN: {job_category} -> {job_sub_category} (group: {nn_group})")
                        print(f"   Online: {online_category} -> {online_subcategory} (group: {online_group})")
                        print("   Using online search result as it's more context-aware for medical specialists")
                        job_category = online_category
                        job_sub_category = online_subcategory
                        lookup_method = 'online_search_override'
                        return job_category, job_sub_category, lookup_method
                    if nn_group == 'tech' and online_group == 'delivery' and has_strong_tech_context:
                        lookup_method = 'validated_online'  # treat as validated; keep NN result
                        return job_category, job_sub_category, lookup_method
                    # Prefer NN 'tech' over online 'other' (e.g., Creator) when strong tech context exists
                    if nn_group == 'tech' and online_group == 'other' and has_strong_tech_context:
                        lookup_method = 'validated_online'
                        return job_category, job_sub_category, lookup_method
                    print("Warning: Category mismatch detected (NN vs Online):")
                    print(f"   NN: {job_category} -> {job_sub_category} (group: {nn_group})")
                    print(f"   Online: {online_category} -> {online_subcategory} (group: {online_group})")
                    print("   Using online search result as it's more context-aware")
                    job_category = online_category
                    job_sub_category = online_subcategory
                    lookup_method = 'online_search_override'
                else:
                    lookup_method = 'validated_online'
            return job_category, job_sub_category, lookup_method

        # Preferred path #2: ML-based classification (removes hardcoded keyword dependency)
        ml_result = self._classify_job_category_ml(text)
        if ml_result:
            job_category, job_sub_category, lookup_method = ml_result
            # If we also have an online result, cross-validate
            if online_result:
                category_groups = {
                    'medical': ['Doctor', 'Nurse', 'Caregiver', 'Physician', 'Healthcare'],
                    'tech': ['Software Developer', 'Data Scientist', 'DevOps Engineer', 'Developer', 'Engineer', 'Programmer'],
                    'delivery': ['Delivery Service', 'Delivery Driver', 'Courier', 'Logistics'],
                    'household': ['Housekeeper', 'Cleaner', 'Cleaning', 'Laundry'],
                    'childcare': ['Child Care Provider', 'Nanny', 'Babysitter'],
                    'event': ['Event Planner', 'Wedding Planner', 'Coordinator'],
                    'trades': ['Plumber', 'Electrician', 'Carpenter', 'Mechanic', 'Technician'],
                    'transportation': ['Driver', 'Chauffeur'],
                    'legal': ['Lawyer', 'Attorney', 'Legal'],
                    'education': ['Teacher', 'Tutor', 'Instructor', 'Educator'],
                    'finance': ['Accountant', 'Bookkeeper', 'Financial']
                }

                def get_category_group(cat):
                    cat_lower = cat.lower()
                    for group, keywords in category_groups.items():
                        if any(kw.lower() in cat_lower for kw in keywords):
                            return group
                    return 'other'

                ml_group = get_category_group(job_category)
                online_group = get_category_group(online_category)

                if ml_group != online_group:
                    print("Warning: Category mismatch detected (ML vs Online):")
                    print(f"   ML: {job_category} -> {job_sub_category} (group: {ml_group})")
                    print(f"   Online: {online_category} -> {online_subcategory} (group: {online_group})")
                    print("   Using online search result as it's more context-aware")
                    job_category = online_category
                    job_sub_category = online_subcategory
                    lookup_method = 'online_search_override'
                else:
                    lookup_method = 'validated_online'
            return job_category, job_sub_category, lookup_method
        
        # DevOps/Infrastructure (check early to avoid misclassification)
        if any(kw in text_lower for kw in ['devops', 'ci/cd', 'infrastructure']) and 'engineer' in text_lower:
            job_category = "Software Developer"
            job_sub_category = "DevOps Engineer"
            return job_category, job_sub_category, lookup_method
        
        # Data Science (check before software developer to avoid misclassification)
        elif any(kw in text_lower for kw in ['data scientist', 'data science', 'data analyst']) and 'software' not in text_lower:
            job_category = "Data Scientist"
            if 'machine learning' in text_lower or 'ml' in text_lower:
                job_sub_category = "Machine Learning Specialist"
            elif 'analytics' in text_lower or 'analyst' in text_lower:
                job_sub_category = "Data Analyst"
            else:
                job_sub_category = "General Data Scientist"
        
        # Delivery/Courier Services (check early to avoid false matches)
        elif any(kw in text_lower for kw in ['deliver', 'delivery', 'courier', 'ship', 'shipping', 'package', 'parcel']):
            job_category = "Delivery Service"
            if any(kw in text_lower for kw in ['food', 'meal', 'restaurant']):
                job_sub_category = "Food Delivery Driver"
            elif any(kw in text_lower for kw in ['package', 'parcel', 'mail']):
                job_sub_category = "Package Delivery / Courier"
            else:
                job_sub_category = "General Delivery Service"
        
        # Software Development Categories
        elif any(kw in text_lower for kw in ['software', 'developer', 'programmer', 'coding', 'application', 'app', 'website', 'web app']):
            job_category = "Software Developer"
            
            # Determine subcategory based on tech stack
            if any(kw in text_lower for kw in ['react', 'vue', 'angular', 'frontend', 'ui', 'ux']):
                if any(kw in text_lower for kw in ['node', 'express', 'backend', 'api']):
                    # Full stack
                    if 'react' in text_lower and ('node' in text_lower or 'express' in text_lower):
                        if 'mongo' in text_lower:
                            job_sub_category = "MERN Stack Developer"
                        else:
                            job_sub_category = "Full Stack Developer (React + Node.js)"
                    elif 'vue' in text_lower:
                        job_sub_category = "Full Stack Developer (Vue.js)"
                    elif 'angular' in text_lower:
                        job_sub_category = "Full Stack Developer (Angular)"
                    else:
                        job_sub_category = "Full Stack Developer"
                else:
                    # Frontend only
                    if 'react' in text_lower:
                        job_sub_category = "Frontend Developer (React)"
                    elif 'vue' in text_lower:
                        job_sub_category = "Frontend Developer (Vue.js)"
                    elif 'angular' in text_lower:
                        job_sub_category = "Frontend Developer (Angular)"
                    else:
                        job_sub_category = "Frontend Developer"
            elif any(kw in text_lower for kw in ['node', 'express', 'django', 'flask', 'fastapi', 'backend', 'api', 'server']):
                if 'node' in text_lower or 'express' in text_lower:
                    job_sub_category = "Backend Developer (Node.js)"
                elif 'django' in text_lower:
                    job_sub_category = "Backend Developer (Django)"
                elif 'flask' in text_lower:
                    job_sub_category = "Backend Developer (Flask)"
                elif 'fastapi' in text_lower:
                    job_sub_category = "Backend Developer (FastAPI)"
                else:
                    job_sub_category = "Backend Developer"
            elif any(kw in text_lower for kw in ['mobile', 'ios', 'android', 'react native', 'flutter']):
                if 'react native' in text_lower:
                    job_sub_category = "Mobile Developer (React Native)"
                elif 'flutter' in text_lower:
                    job_sub_category = "Mobile Developer (Flutter)"
                elif 'ios' in text_lower:
                    job_sub_category = "iOS Developer"
                elif 'android' in text_lower:
                    job_sub_category = "Android Developer"
                else:
                    job_sub_category = "Mobile Developer"
            elif any(kw in text_lower for kw in ['ai', 'ml', 'machine learning', 'data science', 'neural network']):
                job_sub_category = "AI/ML Developer"
            elif any(kw in text_lower for kw in ['data scientist', 'data science', 'analytics']):
                job_sub_category = "Data Scientist"
            elif any(kw in text_lower for kw in ['devops', 'ci/cd', 'pipeline', 'cloud', 'aws', 'azure', 'kubernetes', 'docker']):
                job_sub_category = "DevOps Engineer"
            else:
                job_sub_category = "General Software Developer"
                
        # Medical/Healthcare Categories
        elif any(kw in text_lower for kw in ['doctor', 'physician', 'medical', 'health problem', 'disease', 'illness', 'treatment', 'liver', 'heart', 'stomach', 'pain', 'sick', 'injury', 'fever', 'symptom']):
            job_category = "Doctor"
            
            # Determine medical specialty
            if any(kw in text_lower for kw in ['liver', 'stomach', 'digestive', 'intestine', 'gastro']):
                job_sub_category = "Gastroenterologist"
            elif any(kw in text_lower for kw in ['heart', 'cardiac', 'cardio']):
                job_sub_category = "Cardiologist"
            elif any(kw in text_lower for kw in ['brain', 'neuro', 'nervous system']):
                job_sub_category = "Neurologist"
            elif any(kw in text_lower for kw in ['bone', 'joint', 'orthopedic', 'fracture']):
                job_sub_category = "Orthopedic Surgeon"
            elif any(kw in text_lower for kw in ['skin', 'derma', 'acne', 'rash']):
                job_sub_category = "Dermatologist"
            elif any(kw in text_lower for kw in ['child', 'pediatric', 'baby', 'kid', 'infant']) and 'care' not in text_lower:
                job_sub_category = "Pediatrician"
            elif any(kw in text_lower for kw in ['eye', 'vision', 'ophthalmology']):
                job_sub_category = "Ophthalmologist"
            else:
                job_sub_category = "General Physician"
                
        elif any(kw in text_lower for kw in ['nurse', 'nursing']):
            job_category = "Nurse"
            if any(kw in text_lower for kw in ['registered', 'rn', 'advanced']):
                job_sub_category = "Registered Nurse"
            else:
                job_sub_category = "Licensed Practical Nurse"
                
        # Elderly/Patient Care (home care, caregiver)
        elif any(kw in text_lower for kw in ['look after', 'care for', 'take care of']) and any(kw in text_lower for kw in ['dad', 'mom', 'father', 'mother', 'elderly', 'patient', 'sick', 'disabled']):
            job_category = "Caregiver"
            if any(kw in text_lower for kw in ['cook', 'meal', 'clean', 'housekeep']):
                job_sub_category = "Home Health Aide with Housekeeping"
            else:
                job_sub_category = "Home Health Aide"
                
        # Child Care Categories
        elif any(kw in text_lower for kw in ['child care', 'babysit', 'nanny', 'look after child', 'watch child', 'watch my child', 'look at my child']):
            job_category = "Child Care Provider"
            if 'nanny' in text_lower:
                job_sub_category = "Nanny"
            elif 'babysit' in text_lower:
                job_sub_category = "Babysitter"
            else:
                job_sub_category = "Child Care Specialist"
                
        # Household Services Categories
        elif any(kw in text_lower for kw in ['maid', 'housekeeper', 'cleaning', 'housekeeping', 'laundry', 'wash', 'clean clothes', 'clean my clothes', 'iron', 'dry clean']):
            job_category = "Housekeeper"
            if any(kw in text_lower for kw in ['laundry', 'wash', 'clothes', 'iron', 'dry clean']):
                job_sub_category = "Laundry Service / Housekeeper"
            elif any(kw in text_lower for kw in ['child', 'kid']):
                job_sub_category = "Housekeeper with Child Care"
            else:
                job_sub_category = "General Housekeeper"
        
        # Event Planning (weddings, events, parties)
        elif (any(kw in text_lower for kw in ['wedding', 'event', 'birthday', 'party']) and
              any(kw in text_lower for kw in ['plan', 'organize', 'coordinate'])):
            job_category = "Event Planner"
            if 'wedding' in text_lower:
                job_sub_category = "Wedding Planner"
            else:
                job_sub_category = "Event Coordinator"
                
        # Plumbing and Home Repair (check after software to avoid false matches on 'pipeline')
        elif any(kw in text_lower for kw in ['plumber', 'plumbing']) or (any(kw in text_lower for kw in ['pipe', 'leak', 'drain', 'faucet']) and 'ci/cd' not in text_lower and 'pipeline' not in text_lower):
            job_category = "Plumber"
            if any(kw in text_lower for kw in ['emergency', 'urgent', 'burst']):
                job_sub_category = "Emergency Plumber"
            else:
                job_sub_category = "General Plumber"
                
        elif any(kw in text_lower for kw in ['electrician', 'electrical', 'wiring', 'electric']):
            job_category = "Electrician"
            job_sub_category = "Licensed Electrician"
            
        elif any(kw in text_lower for kw in ['carpenter', 'woodwork', 'furniture']):
            job_category = "Carpenter"
            job_sub_category = "General Carpenter"
            
        # Transportation
        elif any(kw in text_lower for kw in ['driver', 'driving', 'chauffeur', 'transport']):
            job_category = "Driver"
            if any(kw in text_lower for kw in ['uber', 'lyft', 'taxi', 'cab']):
                job_sub_category = "Ride-share Driver"
            elif any(kw in text_lower for kw in ['truck', 'commercial']):
                job_sub_category = "Commercial Driver"
            else:
                job_sub_category = "Personal Driver"
                
        # Legal Services
        elif any(kw in text_lower for kw in ['lawyer', 'attorney', 'legal']):
            job_category = "Lawyer"
            if any(kw in text_lower for kw in ['criminal', 'defense']):
                job_sub_category = "Criminal Defense Attorney"
            elif any(kw in text_lower for kw in ['corporate', 'business']):
                job_sub_category = "Corporate Lawyer"
            else:
                job_sub_category = "General Practice Attorney"
                
        # Education
        elif any(kw in text_lower for kw in ['teacher', 'tutor', 'instructor', 'teaching']):
            job_category = "Teacher"
            if any(kw in text_lower for kw in ['math', 'mathematics']):
                job_sub_category = "Mathematics Teacher"
            elif any(kw in text_lower for kw in ['science', 'physics', 'chemistry', 'biology']):
                job_sub_category = "Science Teacher"
            elif any(kw in text_lower for kw in ['english', 'language']):
                job_sub_category = "Language Arts Teacher"
            else:
                job_sub_category = "General Educator"
                
        # Finance
        elif any(kw in text_lower for kw in ['accountant', 'accounting', 'bookkeeper']):
            job_category = "Accountant"
            if any(kw in text_lower for kw in ['cpa', 'certified']):
                job_sub_category = "Certified Public Accountant"
            else:
                job_sub_category = "General Accountant"
                
        # Default fallback
        else:
            # No pattern matched - rely on online search
            if online_result:
                job_category, job_sub_category = online_category, online_subcategory
                lookup_method = 'online_search'
            else:
                job_category = "General Professional"
                job_sub_category = "Unspecified Specialty"
                lookup_method = 'default_fallback'
        
        # CROSS-VALIDATION: Compare pattern-matched result with online search
        # If they differ significantly, check for potential false positives
        if online_result and lookup_method == 'primary_pattern':
            pattern_category = job_category
            pattern_subcategory = job_sub_category
            
            # Define category mappings for validation (categories that should match)
            category_groups = {
                'medical': ['Doctor', 'Nurse', 'Caregiver', 'Physician', 'Healthcare'],
                'tech': ['Software Developer', 'Data Scientist', 'DevOps Engineer', 'Developer', 'Engineer', 'Programmer'],
                'delivery': ['Delivery Service', 'Delivery Driver', 'Courier', 'Logistics'],
                'household': ['Housekeeper', 'Cleaner', 'Cleaning', 'Laundry'],
                'childcare': ['Child Care Provider', 'Nanny', 'Babysitter'],
                'event': ['Event Planner', 'Wedding Planner', 'Coordinator'],
                'trades': ['Plumber', 'Electrician', 'Carpenter', 'Mechanic', 'Technician'],
                'transportation': ['Driver', 'Chauffeur'],
                'legal': ['Lawyer', 'Attorney', 'Legal'],
                'education': ['Teacher', 'Tutor', 'Instructor', 'Educator'],
                'finance': ['Accountant', 'Bookkeeper', 'Financial']
            }
            
            # Helper function to get group for a category
            def get_category_group(cat):
                cat_lower = cat.lower()
                for group, keywords in category_groups.items():
                    if any(kw.lower() in cat_lower for kw in keywords):
                        return group
                return 'other'
            
            pattern_group = get_category_group(pattern_category)
            online_group = get_category_group(online_category)
            
            # If groups don't match, we may have a false positive
            if pattern_group != online_group:
                # Log the conflict for debugging
                print("Warning: Category mismatch detected:")
                print(f"   Pattern matching: {pattern_category} -> {pattern_subcategory} (group: {pattern_group})")
                print(f"   Online search: {online_category} -> {online_subcategory} (group: {online_group})")
                print(f"   Using online search result as it's more context-aware")
                
                # Override with online search result
                job_category = online_category
                job_sub_category = online_subcategory
                lookup_method = 'online_search_override'
            else:
                # Categories align - pattern matching validated by online search
                lookup_method = 'validated_online'
        
        return job_category, job_sub_category, lookup_method


