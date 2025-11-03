import re
from typing import Dict, List

class ComplexityScorer:
    def __init__(self):
        self.replit_agent_3_baseline = 100
        
        self.complexity_factors = {
            'basic_web': {
                'keywords': [' html', ' css', 'landing page', 'webpage', 'static site'],
                'weight': 30
            },
            'database': {
                'keywords': [' database ', 'postgresql', 'mysql', 'mongodb', 'sqlite', ' orm ', 'migrations', 'schema', 'nosql'],
                'weight': 33
            },
            'api_integration': {
                'keywords': ['rest api', 'graphql', 'webhook', 'third-party', 'oauth', 'stripe', 'payment integration', ' api ', 'endpoints'],
                'weight': 29
            },
            'frontend': {
                'keywords': ['react', 'vue', 'angular', 'frontend', 'responsive', 'javascript', 'typescript', 'next.js', 'svelte'],
                'weight': 24
            },
            'backend': {
                'keywords': ['backend', 'flask', 'django', 'fastapi', 'node.js', 'nodejs', 'express', 'microservice', 'server-side'],
                'weight': 24
            },
            'real_time': {
                'keywords': ['websocket', 'real-time', 'real time', 'streaming', 'socket.io', 'sse', 'collaborative'],
                'weight': 30
            },
            'ai_ml': {
                'keywords': ['machine learning', 'neural network', 'openai', 'ai-powered', 'recommendation engine', 'model training'],
                'weight': 38
            },
            'deployment': {
                'keywords': ['deployment', 'ci/cd', 'docker', 'kubernetes', ' aws ', 'azure', 'cloud platform', 'heroku'],
                'weight': 15
            },
            'security': {
                'keywords': ['security', 'encryption', ' jwt ', 'jwt token', 'authentication', 'authorization', 'password hash', 'user registration', 'login', 'logout'],
                'weight': 28
            },
            'testing': {
                'keywords': ['testing suite', 'unit test', 'integration test', ' e2e ', 'pytest', 'jest', 'test coverage'],
                'weight': 14
            },
            'scalability': {
                'keywords': ['scalable', 'load balancing', 'caching', 'redis', 'performance optimization', ' kafka', 'message queue'],
                'weight': 26
            }
        }
        
        self.task_size_multipliers = {
            'simple': 0.9,
            'moderate': 1.0,
            'complex': 1.0,
            'very_complex': 1.12,
            'expert': 1.25
        }
    
    def analyze_text(self, text: str) -> Dict:
        text_lower = text.lower()
        detected_factors = {}
        total_weight = 0
        
        for factor_name, factor_data in self.complexity_factors.items():
            matches = sum(1 for keyword in factor_data['keywords'] if keyword in text_lower)
            if matches > 0:
                detected_factors[factor_name] = {
                    'matches': matches,
                    'weight': factor_data['weight'],
                    'contribution': factor_data['weight']
                }
                total_weight += factor_data['weight']
        
        task_size = self._estimate_task_size(text, len(detected_factors))
        multiplier = self.task_size_multipliers.get(task_size, 1.0)
        
        scaling_factor = 1.0
        base_score = total_weight * scaling_factor
        
        final_score = base_score * multiplier
        
        difficulty_rating = self._get_difficulty_rating(final_score)
        
        return {
            'complexity_score': round(final_score, 2),
            'baseline_reference': self.replit_agent_3_baseline,
            'detected_factors': detected_factors,
            'task_size': task_size,
            'size_multiplier': multiplier,
            'difficulty_rating': difficulty_rating,
            'summary': self._generate_summary(final_score, detected_factors)
        }
    
    def _estimate_task_size(self, text: str, num_factors: int) -> str:
        word_count = len(text.split())
        text_lower = text.lower()
        
        complexity_indicators = {
            'expert': ['microservice', 'distributed system', 'enterprise-grade'],
            'very_complex': ['architecture', 'comprehensive system'],
            'complex': ['full-stack', 'advanced'],
            'simple': ['simple html', 'basic html', 'basic css', 'landing page']
        }
        
        for size, indicators in complexity_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                return size
        
        if num_factors >= 7:
            return 'very_complex'
        elif num_factors >= 5:
            return 'complex'
        elif num_factors >= 3:
            return 'moderate'
        elif num_factors >= 1:
            return 'moderate'
        else:
            return 'simple'
    
    def _get_difficulty_rating(self, score: float) -> str:
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
    
    def _generate_summary(self, score: float, factors: Dict) -> str:
        factor_names = list(factors.keys())
        if not factor_names:
            return "Low complexity task with minimal technical requirements."
        
        top_factors = sorted(
            factor_names,
            key=lambda x: factors[x]['contribution'],
            reverse=True
        )[:3]
        
        summary = f"Complexity score: {score:.2f}. "
        summary += f"Primary complexity factors: {', '.join(top_factors)}. "
        
        return summary
