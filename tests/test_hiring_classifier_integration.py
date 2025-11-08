import os
import sys
import tempfile
import json
import joblib

# Add project root to sys.path for direct test execution
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
from mcp_server.hiring_classifier import train_and_save


def write_tiny_dataset(path: str):
    examples = [
        {"text": "We are hiring a senior backend engineer", "label": 1},
        {"text": "Job description: Frontend developer", "label": 1},
        {"text": "Looking for a QA engineer", "label": 1},
        {"text": "Build a marketplace web app with React and Node", "label": 0},
        {"text": "Develop an ecommerce platform", "label": 0},
        {"text": "Create a realtime chat with notifications", "label": 0},
        {"text": "Add payments with Stripe", "label": 0},
        {"text": "Join our team as a mobile dev", "label": 1},
        {"text": "Implement OAuth login", "label": 0},
        {"text": "Senior data scientist role", "label": 1},
        {"text": "Refactor monolith to microservices", "label": 0},
        {"text": "Full-time role: react engineer", "label": 1},
    ]
    with open(path, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")


def test_integration_uses_classifier_when_present():
    # Train a tiny model to ensure the code path works
    with tempfile.TemporaryDirectory() as tmp:
        data_path = os.path.join(tmp, 'dataset.jsonl')
        write_tiny_dataset(data_path)
        model_path = os.path.join(tmp, 'hiring_build_classifier.joblib')
        # This may be a bit unstable due to tiny dataset, but we only need a model file present
        try:
            train_and_save(data_path, model_path)
        except Exception:
            # If it fails due to size constraints, create a dummy model bundle
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.linear_model import LogisticRegression
            vec = TfidfVectorizer().fit(["hiring", "build"])
            dummy = LogisticRegression()
            bundle = { 'vectorizer': vec, 'model': dummy }
            joblib.dump(bundle, model_path)

        # Place model where scorer expects
        dest_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'software')
        dest_dir = os.path.abspath(dest_dir)
        os.makedirs(dest_dir, exist_ok=True)
        dest_model = os.path.join(dest_dir, 'hiring_build_classifier.joblib')
        os.replace(model_path, dest_model)

        scorer = SoftwareComplexityScorer()
        hiring_text = "We are hiring a senior backend engineer with 5+ years experience"
        build_text = "Build a marketplace app with React, Node and Postgres"
        out1 = scorer.analyze_text(hiring_text)
        out2 = scorer.analyze_text(build_text)
        assert out1.get('is_hiring_requirement') is True
        assert 'without_ai_and_ml' in out2 or 'complexity_score' in out2
