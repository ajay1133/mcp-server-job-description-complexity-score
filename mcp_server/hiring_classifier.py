import os
from pathlib import Path
from typing import List, Tuple
import json
import joblib
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'software', 'hiring_build_classifier.joblib')
DATA_PATH_ENV = 'HIRING_BUILD_DATA'

@dataclass
class HiringBuildClassifier:
    """Binary classifier to distinguish hiring/job description prompts from build/implementation requirements.

    Expected persisted object: dict { 'vectorizer': TfidfVectorizer, 'model': sklearn estimator }
    """
    model_path: str = MODEL_PATH
    _vectorizer: TfidfVectorizer | None = None
    _model: LogisticRegression | None = None

    def __post_init__(self):
        if os.path.isfile(self.model_path):
            bundle = joblib.load(self.model_path)
            self._vectorizer = bundle['vectorizer']
            self._model = bundle['model']
        else:
            raise FileNotFoundError(f"Hiring/build classifier not found at {self.model_path}. Train it with train_hiring_classifier.py.")

    def predict_proba(self, text: str) -> float:
        if self._vectorizer is None or self._model is None:
            return 0.0
        X = self._vectorizer.transform([text])
        try:
            if hasattr(self._model, 'predict_proba'):
                return float(self._model.predict_proba(X)[0][1])
            # Decision function fallback
            dec = self._model.decision_function(X)
            import numpy as np
            return float(1.0 / (1.0 + np.exp(-float(dec))))
        except Exception:
            # Model not fitted or unsupported; return neutral probability
            return 0.5

# ---------------- Training script helper -----------------

def load_dataset(path: str) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            txt = obj.get('text') or obj.get('requirement')
            label = obj.get('label')
            if txt is None or label is None:
                continue
            texts.append(str(txt))
            labels.append(int(label))
    return texts, labels


def train_and_save(dataset_path: str, model_path: str = MODEL_PATH, min_samples: int = 50):
    texts, labels = load_dataset(dataset_path)
    if len(texts) < min_samples:
        print(f"Warning: Only {len(texts)} examples (recommended: {min_samples}+). Model may not be reliable.")
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1,2), max_features=20000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:,1]

    print(classification_report(y_test, y_pred, digits=3))
    try:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.3f}")
    except Exception:
        pass

    bundle = { 'vectorizer': vectorizer, 'model': model }
    Path(os.path.dirname(model_path)).mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    print(f"Saved hiring/build classifier to {model_path}")

if __name__ == '__main__':
    data_path = os.getenv(DATA_PATH_ENV)
    if not data_path:
        raise SystemExit(f"Set {DATA_PATH_ENV} to the path of your labeled JSONL dataset.")
    train_and_save(data_path)
