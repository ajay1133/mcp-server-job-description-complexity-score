import os
import json
import math
import time
import traceback
import inspect
from pathlib import Path
from typing import Dict, List, Any, Tuple
import threading

# -------------------------- Decorator for instance methods ------------------
def traced(fn):
    """Decorator for instance methods to log call, args, file, and duration."""
    def wrapper(self, *args, **kwargs):
        start_t = time.time()
        cpu_start = time.process_time()
        try:
            import psutil  # optional
            _proc = psutil.Process()
            rss_start = _proc.memory_info().rss
        except Exception:
            rss_start = None
        try:
            return fn(self, *args, **kwargs)
        except Exception as e:
            # Capture error details in trace and re-raise
            tb = traceback.format_exc()
            try:
                src = inspect.getsourcefile(fn) or __file__
            except Exception:
                src = __file__
            cpu_ms = (time.process_time() - cpu_start) * 1000.0
            try:
                import psutil  # optional
                _proc = psutil.Process()
                rss_end = _proc.memory_info().rss
                mem_kb = int(((rss_end - (rss_start or rss_end)))/1024)
            except Exception:
                mem_kb = None
            self._trace_log(
                fn.__name__, args, kwargs, start_t, src,
                cpu_ms=cpu_ms, mem_kb=mem_kb,
                error={"type": type(e).__name__, "message": str(e)}, tb=tb
            )
            raise
        finally:
            if 'e' not in locals():
                try:
                    src = inspect.getsourcefile(fn) or __file__
                except Exception:
                    src = __file__
                cpu_ms = (time.process_time() - cpu_start) * 1000.0
                try:
                    import psutil  # optional
                    _proc = psutil.Process()
                    rss_end = _proc.memory_info().rss
                    mem_kb = int(((rss_end - (rss_start or rss_end)))/1024)
                except Exception:
                    mem_kb = None
                # args already exclude self
                self._trace_log(fn.__name__, args, kwargs, start_t, src, cpu_ms=cpu_ms, mem_kb=mem_kb)
    return wrapper

import joblib
import numpy as np
from .online_tech_inference import infer_technologies_from_web, infer_complexity_multiplier_from_web
try:
    # Optional binary classifier to distinguish hiring vs build prompts
    from .hiring_classifier import HiringBuildClassifier
except Exception:
    HiringBuildClassifier = None  # type: ignore


class SoftwareComplexityScorer:
    """
    Software-only complexity scorer.

    This scorer focuses strictly on software/computer requirements and returns an error
    for non-software prompts based on a trained classifier.
    
    Output schema (final response highlights):
    - Root: technologies (split), microservices, predicted_lines_of_code, data_flow, complexity_score
    - without_ai_and_ml: { time_estimation: { hours_min, hours_avg, hours_max, ... } }
    - with_ai_and_ml: { extra_technologies, speedup_details, time_estimation }
    - Note: system_design_plan and proposed_system_design are not included; only data_flow is exposed at root
    - complexity_score: multi-factor 0–100 scaled metric
    """

    def __init__(self, model_dir: str | None = None):
        # Setup tracing/logging context (per thread)
        self._trace_local = threading.local()
        # Directory for per-request log files (override with SOFTWARE_LOG_DIR)
        # Default to logs/complexity_scorer_logs under project root
        self._log_dir = os.getenv(
            'SOFTWARE_LOG_DIR',
            os.path.join(os.path.dirname(__file__), '..', 'logs', 'complexity_scorer_logs')
        )
        try:
            Path(self._log_dir).mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'software')
        self.model_dir = os.path.abspath(model_dir)
        self._load_models()
        # Optional hiring/build classifier
        self._hiring_clf = None
        if HiringBuildClassifier is not None:
            try:
                self._hiring_clf = HiringBuildClassifier()
            except Exception:
                self._hiring_clf = None
        # Load system design patterns for comprehensive tech inference
        self._system_design_patterns = self._load_system_design_patterns()
        self._technology_difficulty = self._load_technology_difficulty()
        self._realistic_loc_config = self._load_realistic_loc_config()
        
        # Load system design and technology criticality classifiers
        self._system_design_clf = None
        self._system_design_vec = None
        self._tech_criticality_clf = None
        self._tech_criticality_vec = None
        self._tech_loc_overhead_map = {}
        self._load_design_models()

    def _load_realistic_loc_config(self) -> Dict[str, Any]:
        """Load realistic LOC estimation config based on GitHub analysis."""
        config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'realistic_loc_estimates.json')
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            # Fallback to empty config if file doesn't exist
            return {
                "base_component_loc": {},
                "complexity_multipliers": {},
                "ai_coding_speed": {"hours_per_1000_loc": 0.77}
            }

    def _load_design_models(self) -> None:
        """Load system design and technology criticality models."""
        import joblib
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        try:
            # Load system design classifier
            self._system_design_clf = joblib.load(os.path.join(models_dir, 'system_design_classifier.pkl'))
            self._system_design_vec = joblib.load(os.path.join(models_dir, 'system_design_vectorizer.pkl'))
        except Exception:
            pass  # Models optional

    # -------------------------- Mention helpers ---------------------------
    @staticmethod
    def _mention_aliases() -> Dict[str, List[str]]:
        """Aliases to detect tech mentions in prompt text."""
        return {
            'aws_lambda': ['lambda', 'aws lambda', 'lambda function', 'lambda functions'],
            'api_gateway': ['api gateway', 'apigateway', 'api-gateway', 'aws api gateway'],
            'dynamodb': ['dynamodb', 'dynamo db'],
        }

    def _is_tech_mentioned(self, tech: str, prompt: str) -> bool:
        t = (prompt or '').lower()
        candidates = {tech.lower(), tech.replace('_', ' ').lower()}
        aliases = self._mention_aliases().get(tech, [])
        candidates.update(a.lower() for a in aliases)
        return any(c in t for c in candidates)
        
        try:
            # Load technology criticality classifier
            self._tech_criticality_clf = joblib.load(os.path.join(models_dir, 'tech_criticality_classifier.pkl'))
            self._tech_criticality_vec = joblib.load(os.path.join(models_dir, 'tech_criticality_vectorizer.pkl'))
            
            # Load LOC overhead mapping
            loc_map_path = os.path.join(models_dir, 'tech_loc_overhead_map.json')
            with open(loc_map_path, 'r', encoding='utf-8') as f:
                self._tech_loc_overhead_map = json.load(f)
        except Exception:
            pass  # Models optional

    # -------------------------- Tracing utilities ---------------------------
    def _trace_start(self, prompt: str) -> None:
        """Begin a trace session for a request. Creates a unique log file name."""
        ts = time.time()
        # Use high-resolution timestamp for uniqueness
        stamp = time.strftime('%Y%m%dT%H%M%S', time.gmtime(ts)) + f"_{int((ts % 1) * 1000):03d}"
        log_filename = f"log-{stamp}.json"
        log_path = os.path.join(self._log_dir, log_filename)
        self._trace_local.session = {
            'prompt': prompt,
            'start_time': ts,
            'calls': [],
            'log_path': log_path
        }

    def _trace_log(self, name: str, args: tuple, kwargs: dict, start_t: float, file_path: str, *, cpu_ms: float | None = None, mem_kb: int | None = None, error: Dict[str, Any] | None = None, tb: str | None = None) -> None:
        """Append a function call record to the current session."""
        sess = getattr(self._trace_local, 'session', None)
        if sess is None:
            return
        duration_ms = (time.time() - start_t) * 1000.0
        # Sanitize args: keep simple repr, truncate long values
        def _safe(v):
            try:
                r = repr(v)
            except Exception:
                r = str(type(v))
            if len(r) > 256:
                r = r[:256] + '…'
            return r
        call = {
            'name': name,
            'args': [_safe(a) for a in args],
            'kwargs': {k: _safe(v) for k, v in (kwargs or {}).items()},
            'file': self._relpath(file_path),
            'duration_ms': round(duration_ms, 2),
            'ts': time.time()
        }
        if cpu_ms is not None:
            call['cpu_time_ms'] = round(cpu_ms, 2)
        if mem_kb is not None:
            call['mem_delta_kb'] = mem_kb
        if error is not None:
            call['error'] = error
            if tb:
                # Truncate very long tracebacks
                call['traceback'] = tb if len(tb) < 4000 else (tb[:4000] + '…')
        sess['calls'].append(call)

    def _trace_end(self, response: dict) -> None:
        """Finish trace session and write JSONL to log file."""
        sess = getattr(self._trace_local, 'session', None)
        if sess is None:
            return
        end_time = time.time()
        total_ms = (end_time - sess['start_time']) * 1000.0
        # Build summary stats
        calls = sess.get('calls', [])
        total_calls_duration = round(sum(c.get('duration_ms', 0) for c in calls), 2)
        total_calls_cpu = round(sum(c.get('cpu_time_ms', 0) for c in calls), 2)
        breakdown: Dict[str, Dict[str, Any]] = {}
        for c in calls:
            n = c.get('name', 'unknown')
            b = breakdown.setdefault(n, {'count': 0, 'duration_ms': 0.0, 'cpu_time_ms': 0.0})
            b['count'] += 1
            b['duration_ms'] = round(b['duration_ms'] + c.get('duration_ms', 0), 2)
            b['cpu_time_ms'] = round(b['cpu_time_ms'] + c.get('cpu_time_ms', 0), 2)

        record = {
            'prompt': sess.get('prompt'),
            'start_time': sess.get('start_time'),
            'end_time': end_time,
            'total_duration_ms': round(total_ms, 2),
            'calls': calls,
            'summary': {
                'call_count': len(calls),
                'total_calls_duration_ms': total_calls_duration,
                'total_calls_cpu_ms': total_calls_cpu,
                'breakdown_by_function': breakdown
            },
            'response': response,
        }
        log_path = sess.get('log_path')
        try:
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
        finally:
            try:
                del self._trace_local.session
            except Exception:
                pass

    @staticmethod
    def _project_root() -> str:
        # Assume repo root has pyproject.toml at top-level
        cur = Path(__file__).resolve()
        for p in [cur] + list(cur.parents):
            if (p.parent / 'pyproject.toml').exists():
                return str(p.parent)
        # Fallback two levels up
        return str(Path(__file__).resolve().parents[2])

    @classmethod
    def _relpath(cls, file_path: str) -> str:
        try:
            return os.path.relpath(file_path, cls._project_root())
        except Exception:
            return file_path

    @staticmethod
    def _format_time_human_readable(hours: float) -> str:
        """Convert hours to human-readable time string.
        
        Examples:
        - 0.5h -> "30 minutes"
        - 2h -> "2 hours"
        - 48h -> "2 days"
        - 168h -> "1 week"
        - 2190h -> "3 months"
        - 8760h -> "1 year"
        """
        if hours < 1:
            minutes = int(hours * 60)
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif hours < 24:
            h = round(hours, 1)
            return f"{h} hour{'s' if h != 1 else ''}"
        elif hours < 168:  # Less than a week
            days = round(hours / 24, 1)
            return f"{days} day{'s' if days != 1 else ''}"
        elif hours < 730:  # Less than a month (30.4 days)
            weeks = round(hours / 168, 1)
            return f"{weeks} week{'s' if weeks != 1 else ''}"
        elif hours < 8760:  # Less than a year
            months = round(hours / 730, 1)
            return f"{months} month{'s' if months != 1 else ''}"
        else:
            years = round(hours / 8760, 1)
            return f"{years} year{'s' if years != 1 else ''}"

    def _get_human_speed_lines_stats(self) -> Dict[str, float]:
        """Return average and fastest (max) human coding speed (lines/sec) from repo logs.

        Attempts to read structured JSON produced by ratio scripts:
          logs/human_ai_code_ratio/human_ai_code_ratio.json
          human_ai_code_ratio_temp.json
        Falls back to conservative defaults if unavailable.
        Caches results for subsequent calls.
        """
        if self._human_speed_cache is not None:
            return self._human_speed_cache
        root = self._project_root()
        candidate_paths = [
            os.path.join(root, 'logs', 'human_ai_code_ratio', 'human_ai_code_ratio.json'),
            os.path.join(root, 'human_ai_code_ratio_temp.json')
        ]
        speeds: List[float] = []
        for pth in candidate_paths:
            try:
                if not os.path.exists(pth):
                    continue
                with open(pth, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                entries = data.get('repos') if isinstance(data, dict) else data
                if isinstance(entries, list):
                    for e in entries:
                        if isinstance(e, dict):
                            v = e.get('human_lines_per_sec')
                            if isinstance(v, (int, float)) and v > 0:
                                speeds.append(float(v))
                if speeds:
                    break
            except Exception:
                continue
        if not speeds:
            # Defaults derived from documented analysis summary (keeps range meaningful)
            avg_lps = 2.0e-06  # ~median across repos
            max_lps = 1.42e-05  # fastest observed
        else:
            avg_lps = float(sum(speeds) / len(speeds))
            max_lps = float(max(speeds))
        self._human_speed_cache = {
            'avg_lines_per_sec': avg_lps,
            'max_lines_per_sec': max_lps
        }
        return self._human_speed_cache

    def _traced(self, fn):
        """Decorator to trace function calls and durations."""
        def wrapper(*args, **kwargs):
            start_t = time.time()
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    src = inspect.getsourcefile(fn) or __file__
                except Exception:
                    src = __file__
                self._trace_log(fn.__name__, args[1:] if len(args) and args[0] is self else args, kwargs, start_t, src)
        return wrapper

    def _load_models(self) -> None:
        try:
            self.vectorizer = joblib.load(os.path.join(self.model_dir, 'tfidf_vectorizer.joblib'))
            self.software_classifier = joblib.load(os.path.join(self.model_dir, 'software_classifier.joblib'))
            self.tech_classifier = joblib.load(os.path.join(self.model_dir, 'tech_multilabel_classifier.joblib'))
            # Kept for backward compatibility; LOC is no longer surfaced but may influence legacy score model
            self.loc_regressor = joblib.load(os.path.join(self.model_dir, 'loc_regressor.joblib'))
            self.time_regressor = joblib.load(os.path.join(self.model_dir, 'time_regressor.joblib'))
            # optional score model; if missing, we'll compute score heuristically from LOC/tech
            score_path = os.path.join(self.model_dir, 'score_regressor.joblib')
            self.score_regressor = joblib.load(score_path) if os.path.isfile(score_path) else None
        except FileNotFoundError as e:
            raise RuntimeError(f"Software models not found in {self.model_dir}. Run train_software_models.py first.") from e

    def _load_system_design_patterns(self) -> Dict[str, Any]:
        """Load system design patterns from config file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'system_design_patterns.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {"application_patterns": {}, "infrastructure_components": {}}
    
    def _load_technology_difficulty(self) -> Dict[str, Any]:
        """Load technology difficulty ratings from config file."""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'technology_difficulty.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    @traced
    def _predict_is_software(self, text: str) -> float:
        X = self.vectorizer.transform([text])
        if hasattr(self.software_classifier, 'predict_proba'):
            proba = self.software_classifier.predict_proba(X)[0][-1]
        else:
            # Some linear models may not expose predict_proba; use decision_function as proxy
            dec = self.software_classifier.decision_function(X)
            proba = 1.0 / (1.0 + np.exp(-float(dec)))
        # Boost probability if prompt contains software development verbs without requiring "developer"
        t_lower = text.lower()
        software_verbs = ['develop', 'build', 'create', 'implement', 'code', 'program', 'design']
        software_nouns = ['app', 'application', 'website', 'platform', 'system', 'clone', 'api', 'service']
        has_verb = any(verb in t_lower for verb in software_verbs)
        has_noun = any(noun in t_lower for noun in software_nouns)
        # If contains software verb + noun (e.g., "develop a twitter clone"), boost confidence
        if has_verb and has_noun and proba < 0.85:
            proba = max(proba, 0.90)
        return float(proba)

    @traced
    def _predict_technologies(self, text: str) -> List[str]:
        X = self.vectorizer.transform([text])
        if hasattr(self.tech_classifier, 'predict_proba'):
            probs = self.tech_classifier.predict_proba(X)[0]
            tech_labels = getattr(self.tech_classifier, 'classes_', None)
            if tech_labels is None:
                return []
            # Lower threshold for multi-label tech detection
            return [str(tech_labels[i]) for i, p in enumerate(probs) if p >= 0.15]
        # Fallback to binary predictions
        preds = self.tech_classifier.predict(X)[0]
        tech_labels = getattr(self.tech_classifier, 'classes_', None)
        return [str(tech_labels[i]) for i, v in enumerate(preds) if v == 1]



    @staticmethod
    def _is_hiring_requirement(text: str) -> bool:
        """Heuristic detection for hiring/job description style prompts.

        Returns True if multiple common hiring phrases are detected. Intentionally
        lightweight and non-invasive; improves precision by short-circuiting
        estimation for job postings while still allowing skills complexity scoring.
        """
        t = (text or "").lower()
        if not t:
            return False

        signals = [
            'looking for', 'we are hiring', 'hiring', 'join our team', 'seeking', 'candidate',
            'job description', 'jd', 'requirements:', 'qualifications', 'responsibilities',
            'apply', 'send your resume', 'cv', 'salary', 'ctc', 'compensation',
            'years experience', 'years of experience', '+ years', 'yrs', 'experience:',
            'full-time', 'full time', 'part-time', 'part time', 'remote position', 'immediate joiner',
            'should have', 'must have', 'preferred', 'nice to have', 'role:', 'skills:'
        ]

        # Count how many unique signals appear
        count = 0
        for s in signals:
            if s in t:
                count += 1
        
        # Additional pattern: "X+ years" or "X years" e.g., "5+ years", "3+ years", "7 years"
        import re
        if re.search(r'\d+\s*\+?\s*years?', t) or re.search(r'\d+\s*\+?\s*yrs?', t):
            count += 1
        
        # If we have 2+ signals, it's likely a hiring requirement
        if count >= 2:
            return True
        return False

    @traced
    def _predict_is_hiring(self, text: str) -> tuple[bool, float, str]:
        """Predict whether text is a hiring prompt using model if available, with heuristic fallback.

        Returns (is_hiring, proba_hiring, source)
        source in {"model", "heuristic", "model_low_conf+heuristic"}
        """
        t = (text or "").strip()
        if not t:
            return False, 0.0, "heuristic"
        # If optional classifier is available, prefer it when confident
        if self._hiring_clf is not None:
            try:
                proba = float(self._hiring_clf.predict_proba(t))
                if proba >= 0.65:
                    return True, proba, "model"
                if proba <= 0.35:
                    return False, proba, "model"
                # Low confidence, fall back to heuristic
                return self._is_hiring_requirement(t), proba, "model_low_conf+heuristic"
            except Exception:
                pass
        # Heuristic only
        return self._is_hiring_requirement(t), 0.0, "heuristic"

    @staticmethod
    def _extract_experience_requirements(text: str, technologies: List[str]) -> Dict[str, float]:
        """Extract years of experience required from hiring text.
        
        Returns a dict with:
        - 'global': overall experience requirement (if specified)
        - per-tech keys: experience for specific technologies if mentioned
        
        If a global requirement is found but no per-tech specs, assume global applies to all.
        """
        import re
        t = text.lower()
        result: Dict[str, float] = {}
        
        # Patterns for experience: "5+ years", "5 years", "3-5 years", "5+yrs", etc.
        patterns = [
            r'(\d+)\s*\+?\s*years?',
            r'(\d+)\s*\+?\s*yrs?',
            r'(\d+)\s*to\s*(\d+)\s*years?',
            r'(\d+)\s*-\s*(\d+)\s*years?',
        ]
        
        # Find global experience mention
        for pattern in patterns:
            matches = re.findall(pattern, t)
            if matches:
                # Take the first or maximum mentioned
                if isinstance(matches[0], tuple):
                    # Range like "3-5 years"
                    nums = [int(x) for x in matches[0] if x]
                    result['global'] = float(max(nums))
                else:
                    result['global'] = float(matches[0])
                break
        
        # Try to find per-technology experience mentions (e.g., "3 years in React", "5+ years node")
        for tech in technologies:
            tech_lower = tech.lower().replace('_', ' ')
            # Look for patterns like "X years in/with/of <tech>"
            tech_patterns = [
                r'(\d+)\s*\+?\s*years?\s+(?:in|with|of|using)?\s*' + re.escape(tech_lower),
                re.escape(tech_lower) + r'\s*[:\-]?\s*(\d+)\s*\+?\s*years?',
            ]
            for tp in tech_patterns:
                match = re.search(tp, t)
                if match:
                    result[tech] = float(match.group(1))
                    break
        
        # If global found but no per-tech, assume global applies to all techs
        if 'global' in result and not any(k != 'global' for k in result):
            for tech in technologies:
                result[tech] = result['global']
        elif 'global' in result:
            # Fill in missing techs with global
            for tech in technologies:
                if tech not in result:
                    result[tech] = result['global']
        
        return result

    def _predict_loc_and_time(self, text: str) -> tuple[float, float]:
        X = self.vectorizer.transform([text])
        loc = float(self.loc_regressor.predict(X)[0])
        hours = float(self.time_regressor.predict(X)[0])
        # Keep sane bounds
        loc = max(20.0, min(loc, 500_000.0))
        hours = max(1.0, min(hours, 10_000.0))
        return loc, hours

    @staticmethod
    def _compute_loc_based_complexity_score(predicted_loc: float) -> float:
        """Compute complexity score linearly from LOC with Linux 28M LOC -> score 100.

        Formula: score = (predicted_loc / 28_000_000) * 100
        Note: This score is unbounded above if LOC > 28M. For small projects,
        the score may be fractional (< 1).
        """
        try:
            base = 28_000_000.0
            if predicted_loc <= 0:
                return 0.0
            return (float(predicted_loc) / base) * 100.0
        except Exception:
            return 0.0

    def _calculate_total_boilerplate_loc(self, tech_split: Dict[str, List[str]]) -> int:
        """Calculate total boilerplate LOC from CLI scaffolding tools.
        
        This is the code provided by tools like create-react-app, django-admin startproject,
        Spring Initializr, etc. that humans get for free but AI codes from scratch.
        """
        boilerplate_config = self._realistic_loc_config.get("boilerplate_loc", {})
        total_boilerplate = 0
        
        # Check frontend technologies
        for tech in tech_split.get("frontend", []):
            total_boilerplate += boilerplate_config.get("frontend", {}).get(tech, 0)
        
        # Check backend technologies
        for tech in tech_split.get("backend", []):
            total_boilerplate += boilerplate_config.get("backend", {}).get(tech, 0)
        
        # Check infrastructure (docker, kubernetes, etc.)
        # Note: database and mobile typically don't have boilerplate deductions
        # as they're mostly configuration rather than scaffolded code
        for tech in tech_split.get("database", []):
            total_boilerplate += boilerplate_config.get("infrastructure", {}).get(tech, 0)
        
        # Check for common infrastructure patterns in all categories
        all_techs = []
        for category_techs in tech_split.values():
            all_techs.extend(category_techs)
        
        # Infrastructure items like docker, kubernetes might appear in analysis
        infra_config = boilerplate_config.get("infrastructure", {})
        for infra_tech in ["docker", "kubernetes", "terraform", "cicd"]:
            if infra_tech in all_techs:
                total_boilerplate += infra_config.get(infra_tech, 0)
        
        return total_boilerplate

    @staticmethod
    def _sigmoid(x: float, k: float = 1.0, x0: float = 0.0) -> float:
        try:
            import math
            return 1.0 / (1.0 + math.exp(-k * (x - x0)))
        except Exception:
            return 0.0

    def _compute_multifactor_complexity(self, *, text: str, predicted_loc: float, technologies: List[str], tech_split: Dict[str, List[str]], microservices: List[str], domain_multiplier: float, per_tech_complexity: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Compute a richer 0–100 complexity score using multiple normalized dimensions.

        Dimensions (with weights summing to 1.0):
          size (0.15) – log scaled LOC vs 500k cap
          tech_difficulty (0.15) – avg difficulty / 10
          tech_breadth (0.10) – sigmoid over distinct technologies count
          architecture (0.10) – microservice count curve
          feature_richness (0.10) – detected functional features / 10 capped
          domain (0.10) – normalized domain multiplier (1x–3x => 0–1)
          data_complexity (0.10) – presence & diversity of data/storage/search/analytics
          performance (0.10) – latency/throughput/realtime scaling signals
          security (0.05) – auth/encryption/compliance breadth
          ai_ml (0.05) – ML/AI model/training sophistication

        Returns dict with total score, per-dimension subscores (0–1), weighted contributions, and an explanation string.
        """
        import math, re
        t = text.lower()
        # 1. Size (log diminishing returns vs 500k reference for typical complex app domain)
        size_ref = 500_000.0
        size_sub = min(1.0, math.log1p(predicted_loc) / math.log1p(size_ref)) if predicted_loc > 0 else 0.0
        # 2. Tech difficulty
        if per_tech_complexity:
            avg_diff = sum(v.get('difficulty', 5.0) for v in per_tech_complexity.values()) / len(per_tech_complexity)
        else:
            avg_diff = 5.0
        tech_difficulty_sub = max(0.0, min(1.0, avg_diff / 10.0))
        # 3. Tech breadth (sigmoid over technology count with mid ~6)
        tech_count = len(technologies)
        tech_breadth_sub = self._sigmoid(tech_count, k=0.6, x0=6.0)
        # 4. Architecture (microservices complexity curve)
        ms = len(microservices)
        if ms <= 1:
            arch_sub = 0.1
        elif ms <= 3:
            arch_sub = 0.3
        elif ms <= 6:
            arch_sub = 0.55
        elif ms <= 10:
            arch_sub = 0.75
        else:
            arch_sub = 1.0
        # 5. Feature richness (keyword detection reused from LOC logic)
        feature_keywords = {
            "auth": ["auth", "login", "signup", "oauth", "jwt"],
            "payments": ["payment", "stripe", "paypal", "billing", "subscription", "checkout"],
            "search": ["search", "filter", "elasticsearch", "algolia"],
            "chat": ["chat", "messaging", "dm", "conversation"],
            "notifications": ["notification", "alert", "push"],
            "file_upload": ["upload", "file", "image", "storage"],
            "analytics": ["analytics", "dashboard", "metrics", "report", "chart"],
            "admin_panel": ["admin"],
            "realtime": ["websocket", "real-time", "realtime"],
            "video": ["video", "stream", "transcod"],
            "image_processing": ["image processing", "resize", "thumbnail"],
            "ml_inference": ["ml", "machine learning", "model", "inference", "prediction"],
            "recommendation": ["recommendation", "recommend"],
            "geo": ["geospatial", "map", "location"],
            "compliance": ["gdpr", "hipaa", "pci"]
        }
        detected_features = {fname for fname, kws in feature_keywords.items() if any(kw in t for kw in kws)}
        feature_richness_sub = min(1.0, len(detected_features) / 10.0)
        # 6. Domain complexity (normalize 1–3x to 0–1)
        domain_sub = max(0.0, min(1.0, (domain_multiplier - 1.0) / (3.0 - 1.0)))
        # 7. Data complexity signals
        data_signals = 0
        if any(db for db in tech_split.get('database', [])): data_signals += 1
        if any(k in technologies for k in ['redis', 'cassandra', 'elasticsearch']): data_signals += 1
        if any(k in t for k in ['data pipeline', 'etl', 'stream', 'kafka', 'event']): data_signals += 1
        if 'analytics' in detected_features: data_signals += 1
        data_complexity_sub = min(1.0, data_signals / 4.0)
        # 8. Performance / scalability
        perf_keywords = ['high throughput', 'millions of users', 'scale to', 'horizontal scaling', 'low latency', 'sub-second', 'performance', 'benchmark']
        perf_hits = sum(1 for k in perf_keywords if k in t)
        if 'realtime' in detected_features: perf_hits += 1
        performance_sub = min(1.0, perf_hits / 5.0)
        # 9. Security & compliance
        sec_hits = 0
        if any(k in detected_features for k in ['auth', 'admin_panel', 'compliance']): sec_hits += 1
        if any(k in t for k in ['encrypt', 'encryption', 'tls', 'oauth', 'jwt']): sec_hits += 1
        if any(k in t for k in ['audit log', 'auditing']): sec_hits += 1
        security_sub = min(1.0, sec_hits / 3.0)
        # 10. AI / ML sophistication
        ml_hits = 0
        if any(k in t for k in ['machine learning', 'ml', 'model', 'neural', 'transformer']): ml_hits += 1
        if any(k in t for k in ['fine-tune', 'finetune', 'training', 'train model']): ml_hits += 1
        if any(k in t for k in ['reinforcement', 'rlhf']): ml_hits += 1
        ai_ml_sub = min(1.0, ml_hits / 3.0)
        # Weights
        weights = {
            'size': 0.15,
            'tech_difficulty': 0.15,
            'tech_breadth': 0.10,
            'architecture': 0.10,
            'feature_richness': 0.10,
            'domain': 0.10,
            'data_complexity': 0.10,
            'performance': 0.10,
            'security': 0.05,
            'ai_ml': 0.05
        }
        subs = {
            'size': size_sub,
            'tech_difficulty': tech_difficulty_sub,
            'tech_breadth': tech_breadth_sub,
            'architecture': arch_sub,
            'feature_richness': feature_richness_sub,
            'domain': domain_sub,
            'data_complexity': data_complexity_sub,
            'performance': performance_sub,
            'security': security_sub,
            'ai_ml': ai_ml_sub
        }
        # Weighted sum
        raw_score = sum(subs[k] * weights[k] for k in weights)
        score_0_100 = raw_score * 100.0
        # Baseline floor for any legitimate software requirement
        if score_0_100 > 0 and score_0_100 < 5:
            score_0_100 = 5.0
        # Structured explanation list
        explanation = [{
            'dimension': k,
            'subscore': round(subs[k], 3),
            'weight': round(weights[k], 3),
            'weighted_contribution': round(subs[k] * weights[k] * 100.0, 2)
        } for k in weights]
        # Sort by highest contribution first for readability
        explanation.sort(key=lambda d: d['weighted_contribution'], reverse=True)
        # Add percent_of_total for each dimension
        if score_0_100 > 0:
            for item in explanation:
                pct = (item['weighted_contribution'] / score_0_100) * 100.0
                item['percent_of_total'] = round(pct, 2)
        else:
            for item in explanation:
                item['percent_of_total'] = 0.0
        total_obj = {
            'total_score': round(score_0_100, 2),
            'note': 'Multi-factor complexity (0-100) with diminishing returns & domain/context factors'
        }
        return {
            'complexity_score_v2': round(score_0_100, 2),
            'subscores': subs,
            'weights': weights,
            'explanation': explanation,
            'explanation_total': total_obj
        }

    @traced
    def _calculate_realistic_loc(self, text: str, technologies: List[str], tech_split: Dict[str, List[str]], 
                                   services: List[str]) -> float:
        """
        Calculate realistic LOC based on GitHub analysis and system design knowledge.
        
        Uses component-based estimation:
        1. Sum base LOC for each technology/feature
        2. Apply complexity multipliers based on:
           - Number of microservices
           - Integration complexity
           - Architecture style
        3. Clamp to reasonable bounds
        
        Returns: Estimated LOC as float
        """
        config = self._realistic_loc_config
        base_components = config.get("base_component_loc", {})
        multipliers = config.get("complexity_multipliers", {})
        
        # Start with base LOC from components
        total_base_loc = 0
        
        # Add LOC for each technology category
        for category, techs in tech_split.items():
            category_components = base_components.get(category, {})
            for tech in techs:
                tech_loc = category_components.get(tech, 1500)  # Default 1500 LOC per tech
                total_base_loc += tech_loc
        
        # Add LOC for detected features (auth, payments, etc.)
        feature_components = base_components.get("features", {})
        text_lower = text.lower()
        
        # Detect features from text
        feature_keywords = {
            "auth": ["auth", "login", "signup", "oauth", "jwt", "session"],
            "payments": ["payment", "stripe", "paypal", "billing", "subscription", "checkout"],
            "search": ["search", "filter", "query", "elasticsearch", "algolia"],
            "chat": ["chat", "messaging", "dm", "direct message", "conversation"],
            "notifications": ["notification", "alert", "push", "email notification"],
            "file_upload": ["upload", "file upload", "image upload", "s3", "storage"],
            "email": ["email", "sendgrid", "mailgun", "smtp"],
            "analytics": ["analytics", "dashboard", "metrics", "reporting", "charts"],
            "admin_panel": ["admin", "admin panel", "admin dashboard", "backoffice"],
            "websocket": ["websocket", "ws", "real-time", "realtime", "live"],
            "video_processing": ["video", "streaming", "transcode", "video processing"],
            "image_processing": ["image processing", "resize", "thumbnail", "image manipulation"],
            "ml_inference": ["ml", "machine learning", "ai model", "prediction", "inference"]
        }
        
        detected_features = set()
        for feature, keywords in feature_keywords.items():
            if any(kw in text_lower for kw in keywords):
                detected_features.add(feature)
                total_base_loc += feature_components.get(feature, 2000)
        
        # Apply microservices multiplier
        num_services = len(services)
        service_mult = 1.0
        service_ranges = multipliers.get("microservices", {})
        if num_services <= 2:
            service_mult = service_ranges.get("1-2", 1.0)
        elif num_services <= 5:
            service_mult = service_ranges.get("3-5", 1.3)
        elif num_services <= 10:
            service_mult = service_ranges.get("6-10", 1.6)
        elif num_services <= 15:
            service_mult = service_ranges.get("11-15", 2.0)
        else:
            service_mult = service_ranges.get("16+", 2.5)
        
        total_base_loc *= service_mult
        
        # Apply integration complexity multiplier based on number of technologies
        total_techs = sum(len(v) for v in tech_split.values() if isinstance(v, list))
        integration_mult = 1.0
        if total_techs <= 3:
            integration_mult = 1.0  # simple
        elif total_techs <= 6:
            integration_mult = 1.3  # moderate
        elif total_techs <= 10:
            integration_mult = 1.8  # complex
        else:
            integration_mult = 2.5  # enterprise
        
        total_base_loc *= integration_mult
        
        # Apply feature richness multiplier based on detected features
        num_features = len(detected_features)
        feature_mult = 1.0
        if num_features <= 2:
            feature_mult = 0.6  # mvp
        elif num_features <= 5:
            feature_mult = 1.0  # standard
        elif num_features <= 8:
            feature_mult = 1.5  # advanced
        else:
            feature_mult = 2.2  # enterprise
        
        total_base_loc *= feature_mult
        
        # Clamp to reasonable bounds
        # Minimum: 1000 LOC for simplest projects
        # Maximum: 500,000 LOC for largest projects
        final_loc = max(1000, min(total_base_loc, 500_000))
        
        return final_loc

    # ------------------ New Helpers: LOC Breakdown & Tech Complexity ------------------
    @traced
    def _estimate_loc_breakdown(self, total_loc: float, tech_split: Dict[str, List[str]]) -> Dict[str, Any]:
        """Distribute predicted LOC across categories and technologies difficulty-weighted.

        Steps:
        1. Determine active categories present.
        2. Base category weights: backend 0.45, frontend 0.30, database 0.15, mobile 0.10; re-normalize to active.
        3. Inside each category, split cat LOC proportionally to technology difficulty (fallback=5).
        4. Rounding corrections ensure total sums to total_loc.
        """
        base_weights = {'backend': 0.45, 'frontend': 0.30, 'database': 0.15, 'mobile': 0.10}
        active = [c for c, techs in tech_split.items() if techs]
        if not active:
            return {'total_loc': int(total_loc), 'by_category': {}, 'by_technology': {}}
        total_w = sum(base_weights[c] for c in active)
        norm_weights = {c: base_weights[c] / total_w for c in active}

        # Build difficulty lookup from already-computed per-tech complexity (if available on self)
        # Fallback default difficulty = 5
        per_tech_complexity = getattr(self, '_per_tech_complexity_cache', {})
        def _diff(tech: str) -> float:
            meta = per_tech_complexity.get(tech)
            if meta is not None:
                return float(meta.get('difficulty', 5.0))
            return 5.0

        by_category: Dict[str, int] = {}
        by_tech: Dict[str, int] = {}
        # First pass: compute raw allocations (float) then integer rounding
        for cat in active:
            cat_loc = total_loc * norm_weights[cat]
            techs = tech_split.get(cat, [])
            if not techs:
                continue
            diffs = [_diff(t) for t in techs]
            diff_sum = sum(diffs) or len(techs)
            # Raw per-tech loc
            raw_allocs = [cat_loc * (d / diff_sum) for d in diffs]
            int_allocs = [int(round(x)) for x in raw_allocs]
            # Adjust category rounding drift
            cat_int_sum = sum(int_allocs)
            cat_target = int(round(cat_loc))
            drift = cat_target - cat_int_sum
            if drift != 0 and techs:
                # distribute drift across top difficulty techs
                order = sorted(range(len(techs)), key=lambda i: diffs[i], reverse=True)
                sign = 1 if drift > 0 else -1
                for i in range(abs(drift)):
                    idx = order[i % len(order)]
                    int_allocs[idx] += sign
            # Assign
            by_category[cat] = sum(int_allocs)
            for t, val in zip(techs, int_allocs):
                by_tech[t] = val

        # Final global drift correction
        total_assigned = sum(by_tech.values())
        global_drift = int(total_loc) - total_assigned
        if global_drift != 0 and by_tech:
            # Adjust among highest difficulty techs
            ordered = sorted(by_tech.keys(), key=lambda t: _diff(t), reverse=True)
            sign = 1 if global_drift > 0 else -1
            for i in range(abs(global_drift)):
                tech = ordered[i % len(ordered)]
                by_tech[tech] += sign
        return {'total_loc': int(total_loc), 'by_category': by_category, 'by_technology': by_tech}

    @traced
    def _compute_per_technology_complexity(self, technologies: List[str], total_ai_hours: float = 0) -> Dict[str, Dict[str, Any]]:
        """Compute a complexity scorer per technology using difficulty + contextual multipliers.

        Uses `technology_difficulty.json` loaded into `_technology_difficulty`.
        Also estimates time per technology based on difficulty weighting.

        Output per technology:
        {
            tech: {
                'difficulty': float,
                'base_score': int (scaled 10-200),
                'category': str,
                'time_to_productivity': str,
                'estimated_time_hours': float,
                'estimated_time_human': str
            }
        }
        """
        results: Dict[str, Dict[str, Any]] = {}
        if not technologies:
            return results
        # Build reverse index of difficulty entries
        # Each top-level category houses tech entries with 'difficulty'
        difficulty_data = self._technology_difficulty
        for category, entries in difficulty_data.items():
            if category.startswith('_') or category in ['difficulty_multipliers', 'learning_order_recommendations']:
                continue
            if not isinstance(entries, dict):
                continue
            for tech_name, meta in entries.items():
                if tech_name in technologies:
                    diff = float(meta.get('difficulty', 5))
                    # Base complexity formula: map difficulty (1-10) to score 10-200 via quadratic-ish curve
                    # Emphasize higher difficulty disproportionately.
                    # score = 10 + (diff/10)^2 * 190
                    scaled = 10.0 + ((diff / 10.0) ** 2) * 190.0
                    # Provide time-to-productivity if present
                    results[tech_name] = {
                        'difficulty': diff,
                        'base_score': int(round(scaled)),
                        'category': category,
                        'time_to_productivity': meta.get('time_to_productivity', 'n/a'),
                        'reasons': meta.get('reasons', [])[:4]
                    }
        # For technologies not found in difficulty config, assign default medium difficulty
        for t in technologies:
            if t not in results:
                diff = 5.0
                scaled = 10.0 + ((diff / 10.0) ** 2) * 190.0
                results[t] = {
                    'difficulty': diff,
                    'base_score': int(round(scaled)),
                    'category': 'unknown',
                    'time_to_productivity': '3-5 weeks',
                    'reasons': []
                }
        
        # Distribute total AI time proportionally by difficulty
        if total_ai_hours > 0 and results:
            total_difficulty = sum(v['difficulty'] for v in results.values())
            for tech, meta in results.items():
                tech_hours = (meta['difficulty'] / total_difficulty) * total_ai_hours if total_difficulty > 0 else total_ai_hours / len(results)
                meta['estimated_time_hours'] = round(tech_hours, 2)
                meta['estimated_time_human'] = self._format_time_human_readable(tech_hours)
        
        # Cache for downstream loc breakdown weighting
        self._per_tech_complexity_cache = results
        return results

    def _estimate_per_technology_boilerplate_loc(self, tech_split: Dict[str, List[str]]) -> Dict[str, int]:
        """Estimate boilerplate LOC per technology using config boilerplate_loc map.

        Returns mapping: tech -> boilerplate_loc (int)
        """
        out: Dict[str, int] = {}
        try:
            config = self._realistic_loc_config.get("boilerplate_loc", {})
            for category, techs in (tech_split or {}).items():
                # category can be list (old) or already dict (new); handle lists only here
                if isinstance(techs, dict):
                    # If already detailed, collect keys
                    tech_names = list(techs.keys())
                else:
                    tech_names = techs or []
                cat_map = config.get(category, {}) if isinstance(config.get(category), dict) else {}
                for t in tech_names:
                    out[t] = int(cat_map.get(t, 0))
            # Include common infrastructure if present
            infra_map = config.get('infrastructure', {}) if isinstance(config.get('infrastructure'), dict) else {}
            for infra_tech in ["docker", "kubernetes", "terraform", "cicd"]:
                if any(infra_tech in v if isinstance(v, list) else infra_tech in (v or {}) for v in (tech_split or {}).values()):
                    out[infra_tech] = int(infra_map.get(infra_tech, 0))
        except Exception:
            pass
        return out

    @staticmethod
    def _get_alternatives(tech: str) -> List[str]:
        """Return suggested alternatives for a technology."""
        mapping = {
            # Frontend
            'react': ['angular', 'vue', 'svelte', 'nextjs'],
            'nextjs': ['react', 'nuxt', 'sveltekit'],
            'angular': ['react', 'vue', 'svelte'],
            'vue': ['react', 'angular', 'svelte'],
            # Backend
            'node': ['python_fastapi', 'python_django', 'rails', 'golang'],
            'python_fastapi': ['python_django', 'flask', 'node'],
            'python_django': ['python_fastapi', 'flask', 'rails'],
            'flask': ['python_fastapi', 'python_django', 'node'],
            'rails': ['node', 'python_django', 'golang'],
            'aws_lambda': ['node', 'python_fastapi', 'google_cloud_functions', 'azure_functions'],
            'api_gateway': ['kong', 'tyk', 'nginx', 'traefik'],
            # Databases
            'postgres': ['mysql', 'mariadb', 'mongodb'],
            'mysql': ['postgres', 'mariadb'],
            'mongodb': ['postgres', 'mysql', 'cassandra'],
            'redis': ['memcached'],
            'dynamodb': ['mongodb', 'cassandra', 'postgres'],
            # Infra
            'docker': ['podman'],
            'kubernetes': ['nomad', 'ecs'],
        }
        return mapping.get(tech, [])

    def _compute_complexity(self, text: str, technologies: List[str], microservices: List[str], hours: float, ai_hours: float, experience_requirements: Dict[str, float] | None = None) -> float:
        """Compute complexity score based on time, technology diversity, microservices, and experience requirements.
        
        For hiring prompts, experience_requirements can boost the score non-linearly based on required years.
        """
        if self.score_regressor is not None:
            X = self.vectorizer.transform([text])
            score = float(self.score_regressor.predict(X)[0])
            base_score = max(10.0, min(score, 200.0))
        else:
            # Heuristic: combine manual hours, AI hours, tech diversity and microservice count
            manual_time_component = math.log10(max(2.0, hours)) * 11.0  # ~14-35 for typical projects
            ai_time_component = math.log10(max(1.0, ai_hours)) * 7.0    # ~7-22 for typical projects
            tech_component = math.sqrt(len(technologies) + 1) * 6.0     # ~6-30 based on tech count
            ms_component = math.sqrt(len(microservices) + 1) * 8.0      # ~8-32 based on service count
            base_score = float(max(10.0, min(manual_time_component + ai_time_component + tech_component + ms_component, 200.0)))
        
        # Apply experience multiplier for hiring requirements
        if experience_requirements:
            # Compute average required experience across all technologies
            exp_values = [v for k, v in experience_requirements.items() if k != 'global' and v > 0]
            if exp_values:
                avg_exp = sum(exp_values) / len(exp_values)
                # More aggressive non-linear multiplier using log + sqrt hybrid
                # 1 year: 1.0x, 3 years: ~1.28x, 5 years: ~1.45x, 10 years: ~1.73x, 15 years: ~1.95x
                exp_multiplier = 1.0 + math.log10(max(1.0, avg_exp)) * 0.5 + (math.sqrt(max(1.0, avg_exp)) - 1.0) * 0.2
                base_score = base_score * exp_multiplier
        
        return float(max(10.0, min(base_score, 200.0)))

    # Traced wrappers around external inference functions
    @traced
    def _infer_technologies_online(self, text: str) -> List[str]:
        return infer_technologies_from_web(text)

    @traced
    def _infer_domain_multiplier(self, text: str) -> Tuple[float, List[str]]:
        return infer_complexity_multiplier_from_web(text)

    @traced
    def _get_ai_metrics_with_details(self, technologies: List[str], ai_hours: float, manual_hours: float, 
                                      predicted_loc: float, text: str) -> Dict[str, Any]:
        """
        Get AI metrics with details. 
        
        Key principle: Based on analysis of 6 major GitHub repos with adjustment:
        - AI time includes prompt overhead (proportional to project size)
        - Human time not adjusted (breaks apply to both scenarios)
        Result: AI is 98.73x faster than human (human is 1.013% of AI speed)
        Therefore: manual_time = ai_time × 98.73
        
        This method just adds metadata about the project for explanation purposes.
        """
        # Check if AI/ML is required
        ai_ml_techs = {'ai_llm', 'ml', 'cv', 'nlp', 'tensorflow', 'pytorch'}
        is_ai_required = 'ai_llm' in technologies
        is_ml_required = any(t in technologies for t in ['ml', 'cv', 'nlp', 'tensorflow', 'pytorch'])
        
        # Extra technologies needed when using AI/ML tools
        extra_technologies = []
        if is_ai_required or is_ml_required:
            extra_technologies.extend(['langchain', 'vector_db'])
        
        # Calculate average technology difficulty (for display only)
        avg_difficulty = self._calculate_average_tech_difficulty(technologies)
        
        # Determine category based on project characteristics
        text_lower = text.lower()
        
        if any(tech in technologies for tech in ai_ml_techs) or 'video_processing' in technologies:
            category = "expert domains (ML/video)"
        elif any(tech in technologies for tech in ['kubernetes', 'kafka', 'cassandra', 'erlang']):
            category = "distributed systems"
        elif any(word in text_lower for word in ['algorithm', 'workflow', 'recommendation', 'pricing', 'optimization']):
            category = "complex business logic"
        elif 'crud' in text_lower or 'todo' in text_lower or len(technologies) <= 3:
            category = "boilerplate CRUD"
        else:
            category = "standard features"
        
        # Calculate speedup factor (for display)
        speedup_factor = ai_hours / manual_hours if manual_hours > 0 else 0.0007
        time_saved_percent = (1 - speedup_factor) * 100
        
        return {
            'extra_technologies': extra_technologies,
            'time_estimation': round(ai_hours, 2),
            'speedup_factor': round(speedup_factor, 4),
            'speedup_category': category,
            'speedup_details': {
                'avg_tech_difficulty': round(avg_difficulty, 2),
                'predicted_loc': int(predicted_loc),
                'time_saved_percent': round(time_saved_percent, 1),
                'manual_hours': round(manual_hours, 2),
                'ai_hours': round(ai_hours, 2),
                'speed_ratio': '98.73x (AI is 1.013% of human time, includes prompt overhead)',
                'is_ai_required': is_ai_required
            }
        }
    
    def _calculate_average_tech_difficulty(self, technologies: List[str]) -> float:
        """Calculate average difficulty of technologies in the project."""
        if not technologies:
            return 5.0  # Default medium difficulty
        
        difficulties = []
        tech_diff_map = self._technology_difficulty
        
        for tech in technologies:
            # Search through all categories
            found = False
            for category, techs in tech_diff_map.items():
                if isinstance(techs, dict) and tech in techs:
                    if isinstance(techs[tech], dict) and 'difficulty' in techs[tech]:
                        difficulties.append(techs[tech]['difficulty'])
                        found = True
                        break
            
            # Default difficulty if not found
            if not found:
                difficulties.append(5.0)
        
        return sum(difficulties) / len(difficulties) if difficulties else 5.0

    @traced
    def _build_skills_complexity_explanation(self, text: str, skills_score: float, technologies: List[str], services: List[str], experience_requirements: Dict[str, float] | None = None) -> str:
        """Explain how the skills complexity score was derived with a breakdown of contributing factors.

        If a linear score regressor is available, we compute the top contributing
        terms from the TF-IDF vector and the model coefficients. Otherwise we
        provide a heuristic description.
        """
        lines = []
        lines.append("Skills complexity score: {:.2f} (10-200 scale)".format(skills_score))
        lines.append("")
        
        # Add breakdown of factors
        lines.append("Score breakdown:")
        lines.append("  - Technology breadth: {} technologies detected".format(len(technologies)))
        lines.append("    (More technologies = higher complexity)")
        lines.append("  - Architectural scope: {} microservices inferred".format(len(services)))
        lines.append("    (More services = higher complexity)")
        
        # Add experience requirement factor if present
        if experience_requirements:
            exp_values = [v for k, v in experience_requirements.items() if k != 'global' and v > 0]
            if exp_values:
                avg_exp = sum(exp_values) / len(exp_values)
                exp_multiplier = 1.0 + math.log10(max(1.0, avg_exp)) * 0.5 + (math.sqrt(max(1.0, avg_exp)) - 1.0) * 0.2
                lines.append("  - Experience requirement: {:.1f} years average".format(avg_exp))
                lines.append("    (Multiplier: {:.2f}x - higher experience = non-linearly higher complexity)".format(exp_multiplier))
                
                # Show per-technology experience breakdown
                tech_exp_items = [(k, v) for k, v in experience_requirements.items() if k != 'global']
                if tech_exp_items and len(tech_exp_items) > 0:
                    lines.append("    Per-technology experience:")
                    for tech, years in sorted(tech_exp_items, key=lambda x: x[1], reverse=True)[:5]:
                        lines.append("      * {}: {:.0f} years".format(tech, years))
        
        lines.append("")
        
        # Try to show model-based contributions if available
        try:
            model = self.score_regressor
            if model is not None and hasattr(model, 'coef_') and hasattr(self.vectorizer, 'get_feature_names_out'):
                X = self.vectorizer.transform([text])
                feature_names = self.vectorizer.get_feature_names_out()
                coef = getattr(model, 'coef_', None)
                # Support for multi-target or 2D shapes
                if coef is None:
                    raise AttributeError
                coef = coef.ravel()
                # Compute per-feature contribution for this sample
                x_coo = X.tocoo()
                contribs = {}
                for _, j, v in zip(x_coo.row, x_coo.col, x_coo.data):
                    contribs[j] = contribs.get(j, 0.0) + float(v * coef[j])
                # Sort by contribution
                top_positive = sorted(
                    [(idx, val) for idx, val in contribs.items() if val > 0],
                    key=lambda kv: kv[1],
                    reverse=True
                )[:5]
                top_negative = sorted(
                    [(idx, val) for idx, val in contribs.items() if val < 0],
                    key=lambda kv: kv[1]
                )[:3]
                
                if top_positive:
                    lines.append("Top terms increasing complexity:")
                    for idx, val in top_positive:
                        term = feature_names[idx]
                        lines.append("  - {}: +{:.2f}".format(term, val))
                    lines.append("")
                
                if top_negative:
                    lines.append("Terms decreasing complexity:")
                    for idx, val in top_negative:
                        term = feature_names[idx]
                        lines.append("  - {}: {:.2f}".format(term, val))
                    lines.append("")
        except Exception:
            pass
        
        lines.append("Note: This score reflects breadth/depth of skills and tooling indicated by the text.")
        lines.append("No time estimates are used for hiring prompts.")
        return "\n".join(lines)

    @traced
    def _split_technologies(self, all_techs: List[str]) -> Dict[str, List[str]]:
        frontend_set = {'react', 'nextjs', 'vue', 'angular', 'svelte'}
        backend_set = {
            'node', 'python_fastapi', 'python_django', 'flask', 'rails', 'php',
            # Serverless execution / API orchestration considered backend logic carriers
            'aws_lambda', 'api_gateway'
        }
        database_set = {'postgres', 'mysql', 'mongodb', 'redis', 'dynamodb'}
        mobile_set = {'android', 'ios', 'react_native', 'flutter'}

        frontend = [t for t in all_techs if t in frontend_set]
        backend = [t for t in all_techs if t in backend_set]
        database = [t for t in all_techs if t in database_set]
        mobile = [t for t in all_techs if t in mobile_set]
        
        # PHP is full-stack: if present, add to both frontend and backend
        if 'php' in all_techs:
            if 'php' not in frontend:
                frontend.append('php')
            if 'php' not in backend:
                backend.append('php')

        return {
            'frontend': frontend,
            'backend': backend,
            'database': database,
            'mobile': mobile,
        }

    @traced
    def _infer_microservices(self, text: str, techs: List[str]) -> List[str]:
        """Heuristic microservice suggestions based on requirement and techs."""
        t = text.lower()
        services: List[str] = []

        # Core domain hints
        def add(name: str):
            if name not in services:
                services.append(name)

        # First, check for known application patterns from system design knowledge base
        pattern_matched = False
        for pattern_name, pattern_data in self._system_design_patterns.get('application_patterns', {}).items():
            keywords = pattern_data.get('keywords', [])
            if any(kw in t for kw in keywords):
                # Match found! Use comprehensive microservices from pattern
                for svc in pattern_data.get('microservices', []):
                    add(svc)
                pattern_matched = True
                # Note: we could break here but let's continue to catch additional specific mentions
        
        # Then add specific domain hints (these may add to or complement pattern-based services)
        # Auth
        if 'auth' in t or 'jwt' in t or 'login' in t or 'oauth' in t or 'authentication' in t:
            add('auth-service')

        # Payments / billing
        if 'stripe' in t or 'payment' in t or 'billing' in t or 'checkout' in t:
            add('payments-service')
            add('billing-service')

        # Marketplace / ecommerce (unless already matched as pattern)
        if not pattern_matched and ('marketplace' in t or 'ecommerce' in t or 'e-commerce' in t or 'shop' in t or 'store' in t):
            add('catalog-service')
            add('order-service')
            add('inventory-service')

        # Chat / realtime
        if 'chat' in t or 'websocket' in t or 'real-time' in t or 'realtime' in t:
            add('realtime-service')
            add('message-service')

        # File / media (unless already in pattern)
        if ('upload' in t or 'image' in t or 'video' in t or 'file' in t) and 'media-service' not in services:
            add('media-service')

        # Search / recommendation / analytics
        if 'search' in t and 'search-service' not in services:
            add('search-service')
        if ('recommendation' in t or 'recommend' in t) and 'recommendation-service' not in services:
            add('recommendation-service')
        if ('analytics' in t or 'events' in t or 'tracking' in t) and 'analytics-service' not in services:
            add('analytics-service')

        # Generic entities
        if 'user' in t and 'user-service' not in services:
            add('user-service')
        if ('notification' in t or 'email' in t or 'sms' in t or 'push' in t) and 'notification-service' not in services:
            add('notification-service')

        # AI/ML
        if ('ai' in t or 'llm' in t or 'gpt' in t or 'ml' in t or 'machine learning' in t) and 'ml-service' not in services:
            add('ml-service')

        # Default fallbacks based on tech stack
        if not pattern_matched and any(x in techs for x in ['python_fastapi', 'flask', 'python_django', 'node', 'rails']):
            add('api-gateway')

        # If nothing inferred, provide minimal baseline
        if not services:
            services = ['api-gateway']

        return services

    @traced
    def _predict_system_design_architecture(self, text: str) -> Dict[str, Any]:
        """Predict system architecture pattern using ML model."""
        if self._system_design_clf is None or self._system_design_vec is None:
            return {"architecture": "unknown", "confidence": 0.0, "model_used": False}
        
        try:
            X = self._system_design_vec.transform([text])
            pred = self._system_design_clf.predict(X)[0]
            proba = self._system_design_clf.predict_proba(X)[0]
            classes = self._system_design_clf.classes_
            confidence = dict(zip(classes, proba))
            
            return {
                "architecture": pred,
                "confidence": round(confidence[pred], 4),
                "all_probabilities": {k: round(v, 4) for k, v in confidence.items()},
                "model_used": True
            }
        except Exception:
            return {"architecture": "unknown", "confidence": 0.0, "model_used": False}
    
    @traced
    def _analyze_technology_criticality(self, text: str, technologies: List[str]) -> List[Dict[str, Any]]:
        """Analyze each technology for criticality (mandatory/recommended/optional) and overhead."""
        if self._tech_criticality_clf is None or self._tech_criticality_vec is None:
            return []
        
        # Get difficulty ratings for all technologies
        difficulty_map = {}
        difficulty_data = self._technology_difficulty
        for category, entries in difficulty_data.items():
            if category.startswith('_') or category in ['difficulty_multipliers', 'learning_order_recommendations']:
                continue
            if not isinstance(entries, dict):
                continue
            for tech_name, meta in entries.items():
                if tech_name in technologies:
                    difficulty_map[tech_name] = float(meta.get('difficulty', 0))
        
        tech_analysis = []
        for tech in technologies:
            try:
                feature = f"{text} {tech}"
                X = self._tech_criticality_vec.transform([feature])
                pred = self._tech_criticality_clf.predict(X)[0]
                proba = self._tech_criticality_clf.predict_proba(X)[0]
                classes = self._tech_criticality_clf.classes_
                confidence = dict(zip(classes, proba))
                
                # Get LOC overhead from mapping
                loc_overhead = self._tech_loc_overhead_map.get(tech, 0)
                
                # Calculate time overhead using BOTH LOC and difficulty, then take max
                # Time from LOC (coding time)
                loc_based_time = (loc_overhead / 1000.0) * 0.77 if loc_overhead > 0 else 0.0
                
                # Time from difficulty (setup/config time)
                tech_difficulty = difficulty_map.get(tech, 0)
                if tech_difficulty > 0:
                    # Base setup time calculation:
                    # Difficulty 1-3 (easy): 0.5-2 hours
                    # Difficulty 4-6 (medium): 2-6 hours  
                    # Difficulty 7-10 (hard): 6-20 hours
                    # Formula: exponential growth based on difficulty
                    difficulty_based_time = 0.3 * (1.5 ** tech_difficulty)
                else:
                    difficulty_based_time = 0.0
                
                # Take the maximum of both calculations
                # This ensures technologies with small LOC but high difficulty get proper time allocation
                time_overhead_hours = max(loc_based_time, difficulty_based_time)
                
                tech_analysis.append({
                    "technology": tech,
                    "criticality": pred,
                    "confidence": round(confidence[pred], 4),
                    "loc_overhead": int(loc_overhead),
                    "time_overhead_hours": round(time_overhead_hours, 2),
                    "time_overhead_readable": self._format_time_human_readable(time_overhead_hours)
                })
            except Exception:
                # If prediction fails, default to mandatory with no overhead
                tech_analysis.append({
                    "technology": tech,
                    "criticality": "mandatory",
                    "confidence": 0.0,
                    "loc_overhead": 0,
                    "time_overhead_hours": 0.0,
                    "time_overhead_readable": "0 hours"
                })
        
        return tech_analysis

    @traced
    def _build_system_design_plan(self, services: List[str], tech_split: Dict[str, List[str]]) -> Dict[str, Any]:
        components = []
        for s in services:
            comp = {
                'name': s,
                'responsibilities': 'See service name; encapsulate related domain logic and expose REST/JSON APIs.',
                'tech_stack': {
                    'backend': tech_split.get('backend', []),
                    'database': tech_split.get('database', []),
                }
            }
            components.append(comp)

        plan = {
            'architecture_style': 'microservices' if len(services) > 2 else 'modular-monolith',
            'components': components,
            'data_flow': [
                'frontend -> api-gateway',
                'api-gateway -> domain microservices',
                'microservices -> databases'
            ],
            'data_stores': tech_split.get('database', []),
        }
        return plan

    @traced
    def _enrich_technologies(self, text: str, technologies: List[str]) -> Tuple[List[str], Dict[str, Any] | None]:
        """Enrich sparse technology lists with online inference and system design patterns.
        
        Returns:
            Tuple of (enriched_technologies, enrichment_info)
        """
        enrichment_info: Dict[str, Any] | None = None
        added_from_patterns: List[str] = []
        added_from_online: List[str] = []
        
        # First, check for known application patterns and add comprehensive tech stacks
        t = text.lower()
        pattern_sources: List[str] = []
        for pattern_name, pattern_data in self._system_design_patterns.get('application_patterns', {}).items():
            keywords = pattern_data.get('keywords', [])
            if any(kw in t for kw in keywords):
                # Match found! Add all technologies from this pattern
                pattern_techs = pattern_data.get('technologies', {})
                for category, techs in pattern_techs.items():
                    for tech in techs:
                        # Map common technology names to our internal format
                        normalized_tech = self._normalize_technology_name(tech)
                        if normalized_tech and normalized_tech not in technologies:
                            technologies.append(normalized_tech)
                            added_from_patterns.append(normalized_tech)
                pattern_sources.append(pattern_name)
        
        # Then try online enrichment to catch explicit mentions (PHP, MySQL, AWS, etc.)
        online_techs = self._infer_technologies_online(text)
        if online_techs:
            for ot in online_techs:
                if ot not in technologies:
                    technologies.append(ot)
                    added_from_online.append(ot)

        # Heuristic enrichment for AWS serverless stack (lambda, API Gateway, DynamoDB)
        aws_added: List[str] = []
        try:
            tl = t  # already lower-cased text from above
            def _add(tech_id: str):
                if tech_id not in technologies:
                    technologies.append(tech_id)
                    aws_added.append(tech_id)

            if 'dynamodb' in tl:
                _add('dynamodb')
            if 'api gateway' in tl or 'apigateway' in tl or 'api-gateway' in tl:
                _add('api_gateway')
            if 'lambda' in tl:
                _add('aws_lambda')
            # Generic 'serverless' hint
            if 'serverless' in tl:
                # If APIs are mentioned, API Gateway is commonly used
                if ('api' in tl or 'apis' in tl) and 'api_gateway' not in technologies:
                    _add('api_gateway')
                # Default serverless compute
                if 'aws_lambda' not in technologies:
                    _add('aws_lambda')
        except Exception:
            pass
        
        # Build enrichment info
        if added_from_patterns or added_from_online or aws_added:
            enrichment_info = {
                "used": True,
                "sources": []
            }
            if added_from_patterns:
                enrichment_info["sources"].append({
                    "type": "system_design_patterns",
                    "patterns": pattern_sources,
                    "technologies_added": added_from_patterns
                })
            if added_from_online:
                enrichment_info["sources"].append({
                    "type": "online_keywords",
                    "technologies_added": added_from_online
                })
            if aws_added:
                enrichment_info["sources"].append({
                    "type": "aws_serverless_keywords",
                    "technologies_added": aws_added
                })
        
        return technologies, enrichment_info
    
    def _normalize_technology_name(self, tech: str) -> str | None:
        """Normalize technology names from patterns to internal format."""
        # Map common names to internal representation
        mapping = {
            # Frontend
            "react": "react",
            "vue": "vue",
            "angular": "angular",
            "nextjs": "nextjs",
            "typescript": "typescript",
            "redux": "redux",
            "react_native": "mobile",
            "swift": "mobile",
            "kotlin": "mobile",
            "electron": "electron",
            # Backend
            "node": "node",
            "python_fastapi": "python_fastapi",
            "python_django": "python_django",
            "flask": "flask",
            "golang": "golang",
            "java_spring": "java_spring",
            "ruby_rails": "ruby_rails",
            "erlang": "erlang",
            # Database
            "postgres": "postgres",
            "postgresql": "postgres",
            "mysql": "mysql",
            "mongodb": "mongodb",
            "cassandra": "cassandra",
            "postgis": "postgis",
            # Cache
            "redis": "redis",
            "memcached": "memcached",
            "redis_geo": "redis",
            # Storage
            "s3": "s3",
            "cdn": "cdn",
            "cloudfront": "cdn",
            "cloudflare": "cdn",
            "fastly": "cdn",
            # Message Queue
            "kafka": "kafka",
            "rabbitmq": "rabbitmq",
            "sqs": "message_queue",
            # Search
            "elasticsearch": "elasticsearch",
            "search": "search",
            # Real-time
            "websocket": "websocket",
            "xmpp": "realtime",
            # Infrastructure
            "kubernetes": "devops",
            "docker": "docker",
            "nginx": "nginx",
            "haproxy": "nginx",
            "aws_alb": "nginx",
            "ecs": "devops",
            # Monitoring
            "prometheus": "monitoring",
            "grafana": "monitoring",
            "datadog": "monitoring",
            "cloudwatch": "monitoring",
            # Payments
            "stripe": "stripe",
            "paypal": "payments",
            # Maps
            "google_maps": "maps",
            "mapbox": "maps",
            # Video
            "ffmpeg": "video_processing",
            "hls": "streaming",
            "dash": "streaming",
            # ML
            "tensorflow": "ml",
            "pytorch": "ml",
            # Auth
            "signal_protocol": "auth",
            "e2e_encryption": "auth",
            # Image processing
            "pillow": "image_processing",
            "imagemagick": "image_processing",
            # AWS Serverless & Services
            "lambda": "aws_lambda",
            "aws_lambda": "aws_lambda",
            "api_gateway": "api_gateway",
            "api gateway": "api_gateway",
            "apigateway": "api_gateway",
            "aws_api_gateway": "api_gateway",
            "dynamodb": "dynamodb",
            # Misc
            "video_js": "frontend_lib"
        }
        
        tech_lower = tech.lower().replace("-", "_").replace(" ", "_")
        return mapping.get(tech_lower, None)

    @traced
    def _build_time_explanation(self, manual_hours: float, ai_hours: float, microservices: List[str], 
                                 tech_count: int, speedup_category: str, speedup_factor: float,
                                 domain_multiplier: float = 1.0, reference_repos: List[str] = None,
                                 speedup_details: Dict[str, Any] = None) -> str:
        """Generate human-readable explanation of time estimates."""
        lines = []
        lines.append(f"AI-assisted estimate: {ai_hours:.1f} hours")
        lines.append(f"  • Based on ML regression model trained on real project data")
        lines.append(f"  • Accounts for {len(microservices)} microservices and {tech_count} technologies")
        if domain_multiplier > 1.0 and reference_repos:
            lines.append(f"  • Domain complexity multiplier: {domain_multiplier}x (from GitHub: {', '.join(reference_repos[:2])})")
        lines.append("")
        
        # Calculate speedup using baseline (1 line = 60 seconds)
        speedup_ratio = manual_hours / ai_hours if ai_hours > 0 else 1.0
        time_saved_percent = (1 - (ai_hours / manual_hours)) * 100 if manual_hours > 0 else 0
        
        lines.append(f"Manual estimate: {manual_hours:.1f} hours")
        lines.append(f"  • Baseline: 1 line = 60 seconds (industry standard)")
        lines.append(f"  • Range provided: ±20% variation for different developer speeds")
        lines.append(f"  • AI is {speedup_ratio:.1f}x faster")
        lines.append("")
        
        if speedup_details:
            loc = speedup_details.get('predicted_loc', 0)
            avg_diff = speedup_details.get('avg_tech_difficulty', 5.0)
            
            lines.append(f"  Project characteristics:")
            lines.append(f"    • Size: ~{loc:,} lines of code")
            lines.append(f"    • Tech difficulty: {avg_diff:.1f}/10 average")
            lines.append(f"    • Category: {speedup_category}")
            lines.append("")
        
        lines.append("  Key insight:")
        lines.append("    Human baseline: 1 line = 60 seconds (1 minute per line of code)")
        lines.append("    This is an industry-standard assumption for average developer productivity")
        lines.append("    AI coding speed varies by complexity but averages ~0.77 hours per 1000 LOC")
        lines.append("")
        lines.append("Note: Estimates include coding time only. Actual development also includes")
        lines.append("      requirements, design, testing, debugging, and integration.")
        return "\n".join(lines)

    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze software requirement text and return complexity metrics.
        
        Returns:
            Dict with structure:
            {
                "without_ai_and_ml": {
                    "no_of_lines": int,
                    "technologies": List[str],
                    "time_estimation": float (hours)
                },
                "with_ai_and_ml": {
                    "no_of_lines": int,
                    "extra_technologies": List[str],
                    "time_estimation": float (hours)
                },
                "complexity_score": float (10-200 scale)
            }
            
            Or error response:
            {
                "error": str,
                "software_probability": float
            }
        """
        if not text or not text.strip():
            return {"error": "Empty requirement text"}

        # Store prompt for simplified output
        self._last_prompt = text

        # Start per-request trace logging
        self._trace_start(text)

        software_proba = self._predict_is_software(text)
        if software_proba < 0.60:
            # Not a software requirement with sufficient confidence
            resp = {
                "error": "Only computer/software jobs are supported for complexity scoring.",
                "software_probability": round(software_proba, 2)
            }
            self._trace_end(resp)
            return resp

        technologies = self._predict_technologies(text)

        # Detect hiring/job-description style prompts early using classifier + heuristic
        is_hiring, hiring_proba, hiring_source = self._predict_is_hiring(text)
        
        # For hiring prompts, prioritize explicit keyword detection over ML predictions
        if is_hiring:
            online_techs = infer_technologies_from_web(text)
            if online_techs:
                # Use online detection as primary source for hiring
                technologies = online_techs
                enrichment_info = {"used": True, "source": "online_keywords_primary", "technologies": technologies}
            else:
                # Fallback to ML predictions if no keywords found
                technologies, enrichment_info = self._enrich_technologies(text, technologies)
        else:
            # For non-hiring (build requirements), use ML + enrichment
            technologies, enrichment_info = self._enrich_technologies(text, technologies)
        
        # Infer domain complexity multiplier from GitHub
        domain_multiplier, reference_repos = self._infer_domain_multiplier(text)
        
        # Split technologies and infer microservices first (needed for realistic LOC)
        tech_split = self._split_technologies(technologies)
        
        # Remove empty technology categories from tech_split
        tech_split = {k: v for k, v in tech_split.items() if v}
        
        services = self._infer_microservices(text, technologies)
        
        # Calculate realistic LOC based on GitHub analysis and system design
        realistic_loc = self._calculate_realistic_loc(text, technologies, tech_split, services)
        
        # Calculate AI time using researched lines-per-second from GitHub repos
        # From our analysis: AI averages 0.000361 lines/sec (includes prompt overhead)
        # This translates to ~0.77 hours per 1000 LOC
        ai_hours_per_1000_loc = self._realistic_loc_config.get("ai_coding_speed", {}).get("hours_per_1000_loc", 0.77)
        ai_hours = (realistic_loc / 1000.0) * ai_hours_per_1000_loc
        
        # Apply domain multiplier to AI time estimate
        ai_hours = ai_hours * domain_multiplier
        
        # Calculate boilerplate LOC provided by scaffolding tools (e.g., create-react-app, django-admin)
        # Humans get this for free, but AI codes from scratch, so we subtract from human time only
        boilerplate_loc = self._calculate_total_boilerplate_loc(tech_split)
        human_coding_loc = max(0, realistic_loc - boilerplate_loc)  # Never negative
        
        # Calculate manual time using baseline: 1 line = 60 seconds
        # This is the industry-standard assumption for human coding speed
        # Formula: manual_hours = (LOC × 60 seconds) / 3600
        seconds_per_line = 60.0
        manual_hours_baseline = (human_coding_loc * seconds_per_line) / 3600.0
        
        # Provide range: ±20% variation for different developers
        manual_hours_min = manual_hours_baseline * 0.8   # Fast developer (48 sec/line)
        manual_hours_avg = manual_hours_baseline         # Average developer (60 sec/line)
        manual_hours_max = manual_hours_baseline * 1.2   # Slower developer (72 sec/line)

        # Detect hiring/job-description style prompts (already computed above)
        # is_hiring remains as determined via classifier/heuristic
        
        # Get AI metrics with speedup details (use avg time for speedup estimate)
        ai_metrics = self._get_ai_metrics_with_details(technologies, ai_hours, manual_hours_avg, realistic_loc, text)

        # Build time estimation explanation
        time_explanation = self._build_time_explanation(
            manual_hours_avg,
            ai_metrics['time_estimation'],
            services,
            len(technologies),
            ai_metrics['speedup_category'],
            ai_metrics['speedup_factor'],
            domain_multiplier,
            reference_repos,
            ai_metrics.get('speedup_details')
        )
        
        # Predict system design architecture using ML model (kept internal; not surfaced directly)
        system_design_prediction = self._predict_system_design_architecture(text)
        
        # Build per-technology analysis for final response: only time share percentages and mention flag
        # Compute LOC breakdown for percentage shares
        # We'll compute these after loc_breakdown is available (later); placeholder here
        tech_criticality_analysis = []
        
        if is_hiring:
            # For hiring prompts: compute a skills complexity score and return only the minimal schema.
            # Extract experience requirements
            experience_requirements = self._extract_experience_requirements(text, technologies)
            
            skills_score = self._compute_complexity(
                text,
                technologies,
                services,
                1.0,
                1.0,
                experience_requirements
            )

            # Build richer explanation using model feature attributions when available
            complexity_score_explanation = self._build_skills_complexity_explanation(
                text, skills_score, technologies, services, experience_requirements
            )

            # Also include per-technology complexity and estimated LOC breakdown (based on predicted LOC)
            per_tech_complexity = self._compute_per_technology_complexity(technologies, ai_hours)
            loc_breakdown = self._estimate_loc_breakdown(realistic_loc, tech_split)

            # LOC-based complexity score (Linux 28M LOC = 100)
            loc_based_score = self._compute_loc_based_complexity_score(realistic_loc)
            # Multi-factor complexity v2 (use as primary complexity score)
            multifactor = self._compute_multifactor_complexity(
                text=text,
                predicted_loc=realistic_loc,
                technologies=technologies,
                tech_split=tech_split,
                microservices=services,
                domain_multiplier=domain_multiplier,
                per_tech_complexity=per_tech_complexity
            )

            # Build per-technology time share analysis (percentages + mentioned flag)
            loc_by_tech = loc_breakdown.get('by_technology', {}) if isinstance(loc_breakdown, dict) else {}
            per_tech_time_share: List[Dict[str, Any]] = []
            for tech, tech_loc in loc_by_tech.items():
                if tech_loc <= 0:
                    continue
                share_pct = (tech_loc / realistic_loc * 100.0) if realistic_loc > 0 else 0.0
                mentioned = self._is_tech_mentioned(tech, text)
                per_tech_time_share.append({
                    "technology": tech,
                    "time_spent": {
                        "human_percent": round(share_pct, 2),
                        "ai_percent": round(share_pct, 2)
                    },
                    "is_mentioned_in_prompt": bool(mentioned)
                })

            # Build nested technologies with per-tech details (move loc_breakdown + per-tech time here)
            per_tech_boiler = self._estimate_per_technology_boilerplate_loc(tech_split)
            nested_tech: Dict[str, Dict[str, Any]] = {}
            loc_by_tech = loc_breakdown.get('by_technology', {}) if isinstance(loc_breakdown, dict) else {}
            ai_total = ai_hours
            for category, techs in tech_split.items():
                cat_obj: Dict[str, Any] = {}
                for tech in techs:
                    t_loc = int(loc_by_tech.get(tech, 0))
                    share = (t_loc / realistic_loc) if realistic_loc > 0 else 0.0
                    human_avg_hours = manual_hours_avg * share
                    ai_share_hours = ai_total * share
                    t_meta = per_tech_complexity.get(tech, {})
                    mentioned = self._is_tech_mentioned(tech, text)
                    cat_obj[tech] = {
                        "loc": t_loc,
                        "time_spent": {
                            "human": {"hours": round(human_avg_hours, 2), "percent": round(share * 100.0, 2)},
                            "ai": {"hours": round(ai_share_hours, 2), "percent": round(share * 100.0, 2)}
                        },
                        "is_mentioned_in_prompt": bool(mentioned),
                        "boilerplate_loc_deducted": int(per_tech_boiler.get(tech, 0)),
                        "difficulty": float(t_meta.get('difficulty', 5.0)),
                        "complexity_score": int(t_meta.get('base_score', 0)),
                        "alternatives": self._get_alternatives(tech)
                    }
                nested_tech[category] = cat_obj

            result_hiring: Dict[str, Any] = {
                "technologies": nested_tech,
                "predicted_lines_of_code": int(realistic_loc),
                "skills_complexity_score": round(float(skills_score), 2),
                "complexity_score": multifactor['complexity_score_v2'],
                "size_score_linux_ref": round(float(loc_based_score), 2),
                # No separate complexity_v2 or difficulty_summary fields
                "is_hiring_requirement": True,
                "time_estimation": {
                    "ai_hours": round(ai_hours, 2),
                    "ai_human_readable": self._format_time_human_readable(ai_hours),
                    "manual_hours_min": round(manual_hours_min, 2),
                    "manual_hours_avg": round(manual_hours_avg, 2),
                    "manual_hours_max": round(manual_hours_max, 2),
                    "manual_human_readable_min": self._format_time_human_readable(manual_hours_min),
                    "manual_human_readable_avg": self._format_time_human_readable(manual_hours_avg),
                    "manual_human_readable_max": self._format_time_human_readable(manual_hours_max),
                    "boilerplate_loc_deducted": boilerplate_loc,
                    "human_coding_loc": human_coding_loc,
                    "note": "Range based on human_ai_code_ratio.json analysis: min ratio 0.01000830 (fastest), median 0.01007192 (typical), max 0.01051033 (slowest). AI is ~99-100x faster."
                }
            }
            # Move data_flow to root and remove system_design_plan/proposed_system_design from final response
            try:
                plan = self._build_system_design_plan(services, tech_split)
                if isinstance(plan, dict) and 'data_flow' in plan:
                    result_hiring['data_flow'] = plan['data_flow']
            except Exception:
                pass
            # Optionally include detection metadata for debugging
            result_hiring["hiring_detection"] = {"source": hiring_source, "proba": round(hiring_proba, 3)}
            self._trace_end(result_hiring)
            return result_hiring
        # Compute LOC-based linear size score (Linux 28M LOC = 100)
        linux_ref_size_score = self._compute_loc_based_complexity_score(realistic_loc)

        # Build nested technologies with per-tech details (build path)
        per_tech_boiler = self._estimate_per_technology_boilerplate_loc(tech_split)
        nested_tech: Dict[str, Dict[str, Any]] = {}
        # We'll fill after loc_breakdown available, so create placeholder then patch below after computing multifactor
        result: Dict[str, Any] = {
            "technologies": nested_tech,
            "predicted_lines_of_code": int(realistic_loc),
            "microservices": services,
            # per_technology_analysis will be set after loc_breakdown
            "without_ai_and_ml": {
                "time_estimation": {
                    "hours_min": round(manual_hours_min, 2),
                    "hours_avg": round(manual_hours_avg, 2),
                    "hours_max": round(manual_hours_max, 2),
                    "human_readable_min": self._format_time_human_readable(manual_hours_min),
                    "human_readable_avg": self._format_time_human_readable(manual_hours_avg),
                    "human_readable_max": self._format_time_human_readable(manual_hours_max),
                    "boilerplate_loc_deducted": boilerplate_loc,
                    "human_coding_loc": human_coding_loc,
                    "note": f"Baseline: 1 line = 60 seconds. Range ±20% for developer variation (48-72 sec/line). Boilerplate ({boilerplate_loc} LOC) deducted from human time."
                }
            },
            "with_ai_and_ml": {
                **{k: v for k, v in ai_metrics.items() if k not in ['speedup_factor', 'speedup_category', 'time_estimation']},
                "time_estimation": {
                    "hours": round(ai_metrics['time_estimation'], 2),
                    "human_readable": self._format_time_human_readable(ai_metrics['time_estimation'])
                }
            },
            "time_estimation_explanation": time_explanation,
            # system_design_plan removed from final response; expose only data_flow at root
            # Placeholder; will set complexity_score after computing multifactor
            # Placeholder; will set complexity_score after computing multifactor
            "size_score_linux_ref": round(float(linux_ref_size_score), 2)
        }
        # Add per-technology complexity scoring & LOC breakdown
        per_tech_complexity = self._compute_per_technology_complexity(technologies, ai_hours)
        loc_breakdown = self._estimate_loc_breakdown(realistic_loc, tech_split)
        # Compute multifactor after we have per-tech complexity
        multifactor = self._compute_multifactor_complexity(
            text=text,
            predicted_loc=realistic_loc,
            technologies=technologies,
            tech_split=tech_split,
            microservices=services,
            domain_multiplier=domain_multiplier,
            per_tech_complexity=per_tech_complexity
        )
        # Expose complexity_score computed from multi-factor model; do not include separate complexity_v2/difficulty_summary fields
        result['complexity_score'] = multifactor['complexity_score_v2']
        # complexity_explanation removed per schema cleanup (difficulty_summary removed)
        # Keep loc_breakdown; do not expose per_technology_complexity summary structure
        # Build nested technologies structure with LOC & time
        loc_by_tech = loc_breakdown.get('by_technology', {}) if isinstance(loc_breakdown, dict) else {}
        ai_total = ai_hours
        for category, techs in tech_split.items():
            cat_obj: Dict[str, Any] = {}
            for tech in techs:
                t_loc = int(loc_by_tech.get(tech, 0))
                share = (t_loc / realistic_loc) if realistic_loc > 0 else 0.0
                human_avg_hours = manual_hours_avg * share
                ai_share_hours = ai_total * share
                t_meta = per_tech_complexity.get(tech, {})
                mentioned = self._is_tech_mentioned(tech, text)
                cat_obj[tech] = {
                    "loc": t_loc,
                    "time_spent": {
                        "human": {"hours": round(human_avg_hours, 2), "percent": round(share * 100.0, 2)},
                        "ai": {"hours": round(ai_share_hours, 2), "percent": round(share * 100.0, 2)}
                    },
                    "is_mentioned_in_prompt": bool(mentioned),
                    "boilerplate_loc_deducted": int(per_tech_boiler.get(tech, 0)),
                    "difficulty": float(t_meta.get('difficulty', 5.0)),
                    "complexity_score": int(t_meta.get('base_score', 0)),
                    "alternatives": self._get_alternatives(tech)
                }
            nested_tech[category] = cat_obj
        result['technologies'] = nested_tech
        # Expose only data_flow at root level
        try:
            plan = self._build_system_design_plan(services, tech_split)
            if isinstance(plan, dict) and 'data_flow' in plan:
                result['data_flow'] = plan['data_flow']
        except Exception:
            pass
        # Store last result for export helper methods
        self._last_result = result
        if enrichment_info:
            result["enrichment"] = enrichment_info
        # Optionally include detection metadata for debugging in build responses as well
        result["hiring_detection"] = {"source": hiring_source, "proba": round(hiring_proba, 3)}
        self._trace_end(result)
        return result

    # ------------------ Export Helpers ------------------
    def export_last_result_json(self, path: str) -> bool:
        """Export last analysis result to JSON file. Returns True on success."""
        data = getattr(self, '_last_result', None)
        if not data:
            return False
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception:
            return False

    def export_last_result_csv(self, path: str) -> bool:
        """Export per-technology complexity and LOC breakdown to a CSV file."""
        data = getattr(self, '_last_result', None)
        if not data:
            return False
        per_tech = data.get('per_technology_complexity', {})
        loc_map = data.get('loc_breakdown', {}).get('by_technology', {})
        try:
            import csv
            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['technology', 'difficulty', 'base_score', 'category', 'time_to_productivity', 'loc_estimate'])
                for tech, meta in per_tech.items():
                    writer.writerow([
                        tech,
                        meta.get('difficulty'),
                        meta.get('base_score'),
                        meta.get('category'),
                        meta.get('time_to_productivity'),
                        loc_map.get(tech, 0)
                    ])
            return True
        except Exception:
            return False

    def get_simplified_output(self) -> Dict[str, Any]:
        """Return simplified schema with essential metrics only.
        
        Must be called after analyze_text() to access last result.
        Returns clean schema matching user requirements exactly.
        """
        data = getattr(self, '_last_result', None)
        if not data:
            return {"error": "No analysis result available. Call analyze_text() first."}
        
        # Extract core metrics
        realistic_loc = data.get('predicted_lines_of_code', 0)
        without_ai = data.get('without_ai_and_ml', {})
        with_ai = data.get('with_ai_and_ml', {})
        time_est = without_ai.get('time_estimation', {})
        ai_time_est = with_ai.get('time_estimation', {})
        
        # Human/AI speed from config
        ai_speed_config = self._realistic_loc_config.get("ai_coding_speed", {})
        ai_hours_per_1000_loc = ai_speed_config.get("hours_per_1000_loc", 0.77)
        
        # Get ratios from config or fallback
        ratio_min = 0.01000830
        ratio_avg = 0.01007192
        
        # Human speed = AI speed * ratio
        ai_lines_per_sec = (1000.0 / (ai_hours_per_1000_loc * 3600)) if ai_hours_per_1000_loc > 0 else 0
        human_lines_per_sec = ai_lines_per_sec * ratio_avg
        
        # AI time
        ai_hours = ai_time_est.get('hours', 0)
        ai_human_readable = ai_time_est.get('human_readable', '')
        
        # Human times
        manual_hours_min = time_est.get('hours_min', 0)
        manual_hours_avg = time_est.get('hours_avg', 0)
        
        manual_readable_min = time_est.get('human_readable_min', '')
        manual_readable_avg = time_est.get('human_readable_avg', '')
        
        # Per-technology breakdown - only include technologies with LOC > 0
        tech_split = data.get('technologies', {})
        # Support both old (list) and new (nested) structures
        loc_breakdown_map = data.get('loc_breakdown', {}).get('by_technology', {})
        per_tech_complexity = data.get('per_technology_complexity', {})
        complexity_explanation = data.get('complexity_explanation', {})
        
        # Get tech tools map for CLI commands
        tech_tools = {}
        try:
            import json
            import os
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config', 'tech_tools_map.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                tech_tools = json.load(f)
        except Exception:
            pass
        
        # Build technology object - only technologies that contribute to LOC
        technologies_output = {}
        all_techs_in_result = set()
        for category_techs in tech_split.values():
            if isinstance(category_techs, dict):
                all_techs_in_result.update(category_techs.keys())
            else:
                all_techs_in_result.update(category_techs)
        
        # Get mentioned technologies from prompt
        prompt_lower = getattr(self, '_last_prompt', '').lower()
        
        for tech in all_techs_in_result:
            # Prefer new nested "loc" if available
            tech_loc = 0
            for category_techs in tech_split.values():
                if isinstance(category_techs, dict) and tech in category_techs:
                    tech_loc = int(category_techs[tech].get('loc', 0))
                    break
            if tech_loc == 0:
                tech_loc = loc_breakdown_map.get(tech, 0)
            
            # Skip technologies with 0 LOC
            if tech_loc <= 0:
                continue
            
            tech_complexity_data = per_tech_complexity.get(tech, {})
            
            # Calculate per-tech time estimates
            if realistic_loc > 0:
                tech_portion = tech_loc / realistic_loc
                tech_manual_min = manual_hours_min * tech_portion
                tech_manual_avg = manual_hours_avg * tech_portion
            else:
                tech_manual_min = 0
                tech_manual_avg = 0
            
            # Check if explicitly mentioned in prompt
            tech_normalized = tech.replace('_', ' ').lower()
            mentioned = tech_normalized in prompt_lower or tech.lower() in prompt_lower
            
            # Check if recommended by system design (in mandatory categories but not mentioned)
            recommended = not mentioned
            
            # Calculate contribution percentage
            contribution_pct = (tech_loc / realistic_loc * 100) if realistic_loc > 0 else 0
            
            # Get complexity score and explanation
            complexity_score = tech_complexity_data.get('base_score', 0)
            difficulty = tech_complexity_data.get('difficulty', 0)
            reasons = tech_complexity_data.get('reasons', [])
            
            # Build technology entry
            tech_entry = {
                "estimated_lines_of_code": f"{int(tech_loc):,}",
                "estimated_human_time_min": round(tech_manual_min, 2),
                "estimated_human_time_min_readable": self._format_time_human_readable(tech_manual_min),
                "estimated_human_time_average": round(tech_manual_avg, 2),
                "estimated_human_time_average_readable": self._format_time_human_readable(tech_manual_avg),
                "mentioned_in_prompt": mentioned,
                "recommended_in_standard_system_design": recommended,
                "contribution_to_lines_of_code": round(contribution_pct, 2),
                "complexity_score": int(complexity_score)
            }
            
            # Add complexity explanation if available
            if reasons:
                tech_entry["complexity_explanation"] = {
                    "difficulty_level": difficulty,
                    "reasons": reasons
                }
            
            # Add CLI command if available
            if tech in tech_tools and tech_tools[tech]:
                tech_entry["default_cli_code"] = tech_tools[tech][0] if isinstance(tech_tools[tech], list) else str(tech_tools[tech])
            
            # Add difficulty explanation
            time_to_prod = tech_complexity_data.get('time_to_productivity', '')
            if time_to_prod or reasons:
                tech_entry["difficulty_explanation"] = {
                    "time_to_productivity": time_to_prod,
                    "learning_curve_factors": reasons
                }
            
            technologies_output[tech] = tech_entry
        
        # Format LOC as human readable
        if realistic_loc >= 1000:
            loc_readable = f"{realistic_loc:,}"
        else:
            loc_readable = str(int(realistic_loc))
        
        result_dict = {
            "estimated_no_of_lines": loc_readable,
            "human_lines_per_second": round(human_lines_per_sec, 2),
            "ai_lines_per_second": round(ai_lines_per_sec, 2),
            "human_to_ai_ratio_min": round(ratio_min, 2),
            "estimated_ai_time": f"{round(ai_hours, 2)} ({ai_human_readable})",
            "estimated_human_time_min": f"{round(manual_hours_min, 2)} ({manual_readable_min})",
            "estimated_human_time_average": f"{round(manual_hours_avg, 2)} ({manual_readable_avg})",
            "technologies": technologies_output
        }
        
        # Add formatted detailed string
        result_dict["formatted_detailed_string"] = self._format_detailed_response(
            result_dict, 
            getattr(self, '_last_prompt', 'Project')
        )
        
        return result_dict
    
    def _format_detailed_response(self, simplified_data: Dict[str, Any], prompt: str) -> str:
        """Format the simplified output as a detailed, readable string response.
        
        Creates a formatted response similar to a human analyst explaining the project complexity.
        """
        lines = []
        
        # Extract project title from prompt (first few words)
        prompt_words = prompt.strip().split()[:10]
        project_title = ' '.join(prompt_words).capitalize()
        if len(prompt_words) >= 10:
            project_title += "..."
        
        # Header
        lines.append(f"## {project_title}")
        lines.append("")
        
        # Estimated Effort section
        lines.append("### Estimated Effort:")
        lines.append(f"- **Total Lines of Code**: ~{simplified_data['estimated_no_of_lines']} lines")
        lines.append(f"- **AI-Assisted Development**: ~{simplified_data['estimated_ai_time']}")
        lines.append(f"- **Human Development**:")
        lines.append(f"  - Minimum: {simplified_data['estimated_human_time_min']}")
        lines.append(f"  - Average: {simplified_data['estimated_human_time_average']}")
        lines.append("")
        
        # Technology Stack section
        technologies = simplified_data.get('technologies', {})
        if technologies:
            lines.append("### Technology Stack Detected:")
            
            # Separate mentioned and recommended
            mentioned_techs = {k: v for k, v in technologies.items() if v.get('mentioned_in_prompt', False)}
            recommended_techs = {k: v for k, v in technologies.items() if v.get('recommended_in_standard_system_design', False)}
            
            if mentioned_techs:
                lines.append("#### Mentioned in Requirements:")
                for tech, data in mentioned_techs.items():
                    lines.append(f"- **{tech.replace('_', ' ').title()}**")
                    lines.append(f"  - Difficulty: {data.get('complexity_score', 0)}/100")
                    lines.append(f"  - Estimated LOC: {data['estimated_lines_of_code']} ({data['contribution_to_lines_of_code']}%)")
                    lines.append(f"  - Setup Time: {data['estimated_human_time_average_readable']}")
                    
                    if 'default_cli_code' in data:
                        lines.append(f"  - Tools: {data['default_cli_code']}")
                    
                    complexity_exp = data.get('complexity_explanation', {})
                    if complexity_exp and 'reasons' in complexity_exp:
                        lines.append(f"  - Complexity Factors: {', '.join(complexity_exp['reasons'][:3])}")
                    lines.append("")
            
            if recommended_techs:
                lines.append("#### Recommended by System Design:")
                for tech, data in recommended_techs.items():
                    lines.append(f"- **{tech.replace('_', ' ').title()}**")
                    lines.append(f"  - Difficulty: {data.get('complexity_score', 0)}/100")
                    lines.append(f"  - Estimated LOC: {data['estimated_lines_of_code']} ({data['contribution_to_lines_of_code']}%)")
                    lines.append(f"  - Setup Time: {data['estimated_human_time_average_readable']}")
                    
                    if 'default_cli_code' in data:
                        lines.append(f"  - Tools: {data['default_cli_code']}")
                    
                    complexity_exp = data.get('complexity_explanation', {})
                    if complexity_exp and 'reasons' in complexity_exp:
                        lines.append(f"  - Complexity Factors: {', '.join(complexity_exp['reasons'][:3])}")
                    lines.append("")
        
        # Key Components section (generic based on technologies)
        lines.append("### Key Components:")
        tech_categories = {}
        for tech, data in technologies.items():
            complexity_data = data.get('complexity_explanation', {})
            category = 'Other'
            if 'database' in tech.lower() or tech.lower() in ['postgres', 'mysql', 'mongodb', 'redis']:
                category = 'Data Storage'
            elif 'react' in tech.lower() or 'vue' in tech.lower() or 'angular' in tech.lower() or 'nextjs' in tech.lower():
                category = 'Frontend'
            elif 'node' in tech.lower() or 'python' in tech.lower() or 'fastapi' in tech.lower() or 'django' in tech.lower():
                category = 'Backend'
            elif 'auth' in tech.lower():
                category = 'Authentication'
            elif 'docker' in tech.lower() or 'devops' in tech.lower():
                category = 'Infrastructure'
            
            if category not in tech_categories:
                tech_categories[category] = []
            tech_categories[category].append(tech.replace('_', ' ').title())
        
        for category, techs in sorted(tech_categories.items()):
            if techs:
                lines.append(f"- **{category}**: {', '.join(techs)}")
        lines.append("")
        
        # Development Timeline
        lines.append("### Development Timeline:")
        ai_time_match = simplified_data['estimated_ai_time'].split('(')[0].strip()
        human_time_match = simplified_data['estimated_human_time_average'].split('(')[1].strip(')')
        
        lines.append(f"- With AI assistance: ~{ai_time_match} hours")
        lines.append(f"- Traditional development: ~{human_time_match}")
        lines.append(f"- Speed improvement: ~{round(1/simplified_data['human_to_ai_ratio_min'])}x faster with AI")
        lines.append("")
        
        # Challenges section
        lines.append("### Potential Challenges:")
        high_complexity_techs = [(k, v) for k, v in technologies.items() if v.get('complexity_score', 0) >= 50]
        if high_complexity_techs:
            for tech, data in sorted(high_complexity_techs, key=lambda x: x[1].get('complexity_score', 0), reverse=True)[:3]:
                complexity_exp = data.get('complexity_explanation', {})
                if complexity_exp and 'reasons' in complexity_exp:
                    lines.append(f"- **{tech.replace('_', ' ').title()}**: {', '.join(complexity_exp['reasons'])}")
        else:
            lines.append("- Relatively straightforward implementation")
            lines.append("- Focus on code quality and testing")
        
        return '\n'.join(lines)

