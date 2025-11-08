"""Online technology inference for ambiguous prompts.

This module optionally queries public sources (GitHub search) to infer a
recommended stack when the prompt is too vague to extract technologies.

It is used only when ENABLE_ONLINE_TECH_ENRICH=1 is set in the environment.
If any network error or rate limit occurs, it fails closed (returns []).
"""

from __future__ import annotations

import os
import re
import time
import json
import logging
from pathlib import Path
from urllib.parse import urlparse
from typing import List, Dict

try:
    import requests  # optional; only required for remote keyword map fetch
except ImportError:  # graceful degradation
    requests = None


# Simple in-memory cache for GitHub search results (TTL: 5 minutes)
_GITHUB_CACHE: Dict[str, tuple[List[Dict], float]] = {}
_CACHE_TTL = 300  # seconds


# Dynamic keyword map (loaded once, refreshable) ---------------------------------
_KEYWORD_MAP: Dict[str, str] | None = None
_KEYWORD_MAP_LAST_LOAD: float | None = None
_KEYWORD_MAP_TTL = int(os.getenv('SOFTWARE_KEYWORD_MAP_TTL', '300'))  # seconds

def _default_keyword_map() -> Dict[str, str]:
    """Built-in fallback keyword map (minimal)."""
    return {
        # Frontend
        'react': 'react', 'next': 'nextjs', 'next.js': 'nextjs', 'vue': 'vue', 'angular': 'angular', 'svelte': 'svelte',
        # Backend
        'php': 'php', 'laravel': 'php', 'fastapi': 'python_fastapi', 'django': 'python_django', 'flask': 'flask', 'rails': 'rails', 'node': 'node', 'express': 'node',
        # Mobile
        'android': 'android', 'kotlin': 'android', 'ios': 'ios', 'swift': 'ios', 'react native': 'react_native', 'flutter': 'flutter',
        # Databases / cache
        'postgres': 'postgres', 'postgresql': 'postgres', 'mysql': 'mysql', 'mongodb': 'mongodb', 'mongo': 'mongodb', 'redis': 'redis',
        # Features
        'stripe': 'payments', 'payment': 'payments', 'auth': 'auth', 'oauth': 'auth', 'jwt': 'auth',
    }

def _load_keyword_map() -> Dict[str, str]:
    """Load keyword->tag map from env path, env URL, repo config, or fallback.

    Precedence:
      1. SOFTWARE_KEYWORD_MAP_PATH (local JSON file)
      2. SOFTWARE_KEYWORD_MAP_URL  (remote JSON via HTTP GET)
      3. config/keyword_map.json (repo default)
      4. Built-in minimal map

    JSON shape must be {"keyword": "tag", ...}.
    """
    env_path = os.getenv('SOFTWARE_KEYWORD_MAP_PATH')
    env_url = os.getenv('SOFTWARE_KEYWORD_MAP_URL')
    repo_default = Path(__file__).parent.parent / 'config' / 'keyword_map.json'

    # Path override
    if env_path:
        p = Path(env_path)
        if p.is_file():
            try:
                with p.open('r', encoding='utf-8') as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return {str(k).lower(): str(v) for k, v in data.items() if isinstance(v, str)}
            except Exception as e:
                logging.warning(f"Failed to load keyword map from SOFTWARE_KEYWORD_MAP_PATH={env_path}: {e}")

    # URL override
    if env_url and requests:
        try:
            parsed = urlparse(env_url)
            if parsed.scheme in ('http', 'https'):
                resp = requests.get(env_url, timeout=5)
                if resp.status_code == 200:
                    data = resp.json()
                    if isinstance(data, dict):
                        return {str(k).lower(): str(v) for k, v in data.items() if isinstance(v, str)}
                else:
                    logging.warning(f"Non-200 status for SOFTWARE_KEYWORD_MAP_URL={env_url}: {resp.status_code}")
        except Exception as e:
            logging.warning(f"Failed to load keyword map from SOFTWARE_KEYWORD_MAP_URL={env_url}: {e}")
    elif env_url and not requests:
        logging.warning("SOFTWARE_KEYWORD_MAP_URL set but 'requests' not installed; skipping remote fetch.")

    # Repo default
    if repo_default.is_file():
        try:
            with repo_default.open('r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return {str(k).lower(): str(v) for k, v in data.items() if isinstance(v, str)}
        except Exception as e:
            logging.warning(f"Failed to load repo keyword_map.json: {e}")

    return _default_keyword_map()

def _get_keyword_map() -> Dict[str, str]:
    global _KEYWORD_MAP, _KEYWORD_MAP_LAST_LOAD
    now = time.time()
    if _KEYWORD_MAP is None or _KEYWORD_MAP_LAST_LOAD is None or (now - _KEYWORD_MAP_LAST_LOAD) > _KEYWORD_MAP_TTL:
        _KEYWORD_MAP = _load_keyword_map()
        _KEYWORD_MAP_LAST_LOAD = now
    return _KEYWORD_MAP

def reload_keyword_map() -> None:
    """Force reload of keyword map (e.g., after external update)."""
    global _KEYWORD_MAP, _KEYWORD_MAP_LAST_LOAD
    _KEYWORD_MAP = _load_keyword_map()
    _KEYWORD_MAP_LAST_LOAD = time.time()
    logging.info("Keyword map reloaded; entries=%d", len(_KEYWORD_MAP))


def _map_keywords_to_tags(text: str) -> List[str]:
    text_l = text.lower()
    tags: List[str] = []
    kw_map = _get_keyword_map()
    for kw, tag in kw_map.items():
        if kw in text_l and tag not in tags:
            tags.append(tag)
    return tags


def _github_search_repos(query: str, per_page: int = 5) -> List[Dict]:
    # Check cache first
    cache_key = f"{query}:{per_page}"
    now = time.time()
    if cache_key in _GITHUB_CACHE:
        cached_items, cached_time = _GITHUB_CACHE[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_items
    
    url = "https://api.github.com/search/repositories"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "complexity-scorer/1.0",
    }
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    params = {"q": query, "sort": "stars", "order": "desc", "per_page": per_page}
    resp = requests.get(url, headers=headers, params=params, timeout=10)
    if resp.status_code != 200:
        return []
    data = resp.json() or {}
    items = data.get("items", [])
    
    # Cache result
    _GITHUB_CACHE[cache_key] = (items, now)
    return items


def _estimate_project_complexity(repo: Dict) -> float:
    """Estimate project complexity from GitHub repo metadata.
    
    Returns a multiplier (1.0 = baseline, >1.0 = more complex).
    """
    stars = repo.get("stargazers_count", 0)
    forks = repo.get("forks_count", 0)
    size = repo.get("size", 0)  # KB
    
    # Heuristic: popular projects with many forks indicate complexity
    # Stars: >10k = very popular/complex, <100 = toy project
    # Forks: >1k = actively maintained, complex
    # Size: >10MB = substantial codebase
    
    star_score = min(stars / 1000.0, 10.0)  # cap at 10x
    fork_score = min(forks / 100.0, 5.0)    # cap at 5x
    size_score = min(size / 10000.0, 3.0)   # cap at 3x (size in KB, 10MB = 10000KB)
    
    # Weighted average
    multiplier = 1.0 + (star_score * 0.3 + fork_score * 0.4 + size_score * 0.3) / 10.0
    return max(1.0, min(multiplier, 3.0))  # clamp between 1.0 and 3.0


def infer_technologies_from_web(prompt: str, max_repos: int = 5) -> List[str]:
    """Infer recommended technologies from the web (GitHub search) and direct keyword mapping.

    Strategy: 
    1. First apply direct keyword mapping to the prompt text
    2. Then search repos related to the prompt and extract keywords from repo metadata
    3. Map to internal tags and deduplicate
    """
    try:
        # First, check for direct mentions in the prompt itself
        inferred = _map_keywords_to_tags(prompt)
        
        # Then supplement with GitHub search results
        items = _github_search_repos(prompt, per_page=max_repos)
        if items:
            for it in items:
                name = it.get("name") or ""
                desc = it.get("description") or ""
                lang = it.get("language") or ""
                full_text = f"{name} {desc} {lang}"
                tags = _map_keywords_to_tags(full_text)
                for t in tags:
                    if t not in inferred:
                        inferred.append(t)

        return inferred
    except Exception:
        # Fail closed but return at least direct keyword matches
        try:
            return _map_keywords_to_tags(prompt)
        except Exception:
            return []


def infer_complexity_multiplier_from_web(prompt: str, max_repos: int = 5) -> tuple[float, List[str]]:
    """Infer domain complexity multiplier from real GitHub projects.
    
    Returns:
        (multiplier, reference_repos): multiplier is 1.0-3.0, reference_repos are repo names
    """
    try:
        items = _github_search_repos(prompt, per_page=max_repos)
        if not items:
            return 1.0, []
        
        multipliers = []
        repo_names = []
        for it in items:
            mult = _estimate_project_complexity(it)
            multipliers.append(mult)
            repo_names.append(it.get("full_name", "unknown"))
        
        # Average multiplier from top repos
        avg_multiplier = sum(multipliers) / len(multipliers) if multipliers else 1.0
        return round(avg_multiplier, 2), repo_names[:3]  # return top 3 reference repos
    except Exception:
        return 1.0, []
