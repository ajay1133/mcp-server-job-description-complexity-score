# Dynamic Technology Registry - Integration Guide

## Problem
When a new technology emerges (Bun, Deno, htmx, Astro, etc.), the system has hardcoded bottlenecks that require manual updates across multiple files.

## Solution
The `TechRegistry` provides a **self-updating** approach with graceful fallbacks:

1. **Baseline database** - Embedded common techs (offline-first)
2. **External enrichment** - Fetch from GitHub/StackOverflow/npm (optional)
3. **Smart caching** - Cache external data to avoid rate limits
4. **Graceful fallback** - Return sensible defaults for unknown techs

---

## Quick Start

### 1. Basic Usage

```python
from mcp_server.tech_registry import get_tech_registry

registry = get_tech_registry()

# Get info for any tech (known or unknown)
react_info = registry.get_tech_info("react")
# Returns: {"difficulty": 5.2, "category": "frontend", "keywords": [...]}

# Unknown tech gets sensible defaults
new_tech = registry.get_tech_info("super-new-framework-2025")
# Returns: {"difficulty": 5.0, "category": "other", "confidence": 0.3, ...}
```

### 2. Manually Add New Techs

```python
# Add a trending tech immediately
registry.add_custom_tech(
    tech_name="htmx",
    difficulty=4.0,
    category="frontend",
    keywords=["htmx", "hypermedia", "hateoas"]
)

# Add Bun (new JS runtime)
registry.add_custom_tech(
    tech_name="bun",
    difficulty=5.3,
    category="backend",
    keywords=["bun", "bun.js", "bunjs"]
)
```

### 3. Find Similar Technologies

```python
# Get alternatives automatically
alternatives = registry.search_similar_techs("react", top_k=5)
# Returns: ["vue", "svelte", "angular", ...]
```

---

## Integration with Existing Extractors

### SimpleTechExtractor Integration

Replace hardcoded `tech_db` and `tech_keywords` with registry:

```python
# OLD APPROACH (static)
class SimpleTechExtractor:
    def __init__(self):
        self.tech_db = {
            "react": {"difficulty": 5.2, "category": "frontend", ...},
            # ... hundreds of lines
        }
        self.tech_keywords = {
            "react": ["react", "reactjs", ...],
            # ... more static mappings
        }

# NEW APPROACH (dynamic)
from mcp_server.tech_registry import get_tech_registry

class SimpleTechExtractor:
    def __init__(self):
        self.registry = get_tech_registry()

    def extract_technologies(self, text: str) -> Dict[str, Any]:
        # Get all known keywords dynamically
        all_keywords = self.registry.get_all_keywords()

        detected = {}
        text_lower = text.lower()

        for tech_id, keywords in all_keywords.items():
            if any(kw in text_lower for kw in keywords):
                tech_info = self.registry.get_tech_info(tech_id)

                # Get alternatives dynamically
                alternatives_list = self.registry.search_similar_techs(tech_id)
                alternatives = {}
                for alt_id in alternatives_list:
                    alt_info = self.registry.get_tech_info(alt_id)
                    alternatives[alt_id] = {"difficulty": alt_info["difficulty"]}

                detected[tech_id] = {
                    "difficulty": tech_info["difficulty"],
                    "category": tech_info["category"],
                    "alternatives": alternatives,
                    "confidence": tech_info.get("confidence", 1.0),
                }

        return {"technologies": detected}
```

### MLTechExtractor Integration

Update the category_map dynamically:

```python
class MLTechExtractor:
    def __init__(self):
        self.registry = get_tech_registry()

        # Build category map dynamically
        self.category_map = {}
        for tech_id in self.registry.baseline_db.keys():
            tech_info = self.registry.get_tech_info(tech_id)
            self.category_map[tech_id] = tech_info["category"]

    def extract_technologies(self, text: str) -> Dict[str, Any]:
        # ... existing ML extraction logic ...

        for tech_name in extracted_tech_names:
            # Use registry for difficulty
            tech_info = self.registry.get_tech_info(tech_name)
            difficulty = tech_info.get("difficulty", 5.0)

            # Use registry for alternatives
            alternatives_list = self.registry.search_similar_techs(tech_name)
            # ...
```

---

## External Data Sources (Future Enhancement)

The registry is designed to integrate with external APIs:

### GitHub API
```python
def _get_github_stars(self, repo_name: str) -> int:
    """Get GitHub stars as popularity metric."""
    response = requests.get(
        f"https://api.github.com/repos/{org}/{repo}",
        headers={"Authorization": f"token {GITHUB_TOKEN}"}
    )
    return response.json()["stargazers_count"]
```

### StackOverflow API
```python
def _get_so_questions(self, tag: str) -> int:
    """Get StackOverflow question count."""
    response = requests.get(
        f"https://api.stackexchange.com/2.3/tags/{tag}/info",
        params={"site": "stackoverflow"}
    )
    return response.json()["items"][0]["count"]
```

### npm/PyPI Stats
```python
def _get_npm_downloads(self, package: str) -> int:
    """Get npm download count."""
    response = requests.get(
        f"https://api.npmjs.org/downloads/point/last-month/{package}"
    )
    return response.json()["downloads"]
```

### Difficulty Estimation Heuristic
```python
def _estimate_difficulty(self, stars: int, questions: int, downloads: int) -> float:
    """Estimate difficulty from community metrics.

    More stars + docs = easier to learn
    More questions = more complex OR more popular
    High downloads + low questions = mature/stable = easier
    """
    # Example heuristic
    base_difficulty = 5.0

    # High stars = more resources = slightly easier
    if stars > 50000:
        base_difficulty -= 0.5

    # Very high question count might indicate complexity
    if questions > 100000:
        base_difficulty += 0.5

    return max(1.0, min(10.0, base_difficulty))
```

---

## Configuration

Create a config file for API keys and settings:

```python
# config/tech_registry.json
{
    "cache_ttl_hours": 24,
    "external_apis": {
        "github": {
            "enabled": true,
            "token": "ghp_xxxxx"
        },
        "stackoverflow": {
            "enabled": true,
            "key": "xxxxx"
        },
        "npm": {
            "enabled": true
        }
    },
    "fallback_difficulty": 5.0,
    "min_confidence": 0.3
}
```

---

## Migration Steps

### Phase 1: Parallel Run (No Breaking Changes)
1. ✅ Add `tech_registry.py` (done)
2. ✅ Add tests (done)
3. Keep existing extractors working
4. Add opt-in flag to use registry

### Phase 2: Gradual Adoption
1. Update `SimpleTechExtractor` to use registry (behind feature flag)
2. Update `MLTechExtractor` to use registry (behind feature flag)
3. Run A/B tests comparing old vs new
4. Validate accuracy

### Phase 3: Full Migration
1. Make registry the default
2. Remove hardcoded `tech_db` and `tech_keywords`
3. Add admin API to add new techs on-the-fly
4. Enable external API integrations

### Phase 4: Advanced Features
1. Add ML-based difficulty prediction from metrics
2. Auto-detect trending techs from job postings
3. Crowd-source tech additions from users
4. Build web UI for tech database management

---

## Testing

Run registry tests:
```bash
pytest tests/test_tech_registry.py -v
```

Test with new tech:
```python
from mcp_server.tech_registry import get_tech_registry

registry = get_tech_registry()

# Add today's trending tech
registry.add_custom_tech("astro", 4.5, "frontend", ["astro", "astro.js"])

# Verify it works
info = registry.get_tech_info("astro")
print(info)  # {"difficulty": 4.5, "category": "frontend", ...}
```

---

## Benefits

✅ **Future-proof**: New techs get sensible defaults automatically
✅ **Extensible**: Easy to add external data sources
✅ **Cached**: Avoids hitting rate limits
✅ **Graceful**: Falls back when external APIs fail
✅ **Testable**: Clear API for unit tests
✅ **Admin-friendly**: Can add techs via API (no code deploy needed)

## Next Steps

1. Run tests: `pytest tests/test_tech_registry.py`
2. Review integration approach
3. Decide: opt-in feature flag or full migration?
4. Choose external APIs to integrate (GitHub, SO, npm, etc.)
5. Add admin endpoint for manual tech additions
