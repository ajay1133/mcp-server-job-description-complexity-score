# Software-Only Complexity Scorer

This document describes the new **SoftwareComplexityScorer**, which focuses exclusively on software/computer requirements and removes all profession lookup and online search logic.

---

## Overview

### What it does

1. **Software Detection**: Uses a trained binary classifier to determine if a prompt is software-related (no hardcoded keywords).
2. **Technology Prediction**: Multi-label classification to identify relevant technologies (e.g., `react`, `python_fastapi`, `postgres`, `auth`, `ai_llm`).
3. **LOC Estimation**: Predicts approximate lines of code needed.
4. **Time Estimation**: 
   - **Manual coding time** (worst-case: hand-coding from scratch)
   - **AI-accelerated time** (best-case: using AI tools for boilerplate, code generation, etc.)
5. **Complexity Score**: Computed from LOC, time, and technology count (or via optional trained regressor).
6. **Required Tools**: Maps detected technologies to recommended tools/frameworks.
7. **Error Handling**: Returns an error object for non-software prompts.

### What it does NOT do

- No online search or heuristic keyword matching for professions
- No cross-domain job category mapping (doctor, plumber, delivery, etc.)
- No profession availability or scarcity multipliers

---

## Training the Models

### Dataset Format

Prepare a JSONL file (one JSON object per line) with the following schema:

```jsonl
{"text": "Build a React dashboard with Stripe payments", "is_software": true, "technologies": ["react", "auth", "payments", "node", "postgres"], "loc": 2400, "hours": 160, "complexity_score": 115}
{"text": "Need someone to look after my elderly father", "is_software": false}
{"text": "Create FastAPI microservice with PostgreSQL backend", "is_software": true, "technologies": ["python_fastapi", "postgres"], "loc": 1400, "hours": 95, "complexity_score": 90}
```

#### Required Fields

- `text` (string): The requirement prompt
- `is_software` (boolean): `true` for software, `false` for non-software

#### Optional Fields (for software examples)

- `technologies` (array of strings): List of technology tags (see ontology below)
- `loc` (number): Approximate lines of code
- `hours` (number): Manual coding hours estimate
- `complexity_score` (number): Complexity score (10-200 scale)

### Technology Ontology

Use lowercase, snake_case or kebab-case tags:

**Frontend**: `react`, `nextjs`, `vue`, `angular`, `svelte`  
**Backend**: `node`, `python_fastapi`, `python_django`, `flask`, `rails`  
**Database**: `postgres`, `mysql`, `mongodb`, `redis`, `sqlite`  
**Auth**: `auth`, `oauth`, `jwt`  
**Payments**: `payments`, `stripe`  
**DevOps**: `devops`, `docker`, `kubernetes`, `aws`, `azure`, `gcp`  
**AI/ML**: `ai_llm`, `ml`, `cv`, `nlp`  
**Testing**: `testing`, `jest`, `pytest`, `playwright`  
**Real-time**: `websocket`, `realtime`

Extend as needed for your domain.

### Training Command

```bash
python train_software_models.py --data data/software_training_data.jsonl --out models/software
```

This will generate:
- `tfidf_vectorizer.joblib`
- `software_classifier.joblib` (binary: software vs other)
- `tech_multilabel_classifier.joblib` (multi-label tech prediction)
- `loc_regressor.joblib` (LOC estimator)
- `time_regressor.joblib` (hours estimator)
- `score_regressor.joblib` (optional: complexity score)
- `technology_labels.json` (list of tech tags learned)

### Minimum Dataset Size

- **Software examples**: ≥500 (ideally 1000+)
- **Non-software examples**: ≥500 (for robust classifier)
- Balance across technology stacks and complexity levels

---

## Usage

### Basic Example

```python
from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

scorer = SoftwareComplexityScorer(model_dir="models/software")

result = scorer.analyze_text("Build a Next.js dashboard with real-time notifications")

if result.get("ok"):
    print(f"Complexity: {result['complexity_score']}")
    print(f"Technologies: {result['technologies']}")
    print(f"LOC estimate: {result['predicted_loc']}")
    print(f"Manual time: {result['predicted_hours_manual']} hours")
    print(f"With AI tools: {result['predicted_hours_with_ai']} hours")
    print(f"Recommended tools: {result['required_tools']}")
else:
    print(f"Error: {result['error']}")
```

### Output Schema

#### Success Response (software prompt)

```json
{
  "ok": true,
  "software_probability": 0.95,
  "complexity_score": 112.5,
  "predicted_loc": 2200,
  "predicted_hours_manual": 140.0,
  "predicted_hours_with_ai": 49.0,
  "technologies": ["nextjs", "websocket", "auth", "postgres"],
  "required_tools": [
    "Next.js",
    "ESLint + Prettier",
    "Jest/RTL",
    "Tailwind or MUI",
    "OAuth2/OIDC (Auth0/Clerk)",
    "JWT",
    "PostgreSQL",
    "SQLAlchemy/Prisma/TypeORM",
    "pgAdmin"
  ],
  "assumptions": {
    "ai_speedup_note": "AI assistance assumed to speed up common CRUD and boilerplate tasks more than specialized ML or low-level systems work.",
    "loc_basis": "LOC estimate learned from examples; upper/lower bounds applied for sanity."
  }
}
```

#### Error Response (non-software prompt)

```json
{
  "ok": false,
  "error": "Only computer/software jobs are supported for complexity scoring.",
  "software_probability": 0.12
}
```

---

## AI Acceleration Logic

The scorer applies a heuristic speedup factor based on detected technologies:

- **CRUD-heavy stacks** (React, Next.js, FastAPI, Django): **35% of manual time** (AI excels at boilerplate)
- **Specialized stacks** (ML, CV, NLP): **60% of manual time** (AI helps but domain expertise still critical)
- **Default**: **50% of manual time**

This reflects the reality that AI tools like Copilot, Cursor, and ChatGPT accelerate common patterns more than niche systems work.

---

## Tool Mapping

Technologies are automatically mapped to recommended tools:

| Technology | Recommended Tools |
|------------|-------------------|
| `react` / `nextjs` | Vite/Next.js, ESLint+Prettier, Jest/RTL, Tailwind/MUI |
| `python_fastapi` | FastAPI, Uvicorn, PyTest, Docker |
| `postgres` | PostgreSQL, SQLAlchemy/Prisma/TypeORM, pgAdmin |
| `auth` | OAuth2/OIDC (Auth0/Clerk), JWT |
| `payments` | Stripe SDK, Stripe CLI |
| `devops` | GitHub Actions, Docker, Kubernetes |
| `ai_llm` | OpenAI/Anthropic SDK, LangChain/LlamaIndex |

Extend `tech_tools_map` in `SoftwareComplexityScorer` as needed.

---

## Complexity Score Calculation

There are now two scores:

1. **`size_score_linux_ref`** (linear size benchmark):
  - `size_score_linux_ref = (predicted_loc / 28,000,000) * 100`
  - Anchors raw project size against the Linux kernel (~28M LOC). Small projects will naturally score <1 here; it is not intended for human interpretability beyond relative scale.

2. **Primary **`complexity_score`** (multi-factor 0–100): combines 10 dimensions with diminishing returns.

### Multi-Factor Dimensions & Weights

| Dimension | Weight | Subscore (0–1) Basis |
|-----------|--------|----------------------|
| size | 0.15 | `log1p(loc)/log1p(500k)` |
| tech_difficulty | 0.15 | avg difficulty / 10 |
| tech_breadth | 0.10 | sigmoid(count, k=0.6, x0=6) |
| architecture | 0.10 | microservice count curve |
| feature_richness | 0.10 | detected feature tags / 10 |
| domain | 0.10 | normalize (domain_multiplier −1)/(3−1) |
| data_complexity | 0.10 | signals (DB, caches/search, streaming, analytics) / 4 |
| performance | 0.10 | perf/scale keywords hits / 5 |
| security | 0.05 | auth/encryption/compliance/audit hits / 3 |
| ai_ml | 0.05 | ML/AI training/inference novelty hits / 3 |

**Formula:**
```
score_raw = Σ weight_i * subscore_i
complexity_score = max(5, score_raw * 100)  # floor at 5 for any legitimate software requirement
```

No single dimension can dominate; size grows logarithmically, breadth saturates via sigmoid, and architecture uses a stepped curve.

### Why Replace the Old Heuristic?

The previous fallback heuristic (log time + log LOC + sqrt breadth) over-emphasized size and produced unintuitive results for small but intricate systems (e.g., low-latency, high-security, ML-heavy). The multi-factor approach better mirrors real engineering complexity by integrating cross-cutting concerns.

### Optional Score Regressor

If `score_regressor.joblib` exists, it is ignored for the primary `complexity_score` (now deterministic) but may still be used in experiments or for legacy comparison.

### Interpreting Scores

| Complexity Score | Interpretation |
|------------------|---------------|
| 5–15 | Minimal / prototype / CRUD MVP |
| 15–30 | Standard product features (auth, payments, some analytics) |
| 30–50 | Multi-feature app with moderate scale/data needs |
| 50–70 | Distributed/services, richer data & performance constraints |
| 70–85 | High scale, compliance, advanced data + ML components |
| 85–100 | Mission-critical, multi-region, strict SLAs + extensive ML/real-time/security |

The score is not strictly ordinal across all domains (e.g., a deeply specialized ML research stack vs. a regulated fintech platform) but offers a robust comparative baseline across typical web/software products.

---

## Migration from Legacy Scorer

The original `ComplexityScorer` handled both software and non-software jobs via:
- Hardcoded keyword matching
- Online search heuristics
- Profession categories (doctor, plumber, etc.)

**New approach**: Focus exclusively on software prompts with model-based classification.

### Steps to Migrate

1. **Train models** on your software dataset
2. **Switch imports** in your MCP server:
   ```python
   # Old
   from mcp_server.complexity_scorer import ComplexityScorer
   
   # New
   from mcp_server.software_complexity_scorer import SoftwareComplexityScorer
   ```
3. **Update server.py** to use `SoftwareComplexityScorer.analyze_text()`
4. **Handle errors** gracefully when users submit non-software prompts

---

## FAQ

**Q: What if I don't have training data?**  
A: Start with the example dataset, then iteratively expand by:
1. Logging real user prompts
2. Manually labeling software vs non-software
3. For software prompts, add tech tags, LOC, and hours
4. Retrain periodically

**Q: Can I skip the `complexity_score` field in training data?**  
A: Yes. The scorer will compute a heuristic score from LOC/hours/tech. Providing it trains a dedicated regressor for better accuracy.

**Q: How do I add a new technology?**  
A: Add examples with the new tag in your JSONL, retrain, and extend `tech_tools_map` in the scorer.

**Q: What if the classifier wrongly rejects a software prompt?**  
A: Increase confidence threshold in `_predict_is_software()` or add more diverse software examples to training data.

---

## Next Steps

- Curate a production dataset (1000+ examples)
- Train and evaluate models
- Integrate into MCP server
- Monitor misclassifications and expand training data
- Optionally: add embeddings or transformer-based classifier for semantic robustness

---

*For questions or contributions, open an issue or PR on the repository.*
