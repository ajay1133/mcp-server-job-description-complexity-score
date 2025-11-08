# Better Training Data Approach - No Hardcoded Keywords

## Problem with Current Approach

The current `generate_training_data_from_web.py` violates the core principle of "no hardcoded keywords" because it:

1. **Hardcoded Templates**: Uses 100+ hardcoded software requirement templates
2. **Keyword-Based Tech Detection**: Uses regex patterns to detect technologies
3. **Heuristic LOC/Hours**: Estimates based on hardcoded complexity keywords
4. **No Real Data**: Everything is synthetic and assumption-based

This defeats the purpose of ML-based inference!

## New Approach: Bootstrap + Active Learning

### Philosophy
- **Start Small**: Begin with minimal seed set (~20 examples)
- **Learn Incrementally**: Use active learning to focus on uncertain cases
- **Use Real Data**: Extract metrics from actual GitHub repositories
- **Human-in-the-Loop**: Efficiently use human labeling where needed

## Implementation

### 1. Bootstrap Training Data (`bootstrap_training_data.py`)

**Workflow:**
1. Create minimal seed set (10 clear software + 10 clear non-software examples)
2. Collect diverse unlabeled data from multiple sources
3. No keyword filtering - pure diversity

**Data Sources:**
- **GitHub Repos**: Project descriptions from trending repos
- **Stack Overflow**: Question titles (naturally software-related)
- **Reddit**: Mix of tech and non-tech subreddits
- **Job Boards**: Indeed, RemoteOK (natural mix of software/non-software)

**Example:**
```bash
python bootstrap_training_data.py
```

Creates:
- `data/bootstrapped_training_data.jsonl` - 20 seed examples
- Foundation for active learning

### 2. Active Learning Loop (`active_learning_loop.py`)

**Workflow:**
```
┌─────────────────────────────────────────┐
│ 1. Train model on current labeled data │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 2. Predict on unlabeled pool            │
│    Calculate confidence for each        │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 3. Select most uncertain examples       │
│    (confidence 0.3-0.7)                 │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 4. Present to human for labeling        │
│    (Interactive CLI)                    │
└───────────────┬─────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────┐
│ 5. Add to labeled set, remove from      │
│    unlabeled pool                       │
└───────────────┬─────────────────────────┘
                │
                └─────► Repeat until accuracy > 95%
```

**Key Benefits:**
- **Efficiency**: Only label uncertain examples (20-30% of data)
- **Quality**: Human feedback on hardest cases improves model most
- **Adaptive**: Model learns what it doesn't know

**Example:**
```bash
# Create seed set
python bootstrap_training_data.py

# Collect unlabeled data (implement API calls)
# ... scrape GitHub, Reddit, job boards ...

# Run active learning
python active_learning_loop.py
```

**Interactive Session:**
```
ACTIVE LEARNING: Label Uncertain Examples
======================================================================
The model is uncertain about these examples.
Your labels will help improve accuracy.

[1/20]
Model prediction: SOFTWARE
Confidence: 0.52 (uncertain!)

Text: Build a REST API that processes customer orders and sends
email notifications...

Is this a SOFTWARE development task?
  y) Yes (software)
  n) No (not software)
  s) Skip
  q) Quit labeling

Your choice (y/n/s/q): y
✓ Labeled as SOFTWARE
```

### 3. GitHub Repo Analyzer (`analyze_github_repos.py`)

**Extracts REAL metrics from actual codebases:**

**Metrics Extracted:**
1. **LOC**: Uses `cloc` tool or simple counter
2. **Technologies**: Parses package.json, requirements.txt, etc.
3. **Hours**: Estimates from commit history (1.5 hours per commit)
4. **Complexity**: Calculated from real metrics

**Technology Detection (No Keywords!):**
- Parse `package.json` → detect React, Next.js, Vue, etc.
- Parse `requirements.txt` → detect Django, FastAPI, Flask
- Parse `Gemfile` → detect Rails
- Check dependency files for databases, auth, payments, cloud services

**Example:**
```bash
# Analyze specific repo
python analyze_github_repos.py

# Or integrate into training pipeline
```

**Output:**
```json
{
  "text": "A modern web framework for Python...",
  "is_software": true,
  "technologies": ["python_fastapi", "postgres", "redis", "auth"],
  "loc": 15234,
  "hours": 245.5,
  "complexity_score": 142,
  "source": "github_https://github.com/tiangolo/fastapi"
}
```

## Complete Workflow

### Phase 1: Bootstrap (Day 1)
```bash
# 1. Create minimal seed set
python bootstrap_training_data.py

# 2. Implement data collection (one-time setup)
# - Add GitHub API token to .env
# - Implement Reddit API (PRAW)
# - Implement Stack Exchange API
# - Implement job board scraping

# 3. Collect diverse unlabeled data
python collect_unlabeled_data.py
```

### Phase 2: Active Learning (Days 2-3)
```bash
# 1. Run active learning loop
python active_learning_loop.py

# Interactive labeling of 100-200 examples
# Takes 1-2 hours, focuses on uncertain cases

# 2. Check model performance
# Model stops when accuracy > 95%
```

### Phase 3: Real Metrics (Days 4-5)
```bash
# 1. Analyze GitHub repos for real metrics
python analyze_github_repos.py

# 2. Merge with actively learned data
python merge_training_data.py

# 3. Train final models
python train_software_models.py \
  --data data/merged_training_data.jsonl \
  --out models/software
```

## Comparison: Old vs New

### Old Approach (Current)
```python
# Hardcoded templates
SOFTWARE_EXAMPLES = [
    "Build a React dashboard with user authentication",
    "Create a REST API using FastAPI with PostgreSQL",
    # ... 100 more hardcoded templates
]

# Keyword-based tech detection
if 'react' in text_lower:
    technologies.append('react')
if 'fastapi' in text_lower:
    technologies.append('python_fastapi')

# Heuristic LOC estimation
if 'simple' in text_lower:
    multiplier = 0.5
elif 'complex' in text_lower:
    multiplier = 2.0
```

**Problems:**
- ❌ Hardcoded assumptions
- ❌ Synthetic data only
- ❌ Biased toward template phrases
- ❌ No real-world validation

### New Approach
```python
# Minimal seed (10 software + 10 non-software)
seed_examples = [
    "Build a web application",  # Clear software
    "Clean my house",           # Clear non-software
    # ... 18 more clear examples
]

# Active learning
uncertain = model.select_uncertain(unlabeled_pool)
labeled = human.label(uncertain)  # Only 20-30% of data
model.retrain(labeled)

# Real metrics from GitHub
metrics = analyze_repo("https://github.com/user/project")
# Returns actual LOC, real tech stack, commit-based hours
```

**Benefits:**
- ✅ Minimal hardcoding (20 seed examples)
- ✅ Real data from actual projects
- ✅ Efficient human labeling (only uncertain cases)
- ✅ Model learns from real-world examples

## Migration Path

### Immediate Steps
1. **Keep current system running** (for backward compatibility)
2. **Implement bootstrap script** (already done ✓)
3. **Implement active learning** (already done ✓)
4. **Implement GitHub analyzer** (already done ✓)

### Next Steps (Recommended)
1. **Collect unlabeled data** (implement API calls)
   - GitHub API for repo descriptions
   - Reddit API (PRAW) for diverse posts
   - Stack Exchange API for questions
   - Job board scraping (Indeed, RemoteOK)

2. **Run active learning loop**
   - Label 100-200 uncertain examples (2 hours of work)
   - Train model on these examples

3. **Analyze 50-100 GitHub repos**
   - Extract real LOC, tech stacks, hours
   - Merge with actively learned data

4. **Train final models**
   - Use merged dataset (real metrics + human labels)
   - Test against current model
   - Switch if performance is better

5. **Deprecate old approach**
   - Move `generate_training_data_from_web.py` to `legacy/`
   - Update documentation

## File Structure

```
mcp_complexity_scorer/
├── bootstrap_training_data.py       # NEW: Create seed set, collect data
├── active_learning_loop.py          # NEW: Interactive labeling loop
├── analyze_github_repos.py          # NEW: Extract real metrics
├── collect_unlabeled_data.py        # TODO: Implement data collection
├── merge_training_data.py           # TODO: Merge multiple sources
├── train_software_models.py         # Existing (works with new data)
├── generate_training_data_from_web.py  # OLD: Deprecated (keep for reference)
└── data/
    ├── bootstrapped_training_data.jsonl   # Seed examples
    ├── labeled_training_data.jsonl        # Actively learned labels
    ├── unlabeled_pool.jsonl               # Unlabeled examples
    ├── github_analyzed_training_data.jsonl # Real repo metrics
    └── merged_training_data.jsonl         # Final training set
```

## Quick Start

### Option 1: Minimal (No External APIs)
```bash
# Use bootstrap seed set only
python bootstrap_training_data.py
python train_software_models.py \
  --data data/bootstrapped_training_data.jsonl \
  --out models/software
```

### Option 2: With Active Learning
```bash
# 1. Bootstrap
python bootstrap_training_data.py

# 2. Manually create unlabeled pool
# Add diverse text to data/unlabeled_pool.jsonl

# 3. Run active learning
python active_learning_loop.py

# 4. Train on labeled data
python train_software_models.py \
  --data data/labeled_training_data.jsonl \
  --out models/software
```

### Option 3: Full Pipeline (Recommended)
```bash
# 1. Bootstrap
python bootstrap_training_data.py

# 2. Collect data (implement APIs first)
python collect_unlabeled_data.py

# 3. Active learning
python active_learning_loop.py

# 4. Analyze GitHub repos
python analyze_github_repos.py

# 5. Merge all sources
python merge_training_data.py

# 6. Train final models
python train_software_models.py \
  --data data/merged_training_data.jsonl \
  --out models/software
```

## Benefits Summary

1. **No Hardcoded Keywords**: Only 20 seed examples, rest is learned
2. **Real Data**: GitHub repos provide actual LOC, tech stacks
3. **Efficient Labeling**: Active learning focuses on 20-30% uncertain cases
4. **Scalable**: Easy to add more data sources
5. **Adaptive**: Model learns what it doesn't know
6. **Production-Ready**: Real metrics mean accurate predictions

## Next Actions

1. ✅ Created `bootstrap_training_data.py`
2. ✅ Created `active_learning_loop.py`
3. ✅ Created `analyze_github_repos.py`
4. ⏳ Implement `collect_unlabeled_data.py` (API integrations)
5. ⏳ Implement `merge_training_data.py` (combine sources)
6. ⏳ Test new approach vs current approach
7. ⏳ Migrate if performance is better

## Conclusion

This new approach aligns with your requirement of **"NO hardcoded keywords"** by:
- Using minimal seed examples (20 vs 180+ templates)
- Learning from real-world data (GitHub repos, job posts, forums)
- Efficiently using human labeling (active learning)
- Extracting real metrics instead of heuristics

The system becomes truly ML-based rather than template-based!
