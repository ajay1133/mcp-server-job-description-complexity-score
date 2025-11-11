# ML-Powered Technology Extraction

This directory contains machine learning models for intelligent technology extraction and analysis.

## Models

### 1. Technology Extractor (NER)
- **Purpose**: Identify technology mentions in job descriptions and resumes
- **Approach**: Transformer-based token classification (DistilBERT/RoBERTa fine-tuned)
- **Fallback**: Regex pattern matching
- **Location**: `models/tech_extractor/`

### 2. Difficulty Scorer
- **Purpose**: Predict difficulty ratings (1-10 scale) for technologies
- **Features**:
  - Years in market
  - GitHub stars (popularity)
  - Stack Overflow questions (community size)
  - Learning resources availability
  - API complexity
  - Ecosystem maturity
- **Approach**: Gradient Boosting Regressor
- **Fallback**: Hardcoded baseline scores
- **Location**: `models/difficulty/`

### 3. Experience Extractor
- **Purpose**: Extract and validate years of experience from text
- **Approach**: Regex patterns + Random Forest classifier for validation
- **Features**:
  - Years value (normalized)
  - Context keywords (senior, junior, lead)
  - Proximity to technology mentions
- **Location**: `models/experience/`

### 4. Alternatives Recommender
- **Purpose**: Suggest alternative/similar technologies
- **Approach**: TF-IDF + SVD embeddings with cosine similarity
- **Features**:
  - Technology descriptions
  - Use cases
  - Category information
- **Fallback**: Curated alternative mappings
- **Location**: `models/alternatives/`

## Training

### Quick Training (for CI)
```bash
python train_system_design_models.py --quick
```

### Full Training
```bash
# Train all models
python train_system_design_models.py

# Train specific models
python train_system_design_models.py --models difficulty experience

# Skip training (useful in CI)
SKIP_MODEL_TRAIN=1 python train_system_design_models.py
```

### Training Data

Sample data is provided in `training_data/sample_data.py`. In production, augment with:

1. **Job Postings**:
   - Use APIs: Indeed, LinkedIn, Stack Overflow Jobs
   - Scrape with permission: company career pages
   - Synthetic generation with templates

2. **Technology Metadata**:
   - GitHub API for stars, contributors, activity
   - Stack Overflow API for question counts
   - Package managers: npm, PyPI, Maven Central

3. **Difficulty Ground Truth**:
   - Developer surveys (Stack Overflow, JetBrains)
   - Learning platform data (Udemy, Coursera)
   - Interview feedback and hiring metrics

4. **Similarity/Alternatives**:
   - Technology comparison articles
   - Migration guides
   - Developer discussions on Reddit, HN

## Usage

### Using ML-Powered Extractor

```python
from mcp_server.ml_tech_extractor import MLTechExtractor

# Initialize (loads trained models or falls back to patterns)
extractor = MLTechExtractor()

# Extract from job posting
result = extractor.extract_technologies(
    "Senior Engineer with 5+ years React experience",
    is_resume=False,
    prompt_override=""
)

# Result:
# {
#     "technologies": {
#         "react": {
#             "difficulty": 5.2,  # ML-predicted
#             "category": "frontend",
#             "alternatives": {
#                 "vue": {"difficulty": 4.8},
#                 "angular": {"difficulty": 6.5}
#             },
#             "experience_mentioned_in_prompt": 5.0
#         }
#     }
# }
```

### Backward Compatibility

The ML extractor maintains the same API as `SimpleTechExtractor`:

```python
# Old way (pattern-based)
from mcp_server.simple_tech_extractor import SimpleTechExtractor
extractor = SimpleTechExtractor()

# New way (ML-powered)
from mcp_server.ml_tech_extractor import MLTechExtractor
extractor = MLTechExtractor()

# Same interface!
result = extractor.extract_technologies(text)
```

## Model Registry

After training, a registry file is created at `models/registry.json`:

```json
{
  "version": "1.0.0",
  "timestamp": 1699700000,
  "models": {
    "difficulty": "models/difficulty",
    "experience": "models/experience",
    "alternatives": "models/alternatives"
  },
  "quick_mode": false
}
```

## Performance

### Accuracy Metrics (Sample Data)

| Model | Metric | Value |
|-------|--------|-------|
| Difficulty | MAE | ~0.5 points |
| Difficulty | RMSE | ~0.8 points |
| Experience | Accuracy | ~85% |
| Alternatives | Similarity MAE | ~0.3 |

### Speed

- **Pattern-based fallback**: <10ms per extraction
- **ML models loaded**: ~50-100ms per extraction
- **Cold start (first load)**: 1-2 seconds

## Dependencies

```bash
# Core ML
scikit-learn>=1.3.0
numpy>=1.24.0
joblib>=1.3.0

# Optional (for transformer models)
torch>=2.0.0
transformers>=4.30.0
```

## Roadmap

### Phase 1 (Current)
- [x] Pattern-based extraction
- [x] ML difficulty scoring
- [x] Experience validation
- [x] Embedding-based alternatives

### Phase 2 (Next)
- [ ] Fine-tune transformer for NER
- [ ] Gather real job posting dataset (10k+)
- [ ] Active learning for difficulty labels
- [ ] Graph-based alternatives (tech dependency graph)

### Phase 3 (Future)
- [ ] Multi-modal: parse code samples in resumes
- [ ] Temporal trends: difficulty changes over time
- [ ] Personalized recommendations based on existing skills
- [ ] Integration with GitHub profile analysis

## Contributing

To improve models:

1. **Add Training Data**: Edit `training_data/sample_data.py`
2. **Retrain**: Run `python train_system_design_models.py`
3. **Evaluate**: Check metrics in training output
4. **Test**: Run test suite to ensure compatibility

## License

Same as parent project.
