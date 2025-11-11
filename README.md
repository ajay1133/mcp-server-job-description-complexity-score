# Technology Extractor MCP Server

A lightweight MCP (Model Context Protocol) server that extracts required technologies from job descriptions **and resume files** and provides difficulty ratings with alternatives.

## Features

- **Technology Detection**: Automatically detects technologies mentioned in text or resume files
- **Resume File Support**: Parse .txt, .docx, .pdf, and .doc resume files
- **Difficulty Ratings**: Provides difficulty scores (1-10 scale) for each technology
- **Experience Tracking**: Three-tier experience validation:
  - `experience_mentioned_in_prompt`: Years specified in job requirements
  - `experience_accounted_for_in_resume`: Years extracted from resume
  - `experience_validated_via_github`: GitHub-based verification (placeholder for future)
- **Smart Alternatives**: Suggests alternative technologies with their difficulty ratings
- **Simple Schema**: Returns clean, structured JSON with no overhead
- **Request Logging**: Automatic per-request logging with timing, CPU, and memory metrics

## Installation

```bash
# Clone the repository
git clone https://github.com/ajay1133/mcp-server-job-description-complexity-score.git
cd mcp-server-job-description-complexity-score

# Install dependencies (includes resume parsing libraries)
pip install -e .
# OR with uv:
uv pip install -e .
```

**Resume parsing dependencies automatically installed:**
- `python-docx` for .docx files
- `PyPDF2` for .pdf files
- `python-magic-bin` for Windows file type detection

## Usage

### As MCP Server

```bash
python mcp_server/server.py
```

### Self-Test

```bash
python mcp_server/server.py --self-test
```

### As Python Module

```python
from mcp_server.simple_tech_extractor import SimpleTechExtractor

extractor = SimpleTechExtractor()
result = extractor.extract_technologies("Senior Full-Stack Engineer with React and Node.js")

print(result)
# {
#   "technologies": {
#     "react": {
#       "difficulty": 5.2,
#       "experience_required": 2.5,
#       "mentioned_in_prompt": true,
#       "category": "frontend",
#       "alternatives": {
#         "vue": {"difficulty": 4.8, "experience_required": 2.0},
#         "angular": {"difficulty": 6.5, "experience_required": 3.0}
#       }
#     },
#     "node": {
#       "difficulty": 5.0,
#       "experience_required": 2.5,
#       "mentioned_in_prompt": true,
#       "category": "backend",
#       "alternatives": {
#         "python_fastapi": {"difficulty": 4.5, "experience_required": 2.0}
#       }
#     }
#   }
# }
```

## Response Schema

```json
{
  "technologies": {
    "<tech_name>": {
      "difficulty": 5.2,                    // 1-10 scale
      "mentioned_in_prompt": true,          // boolean
      "category": "frontend",               // category
      "alternatives": {
        "<alt_tech_name>": {
          "difficulty": 4.8
        }
      },
      "experience_required": 5.0            // Optional: only if explicitly mentioned (e.g., "5+ years React")
    }
  }
}
```

**Note**: `experience_required` is only included when explicitly mentioned in the prompt (e.g., "5+ years React" or "3 years Node.js experience").

## Supported Technologies

### Frontend
- React, Vue, Angular, Next.js, Svelte, TypeScript

### Backend
- Node.js, FastAPI, Flask, Django, Golang, Java Spring, Ruby on Rails

### Database
- PostgreSQL, MySQL, MongoDB, Redis, DynamoDB, Cassandra

### Infrastructure
- Docker, Kubernetes, AWS, Lambda

### Messaging
- Kafka, RabbitMQ

### Search
- Elasticsearch

## License

MIT
python active_learning_loop.py
```

4) (Optional) Analyze real GitHub repos for actual LOC/tech/hours

```powershell
python analyze_github_repos.py
```

5) Merge all sources and de-duplicate by text

```powershell
python merge_training_data.py --out data\merged_training_data.jsonl
```

6) Train models on merged dataset

```powershell
python train_software_models.py --data data\merged_training_data.jsonl --out models\software
```

7) Validate and run server

```powershell
python test_new_schema.py
python mcp_server\server.py --self-test
python mcp_server\server.py  # long-running MCP server
```

### Training with System Design Patterns (Recommended)

**NEW**: Generate high-quality training data from system design patterns:

```powershell
# 1. Generate training data from patterns (47 examples covering 10 major apps)
python generate_training_from_patterns.py --output data/training_from_patterns.jsonl

# 2. Merge with existing data
python merge_training_data.py --inputs data/training_from_patterns.jsonl data/software_training_data.jsonl --out data/merged_training_data.jsonl

# 3. Train models with enriched dataset
python train_software_models.py --data data/merged_training_data.jsonl --out models/software

# 4. Test the improved models
python run_requirements_cli.py --text "Build a Twitter clone with real-time feeds"
```

This approach uses the comprehensive system design knowledge base to generate realistic training examples for:
- Twitter, Instagram, YouTube, WhatsApp, Uber, Netflix, Airbnb, E-commerce, Slack, TikTok patterns
- Comprehensive tech stacks (15-22 technologies per example)
- Production-ready microservice architectures (10-15 services)
- Accurate complexity estimates (500-1200 hours for platforms)

See [AI_MODEL_TRAINING_SUMMARY.md](AI_MODEL_TRAINING_SUMMARY.md) for complete details.

### Current Status

✅ **Models trained with system design knowledge** - Ready to use!

- **401 training examples** (pattern-based + existing data)
- **49 technology labels** (including infrastructure: kafka, redis, docker, cdn, monitoring, etc.)
- **10 application patterns** recognized (Twitter, YouTube, Uber, etc.)
- **Production-ready** architecture recommendations

**Test the system**:
   ```bash
   python test_software_scorer.py
   ```

The MCP server (`server.py`) is already configured to use `SoftwareComplexityScorer` but will fail until models are trained.

### Optional: Hiring vs Build classifier (binary)

To cleanly separate output schemas for job descriptions vs build requirements, you can train a small binary classifier:

Dataset (JSONL): one object per line with fields `{ "text": str, "label": int }` where `label=1` for hiring/job-description and `label=0` for build/implementation. An example is in `data/hiring_build_training_data.example.jsonl`.

Train (PowerShell):

```powershell
$env:HIRING_BUILD_DATA = "data/hiring_build_training_data.jsonl"
python train_hiring_classifier.py
```

This writes `models/software/hiring_build_classifier.joblib`. When present, `SoftwareComplexityScorer` will use it to detect hiring prompts with confidence thresholds (>=0.65 → hiring, <=0.35 → build) and fall back to the existing heuristic when confidence is low.

Recommended: curate ~500–1,000 labeled examples for good separation. You can bootstrap with heuristics and then hand-correct.

#### Workflow: evaluation, active learning, and threshold tuning

**1. Evaluate classifier vs heuristic baseline:**

After training with at least 100–200 examples, measure performance:

```powershell
$env:HIRING_BUILD_DATA = "data/hiring_build_training_data.jsonl"
python evaluate_hiring_classifier.py --test-size 0.2
```

This generates:
- Precision/recall/F1 report for model and heuristic
- `logs/hiring_classifier_pr_curve.png` and `logs/hiring_classifier_roc_curve.png`
- `logs/hiring_classifier_evaluation.json` with AUC, AP, recommended thresholds

**2. Active learning to grow dataset efficiently:**

Surface uncertain examples (probabilities 0.35–0.65) for manual labeling:

```powershell
python active_learning_hiring.py --unlabeled data/unlabeled_prompts.txt --limit 50 --out data/uncertain_samples.jsonl
```

Manually edit `data/uncertain_samples.jsonl` to add `"label": 0` or `"label": 1`, then merge:

```powershell
type data\hiring_build_training_data.jsonl data\uncertain_samples.jsonl > data\merged.jsonl
$env:HIRING_BUILD_DATA = "data\merged.jsonl"
python train_hiring_classifier.py
```

**3. Tune decision threshold to minimize misclassification cost:**

If false negatives (hiring → build) are more expensive than false positives (build → hiring), tune the threshold:

```powershell
python tune_hiring_threshold.py --data data/hiring_build_training_data.jsonl --cost-fp 1.0 --cost-fn 2.0 --write-config
```

This finds the optimal threshold and writes it to `config/hiring_threshold.json`. Update `_predict_is_hiring` in the scorer to use the new threshold (default is 0.65 for hiring, 0.35 for build).

Example: if tuning suggests 0.72 for hiring, change:
```python
if proba >= 0.72:  # was 0.65
    return True, proba, "model"
```

**Iterative improvement loop:**

1. Train initial model with ~100 examples
2. Evaluate and identify weak areas
3. Use active learning to label uncertain examples
4. Retrain with expanded data
5. Tune threshold for production cost function
6. Repeat until precision/recall targets are met

---

## Legacy Multi-Profession Scorer (deprecated)

The original `ComplexityScorer` handled both software and non-software jobs with online search heuristics and profession categorization. This approach is being phased out.

An MCP (Model Context Protocol) server that predicts the complexity of programming tasks and job requirements using a machine learning model. Scores are calibrated around a baseline of 100 (roughly "Replit Agent 3" difficulty) and include estimated completion time.

### Features (legacy scorer)

- ML-based predictions for complexity score and time-to-complete
- Calibrated scoring (baseline 100) with human-friendly difficulty labels
- Detected factor hints (frontend, backend, database, real-time, etc.)
- Time estimates in hours/days/weeks with uncertainty range
- **Duration extraction**: Automatically detects time requirements from user prompts (e.g., "couple of days", "3 weeks")
- **Smart time calculation**: Distinguishes between project deadlines (8-hour workdays) and continuous care (24/7)
- **Job categorization**: Automatically deduces job category and sub-category from requirements
- **Extended profession database**: Falls back to comprehensive profession database (100+ professions) when primary categorization fails
- MCP tool integration for assistants that support MCP

## Prerequisites

- **Python**: Version 3.11 or higher
- **pip**: Python package installer (usually comes with Python)
- **uv** (recommended): Fast Python package installer and resolver
  - Install via PowerShell: `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
  - Or via pip: `pip install uv`

## Installation

1. **Clone the repository** (if not already done):
   ```powershell
   git clone https://github.com/ajay1133/mcp-server-job-description-complexity-score.git
   cd mcp-server-job-description-complexity-score
   ```

2. **Install dependencies using uv** (recommended):
   ```powershell
   uv pip install -e .
   ```

  Or using pip:
   ```powershell
   pip install -e .
   ```

## Usage

### 1) Train the model (first-time or after updating data)

Models are not committed; you must train locally before use:

```powershell
uv run train-model
```

Alternative:

```powershell
python train_model.py
```

This creates the following files under `models/`:
- `tfidf_vectorizer.joblib`
- `score_model.joblib`
- `time_model.joblib`

### 2) Running the MCP Server

Start the MCP server by running:

```powershell
uv run mcp-server
```

Alternative:

```powershell
python -m mcp_server.server
```

If models are missing, the server will warn you to run `train_model.py` first.

### 2b) Inspect the server with MCP Inspector (recommended)

Requires Node.js with `npx`:

```powershell
uv run mcp-inspect-server
```

This launches the MCP Inspector and spawns the server via uv (stdio), ensuring it uses the same Python environment and dependencies managed by uv. Try the `score_complexity` tool interactively.

### 3) Using as a Standalone Tool

You can also import and use the complexity scorer in your Python code:

```python
from mcp_server.complexity_scorer import ComplexityScorer

scorer = ComplexityScorer()
result = scorer.analyze_text("Build a full-stack web application with React frontend and Django backend")

print(f"Complexity Score: {result['complexity_score']}")
print(f"Difficulty: {result['difficulty_rating']}")
print(f"Summary: {result['summary']}")
```

### 4) Running Tests

Run the test suite:

```powershell
python test_scoring.py
```

### 5) Running the Demo

See the time estimation feature in action:

```powershell
uv run demo
```

This prints several examples from simple to expert-level and shows predicted time ranges.

## MCP Tool: `score_complexity`

### Description
Analyzes programming requirements or job descriptions and provides a complexity score calibrated against Replit Agent 3's capabilities.

### Parameters
- `requirement` (string): A text description of the programming requirement or job description

### Returns
A dictionary containing:
- `complexity_score`: Numerical score (baseline reference = 100)
- `detected_factors`: Map of factors and relevance signals (e.g., matches, relevance)
- `task_size`: simple | moderate | complex | very_complex | expert
- `difficulty_rating`: Human-friendly description
- `job_category`: Deduced job category (e.g., "Software Developer", "Doctor", "Plumber")
- `job_sub_category`: Deduced job specialization (e.g., "Full Stack Developer (React + Node.js)", "Gastroenterologist", "General Plumber")
- `category_lookup_method`: How the category was determined - "primary_pattern", "extended_database", "online_search", or "default_fallback"
- `estimated_completion_time`: Object with `hours`, `days`, `weeks`, `best_estimate`, `time_range`, `assumptions`
- `summary`: Brief summary including time
- `model_type`: Always `"machine_learning"` for this version

### Example Request
```json
{
  "requirement": "Create a RESTful API with PostgreSQL database, user authentication, and real-time notifications"
}
```

### Example Response
```json
{
  "complexity_score": 109.8,
  "baseline_reference": 100,
  "detected_factors": {
    "database": {"matches": 1, "relevance": 0.22},
    "api_integration": {"matches": 1, "relevance": 0.18},
    "security": {"matches": 1, "relevance": 0.14}
  },
  "task_size": "complex",
  "difficulty_rating": "Similar to Replit Agent 3 capabilities",
  "job_category": "Software Developer",
  "job_sub_category": "Backend Developer",
  "category_lookup_method": "primary_pattern",
  "estimated_completion_time": {
    "hours": 9.1,
    "days": 1.14,
    "weeks": 0.23,
    "best_estimate": "1.1 days",
    "time_range": "1.1-1.4 days",
    "assumptions": "Time estimate based on task complexity and typical completion times for similar requirements"
  },
  "summary": "Complexity score: 109.80. Complex task. Primary complexity factors: database, api integration, security. Estimated completion time: 1.1 days.",
  "model_type": "machine_learning"
}
```

## Complexity Factors

The model reports hints for the following factor categories (non-exhaustive examples):

- Basic Web: HTML, CSS, static sites
- Database: PostgreSQL, MySQL, MongoDB, ORMs
- API Integration: REST, GraphQL, webhooks, OAuth
- Frontend: React, Vue, Angular, TypeScript
- Backend: Django, Flask, FastAPI, Node.js
- Real-time: WebSockets, streaming, collaborative features
- AI/ML: ML pipelines, model training, OpenAI, NLP
- Deployment: CI/CD, Docker, Kubernetes, cloud platforms
- Security: Auth, encryption, JWT, RBAC
- Testing: Unit, integration, E2E, coverage
- Scalability: Caching, queues, load balancing, distributed systems

### Configure factor categories (no hardcoded lists)

You can customize the factor categories without code changes:

- Edit `config/complexity_factors.json` to add/remove categories or keywords
- Or set `MCP_COMPLEXITY_FACTORS` to point to a custom JSON file
- Or pass a mapping directly when constructing the scorer: `ComplexityScorer(complexity_factors=...)`

If no config is found, sensible built-in defaults are used.

## Job Categorization

The scorer automatically deduces the job category and sub-category from the requirement text. This feature works for both software development roles and general professions.

### Supported Categories

**Software Development:**
- Full Stack Developer (React + Node.js, Vue.js, Angular, MERN Stack)
- Frontend Developer (React, Vue, Angular)
- Backend Developer (Node.js, Django, Flask, FastAPI)
- Mobile Developer (React Native, Flutter, iOS, Android)
- AI/ML Developer
- Data Scientist
- DevOps Engineer

**Healthcare:**
- Doctor (Gastroenterologist, Cardiologist, Neurologist, Orthopedic Surgeon, Dermatologist, Pediatrician, Ophthalmologist, General Physician)
- Nurse (Registered Nurse, Licensed Practical Nurse)

**Trades & Services:**
- Plumber (Emergency Plumber, General Plumber)
- Electrician
- Carpenter

**Child & Home Care:**
- Child Care Provider (Nanny, Babysitter, Child Care Specialist)
- Housekeeper
- **Caregiver** (Home Health Aide, Home Health Aide with Housekeeping)

**Professional Services:**
- Lawyer (Criminal Defense, Corporate, General Practice)
- Teacher (Mathematics, Science, Language Arts)
- Accountant (CPA, General)
- Driver (Ride-share, Commercial, Personal)

**Extended Professions** (100+ supported via fallback database):
- Medical: Veterinarian, Dentist, Therapist, Psychologist, Pharmacist, Paramedic
- Creative: Photographer, Videographer, Graphic Designer, Writer, Musician, Artist
- Culinary: Chef, Cook, Baker, Bartender
- Trades: Mechanic, HVAC Technician, Welder, Mason, Roofer, Painter
- Services: Hairdresser, Barber, Massage Therapist, Personal Trainer
- Real Estate: Architect, Engineer, Surveyor, Contractor, Realtor
- Business: Consultant, Analyst, Banker, Broker
- Security: Security Guard, Firefighter, Police Officer
- Transportation: Pilot, Flight Attendant, Delivery Driver
- Agriculture: Farmer, Gardener, Landscaper, Florist
- And many more...

### Example Job Category Deductions

| Requirement | Job Category | Job Sub-Category |
|-------------|--------------|------------------|
| "I need a software developer who can develop a video streaming application in React Js and Node Js" | Software Developer | Full Stack Developer (React + Node.js) |
| "I have problems with my liver" | Doctor | Gastroenterologist |
| "I need someone who can look at my child while I am gone for work" | Child Care Provider | Child Care Specialist |
| "Looking for a data scientist with machine learning experience" | Data Scientist | Machine Learning Specialist |
| "Need a mobile app developer for iOS and Android using Flutter" | Software Developer | Mobile Developer (Flutter) |
| "I need someone to look after my dad, he can barely walk due to diabetes and cannot cook his meals" | Caregiver | Home Health Aide with Housekeeping |

## Category Lookup Method

The `category_lookup_method` field tracks how the job category was determined, providing transparency and auditability.

### Lookup Methods

| Method | Description | Icon | Example |
|--------|-------------|------|---------|
| `primary_pattern` | Detected using primary pattern matching for common professions with context-aware subcategories | 🎯 | Software Developer → "Full Stack Developer (React + Node.js)" |
| `extended_database` | Found in extended profession database containing 100+ professions across multiple domains | 📚 | Veterinarian, Photographer, Chef, Mechanic |
| `online_search` | Retrieved via online keyword matching fallback when primary patterns and extended database don't match | 🌐 | Uncommon professions detected via action keywords (repair, design, install, etc.) |
| `default_fallback` | No match found in any database or online search, using generic categorization | ❓ | Very uncommon or vague requirements |

### Online Search Capability

When the primary patterns and extended database fail to categorize a profession, the system automatically uses online search logic:

**Triggers:**
- When no primary pattern matches the requirement
- When `detected_factors` is empty (no technical keywords found)
- For default fallback cases before settling on "General Professional"

**Search Strategy:**
1. **Primary Search**: Attempts DuckDuckGo API instant answer lookup (when available)
2. **Keyword Fallback**: Matches action-based keywords in the text:
   - Action verbs: repair, fix, install, design, build, create, organize, plan, manage, coordinate, help, service
   - Rare professions: sommelier, curator, librarian, auctioneer, appraiser, jeweler, locksmith, upholsterer, taxidermist, mortician, interpreter, translator, stenographer, actuary, statistician, meteorologist, geologist, astronomer, botanist, zoologist, ecologist

**Examples:**
- "I need someone to repair my antique clock" → `Repair Technician` (via "repair" keyword) ✅ online_search
- "Need a sommelier for my restaurant" → `Sommelier` (via profession keyword) ✅ online_search
- "Help me organize my event" → `Event Organizer` (via "organize" keyword) ✅ online_search
- "I need to design a logo" → `Designer` (via "design" keyword) ✅ online_search

### Usage

The `category_lookup_method` field helps you:
- **Track categorization performance**: See which method was used for each request
- **Identify gaps**: Find professions frequently hitting extended database or fallback
- **Audit accuracy**: Verify categorization logic is working as expected
- **Improve coverage**: Add frequently-requested professions to primary patterns

### Example

```json
{
  "job_category": "Veterinarian",
  "job_sub_category": "Animal Healthcare Professional",
  "category_lookup_method": "extended_database"
}
```

This indicates the profession was found in the extended database (Tier 2), not through primary pattern matching.

## Duration Extraction

The scorer automatically detects duration requirements mentioned in the user's prompt and adjusts time estimates accordingly.

### Supported Duration Patterns

**Numeric Patterns:**
- "3 hours", "2 days", "4 weeks", "2 months"

**Word-based Patterns:**
- "couple of days" = 2 days
- "few days" = 3 days
- "couple of weeks" = 2 weeks
- "weekend" = 2 days

**Special Cases:**
- "overnight" = 12 hours
- "all day" / "full day" = 8 hours
- "half day" = 4 hours

### Project Deadlines vs Continuous Care

The system intelligently distinguishes between:

1. **Project Deadlines** (8-hour workdays):
   - Detected when phrases like "needs to be done in", "deadline", "complete in" are present
   - Example: "Build a React app, needs to be done in 3 days" = 24 work hours (3 days × 8 hours)

2. **Continuous Care** (24/7):
   - Applied to Caregiver, Nurse, Child Care Provider, Housekeeper categories
   - Example: "Need someone to look after my dad for couple of days" = 48 hours (2 days × 24 hours)

### Duration Extraction Examples

| Prompt | Detected Duration | Job Category | Time Estimate |
|--------|------------------|--------------|---------------|
| "I need someone to look after my dad... I will be gone for a couple of days" | "couple of day" | Caregiver | 2.0 days (continuous care) |
| "Build a React web app, needs to be done in 3 days" | "3 day" | Software Developer | 3.0 days |
| "Need a babysitter for the weekend" | "weekend" | Child Care Provider | 2.0 days (continuous care) |
| "Need a nurse for my mother's recovery, will need help for 2 weeks" | "2 week" | Nurse | 2.0 weeks (continuous care) |

## Scoring Interpretation

- **< 50**: Much easier than Replit Agent 3 capabilities
- **50-80**: Easier than Replit Agent 3 capabilities
- **80-120**: Similar to Replit Agent 3 capabilities (baseline)
- **120-150**: More challenging than Replit Agent 3 capabilities
- **> 150**: Significantly more challenging than Replit Agent 3 capabilities

## Time Estimation

The scorer provides completion time estimates based on the complexity score, assuming the developer is skilled in using AI coding agents like Replit.

### Estimation Logic

1. **Baseline**: A task with a complexity score of 100 is estimated at 8 hours (1 working day)
2. **Linear Scaling**: Time scales proportionally with complexity score
3. **Task Size Adjustments**: Non-linear adjustments based on task complexity:
   - Simple tasks: 0.6x multiplier (complete faster than linear)
   - Moderate tasks: 0.8x multiplier
   - Complex tasks: 1.0x multiplier (linear)
   - Very complex tasks: 1.3x multiplier (extra coordination overhead)
   - Expert tasks: 1.6x multiplier (significant architectural overhead)
4. **Time Range**: Provides a range (best estimate to 1.3x) to account for uncertainty

### Time Format

The estimate automatically selects the most appropriate time unit:
- **Minutes**: For tasks under 1 hour
- **Hours**: For tasks 1-8 hours
- **Days**: For tasks 1-5 days (8-hour workdays)
- **Weeks**: For tasks over 5 days (5-day work weeks)

### Example Time Estimates

| Complexity Score | Task Size | Estimated Time |
|-----------------|-----------|----------------|
| 30 | Simple | ~1.4 hours |
| 75 | Moderate | ~4.8 hours |
| 100 | Complex | ~8 hours (1 day) |
| 150 | Very Complex | ~2.4 days |
| 200 | Expert | ~3.2 days |

## Project Structure

```
mcp_complexity_scorer/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py              # MCP server implementation
│   └── complexity_scorer.py   # Core ML-based scoring logic
├── complexity_mcp_project/
│   ├── __init__.py
│   ├── settings.py            # Django settings
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── models/                   # Trained artifacts (created by training)
│   ├── tfidf_vectorizer.joblib
│   ├── score_model.joblib
│   └── time_model.joblib
├── train_model.py            # Train TF-IDF + regressors
├── training_data.py          # Labeled examples and validation ranges
├── demo_time_estimation.py   # Demo runner printing examples
├── pyproject.toml            # Project dependencies
├── test_scoring.py           # Scripted tests (ranges)
└── README.md                 # This file
```

## Development

### Add or refine training data

1. Edit `training_data.py` and append new labeled examples.
2. Retrain models:

```powershell
uv run train-model
```

3. Re-run demo/tests.

## License

See repository for license information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

[https://github.com/ajay1133/mcp-server-job-description-complexity-score](https://github.com/ajay1133/mcp-server-job-description-complexity-score)

---

## Docker Deployment

The project includes Docker support for containerized deployment with both MCP server and Flask API modes.

### Building the Docker Image

```powershell
# Build the image
docker build -t mcp-complexity-scorer:latest .

# Or build with a specific tag
docker build -t your-dockerhub-username/mcp-complexity-scorer:dev .
```

### Running with Docker

**MCP Server Mode (default):**
```powershell
docker run -p 8000:8000 mcp-complexity-scorer:latest
```

**Flask API Mode:**
```powershell
docker run -p 8000:8000 -e FLASK_MODE=1 mcp-complexity-scorer:latest
```

**With custom configuration:**
```powershell
docker run -p 8000:8000 `
  -e FLASK_MODE=1 `
  -e HOST=0.0.0.0 `
  -e PORT=8000 `
  -v ${PWD}/logs:/app/logs `
  mcp-complexity-scorer:latest
```

### Docker Compose

For local development with logs mounted:

```powershell
docker-compose up
```

The `docker-compose.yml` includes:
- Volume mounts for logs persistence
- Environment variables for configuration
- Port mapping to localhost:8000

### Flask API Endpoints

When running in Flask mode (`FLASK_MODE=1`):

- **GET `/health`** - Health check endpoint
  ```bash
  curl http://localhost:8000/health
  ```

- **POST `/score`** - Analyze complexity
  ```bash
  curl -X POST http://localhost:8000/score \
    -H "Content-Type: application/json" \
    -d '{"requirement": "Build a React dashboard with Stripe payments"}'
  ```

---

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration and deployment across multiple environments.

### Pipeline Overview

**Triggers:**
- Push to: `master`, `development`, `qa`, `uat` branches
- Pull requests to any of these branches

**Jobs:**
1. **Test** - Runs on Python 3.10, 3.11, 3.12
2. **Lint** - Code quality checks (flake8, black, isort)
3. **Docker** - Build and push images (on push only)
4. **Deploy** - Environment-specific deployment (placeholder)

### Branch → Environment Mapping

| Branch | Environment | Docker Tag | Description |
|--------|-------------|------------|-------------|
| `master` | Production | `prod-{sha}`, `prod-latest` | Stable production releases |
| `development` | Development | `dev-{sha}`, `dev-latest` | Active development |
| `qa` | QA | `qa-{sha}`, `qa-latest` | Quality assurance testing |
| `uat` | UAT | `uat-{sha}`, `uat-latest` | User acceptance testing |

### Setting Up CI/CD

**1. Configure Docker Hub Secrets**

Add these secrets to your GitHub repository (Settings → Secrets and variables → Actions):

- **`DOCKER_USERNAME`**: Your Docker Hub username
- **`DOCKER_PASSWORD`**: Docker Hub access token (recommended) or password

**To create a Docker Hub access token:**
1. Log in to https://hub.docker.com
2. Go to Account Settings → Security → Access Tokens
3. Click "New Access Token"
4. Set permissions: Read, Write, Delete
5. Copy the token and add as `DOCKER_PASSWORD` secret

**2. Workflow Steps**

The pipeline automatically:
- ✅ Runs tests across Python 3.10, 3.11, 3.12
- ✅ Checks code style with flake8, black, isort
- ✅ Builds Docker images with environment-specific tags
- ✅ Pushes to Docker Hub on successful builds
- ℹ️ Notifies deployment (customize for your infrastructure)

**3. Running Locally**

Test the CI steps locally before pushing:

```powershell
# Install dev dependencies
pip install pytest pytest-cov flake8 black isort

# Run tests
pytest tests/ -v --cov=mcp_server

# Check linting
flake8 mcp_server/
black --check mcp_server/
isort --check-only mcp_server/

# Build Docker
docker build -t mcp-complexity-scorer:test .
```

### Deployment

After Docker images are pushed, customize the `deploy` job in `.github/workflows/ci-cd.yml` to:
- Deploy to Kubernetes clusters
- Update ECS task definitions
- Trigger Azure Container Instances
- Or your preferred container orchestration platform

**Example Kubernetes deployment:**
```yaml
- name: Deploy to Kubernetes
  run: |
    kubectl set image deployment/complexity-scorer \
      app=${{ secrets.DOCKER_USERNAME }}/mcp-complexity-scorer:${{ env.IMAGE_TAG }} \
      --namespace=${{ env.ENV_NAME }}
```

### Monitoring CI/CD

- View workflow runs: Repository → Actions tab
- Check build logs: Click on any workflow run
- Docker images: https://hub.docker.com/r/YOUR_USERNAME/mcp-complexity-scorer

---
