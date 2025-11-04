# MCP Complexity Scorer

An MCP (Model Context Protocol) server that predicts the complexity of programming tasks and job requirements using a machine learning model. Scores are calibrated around a baseline of 100 (roughly “Replit Agent 3” difficulty) and include estimated completion time.

## Features

- ML-based predictions for complexity score and time-to-complete
- Calibrated scoring (baseline 100) with human-friendly difficulty labels
- Detected factor hints (frontend, backend, database, real-time, etc.)
- Time estimates in hours/days/weeks with uncertainty range
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
python train_model.py
```

This creates the following files under `models/`:
- `tfidf_vectorizer.joblib`
- `score_model.joblib`
- `time_model.joblib`

### 2) Running the MCP Server

Start the MCP server by running:

```powershell
python mcp_server/server.py
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

Fallback (without uv scripts):

```powershell
uv run demo_time_estimation.py
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
  "estimated_completion_time": {
    "hours": 9.1,
    "days": 1.14,
    "weeks": 0.23,
    "best_estimate": "1.1 days",
    "time_range": "1.1-1.4 days",
    "assumptions": "Assumes developer skilled in using AI coding agents like Replit"
  },
  "summary": "Complexity score: 109.80. Primary complexity factors: database, api integration, security. Estimated completion time: 1.1 days.",
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
python train_model.py
```

3. Re-run demo/tests.

## License

See repository for license information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

[https://github.com/ajay1133/mcp-server-job-description-complexity-score](https://github.com/ajay1133/mcp-server-job-description-complexity-score)
