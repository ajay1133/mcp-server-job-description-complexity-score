# MCP Complexity Scorer

A Model Context Protocol (MCP) server that analyzes programming requirements and job descriptions to provide complexity scores. The scoring system uses Replit Agent 3's capabilities as a baseline (score of 100), helping you assess task difficulty and time estimates.

## Features

- **Complexity Analysis**: Analyzes text to detect technical complexity factors
- **Calibrated Scoring**: Uses Replit Agent 3 as a baseline (score 100) for relative difficulty assessment
- **Detailed Breakdown**: Provides detected factors, task size, difficulty rating, and summary
- **MCP Integration**: Works as an MCP server for integration with compatible AI assistants

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

### Running the MCP Server

Start the MCP server by running:

```powershell
python mcp_server/server.py
```

The server will start and listen for MCP tool requests.

### Using as a Standalone Tool

You can also import and use the complexity scorer in your Python code:

```python
from mcp_server.complexity_scorer import ComplexityScorer

scorer = ComplexityScorer()
result = scorer.analyze_text("Build a full-stack web application with React frontend and Django backend")

print(f"Complexity Score: {result['complexity_score']}")
print(f"Difficulty: {result['difficulty_rating']}")
print(f"Summary: {result['summary']}")
```

### Running Tests

Run the test suite:

```powershell
python test_scoring.py
```

## MCP Tool: `score_complexity`

### Description
Analyzes programming requirements or job descriptions and provides a complexity score calibrated against Replit Agent 3's capabilities.

### Parameters
- `requirement` (string): A text description of the programming requirement or job description

### Returns
A dictionary containing:
- `complexity_score`: Numerical score relative to Replit Agent 3 (baseline 100)
- `detected_factors`: Technical complexity factors identified in the text
- `task_size`: Estimated task size (simple, moderate, complex, very_complex, expert)
- `difficulty_rating`: Human-readable difficulty assessment
- `summary`: Brief summary of the analysis

### Example Request
```json
{
  "requirement": "Create a RESTful API with PostgreSQL database, user authentication, and real-time notifications"
}
```

### Example Response
```json
{
  "complexity_score": 115.5,
  "baseline_reference": 100,
  "detected_factors": {
    "database": {"matches": 1, "weight": 33, "contribution": 33},
    "api_integration": {"matches": 1, "weight": 29, "contribution": 29},
    "security": {"matches": 1, "weight": 28, "contribution": 28},
    "real_time": {"matches": 1, "weight": 30, "contribution": 30}
  },
  "task_size": "complex",
  "size_multiplier": 1.0,
  "difficulty_rating": "Similar to Replit Agent 3 capabilities",
  "summary": "Complexity score: 115.50. Primary complexity factors: database, real_time, api_integration."
}
```

## Complexity Factors

The scorer evaluates the following complexity factors:

- **Basic Web** (weight: 30): HTML, CSS, static sites
- **Database** (weight: 33): PostgreSQL, MySQL, MongoDB, ORMs
- **API Integration** (weight: 29): REST APIs, GraphQL, webhooks, OAuth
- **Frontend** (weight: 24): React, Vue, Angular, TypeScript
- **Backend** (weight: 24): Django, Flask, FastAPI, Node.js
- **Real-time** (weight: 30): WebSockets, streaming, collaborative features
- **AI/ML** (weight: 38): Machine learning, neural networks, AI-powered features
- **Deployment** (weight: 15): CI/CD, Docker, Kubernetes, cloud platforms
- **Security** (weight: 28): Authentication, encryption, JWT, user management
- **Testing** (weight: 14): Unit tests, integration tests, test coverage
- **Scalability** (weight: 26): Load balancing, caching, message queues

## Scoring Interpretation

- **< 50**: Much easier than Replit Agent 3 capabilities
- **50-80**: Easier than Replit Agent 3 capabilities
- **80-120**: Similar to Replit Agent 3 capabilities (baseline)
- **120-150**: More challenging than Replit Agent 3 capabilities
- **> 150**: Significantly more challenging than Replit Agent 3 capabilities

## Project Structure

```
mcp_complexity_scorer/
├── mcp_server/
│   ├── __init__.py
│   ├── server.py              # MCP server implementation
│   └── complexity_scorer.py   # Core scoring logic
├── complexity_mcp_project/
│   ├── __init__.py
│   ├── settings.py            # Django settings
│   ├── urls.py
│   ├── wsgi.py
│   └── asgi.py
├── manage.py                  # Django management script
├── pyproject.toml            # Project dependencies
├── test_scoring.py           # Test suite
└── README.md                 # This file
```

## Development

### Adding New Complexity Factors

Edit `mcp_server/complexity_scorer.py` and add new factors to the `complexity_factors` dictionary:

```python
'your_factor': {
    'keywords': ['keyword1', 'keyword2'],
    'weight': 25
}
```

### Adjusting Weights

Modify the `weight` values in `complexity_factors` to fine-tune the scoring system based on your experience.

## License

See repository for license information.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Repository

[https://github.com/ajay1133/mcp-server-job-description-complexity-score](https://github.com/ajay1133/mcp-server-job-description-complexity-score)
