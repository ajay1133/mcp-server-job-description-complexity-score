# MCP Complexity Scorer

## Overview
This project is a **Model Context Protocol (MCP) server** built with Python and Django that provides a tool for scoring the complexity of programming requirements and job descriptions. The scoring system uses **Replit Agent 3's capabilities as a baseline reference** (assigned a score of 100).

## Purpose
The MCP server exposes a `score_complexity` tool that analyzes programming requirements and provides:
- Numerical complexity score (relative to Replit Agent 3's baseline of 100)
- Detected technical complexity factors
- Task size estimation
- Difficulty rating with human-readable description
- Summary of the analysis

## Recent Changes
- **Oct 31, 2025**: Initial project setup
  - Created MCP server using FastMCP from the official Python SDK
  - Implemented complexity scoring algorithm with 11 technical factors
  - Set up Django project structure
  - Configured workflow for running the MCP server
  - Calibrated scoring system with Replit Agent 3 as baseline (100)

## Project Architecture

### Directory Structure
```
.
├── complexity_mcp_project/      # Django project configuration
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
├── mcp_server/                  # MCP server implementation
│   ├── __init__.py
│   ├── server.py               # FastMCP server with tool definitions
│   └── complexity_scorer.py    # Complexity scoring algorithm
├── manage.py                    # Django management script
├── pyproject.toml              # Python project dependencies
└── replit.md                   # This documentation
```

### Key Components

#### 1. ComplexityScorer (`mcp_server/complexity_scorer.py`)
The scoring algorithm analyzes text based on:
- **10 Technical Factors**: Database, API Integration, Frontend, Backend, Real-time, AI/ML, Deployment, Security, Testing, Scalability
- **Task Size Estimation**: Simple, Moderate, Complex, Very Complex, Expert
- **Weighted Scoring**: Each factor has a weight contribution to the final score
- **Multipliers**: Task size affects the final complexity score

#### 2. MCP Server (`mcp_server/server.py`)
- Uses FastMCP from the official `mcp` Python SDK (v1.20.0)
- Exposes one tool: `score_complexity(requirement: str) -> dict`
- Runs over stdio transport (standard for MCP servers)
- Returns structured JSON response with complexity analysis

## How to Use

### Running the MCP Server
The server is configured to run automatically via the workflow:
```bash
python mcp_server/server.py
```

### MCP Tool: `score_complexity`

**Purpose**: Analyzes programming requirements or job descriptions and provides a complexity score calibrated to Replit Agent 3's capabilities (baseline 100).

**Input**:
- `requirement` (string): Text description of a programming requirement or job description

**Output** (JSON):
```json
{
  "complexity_score": 138.0,
  "baseline_reference": 100,
  "detected_factors": {
    "api_integration": {
      "matches": 2,
      "weight": 29,
      "contribution": 29
    },
    "frontend": {
      "matches": 1,
      "weight": 24,
      "contribution": 24
    },
    "backend": {
      "matches": 1,
      "weight": 24,
      "contribution": 24
    }
  },
  "task_size": "complex",
  "size_multiplier": 1.0,
  "difficulty_rating": "More challenging than Replit Agent 3 capabilities",
  "summary": "Complexity score: 138.00. Primary complexity factors: api_integration, frontend, backend."
}
```

### Complexity Scoring System

**Baseline**: Replit Agent 3 = **100 points**

**Score Interpretation**:
- **< 50**: Much easier than Replit Agent 3 capabilities
- **50-80**: Easier than Replit Agent 3 capabilities
- **80-120**: Similar to Replit Agent 3 capabilities
- **120-150**: More challenging than Replit Agent 3 capabilities
- **> 150**: Significantly more challenging than Replit Agent 3 capabilities

**Technical Factors** (with weights):
1. **AI/ML** (weight: 38) - Machine learning, neural networks, LLMs, AI-powered systems
2. **Database** (weight: 33) - PostgreSQL, MySQL, MongoDB, SQLite, ORM, migrations
3. **Real-time** (weight: 30) - WebSockets, streaming, live updates, collaborative features
4. **Basic Web** (weight: 30) - HTML, CSS, landing pages, static sites
5. **API Integration** (weight: 29) - REST APIs, GraphQL, OAuth, third-party integrations
6. **Security** (weight: 28) - Encryption, JWT, authentication, authorization, password hashing
7. **Scalability** (weight: 26) - Load balancing, caching, Redis, Kafka, performance optimization
8. **Frontend** (weight: 24) - React, Vue, Angular, responsive design, modern frameworks
9. **Backend** (weight: 24) - Flask, Django, FastAPI, Node.js, Express, microservices
10. **Deployment** (weight: 15) - CI/CD, Docker, Kubernetes, AWS, Azure, cloud platforms
11. **Testing** (weight: 14) - Unit tests, integration tests, E2E testing, test coverage

**Task Size Multipliers**:
- **Simple** (0.9x) - Basic HTML/CSS, minimal features
- **Moderate** (1.0x) - Standard CRUD apps, authentication systems
- **Complex** (1.0x) - Full-stack apps with multiple integrations
- **Very Complex** (1.12x) - Comprehensive systems with advanced architecture
- **Expert** (1.25x) - Enterprise-grade, microservices, AI-powered systems

## Testing the MCP Server

### Using MCP Inspector (Recommended)
```bash
npx @modelcontextprotocol/inspector python mcp_server/server.py
```

This opens a browser interface at `http://localhost:5173` for interactive testing.

### Example Requirements to Test

**Simple Task (Expected: < 50)**:
```
Create a basic HTML page with a contact form
```

**Moderate Task (Expected: 80-120)**:
```
Build a REST API with user authentication using JWT tokens and PostgreSQL database
```

**Complex Task (Expected: 120-150)**:
```
Develop a full-stack real-time chat application with WebSocket support, React frontend, Node.js backend, MongoDB database, and user authentication
```

**Very Complex Task (Expected: > 150)**:
```
Create a scalable microservices architecture with AI-powered recommendations, real-time data streaming, OAuth integration, Kubernetes deployment, comprehensive security, load balancing, and machine learning model training pipeline
```

## Dependencies
- **Python 3.11**
- **mcp** (v1.20.0) - Official Model Context Protocol SDK
- **Django** (v5.2.7) - Web framework
- **FastMCP** - High-level MCP server API (included in `mcp` package)

## Technical Notes

### MCP Transport
The server uses **stdio** (standard input/output) transport, which is the standard for MCP servers. This allows the server to:
- Communicate with MCP clients via JSON-RPC
- Work with Claude Desktop and other MCP-compatible clients
- Run as a subprocess managed by the client

### Future Enhancements
- Add HTTP/SSE transport for remote connections
- Integrate with LLM for more sophisticated requirement analysis
- Create web dashboard for viewing scoring history
- Support batch processing of multiple requirements
- Add detailed breakdown of complexity sub-factors
- Provide recommendations for reducing complexity

## User Preferences
- Using official MCP Python SDK with FastMCP
- Calibrated scoring system with Replit Agent 3 as reference point
- Django framework for project structure
