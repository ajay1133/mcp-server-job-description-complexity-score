# Testing the MCP Complexity Scorer

This document explains how to test the MCP server and provides example requirements to evaluate.

## Test Setup

### Option 1: Using MCP Inspector (Interactive Testing)
The easiest way to test your MCP server is using the official MCP Inspector:

```bash
npx @modelcontextprotocol/inspector python mcp_server/server.py
```

This will:
1. Start the MCP server
2. Open a web interface at `http://localhost:5173`
3. Allow you to interactively call the `score_complexity` tool

### Option 2: Programmatic Testing with MCP Client
You can also create a Python client to test the server programmatically.

## Example Test Cases

### Test Case 1: Simple Web Page (Expected Score: < 50)
**Requirement**:
```
Create a simple HTML landing page with a header, footer, and contact form. Style it with basic CSS.
```

**Expected Output**:
- Complexity Score: 20-40
- Detected Factors: frontend
- Task Size: simple
- Difficulty: Much easier than Replit Agent 3

---

### Test Case 2: Basic CRUD API (Expected Score: 60-80)
**Requirement**:
```
Build a REST API for a todo list application with create, read, update, and delete endpoints. Use SQLite database.
```

**Expected Output**:
- Complexity Score: 60-80
- Detected Factors: backend, database, api_integration
- Task Size: moderate
- Difficulty: Easier than Replit Agent 3

---

### Test Case 3: Authentication System (Expected Score: 90-110)
**Requirement**:
```
Implement a user authentication system with JWT tokens, password hashing, user registration, login, and logout endpoints using PostgreSQL database.
```

**Expected Output**:
- Complexity Score: 90-110
- Detected Factors: security, backend, database, api_integration
- Task Size: moderate
- Difficulty: Similar to Replit Agent 3

---

### Test Case 4: Full-Stack App (Expected Score: 120-140)
**Requirement**:
```
Develop a full-stack e-commerce application with React frontend, Node.js backend, MongoDB database, Stripe payment integration, user authentication, and responsive design.
```

**Expected Output**:
- Complexity Score: 120-140
- Detected Factors: frontend, backend, database, api_integration, security
- Task Size: complex
- Difficulty: More challenging than Replit Agent 3

---

### Test Case 5: Real-Time Application (Expected Score: 140-160)
**Requirement**:
```
Create a real-time collaborative whiteboard application with WebSocket support, React frontend, user authentication, PostgreSQL database for persistence, and deployment on AWS with load balancing.
```

**Expected Output**:
- Complexity Score: 140-160
- Detected Factors: real_time, frontend, backend, database, security, deployment, scalability
- Task Size: complex
- Difficulty: More challenging than Replit Agent 3

---

### Test Case 6: AI-Powered System (Expected Score: > 180)
**Requirement**:
```
Build a scalable AI-powered recommendation engine with machine learning model training, real-time data streaming using Kafka, microservices architecture, Kubernetes deployment, comprehensive testing suite, OAuth integration, Redis caching, and monitoring dashboard.
```

**Expected Output**:
- Complexity Score: > 180
- Detected Factors: ai_ml, real_time, scalability, backend, deployment, security, testing
- Task Size: very_complex or expert
- Difficulty: Significantly more challenging than Replit Agent 3

---

## Using the Tool via MCP Inspector

1. Start the inspector:
   ```bash
   npx @modelcontextprotocol/inspector python mcp_server/server.py
   ```

2. In the web interface:
   - Navigate to "Tools"
   - Select `score_complexity`
   - Enter your requirement text in the `requirement` field
   - Click "Execute"
   - View the detailed JSON response

## Understanding the Results

The tool returns a JSON object with these fields:

- **complexity_score**: The numerical score (higher = more complex)
- **baseline_reference**: Always 100 (Replit Agent 3 baseline)
- **detected_factors**: Dictionary of identified technical factors with their contributions
- **task_size**: Estimated size category (simple, moderate, complex, very_complex, expert)
- **size_multiplier**: Multiplier applied based on task size
- **difficulty_rating**: Human-readable assessment
- **summary**: Brief text summary of the analysis

## Interpreting Scores

| Score Range | Meaning |
|-------------|---------|
| 0-50 | Much easier than Replit Agent 3 - basic tasks |
| 50-80 | Easier than Replit Agent 3 - simple applications |
| 80-120 | Similar to Replit Agent 3 - standard web apps |
| 120-150 | More challenging - complex integrations |
| 150+ | Significantly more challenging - enterprise systems |

## Tips for Testing

1. **Start Simple**: Test with basic requirements first to verify the tool works
2. **Add Complexity Gradually**: Incrementally add technical requirements to see how the score changes
3. **Use Keywords**: The algorithm looks for specific technical terms, so be explicit about requirements
4. **Check Factor Detection**: Review which factors were detected to understand the scoring
5. **Compare Similar Requirements**: Test variations of the same task to see scoring differences
