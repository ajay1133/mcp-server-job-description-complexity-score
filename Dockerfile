FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Copy application code
COPY mcp_server/ ./mcp_server/
COPY config/ ./config/
COPY models/ ./models/
COPY data/ ./data/

# Create logs directory
RUN mkdir -p logs/complexity_scorer_logs

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SOFTWARE_LOG_DIR=/app/logs/complexity_scorer_logs

# Expose port (if needed for future HTTP interface)
EXPOSE 8000

# Default command can be overridden at runtime.
# Use FLASK_MODE=1 to start Flask API server instead of MCP server.
ENV FLASK_MODE=0

CMD ["/bin/sh", "-c", "if [ \"$FLASK_MODE\" = \"1\" ]; then python -m mcp_server.flask_app; else python mcp_server/server.py; fi"]
