# syntax=docker/dockerfile:1.7

# Base image
ARG PYTHON_VERSION=3.11
FROM python:${PYTHON_VERSION}-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Minimal system deps (OpenMP runtime for scikit-learn)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency manifest first to leverage Docker cache
COPY pyproject.toml ./

# Install project dependencies
RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install .

# Copy application source
COPY mcp_server ./mcp_server
COPY training_data ./training_data
COPY train_system_design_models.py ./

# Logs directory
RUN mkdir -p logs/complexity_scorer_logs
ENV SOFTWARE_LOG_DIR=/app/logs/complexity_scorer_logs

# Runtime configuration
ENV HOST=0.0.0.0 \
    PORT=8000 \
    FLASK_MODE=0

EXPOSE 8000

# Switch between Flask API and MCP server via FLASK_MODE
CMD ["/bin/sh", "-c", "if [ \"$FLASK_MODE\" = \"1\" ]; then python -m mcp_server.flask_app; else python -m mcp_server.server; fi"]
