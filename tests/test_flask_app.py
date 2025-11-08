import json
import os
import types

import pytest

# Ensure Flask is importable and skip tests if not installed
pytestmark = pytest.mark.skipif(
    not bool(__import__("importlib").util.find_spec("flask")),
    reason="Flask not installed in test environment",
)

from mcp_server.flask_app import app  # noqa: E402


def test_health():
    with app.test_client() as c:
        r = c.get("/health")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"


def test_score_minimal():
    with app.test_client() as c:
        payload = {"requirement": "Build a Flask REST API with CRUD for todos using SQLite"}
        r = c.post("/score", data=json.dumps(payload), content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        # Basic shape checks without over-constraining
        assert isinstance(data, dict)
        assert "complexity_score" in data or "complexity_score_v2" in data
        assert "technologies" in data or "without_ai_and_ml" in data or "with_ai_and_ml" in data
