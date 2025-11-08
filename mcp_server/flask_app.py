from __future__ import annotations

import os
from typing import Any, Dict

from flask import Flask, jsonify, request

from .software_complexity_scorer import SoftwareComplexityScorer

app = Flask(__name__)
_scorer = SoftwareComplexityScorer()


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.post("/score")
def score() -> Any:
    try:
        data: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
        requirement = data.get("requirement") or data.get("text") or ""
        if not isinstance(requirement, str) or not requirement.strip():
            return jsonify({"error": "Missing 'requirement' in JSON body"}), 400
        result = _scorer.analyze_text(requirement)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
