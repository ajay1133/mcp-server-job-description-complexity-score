#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer


def iter_jsonl(path: str):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def main():
    ap = argparse.ArgumentParser(description="Recalculate complexity scores for a dataset of prompts.")
    ap.add_argument("--input", required=True, help="Input JSONL file with at least a 'text' field per line")
    ap.add_argument("--out", required=True, help="Output JSONL file with appended scores")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit of samples to process")
    args = ap.parse_args()

    scorer = SoftwareComplexityScorer()

    count = 0
    processed = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as out:
        for ex in iter_jsonl(args.input):
            count += 1
            if args.limit and processed >= args.limit:
                break
            text = ex.get('text') or ex.get('requirement')
            if not text:
                continue
            result = scorer.analyze_text(text)
            # Attach key scores to original example for traceability
            payload: Dict[str, Any] = {
                **ex,
                "_complexity": {
                    "complexity_score": result.get("complexity_score"),
                    "size_score_linux_ref": result.get("size_score_linux_ref"),
                    "predicted_lines_of_code": result.get("predicted_lines_of_code"),
                    "complexity_v2": result.get("complexity_v2"),
                    "technologies": result.get("technologies"),
                    "microservices": result.get("microservices"),
                }
            }
            out.write(json.dumps(payload, ensure_ascii=False) + "\n")
            processed += 1
    print(f"Processed {processed} / {count} examples. Output -> {args.out}")


if __name__ == "__main__":
    main()
