#!/usr/bin/env python3
"""Entry point for the MCP-based Technology Extractor server.

Provides an MCP tool `extract_technologies(requirement: str)` which returns
detected technologies with difficulty ratings and alternatives.
Use `--self-test` to quickly test the extraction logic locally.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from mcp_server.simple_tech_extractor import SimpleTechExtractor  # noqa: E402
from mcp_server.logging_utils import RequestLogger  # noqa: E402
from mcp_server.resume_parser import parse_resume_file  # noqa: E402

extractor = SimpleTechExtractor()


def extract_technologies_(
    requirement: str = "",
    resume: str = "",
    resume_file: str = "",
) -> dict:
    """Extract required technologies and emit a detailed per-request log file.

    Creates a JSON log under `logs/responses_traces/<timestamp>.json` capturing:
      - Each function call trace (here: extractor.extract_technologies)
      - Args sample, timing, CPU/memory deltas
      - Request start/end timestamps & total duration
      - Final response object

    Args:
        requirement: Job description or additional prompt context
        resume: Resume text content directly
        resume_file: Path to resume file (.txt, .docx, .pdf, .doc)
    """
    logger = RequestLogger()

    # Parse resume file if provided
    resume_text = resume
    if resume_file:
        try:
            resume_text = logger.trace_call(parse_resume_file, resume_file)
        except Exception as e:
            result = {"error": f"Failed to parse resume file: {e}"}
            logger.finalize(result)
            return result

    # Determine if we have resume content
    has_resume = bool(resume_text)

    if has_resume:
        # Extract from resume with optional prompt override
        result = logger.trace_call(
            extractor.extract_technologies, resume_text, is_resume=True, prompt_override=requirement
        )
    elif requirement:
        # Extract from requirement only
        result = logger.trace_call(extractor.extract_technologies, requirement, is_resume=False, prompt_override="")
    else:
        result = {"error": "Either requirement, resume, or resume_file must be provided"}

    logger.finalize(result)
    return result


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Technology Extractor MCP Server")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a quick self test and exit",
    )
    parser.add_argument(
        "--example",
        type=str,
        default=("Senior Full-Stack Engineer with React, Node.js, and PostgreSQL experience"),
        help="Example text for --self-test",
    )
    args = parser.parse_args(argv)

    if args.self_test:
        result = extractor.extract_technologies(args.example)
        print(json.dumps(result, indent=2))
        return 0

    mcp = FastMCP("tech-extractor")
    mcp.tool()(extract_technologies_)
    print("[tech-extractor] Starting MCP server... (Ctrl+C to stop)")
    print("[tech-extractor] Tool registered: extract_technologies(requirement: str, resume, resume_file) -> dict")
    mcp.run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
