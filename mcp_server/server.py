#!/usr/bin/env python3
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp_server.complexity_scorer import ComplexityScorer

mcp = FastMCP("complexity-scorer")

scorer = ComplexityScorer()

@mcp.tool()
def score_complexity(requirement: str) -> dict:
    """
    Analyzes programming requirements or job descriptions and provides a complexity score.
    
    The scoring is calibrated with Replit Agent 3's capabilities as a baseline (score of 100).
    Scores below 100 indicate tasks easier than what Replit Agent 3 typically handles.
    Scores above 100 indicate more challenging tasks.
    
    Args:
        requirement: A text description of the programming requirement or job description
    
    Returns:
        A detailed analysis including:
        - complexity_score: Numerical score relative to Replit Agent 3 (baseline 100)
        - detected_factors: Technical complexity factors identified
        - task_size: Estimated task size (simple, moderate, complex, etc.)
        - difficulty_rating: Human-readable difficulty assessment
        - best_completion_time: Estimated time to complete the task assuming candidate is good 
        in using AI agents like Replit for coding and making apps
        - summary: Brief summary of the analysis
    """
    result = scorer.analyze_text(requirement)
    return result

if __name__ == "__main__":
    mcp.run()
