#!/usr/bin/env python3
"""
Scrape 100 public GitHub repos to calculate actual human vs AI coding speed ratio.

For each repo, calculate:
- total_lines: total lines of code
- num_contributors: number of contributors
- human_time_taken: seconds between repo creation and last commit (HALVED to account for breaks)
- human_lines_per_sec: total_lines / (num_contributors * human_time_taken)
- ai_time_taken: estimated time for AI to code the entire repo
  - Includes base AI generation time
  - PLUS prompt overhead: extra_time = total_lines / (100 * human_lines_per_sec)
- ai_lines_per_sec: total_lines / ai_time_taken (includes prompt overhead)
- time_ratio: human_lines_per_sec / ai_lines_per_sec

Adjustments:
1. Human time halved to account for breaks and non-coding activities
2. AI time includes overhead for user giving prompts (proportional to project size and human speed)

Output: human_ai_code_ratio.log with all data + average_time_ratio
"""

import os
import sys
import time
import json
import requests
from datetime import datetime
from typing import Dict, List, Optional
import re

# Add parent directory to path for SoftwareComplexityScorer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server.software_complexity_scorer import SoftwareComplexityScorer

# GitHub API token (optional but recommended to avoid rate limits)
GITHUB_TOKEN = os.getenv('GITHUB_TOKEN', '')

def get_github_headers() -> Dict[str, str]:
    """Get headers for GitHub API requests."""
    headers = {
        'Accept': 'application/vnd.github.v3+json',
    }
    if GITHUB_TOKEN:
        headers['Authorization'] = f'token {GITHUB_TOKEN}'
    return headers

def search_popular_repos(min_size_kb: int = 1000, count: int = 20) -> List[str]:
    """
    Search for popular repos with >1MB size (roughly >10K lines).
    Returns list of repo full names (owner/repo).
    """
    print(f"Searching for {count} popular repos with >{min_size_kb}KB size...")
    
    # Search for repos with high stars across different languages
    languages = ['python', 'javascript', 'typescript', 'java', 'go', 'rust', 'cpp', 'ruby', 'php', 'csharp']
    repos = []
    
    for lang in languages:
        if len(repos) >= count:
            break
            
        url = 'https://api.github.com/search/repositories'
        params = {
            'q': f'language:{lang} stars:>500 size:>{min_size_kb}',
            'sort': 'stars',
            'order': 'desc',
            'per_page': min(30, count - len(repos))
        }
        
        try:
            response = requests.get(url, headers=get_github_headers(), params=params, timeout=10)
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', []):
                    if item['full_name'] not in repos:  # Avoid duplicates
                        repos.append(item['full_name'])
                        if len(repos) >= count:
                            break
                print(f"  Found {len(repos)} repos so far (language: {lang})...")
                time.sleep(1)  # Rate limiting
            elif response.status_code == 403:
                print(f"  Rate limited. Waiting 60s...")
                time.sleep(60)
            else:
                print(f"  Error searching {lang}: {response.status_code}")
        except Exception as e:
            print(f"  Error searching {lang}: {e}")
    
    return repos[:count]

def count_lines_in_repo(repo_full_name: str) -> int:
    """
    Count total lines of code in a repo using GitHub API.
    Uses languages endpoint which gives byte counts, then estimates lines.
    """
    url = f'https://api.github.com/repos/{repo_full_name}/languages'
    
    try:
        response = requests.get(url, headers=get_github_headers(), timeout=10)
        if response.status_code == 200:
            languages = response.json()
            total_bytes = sum(languages.values())
            # Estimate: average 40 bytes per line (typical for most languages)
            estimated_lines = total_bytes // 40
            return estimated_lines
        return 0
    except Exception as e:
        print(f"    Error counting lines: {e}")
        return 0

def get_repo_description(repo_full_name: str, repo_data: Dict) -> str:
    """Get a brief description of the repo for AI time estimation."""
    description = repo_data.get('description', '') or ''
    language = repo_data.get('language', '') or ''
    
    # Build a simple description
    if description:
        return f"{language} project: {description}"
    else:
        return f"{language} project"

def estimate_ai_time_for_repo(repo_full_name: str, repo_data: Dict, total_lines: int) -> float:
    """
    Estimate AI time using our trained model.
    
    Returns: estimated seconds for AI to complete this project
    """
    try:
        # Get repo description
        description = get_repo_description(repo_full_name, repo_data)
        
        # Use our scorer to estimate
        scorer = SoftwareComplexityScorer()
        result = scorer.analyze_text(description)
        
        if result.get('is_software', False):
            # Get AI hours estimate
            ai_hours = result.get('ai_hours', 0)
            ai_seconds = ai_hours * 3600
            
            # If estimate seems too low compared to LOC, scale it up
            # Assume AI can generate ~100 lines per hour with human prompting
            min_ai_seconds = (total_lines / 100) * 3600
            ai_seconds = max(ai_seconds, min_ai_seconds)
            
            return ai_seconds
        else:
            # Fallback: assume AI generates 100 lines/hour
            return (total_lines / 100) * 3600
            
    except Exception as e:
        print(f"    Warning: Error estimating AI time: {e}")
        # Fallback calculation
        return (total_lines / 100) * 3600

def get_repo_stats(repo_full_name: str) -> Optional[Dict]:
    """
    Get repository statistics from GitHub API.
    
    Returns dict with:
    - repo_name
    - repo_creation_time
    - last_commit_time
    - total_lines
    - num_contributors
    - human_time_taken_seconds
    - ai_time_taken_seconds
    - human_lines_per_sec
    - ai_lines_per_sec
    - time_ratio
    """
    print(f"  Analyzing: {repo_full_name}")
    
    # Get basic repo info
    url = f'https://api.github.com/repos/{repo_full_name}'
    try:
        response = requests.get(url, headers=get_github_headers(), timeout=10)
        
        if response.status_code == 403:
            print(f"    Rate limited. Waiting 60s...")
            time.sleep(60)
            response = requests.get(url, headers=get_github_headers(), timeout=10)
        
        if response.status_code != 200:
            print(f"    Error: HTTP {response.status_code}")
            return None
        
        repo_data = response.json()
        
        # Get creation time
        created_at = datetime.strptime(repo_data['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        
        # Get last commit time from default branch
        default_branch = repo_data.get('default_branch', 'main')
        commits_url = f'https://api.github.com/repos/{repo_full_name}/commits/{default_branch}'
        commits_response = requests.get(commits_url, headers=get_github_headers(), timeout=10)
        
        if commits_response.status_code != 200:
            print(f"    Error getting commits: HTTP {commits_response.status_code}")
            return None
        
        commit_data = commits_response.json()
        last_commit_date = datetime.strptime(
            commit_data['commit']['committer']['date'], 
            '%Y-%m-%dT%H:%M:%SZ'
        )
        
        # Get contributors count
        # Use contributors endpoint with pagination
        contributors_url = f'https://api.github.com/repos/{repo_full_name}/contributors'
        contributors_response = requests.get(
            contributors_url, 
            headers=get_github_headers(), 
            params={'per_page': 1, 'anon': 'true'}, 
            timeout=10
        )
        
        if contributors_response.status_code != 200:
            print(f"    Error getting contributors: HTTP {contributors_response.status_code}")
            # Use fallback from repo data
            num_contributors = max(1, repo_data.get('contributors_count', 1))
        else:
            # Parse total count from Link header
            link_header = contributors_response.headers.get('Link', '')
            num_contributors = 1
            
            if 'last' in link_header:
                # Parse last page number from link header
                match = re.search(r'page=(\d+)>; rel="last"', link_header)
                if match:
                    num_contributors = int(match.group(1))
        
        # Count lines
        total_lines = count_lines_in_repo(repo_full_name)
        
        if total_lines < 10000:
            print(f"    Skipping: only {total_lines:,} lines (need >10K)")
            return None
        
        # Calculate human time
        human_time_seconds_raw = (last_commit_date - created_at).total_seconds()
        
        if human_time_seconds_raw <= 0:
            print(f"    Skipping: invalid time range")
            return None
        
        # ADJUSTMENT 1: Halve human time to account for breaks/non-coding time
        human_time_seconds = human_time_seconds_raw / 2.0
        
        # Calculate initial human coding speed
        # Human lines per second = total lines / (contributors * time)
        human_lines_per_second = total_lines / (num_contributors * human_time_seconds)
        
        # Estimate base AI time using our model
        ai_time_seconds_base = estimate_ai_time_for_repo(repo_full_name, repo_data, total_lines)
        
        # ADJUSTMENT 2: Add time for user to give prompts to AI
        # extra_user_prompt_to_ai_time = no_of_lines / (100 * human_lines_per_sec)
        extra_user_prompt_to_ai_time = total_lines / (100 * human_lines_per_second)
        
        # Add prompt overhead to AI time
        ai_time_seconds = ai_time_seconds_base + extra_user_prompt_to_ai_time
        
        # Recalculate AI lines per second with prompt overhead included
        ai_lines_per_second = total_lines / ai_time_seconds if ai_time_seconds > 0 else 0
        
        # Time ratio = human speed / AI speed
        # If ratio = 0.001, it means human is 0.1% as fast as AI (AI is 1000x faster)
        time_ratio = human_lines_per_second / ai_lines_per_second if ai_lines_per_second > 0 else 0
        
        print(f"    ✓ {total_lines:,} lines, {num_contributors} contributors")
        print(f"    Human: {human_time_seconds/86400:.1f} days (adjusted), {human_lines_per_second:.6f} lines/sec")
        print(f"    AI base: {ai_time_seconds_base/3600:.1f} hours, prompt overhead: {extra_user_prompt_to_ai_time/3600:.2f} hours")
        print(f"    AI total: {ai_time_seconds/3600:.1f} hours, {ai_lines_per_second:.6f} lines/sec")
        print(f"    Ratio: {time_ratio:.6f} (human is {time_ratio*100:.4f}% of AI speed)")
        
        return {
            'repo_name': repo_full_name,
            'repo_creation_time': created_at.isoformat(),
            'last_commit_time': last_commit_date.isoformat(),
            'total_lines': total_lines,
            'num_contributors': num_contributors,
            'human_time_taken_seconds': human_time_seconds,
            'human_time_taken_days': human_time_seconds / 86400,
            'human_time_raw_seconds': human_time_seconds_raw,
            'human_time_raw_days': human_time_seconds_raw / 86400,
            'ai_time_base_seconds': ai_time_seconds_base,
            'ai_time_base_hours': ai_time_seconds_base / 3600,
            'prompt_overhead_seconds': extra_user_prompt_to_ai_time,
            'prompt_overhead_hours': extra_user_prompt_to_ai_time / 3600,
            'ai_time_taken_seconds': ai_time_seconds,
            'ai_time_taken_hours': ai_time_seconds / 3600,
            'human_lines_per_sec': human_lines_per_second,
            'ai_lines_per_sec': ai_lines_per_second,
            'time_ratio': time_ratio
        }
        
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function to scrape repos and calculate ratios."""
    print("=" * 100)
    print("GitHub Repo Analysis: Human vs AI Coding Speed")
    print("=" * 100)
    print()
    
    if not GITHUB_TOKEN:
        print("⚠️  WARNING: No GITHUB_TOKEN environment variable set.")
        print("   You may hit rate limits. Set GITHUB_TOKEN for better results.")
        print("   Get a token at: https://github.com/settings/tokens")
        print()
        
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting. Please set GITHUB_TOKEN and try again.")
            return
        print()
    
    # Search for repos
    print("Step 1: Searching for repositories...")
    repo_names = search_popular_repos(min_size_kb=1000, count=20)
    print(f"\n✓ Found {len(repo_names)} repos to analyze\n")
    
    # Analyze each repo
    print("Step 2: Analyzing repositories...")
    print()
    results = []
    
    for i, repo_name in enumerate(repo_names, 1):
        print(f"[{i}/{len(repo_names)}]", end=" ")
        
        stats = get_repo_stats(repo_name)
        if stats:
            results.append(stats)
        
        # Rate limiting
        time.sleep(1)
        
        # Save intermediate results every 10 repos
        if i % 10 == 0:
            print(f"\n  💾 Saving intermediate results ({len(results)} valid repos so far)...\n")
            with open('human_ai_code_ratio_temp.json', 'w') as f:
                json.dump(results, f, indent=2)
        
        print()
    
    print()
    print("=" * 100)
    print(f"Analysis Complete: {len(results)} valid repos analyzed")
    print("=" * 100)
    print()
    
    if not results:
        print("❌ No valid results to analyze")
        return
    
    # Calculate average ratio
    average_time_ratio = sum(r['time_ratio'] for r in results) / len(results)
    
    # Calculate statistics
    time_ratios = [r['time_ratio'] for r in results]
    time_ratios_sorted = sorted(time_ratios)
    median_ratio = time_ratios_sorted[len(time_ratios_sorted)//2]
    min_ratio = min(time_ratios)
    max_ratio = max(time_ratios)
    
    # Write detailed log
    print("Step 3: Writing results to human_ai_code_ratio.log...")
    
    with open('human_ai_code_ratio.log', 'w', encoding='utf-8') as f:
        f.write("=" * 140 + "\n")
        f.write("GitHub Repository Analysis: Human vs AI Coding Speed (20 repos)\n")
        f.write("=" * 140 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().isoformat()}\n")
        f.write(f"Total repos analyzed: {len(results)}\n\n")
        
        f.write("KEY FINDINGS:\n")
        f.write(f"  Average time ratio (human/AI): {average_time_ratio:.6f}\n")
        f.write(f"  This means: Human coding speed is {average_time_ratio*100:.4f}% of AI speed\n")
        f.write(f"  Or equivalently: AI is {1/average_time_ratio:.2f}x faster than human\n\n")
        
        f.write("STATISTICS:\n")
        f.write(f"  Median ratio: {median_ratio:.6f}\n")
        f.write(f"  Min ratio: {min_ratio:.6f} (fastest human relative to AI)\n")
        f.write(f"  Max ratio: {max_ratio:.6f} (slowest human relative to AI)\n\n")
        
        f.write("-" * 140 + "\n")
        f.write(f"{'Repo Name':<45} {'Created':<12} {'Last Commit':<12} {'Lines':>10} {'Contrib':>7} "
                f"{'Human Days':>12} {'AI Base (h)':>12} {'Prompt (h)':>11} {'AI Total (h)':>12} {'Ratio':>10}\n")
        f.write("-" * 140 + "\n")
        
        for r in results:
            f.write(
                f"{r['repo_name']:<45} "
                f"{r['repo_creation_time'][:10]:<12} "
                f"{r['last_commit_time'][:10]:<12} "
                f"{r['total_lines']:>10,} "
                f"{r['num_contributors']:>7} "
                f"{r['human_time_taken_days']:>12.1f} "
                f"{r['ai_time_base_hours']:>12.1f} "
                f"{r['prompt_overhead_hours']:>11.2f} "
                f"{r['ai_time_taken_hours']:>12.1f} "
                f"{r['time_ratio']:>10.6f}\n"
            )
        
        f.write("-" * 140 + "\n\n")
        
        f.write("=" * 140 + "\n")
        f.write("FINAL RESULTS\n")
        f.write("=" * 140 + "\n\n")
        
        f.write(f"AVERAGE TIME RATIO: {average_time_ratio:.8f}\n\n")
        
        f.write("Interpretation:\n")
        f.write(f"  - Human coding speed is {average_time_ratio*100:.6f}% of AI coding speed\n")
        f.write(f"  - AI is {1/average_time_ratio:.2f}x faster than human coding\n")
        f.write(f"  - For every 1 line AI writes, human writes {average_time_ratio:.6f} lines in the same time\n")
        f.write(f"  - If human takes 1 day, AI takes {average_time_ratio * 24:.4f} hours\n\n")
        
        f.write("Recommendation for model:\n")
        f.write(f"  Use formula: manual_time = ai_time × {1/average_time_ratio:.2f}\n")
        f.write(f"  Or equivalently: ai_time = manual_time × {average_time_ratio:.8f}\n\n")
        
        f.write("ADJUSTMENTS APPLIED:\n")
        f.write("  1. Human time halved to account for breaks/non-coding time\n")
        f.write("  2. AI time includes prompt overhead: extra_time = total_lines / (100 × human_lines_per_sec)\n")
        f.write("     This accounts for time user spends giving prompts to AI\n")
    
    # Also save as JSON for further analysis
    with open('human_ai_code_ratio.json', 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_repos': len(results),
                'average_time_ratio': average_time_ratio,
                'median_time_ratio': median_ratio,
                'min_time_ratio': min_ratio,
                'max_time_ratio': max_ratio,
                'ai_speed_multiplier': 1 / average_time_ratio,
                'human_as_percent_of_ai': average_time_ratio * 100,
                'recommended_formula': f'manual_time = ai_time × {1/average_time_ratio:.2f}'
            },
            'repos': results
        }, f, indent=2)
    
    print(f"✓ Results saved to human_ai_code_ratio.log")
    print(f"✓ JSON data saved to human_ai_code_ratio.json")
    print()
    
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    print(f"Repos analyzed: {len(results)}")
    print(f"Average time ratio (human/AI): {average_time_ratio:.8f}")
    print(f"Human coding speed is {average_time_ratio * 100:.6f}% of AI speed")
    print(f"AI is {1/average_time_ratio:.2f}x faster than human")
    print()
    print("Recommended formula for your model:")
    print(f"  manual_time = ai_time × {1/average_time_ratio:.2f}")
    print("=" * 100)

if __name__ == '__main__':
    main()
