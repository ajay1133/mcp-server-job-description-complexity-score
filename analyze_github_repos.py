#!/usr/bin/env python3
"""Extract real LOC, technologies, and time estimates from GitHub repositories.

This analyzes actual codebases to get accurate training data instead of heuristics.

Features:
- Clone repo and count LOC using pygount/cloc
- Detect technologies from package files (package.json, requirements.txt, etc.)
- Estimate hours from commit history and contributor activity
- Extract complexity indicators from code structure
"""

import os
import json
import subprocess
import tempfile
import shutil
from typing import Dict, List, Any, Optional
from pathlib import Path
import re


class GitHubRepoAnalyzer:
    """Analyze GitHub repositories to extract real metrics."""
    
    TECH_FILE_PATTERNS = {
        'package.json': ['react', 'nextjs', 'vue', 'angular', 'svelte', 'node'],
        'requirements.txt': ['python_django', 'python_fastapi', 'flask'],
        'Gemfile': ['rails'],
        'go.mod': ['go'],
        'Cargo.toml': ['rust'],
        'pom.xml': ['java'],
        'build.gradle': ['java'],
        'composer.json': ['php'],
    }
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
    
    def analyze_repo(self, repo_url: str) -> Optional[Dict[str, Any]]:
        """
        Analyze a GitHub repository.
        
        Returns:
            {
                'text': repo description,
                'is_software': True,
                'technologies': [...],
                'loc': int,
                'hours': float,
                'complexity_score': int
            }
        """
        print(f"\nAnalyzing: {repo_url}")
        
        # Create temp directory
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Clone repo (shallow clone for speed)
                print("  Cloning repository...")
                result = subprocess.run(
                    ['git', 'clone', '--depth', '1', repo_url, temp_dir],
                    capture_output=True,
                    text=True,
                    timeout=60
                )
                
                if result.returncode != 0:
                    print(f"  ✗ Failed to clone: {result.stderr}")
                    return None
                
                # Get repo description from README
                description = self._extract_description(temp_dir)
                
                # Count LOC
                loc = self._count_loc(temp_dir)
                print(f"  LOC: {loc:,}")
                
                # Detect technologies
                technologies = self._detect_technologies(temp_dir)
                print(f"  Technologies: {', '.join(technologies)}")
                
                # Estimate hours from commit history
                hours = self._estimate_hours_from_commits(temp_dir)
                print(f"  Estimated hours: {hours:.1f}")
                
                # Calculate complexity
                complexity = self._calculate_complexity(loc, len(technologies), hours)
                
                return {
                    'text': description,
                    'is_software': True,
                    'technologies': technologies,
                    'loc': loc,
                    'hours': hours,
                    'complexity_score': complexity,
                    'source': f'github_{repo_url}'
                }
                
            except Exception as e:
                print(f"  ✗ Error analyzing repo: {e}")
                return None
    
    def _extract_description(self, repo_path: str) -> str:
        """Extract project description from README."""
        readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
        
        for readme in readme_files:
            readme_path = Path(repo_path) / readme
            if readme_path.exists():
                try:
                    content = readme_path.read_text(encoding='utf-8', errors='ignore')
                    # Get first paragraph (rough description)
                    lines = [l.strip() for l in content.split('\n') if l.strip()]
                    # Skip title and get description
                    for i, line in enumerate(lines):
                        if not line.startswith('#') and len(line) > 50:
                            return line[:500]  # First substantial line
                except Exception as e:
                    print(f"    Warning: Error reading README: {e}")
        
        return "Project repository"
    
    def _count_loc(self, repo_path: str) -> int:
        """
        Count lines of code using cloc or pygount.
        Excludes comments, blanks, and generated code.
        """
        try:
            # Try using cloc (fast, accurate)
            result = subprocess.run(
                ['cloc', repo_path, '--json'],
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return data.get('SUM', {}).get('code', 0)
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            pass
        
        # Fallback: simple file counting
        return self._count_loc_simple(repo_path)
    
    def _count_loc_simple(self, repo_path: str) -> int:
        """Simple LOC counter as fallback."""
        total_lines = 0
        
        code_extensions = {
            '.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.c', '.h',
            '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.cs', '.html', '.css',
            '.vue', '.svelte'
        }
        
        for root, dirs, files in os.walk(repo_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build'}]
            
            for file in files:
                if Path(file).suffix in code_extensions:
                    try:
                        file_path = Path(root) / file
                        lines = len(file_path.read_text(encoding='utf-8', errors='ignore').splitlines())
                        total_lines += lines
                    except Exception:
                        continue
        
        return total_lines
    
    def _detect_technologies(self, repo_path: str) -> List[str]:
        """Detect technologies from package files and code structure."""
        technologies = []
        
        # Check for package files
        for filename, techs in self.TECH_FILE_PATTERNS.items():
            file_path = Path(repo_path) / filename
            if file_path.exists():
                technologies.extend(self._parse_package_file(file_path, techs))
        
        # Check for database files or configs
        db_patterns = {
            'postgres': ['postgres', 'postgresql', 'pg'],
            'mysql': ['mysql'],
            'mongodb': ['mongo', 'mongodb'],
            'redis': ['redis']
        }
        
        for tech, patterns in db_patterns.items():
            if self._file_contains_patterns(repo_path, patterns):
                technologies.append(tech)
        
        # Check for auth implementations
        auth_patterns = ['jwt', 'oauth', 'passport', 'auth0', 'firebase/auth']
        if self._file_contains_patterns(repo_path, auth_patterns):
            technologies.append('auth')
        
        # Check for payment integrations
        payment_patterns = ['stripe', 'paypal', 'braintree']
        if self._file_contains_patterns(repo_path, payment_patterns):
            technologies.append('payments')
        
        # Check for cloud services
        if self._file_contains_patterns(repo_path, ['aws-sdk', 'boto3', '@aws']):
            technologies.append('aws')
        if self._file_contains_patterns(repo_path, ['azure', '@azure']):
            technologies.append('azure')
        if self._file_contains_patterns(repo_path, ['google-cloud', 'gcloud']):
            technologies.append('gcp')
        
        # Check for ML/AI
        ml_patterns = ['tensorflow', 'pytorch', 'scikit-learn', 'keras']
        if self._file_contains_patterns(repo_path, ml_patterns):
            technologies.append('ml')
        
        ai_patterns = ['openai', 'anthropic', 'langchain', 'gpt']
        if self._file_contains_patterns(repo_path, ai_patterns):
            technologies.append('ai_llm')
        
        return list(set(technologies))
    
    def _parse_package_file(self, file_path: Path, potential_techs: List[str]) -> List[str]:
        """Parse package file to detect specific technologies."""
        detected = []
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
            
            if file_path.name == 'package.json':
                # Parse JSON for specific packages
                data = json.loads(content)
                deps = {**data.get('dependencies', {}), **data.get('devDependencies', {})}
                
                if 'react' in deps or 'react-dom' in deps:
                    detected.append('react')
                if 'next' in deps:
                    detected.append('nextjs')
                if 'vue' in deps:
                    detected.append('vue')
                if '@angular/core' in deps:
                    detected.append('angular')
                if 'svelte' in deps:
                    detected.append('svelte')
                if 'express' in deps:
                    detected.append('node')
            
            elif file_path.name == 'requirements.txt':
                if 'django' in content:
                    detected.append('python_django')
                if 'fastapi' in content:
                    detected.append('python_fastapi')
                if 'flask' in content:
                    detected.append('flask')
            
            elif file_path.name == 'Gemfile':
                if 'rails' in content:
                    detected.append('rails')
        
        except Exception as e:
            print(f"    Warning: Error parsing {file_path.name}: {e}")
        
        return detected
    
    def _file_contains_patterns(self, repo_path: str, patterns: List[str]) -> bool:
        """Check if any file in repo contains any of the patterns."""
        for root, dirs, files in os.walk(repo_path):
            # Skip large directories
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', 'venv', '__pycache__'}]
            
            for file in files[:100]:  # Limit files to check
                if file.endswith(('.json', '.txt', '.py', '.js', '.ts', '.rb', '.go')):
                    try:
                        file_path = Path(root) / file
                        content = file_path.read_text(encoding='utf-8', errors='ignore').lower()
                        
                        for pattern in patterns:
                            if pattern.lower() in content:
                                return True
                    except Exception:
                        continue
        
        return False
    
    def _estimate_hours_from_commits(self, repo_path: str) -> float:
        """
        Estimate development hours from git commit history.
        Simple heuristic: 1 commit ≈ 1-2 hours on average.
        """
        try:
            result = subprocess.run(
                ['git', 'rev-list', '--count', 'HEAD'],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                commit_count = int(result.stdout.strip())
                # Rough estimate: 1.5 hours per commit
                estimated_hours = commit_count * 1.5
                return max(5.0, min(estimated_hours, 5000.0))
        except Exception as e:
            print(f"    Warning: Error counting commits: {e}")
        
        # Fallback based on LOC
        return 50.0
    
    def _calculate_complexity(self, loc: int, tech_count: int, hours: float) -> int:
        """Calculate complexity score from metrics."""
        import math
        
        # Log-based formula
        complexity = (
            10 * math.log10(max(loc, 1)) +
            8 * math.log10(max(hours, 1)) +
            6 * math.sqrt(tech_count + 1)
        )
        
        return max(10, min(int(complexity), 200))


def analyze_popular_repos():
    """Analyze popular GitHub repos to build training data."""
    
    # Popular repos across different categories
    repos = [
        # Frontend
        "https://github.com/facebook/react",
        "https://github.com/vuejs/vue",
        "https://github.com/angular/angular",
        
        # Backend
        "https://github.com/django/django",
        "https://github.com/tiangolo/fastapi",
        "https://github.com/rails/rails",
        
        # Full-stack
        "https://github.com/vercel/next.js",
        "https://github.com/nuxt/nuxt",
        
        # Add more diverse repos...
    ]
    
    analyzer = GitHubRepoAnalyzer()
    training_data = []
    
    print("=" * 70)
    print("GitHub Repository Analyzer")
    print("=" * 70)
    print(f"Analyzing {len(repos)} repositories...\n")
    
    for repo_url in repos:
        result = analyzer.analyze_repo(repo_url)
        if result:
            training_data.append(result)
            print("  ✓ Success\n")
        else:
            print("  ✗ Failed\n")
    
    # Save results
    output_file = "data/github_analyzed_training_data.jsonl"
    os.makedirs("data", exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in training_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("=" * 70)
    print(f"Analyzed {len(training_data)} repositories successfully")
    print(f"Saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    analyze_popular_repos()
