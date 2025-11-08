#!/usr/bin/env python3
"""Interactive training data curator for SoftwareComplexityScorer.

This script helps you build a training dataset by:
1. Accepting prompts interactively
2. Asking you to label: software vs non-software
3. For software: collecting tech tags, LOC, hours estimates
4. Appending to a JSONL file

Run: python curate_training_data.py
"""

import json
import os
from typing import Dict, Any, List


def suggest_technologies(text: str) -> List[str]:
    """Suggest technology tags based on prompt keywords."""
    text_lower = text.lower()
    suggestions = []
    
    tech_map = {
        'react': ['react'],
        'next': ['nextjs'],
        'vue': ['vue'],
        'angular': ['angular'],
        'node': ['node'],
        'express': ['node'],
        'fastapi': ['python_fastapi'],
        'django': ['python_django'],
        'flask': ['flask'],
        'rails': ['rails'],
        'postgres': ['postgres'],
        'mysql': ['mysql'],
        'mongodb': ['mongodb'],
        'redis': ['redis'],
        'auth': ['auth'],
        'login': ['auth'],
        'payment': ['payments'],
        'stripe': ['payments'],
        'docker': ['devops'],
        'kubernetes': ['devops'],
        'ci/cd': ['devops'],
        'openai': ['ai_llm'],
        'llm': ['ai_llm'],
        'machine learning': ['ml'],
        'websocket': ['websocket'],
        'real-time': ['realtime'],
    }
    
    for keyword, tags in tech_map.items():
        if keyword in text_lower:
            suggestions.extend(tags)
    
    return list(set(suggestions))


def curate_entry() -> Dict[str, Any] | None:
    """Collect one training example interactively."""
    print("\n" + "=" * 70)
    text = input("Enter requirement prompt (or 'quit' to exit):\n> ").strip()
    
    if text.lower() in ['quit', 'exit', 'q']:
        return None
    
    if not text:
        print("Empty prompt, skipping.")
        return curate_entry()
    
    is_software = input("Is this a software requirement? (y/n): ").strip().lower()
    
    if is_software not in ['y', 'n']:
        print("Invalid input. Use 'y' or 'n'.")
        return curate_entry()
    
    entry: Dict[str, Any] = {
        "text": text,
        "is_software": is_software == 'y'
    }
    
    if entry["is_software"]:
        # Suggest technologies
        suggested = suggest_technologies(text)
        if suggested:
            print(f"\nSuggested technologies: {', '.join(suggested)}")
        
        tech_input = input("Enter technology tags (comma-separated, or press Enter to skip):\n> ").strip()
        if tech_input:
            technologies = [t.strip() for t in tech_input.split(',') if t.strip()]
            entry["technologies"] = technologies
        elif suggested:
            entry["technologies"] = suggested
        else:
            entry["technologies"] = []
        
        # LOC estimate
        loc_input = input("Estimated lines of code (or press Enter to skip): ").strip()
        if loc_input.isdigit():
            entry["loc"] = int(loc_input)
        
        # Hours estimate
        hours_input = input("Estimated manual coding hours (or press Enter to skip): ").strip()
        if hours_input:
            try:
                entry["hours"] = float(hours_input)
            except ValueError:
                pass
        
        # Optional complexity score
        score_input = input("Optional complexity score 10-200 (or press Enter to skip): ").strip()
        if score_input.isdigit():
            entry["complexity_score"] = int(score_input)
    
    return entry


def main():
    output_file = "data/software_training_data.jsonl"
    os.makedirs("data", exist_ok=True)
    
    print("=" * 70)
    print("Training Data Curator for SoftwareComplexityScorer")
    print("=" * 70)
    print(f"\nOutput file: {output_file}")
    print("\nTips:")
    print("- Aim for 1000+ software examples + 500+ non-software examples")
    print("- Balance across tech stacks (web, mobile, ML, DevOps, etc.)")
    print("- Include varying complexity levels")
    print("- You can edit the JSONL file directly later if needed")
    print("\nPress Ctrl+C to quit anytime")
    
    count = 0
    
    try:
        while True:
            entry = curate_entry()
            if entry is None:
                break
            
            # Append to file
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            count += 1
            print(f"\n✓ Saved entry #{count}")
            
            if count % 10 == 0:
                print(f"\n📊 Progress: {count} entries collected so far")
                cont = input("Continue? (y/n): ").strip().lower()
                if cont != 'y':
                    break
    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    
    print(f"\n{'=' * 70}")
    print(f"Collected {count} entries total")
    print(f"Saved to: {output_file}")
    print(f"\nNext steps:")
    print(f"1. Review and edit {output_file} if needed")
    print(f"2. Train models: python train_software_models.py --data {output_file} --out models/software")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
