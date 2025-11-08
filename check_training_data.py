#!/usr/bin/env python3
"""Check training data quality."""

import json

with open('data/software_training_data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

software_entries = [d for d in data if d.get('is_software')]
with_tech = [d for d in software_entries if d.get('technologies')]

print(f"Total entries: {len(data)}")
print(f"Software entries: {len(software_entries)}")
print(f"Entries with technologies: {len(with_tech)}")

if with_tech:
    print(f"\nSample entry:")
    sample = with_tech[0]
    print(f"  Text: {sample['text'][:80]}...")
    print(f"  Technologies: {sample['technologies']}")
    print(f"  LOC: {sample['loc']}")
    
    # Count technology occurrences
    all_techs = {}
    for entry in with_tech:
        for tech in entry.get('technologies', []):
            all_techs[tech] = all_techs.get(tech, 0) + 1
    
    print(f"\nTechnology distribution:")
    for tech, count in sorted(all_techs.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {tech}: {count}")
