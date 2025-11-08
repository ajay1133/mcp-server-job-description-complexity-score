#!/usr/bin/env python3
"""Merge multiple JSONL training datasets into one, without keyword heuristics.

- De-duplicates by normalized text (casefolded, stripped)
- Prefers labeled examples over unlabeled duplicates
- Preserves numeric fields when present (loc, hours, complexity_score)
- Writes a clean merged file ready for training

Usage:
  python merge_training_data.py \
    --inputs data/software_training_data.jsonl data/bootstrapped_training_data.jsonl \
    --out data/merged_training_data.jsonl

If --inputs is omitted, will merge all .jsonl files under ./data.
"""

import os
import json
import argparse
from typing import List, Dict, Any


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            text = obj.get('text')
            if not isinstance(text, str) or not text.strip():
                continue
            obj['text'] = text.strip()
            items.append(obj)
    return items


essential_order = (
    'text', 'is_software', 'technologies', 'loc', 'hours', 'complexity_score'
)


def merge_records(existing: Dict[str, Any], incoming: Dict[str, Any]) -> Dict[str, Any]:
    """Merge two records with the same text.

    Priority rules:
      - If one has is_software and the other doesn't, keep the labeled
      - If both labeled and conflicting, prefer True (software) to avoid excluding positives
      - Keep non-null numeric fields when present
      - For technologies, use the longer list
    """
    merged = dict(existing)
    # is_software
    e_lab = existing.get('is_software')
    i_lab = incoming.get('is_software')
    if i_lab is not None:
        if e_lab is None:
            merged['is_software'] = bool(i_lab)
        elif e_lab is False and i_lab is True:
            merged['is_software'] = True
    # technologies
    e_tech = existing.get('technologies')
    i_tech = incoming.get('technologies')
    if isinstance(i_tech, list):
        if not isinstance(e_tech, list) or len(i_tech) > len(e_tech):
            merged['technologies'] = i_tech
    # numeric fields
    for k in ('loc', 'hours', 'complexity_score'):
        if merged.get(k) is None and isinstance(incoming.get(k), (int, float)):
            merged[k] = incoming[k]
    return merged


def discover_inputs(data_dir: str) -> List[str]:
    return [
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir)
        if name.lower().endswith('.jsonl')
    ]


def main():
    ap = argparse.ArgumentParser(description='Merge multiple training datasets into one JSONL file.')
    ap.add_argument('--inputs', nargs='*', help='Input JSONL files (default: all data/*.jsonl)')
    ap.add_argument('--out', default='data/merged_training_data.jsonl', help='Output JSONL file')
    args = ap.parse_args()

    inputs = args.inputs or discover_inputs('data')
    inputs = [p for p in inputs if os.path.exists(p)]
    if not inputs:
        raise SystemExit('No input files found to merge.')

    print('Merging files:')
    for p in inputs:
        print(f'  - {p}')

    merged_map: Dict[str, Dict[str, Any]] = {}
    for p in inputs:
        for rec in read_jsonl(p):
            key = rec['text'].casefold()
            if key in merged_map:
                merged_map[key] = merge_records(merged_map[key], rec)
            else:
                merged_map[key] = rec

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for _, rec in merged_map.items():
            # prune keys and keep stable order
            ordered = {k: rec[k] for k in essential_order if k in rec}
            # include any extra keys at the end
            for k, v in rec.items():
                if k not in ordered:
                    ordered[k] = v
            f.write(json.dumps(ordered, ensure_ascii=False) + '\n')

    print(f'Wrote merged dataset: {args.out}')
    print(f'Total unique examples: {len(merged_map)}')


if __name__ == '__main__':
    main()
