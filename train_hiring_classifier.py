#!/usr/bin/env python3
"""CLI to train the hiring vs build classifier.

Dataset format: JSONL, each line like:
{"text": "looking for a senior react engineer with 5+ years experience", "label": 1}
{"text": "build a marketplace web app with react node postgres stripe", "label": 0}

label: 1 = hiring/job description, 0 = build/implementation requirement.

Usage (PowerShell):
  $env:HIRING_BUILD_DATA="data/hiring_build_training_data.jsonl"
  python train_hiring_classifier.py

Or specify path:
  python train_hiring_classifier.py --data data/hiring_build_training_data.jsonl --out models/software/hiring_build_classifier.joblib
"""
import argparse
import os
from mcp_server.hiring_classifier import train_and_save


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default=os.getenv('HIRING_BUILD_DATA'), help='Path to labeled JSONL dataset')
    p.add_argument('--out', type=str, default=os.path.join('models', 'software', 'hiring_build_classifier.joblib'), help='Output path for model bundle')
    return p.parse_args()


def main():
    args = parse_args()
    if not args.data or not os.path.isfile(args.data):
        raise SystemExit(f"Dataset file not found: {args.data}")
    train_and_save(args.data, args.out)

if __name__ == '__main__':
    main()
