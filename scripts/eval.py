#!/usr/bin/env python3
"""
Evaluation CLI entrypoint.

Usage:
    python scripts/eval.py [--checkpoint PATH] [--sequences PATH] [--batch-size SIZE] [--device DEVICE]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nfl_bdb.cli import eval_cli
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate NFL trajectory prediction model")
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--sequences', type=str, help='Sequences pickle path')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    args = parser.parse_args()
    
    eval_cli(args)

