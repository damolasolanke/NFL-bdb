#!/usr/bin/env python3
"""
Kaggle submission generation CLI entrypoint.

Usage:
    python scripts/submit.py [--input PATH] [--output PATH]
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nfl_bdb.cli import submit_cli
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Kaggle submission file")
    parser.add_argument('--input', type=str, help='Input CSV path')
    parser.add_argument('--output', type=str, help='Output parquet path')
    args = parser.parse_args()
    
    submit_cli(args)

