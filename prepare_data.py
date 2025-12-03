#!/usr/bin/env python3
"""
Prepare data directory structure for NFL Big Data Bowl 2026 project.

This script creates the necessary directories if they don't exist.
"""

import os
from pathlib import Path
from config import DATA_DIR, TRAIN_DIR, TEST_SAMPLE_DIR, METADATA_DIR


def prepare_data_directories():
    """
    Create all necessary data directories if they don't exist.
    """
    directories = [
        DATA_DIR,
        TRAIN_DIR,
        TEST_SAMPLE_DIR,
        METADATA_DIR,
    ]
    
    created_dirs = []
    existing_dirs = []
    
    for directory in directories:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created_dirs.append(directory)
            print(f"✓ Created directory: {directory}")
        else:
            existing_dirs.append(directory)
            print(f"✓ Directory already exists: {directory}")
    
    print(f"\n{'='*60}")
    print("Data directory structure ready!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Place your input_*.csv files in: {TRAIN_DIR}")
    print(f"2. Place your output_*.csv files in: {TRAIN_DIR}")
    print(f"3. Run the preprocessing notebook to load and process the data.")
    print(f"\nYou can modify the data path in config.py if needed.")
    
    return created_dirs, existing_dirs


if __name__ == "__main__":
    prepare_data_directories()




