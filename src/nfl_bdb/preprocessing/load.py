"""
Load input and output CSV files for training data.
"""

import pandas as pd
import glob
import os
from pathlib import Path
from typing import Optional
from .utils import sort_dataframe


def load_all_train_inputs(data_path: str) -> pd.DataFrame:
    """
    Load all input_*.csv files from the training directory.
    
    Args:
        data_path: Path to the data directory containing train/ subdirectory
        
    Returns:
        Concatenated DataFrame with all input data, sorted deterministically
        
    Raises:
        FileNotFoundError: If training directory doesn't exist or no files found.
                          Prints helpful message with instructions.
    """
    train_path = os.path.join(data_path, 'train')
    
    # Check if directory exists
    if not os.path.exists(train_path):
        print(f"\n{'='*60}")
        print("ERROR: Training directory not found")
        print(f"{'='*60}")
        print(f"Expected path: {train_path}")
        print(f"\nPlease ensure the directory exists and contains input_*.csv files.")
        print(f"You can set the data path in config.py (DATA_PATH variable).")
        print(f"{'='*60}\n")
        raise FileNotFoundError(
            f"Training directory not found: {train_path}\n"
            f"Please create the directory and add your input_*.csv files."
        )
    
    # Find all input CSV files
    input_files = glob.glob(os.path.join(train_path, 'input_*.csv'))
    
    if not input_files:
        print(f"\n{'='*60}")
        print("ERROR: No input CSV files found")
        print(f"{'='*60}")
        print(f"Directory exists: {train_path}")
        print(f"Expected files: input_*.csv")
        print(f"\nPlease ensure training data files are placed in:")
        print(f"  {train_path}")
        print(f"\nYou can set the data path in config.py (DATA_PATH variable).")
        print(f"{'='*60}\n")
        raise FileNotFoundError(
            f"No input_*.csv files found in {train_path}. "
            "Please ensure training data is placed in the train/ directory."
        )
    
    # Sort files for deterministic loading
    input_files = sorted(input_files)
    
    # Load and concatenate all input files
    dfs = []
    for file_path in input_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort deterministically
    combined_df = sort_dataframe(combined_df)
    
    return combined_df


def load_all_train_outputs(data_path: str) -> pd.DataFrame:
    """
    Load all output_*.csv files from the training directory.
    
    Args:
        data_path: Path to the data directory containing train/ subdirectory
        
    Returns:
        Concatenated DataFrame with all output data, sorted deterministically
        
    Raises:
        FileNotFoundError: If training directory doesn't exist or no files found.
                          Prints helpful message with instructions.
    """
    train_path = os.path.join(data_path, 'train')
    
    # Check if directory exists
    if not os.path.exists(train_path):
        print(f"\n{'='*60}")
        print("ERROR: Training directory not found")
        print(f"{'='*60}")
        print(f"Expected path: {train_path}")
        print(f"\nPlease ensure the directory exists and contains output_*.csv files.")
        print(f"You can set the data path in config.py (DATA_PATH variable).")
        print(f"{'='*60}\n")
        raise FileNotFoundError(
            f"Training directory not found: {train_path}\n"
            f"Please create the directory and add your output_*.csv files."
        )
    
    # Find all output CSV files
    output_files = glob.glob(os.path.join(train_path, 'output_*.csv'))
    
    if not output_files:
        print(f"\n{'='*60}")
        print("ERROR: No output CSV files found")
        print(f"{'='*60}")
        print(f"Directory exists: {train_path}")
        print(f"Expected files: output_*.csv")
        print(f"\nPlease ensure training data files are placed in:")
        print(f"  {train_path}")
        print(f"\nYou can set the data path in config.py (DATA_PATH variable).")
        print(f"{'='*60}\n")
        raise FileNotFoundError(
            f"No output_*.csv files found in {train_path}. "
            "Please ensure training data is placed in the train/ directory."
        )
    
    # Sort files for deterministic loading
    output_files = sorted(output_files)
    
    # Load and concatenate all output files
    dfs = []
    for file_path in output_files:
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Sort deterministically
    combined_df = sort_dataframe(combined_df)
    
    return combined_df

