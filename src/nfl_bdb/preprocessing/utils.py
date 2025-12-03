"""
Utility functions for preprocessing operations.
"""

import pandas as pd
import numpy as np
from typing import List, Optional


def validate_required_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    """
    Validate that all required columns exist in the dataframe.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        
    Raises:
        ValueError: If any required column is missing
    """
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def ensure_numeric_types(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    """
    Ensure specified columns are numeric types.
    
    Args:
        df: DataFrame to process
        numeric_cols: List of column names that should be numeric
        
    Returns:
        DataFrame with numeric columns converted
    """
    df = df.copy()
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


def validate_coordinates(df: pd.DataFrame, coord_cols: List[str]) -> None:
    """
    Validate that coordinate columns are non-negative after normalization.
    
    Args:
        df: DataFrame to validate
        coord_cols: List of coordinate column names (e.g., ['x', 'y'])
        
    Raises:
        ValueError: If any coordinate is negative
    """
    for col in coord_cols:
        if col in df.columns:
            negative_count = (df[col] < 0).sum()
            if negative_count > 0:
                # Allow small negative values due to floating point precision
                if (df[col] < -0.01).sum() > 0:
                    raise ValueError(
                        f"Column {col} has {negative_count} negative values. "
                        "Coordinates should be non-negative after normalization."
                    )


def sort_dataframe(
    df: pd.DataFrame,
    sort_cols: Optional[List[str]] = None,
    ascending: bool = True
) -> pd.DataFrame:
    """
    Sort dataframe deterministically.
    
    Args:
        df: DataFrame to sort
        sort_cols: Columns to sort by. Default: ['game_id', 'play_id', 'frame_id', 'nfl_id']
        ascending: Whether to sort in ascending order
        
    Returns:
        Sorted DataFrame
    """
    if sort_cols is None:
        # Default sort order for NFL tracking data
        sort_cols = ['game_id', 'play_id', 'frame_id', 'nfl_id']
    
    # Only use columns that exist
    sort_cols = [col for col in sort_cols if col in df.columns]
    
    if sort_cols:
        df = df.sort_values(by=sort_cols, ascending=ascending).reset_index(drop=True)
    
    return df




