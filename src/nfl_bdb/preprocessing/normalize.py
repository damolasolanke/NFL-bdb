"""
Normalize field coordinates so offense always moves left to right.
"""

import pandas as pd
import numpy as np
from .utils import validate_required_columns


def normalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize field coordinates so offense always moves to the right.
    
    Rule: If play_direction == "left", flip coordinates:
    - x = 120 - x
    - y = 53.3 - y
    - ball_land_x = 120 - ball_land_x (if exists)
    - ball_land_y = 53.3 - ball_land_y (if exists)
    
    Args:
        df: DataFrame with columns including 'play_direction', 'x', 'y',
            and optionally 'ball_land_x', 'ball_land_y'
            
    Returns:
        DataFrame with normalized coordinates
    """
    df = df.copy()
    
    # Check if play_direction column exists
    if 'play_direction' not in df.columns:
        # If no play_direction, assume all plays are already normalized
        # or we need to infer from other data
        return df
    
    # Field dimensions
    FIELD_LENGTH = 120.0  # yards
    FIELD_WIDTH = 53.3    # yards
    
    # Identify plays going left
    left_mask = df['play_direction'] == 'left'
    
    if left_mask.sum() > 0:
        # Flip x coordinate: x' = 120 - x
        df.loc[left_mask, 'x'] = FIELD_LENGTH - df.loc[left_mask, 'x']
        
        # Flip y coordinate: y' = 53.3 - y
        df.loc[left_mask, 'y'] = FIELD_WIDTH - df.loc[left_mask, 'y']
        
        # Flip ball landing coordinates if they exist
        if 'ball_land_x' in df.columns:
            df.loc[left_mask, 'ball_land_x'] = (
                FIELD_LENGTH - df.loc[left_mask, 'ball_land_x']
            )
        
        if 'ball_land_y' in df.columns:
            df.loc[left_mask, 'ball_land_y'] = (
                FIELD_WIDTH - df.loc[left_mask, 'ball_land_y']
            )
    
    return df




