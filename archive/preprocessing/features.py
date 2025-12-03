"""
Feature engineering for NFL tracking data.
"""

import pandas as pd
import numpy as np
from .utils import ensure_numeric_types, validate_required_columns


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the dataframe.
    
    Features added:
    1. Geometry relative to ball landing:
       - dx_ball: x distance to ball landing
       - dy_ball: y distance to ball landing
       - dist_ball: Euclidean distance to ball landing
    
    2. Velocity components:
       - vx: velocity in x direction (s * cos(dir_rad))
       - vy: velocity in y direction (s * sin(dir_rad))
    
    3. Angle conversions:
       - orientation_rad: orientation in radians
       - direction_rad: direction in radians
    
    4. Categorical encodings (one-hot):
       - player_role_*: one-hot encoded player_role
       - player_side_*: one-hot encoded player_side
       - player_position_*: one-hot encoded player_position (if exists)
    
    Args:
        df: DataFrame with tracking data
        
    Returns:
        DataFrame with additional feature columns
    """
    df = df.copy()
    
    # Ensure numeric types for key columns
    numeric_cols = ['x', 'y', 's', 'a', 'dis', 'o', 'dir']
    if 'ball_land_x' in df.columns:
        numeric_cols.append('ball_land_x')
    if 'ball_land_y' in df.columns:
        numeric_cols.append('ball_land_y')
    
    df = ensure_numeric_types(df, numeric_cols)
    
    # 1. Geometry relative to ball landing
    if 'ball_land_x' in df.columns and 'ball_land_y' in df.columns:
        df['dx_ball'] = df['ball_land_x'] - df['x']
        df['dy_ball'] = df['ball_land_y'] - df['y']
        df['dist_ball'] = np.sqrt(df['dx_ball']**2 + df['dy_ball']**2)
    else:
        # If ball landing not available, set to NaN
        df['dx_ball'] = np.nan
        df['dy_ball'] = np.nan
        df['dist_ball'] = np.nan
    
    # 2. Convert angles from degrees to radians
    if 'o' in df.columns:
        df['orientation_rad'] = np.deg2rad(df['o'])
    else:
        df['orientation_rad'] = np.nan
    
    if 'dir' in df.columns:
        df['direction_rad'] = np.deg2rad(df['dir'])
    else:
        df['direction_rad'] = np.nan
    
    # 3. Velocity components
    if 's' in df.columns and 'direction_rad' in df.columns:
        df['vx'] = df['s'] * np.cos(df['direction_rad'])
        df['vy'] = df['s'] * np.sin(df['direction_rad'])
    else:
        df['vx'] = np.nan
        df['vy'] = np.nan
    
    # 4. One-hot encode categorical variables
    if 'player_role' in df.columns:
        role_dummies = pd.get_dummies(df['player_role'], prefix='player_role')
        df = pd.concat([df, role_dummies], axis=1)
    
    if 'player_side' in df.columns:
        side_dummies = pd.get_dummies(df['player_side'], prefix='player_side')
        df = pd.concat([df, side_dummies], axis=1)
    
    if 'player_position' in df.columns:
        position_dummies = pd.get_dummies(
            df['player_position'], prefix='player_position'
        )
        df = pd.concat([df, position_dummies], axis=1)
    
    return df




