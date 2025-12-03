"""
Construct sequences for training and prediction.

For each (game_id, play_id, nfl_id):
- input: last frame before pass release (from input data)
- target: full trajectory while ball is in the air (from output data)
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def build_sequences(
    input_df: pd.DataFrame,
    output_df: pd.DataFrame
) -> Dict[Tuple[int, int, int], Dict[str, any]]:
    """
    Build sequences dictionary from input and output dataframes.
    
    For each unique (game_id, play_id, nfl_id):
    - Extract the last frame from input data (frame at QB release)
    - Extract all frames from output data (trajectory while ball is in air)
    
    Args:
        input_df: DataFrame with input data (pre-release frames)
        output_df: DataFrame with output data (post-release frames)
        
    Returns:
        Dictionary mapping (game_id, play_id, nfl_id) to:
        {
            "input": pd.Series with features from last input frame,
            "target": np.ndarray of shape (T, 2) with [x, y] trajectory
        }
    """
    sequences = {}
    
    # Get unique combinations of game_id, play_id, nfl_id from input
    input_keys = input_df[['game_id', 'play_id', 'nfl_id']].drop_duplicates()
    
    for _, row in input_keys.iterrows():
        game_id = row['game_id']
        play_id = row['play_id']
        nfl_id = row['nfl_id']
        key = (game_id, play_id, nfl_id)
        
        # Get all input frames for this player
        input_mask = (
            (input_df['game_id'] == game_id) &
            (input_df['play_id'] == play_id) &
            (input_df['nfl_id'] == nfl_id)
        )
        input_frames = input_df[input_mask].copy()
        
        if input_frames.empty:
            continue
        
        # Get the last frame (highest frame_id) as input
        input_frames = input_frames.sort_values('frame_id', ascending=True)
        last_input_frame = input_frames.iloc[-1]
        
        # Get all output frames for this player (trajectory)
        output_mask = (
            (output_df['game_id'] == game_id) &
            (output_df['play_id'] == play_id) &
            (output_df['nfl_id'] == nfl_id)
        )
        output_frames = output_df[output_mask].copy()
        
        if output_frames.empty:
            # If no output data, skip this sequence
            continue
        
        # Sort output frames by frame_id
        output_frames = output_frames.sort_values('frame_id', ascending=True)
        
        # Extract trajectory (x, y coordinates)
        if 'x' not in output_frames.columns or 'y' not in output_frames.columns:
            continue
        
        trajectory = output_frames[['x', 'y']].values
        
        # Store sequence
        sequences[key] = {
            "input": last_input_frame,
            "target": trajectory
        }
    
    return sequences




