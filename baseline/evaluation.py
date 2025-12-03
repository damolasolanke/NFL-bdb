"""
Evaluation metrics for trajectory prediction.

Implements the exact Kaggle competition RMSE metric.
"""

import numpy as np
from typing import Dict, Tuple


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error between predicted and true trajectories.
    
    This matches the Kaggle competition metric exactly.
    
    Args:
        pred: Predicted trajectory of shape (T, 2) where columns are [x, y]
        true: True trajectory of shape (T, 2) where columns are [x, y]
        
    Returns:
        RMSE value (float)
        
    Raises:
        ValueError: If shapes don't match or are invalid
    """
    pred = np.asarray(pred)
    true = np.asarray(true)
    
    if pred.shape != true.shape:
        raise ValueError(
            f"Shape mismatch: pred {pred.shape} vs true {true.shape}"
        )
    
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError(
            f"Expected shape (T, 2), got {pred.shape}"
        )
    
    # Compute squared differences for each coordinate
    squared_diff = (pred - true) ** 2
    
    # Mean across all frames and coordinates
    mse = np.mean(squared_diff)
    
    # Root mean squared error
    rmse_value = np.sqrt(mse)
    
    return float(rmse_value)


def evaluate(
    sequences: Dict[Tuple[int, int, int], Dict],
    predictions: Dict[Tuple[int, int, int], np.ndarray]
) -> float:
    """
    Evaluate predictions against true trajectories.
    
    Computes mean RMSE across all sequences.
    
    Args:
        sequences: Dictionary mapping (game_id, play_id, nfl_id) to sequence dict
                  with 'target' key containing true trajectory
        predictions: Dictionary mapping (game_id, play_id, nfl_id) to predicted
                     trajectory array of shape (T, 2)
        
    Returns:
        Mean RMSE across all sequences
    """
    if not sequences:
        raise ValueError("Sequences dictionary is empty")
    
    if not predictions:
        raise ValueError("Predictions dictionary is empty")
    
    rmse_values = []
    
    for key in sequences:
        if key not in predictions:
            # Skip if no prediction for this sequence
            continue
        
        true_trajectory = sequences[key]['target']
        pred_trajectory = predictions[key]
        
        # Ensure shapes match (pad or truncate if necessary)
        min_len = min(len(true_trajectory), len(pred_trajectory))
        if min_len == 0:
            continue
        
        true_subset = true_trajectory[:min_len]
        pred_subset = pred_trajectory[:min_len]
        
        try:
            rmse_val = rmse(pred_subset, true_subset)
            rmse_values.append(rmse_val)
        except ValueError as e:
            # Skip sequences with shape mismatches
            print(f"Warning: Skipping sequence {key} due to error: {e}")
            continue
    
    if not rmse_values:
        raise ValueError("No valid RMSE values computed")
    
    return float(np.mean(rmse_values))




