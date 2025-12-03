"""
Straight-line baseline predictor.

Assumes player continues moving in a straight line from release point
toward ball landing at current speed.
"""

import numpy as np
import pandas as pd


def straight_line_baseline(input_row: pd.Series, T: int) -> np.ndarray:
    """
    Predict trajectory using straight-line movement baseline.
    
    Model: Player runs in a straight line from release point toward
    ball landing at current speed.
    
    Args:
        input_row: Series containing features from last input frame.
                  Must include: x, y, s (speed), ball_land_x, ball_land_y
        T: Number of frames to predict
        
    Returns:
        Predicted trajectory as numpy array of shape (T, 2) with [x, y] coordinates
    """
    # Extract initial position
    x0 = float(input_row['x'])
    y0 = float(input_row['y'])
    
    # Extract speed (yards per second)
    speed = float(input_row.get('s', 0.0))
    
    # Extract ball landing position
    ball_land_x = float(input_row.get('ball_land_x', x0))
    ball_land_y = float(input_row.get('ball_land_y', y0))
    
    # Compute direction vector toward ball landing
    dx = ball_land_x - x0
    dy = ball_land_y - y0
    dist_to_ball = np.sqrt(dx**2 + dy**2)
    
    # Normalize direction (unit vector)
    if dist_to_ball > 1e-6:
        direction_x = dx / dist_to_ball
        direction_y = dy / dist_to_ball
    else:
        # If already at ball landing, use current velocity direction if available
        if 'vx' in input_row and 'vy' in input_row:
            vx = float(input_row['vx'])
            vy = float(input_row['vy'])
            v_mag = np.sqrt(vx**2 + vy**2)
            if v_mag > 1e-6:
                direction_x = vx / v_mag
                direction_y = vy / v_mag
            else:
                direction_x = 0.0
                direction_y = 0.0
        else:
            direction_x = 0.0
            direction_y = 0.0
    
    # Speed in yards per frame (assuming 10 FPS)
    YARDS_PER_FRAME = speed / 10.0
    
    # Generate trajectory
    trajectory = np.zeros((T, 2))
    
    for t in range(T):
        # Distance traveled in this frame
        distance = YARDS_PER_FRAME
        
        # Update position
        x_new = x0 + direction_x * distance * (t + 1)
        y_new = y0 + direction_y * distance * (t + 1)
        
        trajectory[t, 0] = x_new
        trajectory[t, 1] = y_new
    
    return trajectory




