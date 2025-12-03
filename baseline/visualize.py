"""
Visualization utilities for trajectory predictions.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional


def plot_trajectory_comparison(
    true_trajectory: np.ndarray,
    pred_trajectory: np.ndarray,
    start_pos: Optional[Tuple[float, float]] = None,
    title: str = "Trajectory Comparison",
    save_path: Optional[str] = None
) -> None:
    """
    Plot comparison between true and predicted trajectories.
    
    Args:
        true_trajectory: True trajectory of shape (T, 2) with [x, y] coordinates
        pred_trajectory: Predicted trajectory of shape (T, 2) with [x, y] coordinates
        start_pos: Optional starting position (x, y) to mark
        title: Plot title
        save_path: Optional path to save the figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true trajectory
    ax.plot(
        true_trajectory[:, 0],
        true_trajectory[:, 1],
        'b-o',
        label='True Trajectory',
        markersize=4,
        linewidth=2,
        alpha=0.7
    )
    
    # Plot predicted trajectory
    ax.plot(
        pred_trajectory[:, 0],
        pred_trajectory[:, 1],
        'r--s',
        label='Predicted Trajectory',
        markersize=4,
        linewidth=2,
        alpha=0.7
    )
    
    # Mark start position if provided
    if start_pos is not None:
        ax.plot(
            start_pos[0],
            start_pos[1],
            'go',
            markersize=10,
            label='Start Position',
            zorder=5
        )
    
    # Mark end positions
    ax.plot(
        true_trajectory[-1, 0],
        true_trajectory[-1, 1],
        'b*',
        markersize=12,
        label='True End',
        zorder=5
    )
    
    ax.plot(
        pred_trajectory[-1, 0],
        pred_trajectory[-1, 1],
        'r*',
        markersize=12,
        label='Predicted End',
        zorder=5
    )
    
    ax.set_xlabel('X Position (yards)', fontsize=12)
    ax.set_ylabel('Y Position (yards)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()




