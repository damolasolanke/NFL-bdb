"""
NFL Big Data Bowl 2026 - Baseline Models

This module contains baseline prediction models and evaluation metrics.
"""

from .straight_line import straight_line_baseline
from .evaluation import rmse, evaluate
from .visualize import plot_trajectory_comparison

__all__ = [
    'straight_line_baseline',
    'rmse',
    'evaluate',
    'plot_trajectory_comparison',
]




