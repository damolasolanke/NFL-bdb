"""
Training module for NFL trajectory prediction models.
"""

from .train import main as train_main, MaskedMSELoss

__all__ = ['train_main', 'MaskedMSELoss']
