"""
Inference module for NFL trajectory prediction.
"""

from .run_inference import run_inference
from .inference_dataset import InferenceDataset

__all__ = ['run_inference', 'InferenceDataset']

