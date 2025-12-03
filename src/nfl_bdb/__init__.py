"""
NFL Big Data Bowl 2026 - Main Package

Spatio-Temporal Graph Neural Network for NFL player trajectory prediction.
"""

__version__ = "0.1.0"

from . import models
from . import utils
from . import preprocessing
from . import training
from . import inference
from . import evaluation

__all__ = [
    'models',
    'utils',
    'preprocessing',
    'training',
    'inference',
    'evaluation',
]

