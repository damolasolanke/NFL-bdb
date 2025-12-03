"""
Utility functions for baseline models.
"""

import numpy as np
from typing import Dict, Tuple


def get_sequence_length(sequences: Dict[Tuple[int, int, int], Dict]) -> int:
    """
    Get the maximum sequence length from all sequences.
    
    Args:
        sequences: Dictionary of sequences
        
    Returns:
        Maximum target trajectory length
    """
    if not sequences:
        return 0
    
    lengths = [len(seq['target']) for seq in sequences.values()]
    return max(lengths) if lengths else 0




