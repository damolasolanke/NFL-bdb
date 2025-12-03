"""
Save sequences to pickle file for efficient loading during training.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple
import config


def save_sequences(
    sequences: Dict[Tuple[int, int, int], Dict],
    output_path: str = None
) -> Path:
    """
    Save sequences dictionary to pickle file.
    
    Args:
        sequences: Dictionary mapping (game_id, play_id, nfl_id) to sequence dict
        output_path: Optional custom output path. Default: config.PROCESSED_DIR / "sequences.pkl"
        
    Returns:
        Path to saved pickle file
    """
    if output_path is None:
        # Ensure processed directory exists
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = config.PROCESSED_DIR / "sequences.pkl"
    else:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save sequences
    with open(output_path, 'wb') as f:
        pickle.dump(sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✓ Saved {len(sequences):,} sequences to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    return output_path


def load_sequences(input_path: str = None) -> Dict[Tuple[int, int, int], Dict]:
    """
    Load sequences from pickle file.
    
    Args:
        input_path: Optional custom input path. Default: config.PROCESSED_DIR / "sequences.pkl"
        
    Returns:
        Dictionary of sequences
    """
    if input_path is None:
        input_path = config.PROCESSED_DIR / "sequences.pkl"
    else:
        input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(
            f"Sequences file not found: {input_path}\n"
            f"Please run preprocessing first to generate sequences.pkl"
        )
    
    with open(input_path, 'rb') as f:
        sequences = pickle.load(f)
    
    print(f"✓ Loaded {len(sequences):,} sequences from: {input_path}")
    
    return sequences




