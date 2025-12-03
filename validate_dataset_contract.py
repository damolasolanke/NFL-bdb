"""
Validation script to ensure InferenceDataset matches NFLTrajectoryDataset contract exactly.

Compares shapes, dtypes, and keys (excluding target) between training and inference datasets.
"""

import sys
from pathlib import Path
import pickle
import pandas as pd
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.utils.datasets import NFLTrajectoryDataset
from src.inference.inference_dataset import InferenceDataset
import config


def load_training_sample():
    """Load one sample from training dataset."""
    print("[+] Loading training sequences...")
    train_seq_path = config.PROCESSED_DIR / "sequences.pkl"
    with open(train_seq_path, "rb") as f:
        train_sequences = pickle.load(f)
    
    print(f"[+] Loaded {len(train_sequences)} training sequences")
    
    # Load all_input_df for graph construction
    print("[+] Loading training input dataframe...")
    frames = []
    for csv in sorted(config.TRAIN_DIR.glob("input_*.csv"))[:5]:  # Load first 5 for speed
        frames.append(pd.read_csv(csv))
    all_input_df = pd.concat(frames, ignore_index=True)
    print(f"[+] Loaded {len(all_input_df)} rows from training data")
    
    # Create training dataset
    train_dataset = NFLTrajectoryDataset(
        sequences=train_sequences,
        all_input_df=all_input_df,
        max_players=config.MAX_PLAYERS,
        max_trajectory_length=config.MAX_TRAJECTORY_LENGTH,
        radius=config.RADIUS
    )
    
    # Get first sample
    train_sample = train_dataset[0]
    print(f"[✓] Loaded training sample with key: {train_sample['key']}")
    
    return train_sample


def load_inference_sample():
    """Load one sample from inference dataset."""
    print("[+] Loading test sequences...")
    test_seq_path = config.PROCESSED_DIR / "test_sequences.pkl"
    with open(test_seq_path, "rb") as f:
        test_sequences = pickle.load(f)
    
    print(f"[+] Loaded {len(test_sequences)} test sequences")
    
    # Optionally load test input dataframe (if available)
    # For now, use None (fallback to single player)
    all_input_df = None
    
    # Create inference dataset
    inference_dataset = InferenceDataset(
        sequences=test_sequences,
        all_input_df=all_input_df,
        max_players=config.MAX_PLAYERS,
        max_trajectory_length=config.MAX_TRAJECTORY_LENGTH,
        radius=config.RADIUS
    )
    
    # Get first sample
    inference_sample = inference_dataset[0]
    print(f"[✓] Loaded inference sample with key: {inference_sample['key']}")
    
    return inference_sample


def validate_contract(train_sample, inference_sample):
    """Validate that inference sample matches training contract (excluding target)."""
    print("\n" + "="*80)
    print("VALIDATING DATASET CONTRACT")
    print("="*80)
    
    # Keys that should match (excluding target)
    keys_to_validate = [
        'node_feats',
        'edge_index',
        'edge_attr',
        'initial_pos',
        'mask',
        'key',
        'trajectory_length'
    ]
    
    # Check that all required keys exist
    print("\n[1] Checking keys...")
    train_keys = set(train_sample.keys())
    inference_keys = set(inference_sample.keys())
    
    # Training has 'target', inference doesn't - that's expected
    expected_train_only = {'target'}
    expected_inference_only = set()
    
    train_only = train_keys - inference_keys - expected_train_only
    inference_only = inference_keys - train_keys - expected_inference_only
    
    if train_only:
        print(f"❌ Keys in training but not inference (unexpected): {train_only}")
        return False
    if inference_only:
        print(f"❌ Keys in inference but not training (unexpected): {inference_only}")
        return False
    
    print("✓ All expected keys present")
    
    # Validate shapes and dtypes
    print("\n[2] Validating shapes and dtypes...")
    all_match = True
    
    for key in keys_to_validate:
        if key not in train_sample:
            print(f"⚠️  Key '{key}' not in training sample (skipping)")
            continue
        if key not in inference_sample:
            print(f"❌ Key '{key}' missing in inference sample")
            all_match = False
            continue
        
        train_val = train_sample[key]
        inference_val = inference_sample[key]
        
        # Check if both are tensors
        if isinstance(train_val, torch.Tensor) and isinstance(inference_val, torch.Tensor):
            train_shape = train_val.shape
            inference_shape = inference_val.shape
            train_dtype = train_val.dtype
            inference_dtype = inference_val.dtype
            
            shape_match = train_shape == inference_shape
            dtype_match = train_dtype == inference_dtype
            
            status = "✓" if (shape_match and dtype_match) else "❌"
            print(f"{status} {key:20s} | Shape: {str(train_shape):30s} == {str(inference_shape):30s} | "
                  f"Dtype: {train_dtype} == {inference_dtype}")
            
            if not shape_match:
                print(f"   ❌ Shape mismatch!")
                all_match = False
            if not dtype_match:
                print(f"   ❌ Dtype mismatch!")
                all_match = False
        elif key == 'key':
            # Key is a tuple, just check it exists
            print(f"✓ {key:20s} | Training: {train_val} | Inference: {inference_val}")
        elif key == 'trajectory_length':
            # trajectory_length is an int
            train_val_int = int(train_val) if isinstance(train_val, torch.Tensor) else train_val
            inference_val_int = int(inference_val) if isinstance(inference_val, torch.Tensor) else inference_val
            match = train_val_int == inference_val_int
            status = "✓" if match else "❌"
            print(f"{status} {key:20s} | Training: {train_val_int} | Inference: {inference_val_int}")
            if not match:
                print(f"   ⚠️  Note: For inference, trajectory_length=22 (1 initial + 21 predictions) is expected")
        else:
            print(f"⚠️  {key:20s} | Unexpected type: {type(train_val)}")
    
    print("\n" + "="*80)
    if all_match:
        print("✅ VALIDATION PASSED: InferenceDataset matches training contract!")
    else:
        print("❌ VALIDATION FAILED: Mismatches found above")
    print("="*80)
    
    return all_match


def main():
    """Run validation."""
    try:
        train_sample = load_training_sample()
        inference_sample = load_inference_sample()
        
        success = validate_contract(train_sample, inference_sample)
        
        if success:
            print("\n✅ All checks passed. InferenceDataset is contract-compliant.")
            return 0
        else:
            print("\n❌ Validation failed. Please fix the mismatches above.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error during validation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())



