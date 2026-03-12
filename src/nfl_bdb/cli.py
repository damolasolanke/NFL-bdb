"""
Command-line interface for NFL Big Data Bowl 2026 project.

Provides entrypoints for training, inference, evaluation, and submission generation.
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import configs.default as config
from nfl_bdb.training import train_main
from nfl_bdb.inference import run_inference
from nfl_bdb.evaluation import run_eval


def train_cli(args):
    """CLI entrypoint for training."""
    # Override config if provided
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.learning_rate:
        config.LEARNING_RATE = args.learning_rate
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.device:
        config.DEVICE = args.device
    
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Training")
    print("=" * 60)
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.LEARNING_RATE}")
    print(f"Epochs: {config.EPOCHS}")
    print(f"Device: {config.DEVICE}")
    print("=" * 60)
    print()
    
    train_main()


def infer_cli(args):
    """CLI entrypoint for inference."""
    ckpt_path = args.checkpoint or str(config.MODEL_DIR / "best_model.pt")
    seq_path = args.sequences or str(config.PROCESSED_DIR / "test_sequences.pkl")
    output_path = args.output or "submission.csv"
    batch_size = args.batch_size or config.BATCH_SIZE
    device = args.device or ("cuda" if config.DEVICE.startswith("cuda") else "cpu")
    
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Inference")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sequences: {seq_path}")
    print(f"Output: {output_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    run_inference(
        ckpt_path=ckpt_path,
        seq_path=seq_path,
        submission_path=output_path,
        batch_size=batch_size,
        device=device,
        all_input_df=None
    )


def eval_cli(args):
    """CLI entrypoint for evaluation."""
    ckpt_path = args.checkpoint or str(config.MODEL_DIR / "best_model.pt")
    seq_path = args.sequences or str(config.PROCESSED_DIR / "sequences.pkl")
    batch_size = args.batch_size or config.BATCH_SIZE
    device = args.device or ("cuda" if config.DEVICE.startswith("cuda") else "cpu")
    
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Evaluation")
    print("=" * 60)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Sequences: {seq_path}")
    print(f"Batch size: {batch_size}")
    print(f"Device: {device}")
    print("=" * 60)
    print()
    
    run_eval(
        ckpt_path=ckpt_path,
        seq_path=seq_path,
        batch_size=batch_size,
        device=device
    )


def submit_cli(args):
    """CLI entrypoint for Kaggle submission generation."""
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np
    
    input_path = args.input or os.environ.get(
        "KAGGLE_INPUT_PATH",
        str(config.TEST_SAMPLE_DIR / "test_input.csv")
    )
    output_path = args.output or os.environ.get(
        "KAGGLE_OUTPUT_PATH",
        "submission.parquet"
    )
    
    # Fallback for local testing
    if not Path(input_path).exists():
        input_path = str(config.TEST_SAMPLE_DIR / "test_input.csv")
    
    print("=" * 60)
    print("NFL Big Data Bowl 2026 - Submission Generation")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print("=" * 60)
    print()
    
    # Load and filter data
    print("[+] Loading test_input.csv...")
    df = pd.read_csv(input_path)
    print(f"[✓] Loaded {len(df):,} total rows")
    
    # Filter to player_to_predict == True
    print("\n[+] Filtering to player_to_predict == True...")
    if df['player_to_predict'].dtype == 'object':
        submission_df = df[df['player_to_predict'].astype(str).str.lower() == 'true'].copy()
    else:
        submission_df = df[df['player_to_predict'] == True].copy()
    print(f"[✓] Filtered to {len(submission_df):,} rows")
    
    # Construct row_id
    print("\n[+] Constructing row_id...")
    game_id_int = submission_df['game_id'].astype('int64')
    play_id_int = submission_df['play_id'].astype('int64')
    nfl_id_int = submission_df['nfl_id'].astype('int64')
    frame_id_int = submission_df['frame_id'].astype('int64')
    
    row_id = (
        game_id_int.astype(str) + '_' +
        play_id_int.astype(str) + '_' +
        nfl_id_int.astype(str) + '_' +
        frame_id_int.astype(str)
    ).reset_index(drop=True).astype(str).astype(object)
    
    # Add dummy predictions (replace with actual model predictions)
    x_pred = pd.Series([0.0] * len(submission_df), dtype='float64')
    y_pred = pd.Series([0.0] * len(submission_df), dtype='float64')
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'row_id': row_id,
        'x': x_pred,
        'y': y_pred
    })[['row_id', 'x', 'y']]
    
    # Define schema
    schema = pa.schema([
        pa.field('row_id', pa.string()),
        pa.field('x', pa.float64()),
        pa.field('y', pa.float64()),
    ])
    
    # Write parquet
    print("\n[+] Writing parquet file...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(submission, schema=schema)
    pq.write_table(
        table,
        output_path,
        compression='snappy',
        use_dictionary=False,
        write_statistics=False
    )
    
    print(f"[✓] Parquet file written to: {output_path}")
    print(f"[✓] Total rows: {len(submission):,}")


def main():
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="NFL Big Data Bowl 2026 - Trajectory Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Inference command
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    infer_parser.add_argument('--sequences', type=str, help='Test sequences pickle path')
    infer_parser.add_argument('--output', type=str, help='Output CSV path')
    infer_parser.add_argument('--batch-size', type=int, help='Batch size')
    infer_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Evaluation command
    eval_parser = subparsers.add_parser('eval', help='Evaluate the model')
    eval_parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    eval_parser.add_argument('--sequences', type=str, help='Sequences pickle path')
    eval_parser.add_argument('--batch-size', type=int, help='Batch size')
    eval_parser.add_argument('--device', type=str, help='Device (cuda/cpu)')
    
    # Submission command
    submit_parser = subparsers.add_parser('submit', help='Generate Kaggle submission')
    submit_parser.add_argument('--input', type=str, help='Input CSV path')
    submit_parser.add_argument('--output', type=str, help='Output parquet path')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    if args.command == 'train':
        train_cli(args)
    elif args.command == 'infer':
        infer_cli(args)
    elif args.command == 'eval':
        eval_cli(args)
    elif args.command == 'submit':
        submit_cli(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

