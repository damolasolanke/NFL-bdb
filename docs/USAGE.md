# Usage Guide

## Table of Contents

1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training](#training)
4. [Evaluation](#evaluation)
5. [Inference](#inference)
6. [Submission Generation](#submission-generation)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)

## Installation

See [README.md](../README.md#installation) for installation instructions.

## Data Preparation

### 1. Organize Data

Place your training data in `data/train/`:

```
data/train/
├── input_2023_w01.csv
├── input_2023_w02.csv
├── ...
├── output_2023_w01.csv
├── output_2023_w02.csv
└── ...
```

### 2. Preprocess Data

Build sequences from raw CSV files:

```bash
python scripts/build_sequences.py
```

This will:
- Load all input/output CSVs
- Normalize coordinates (offense → right)
- Add engineered features
- Build sequences dictionary
- Save to `data/processed/sequences.pkl`

**Output**: `data/processed/sequences.pkl`

## Training

### Basic Training

```bash
python scripts/train.py
```

Uses default configuration from `configs/default.py`.

### Custom Training

```bash
python scripts/train.py \
    --batch-size 64 \
    --learning-rate 1e-4 \
    --epochs 100 \
    --device cuda
```

### Training Options

- `--batch-size`: Batch size (default: 32)
- `--learning-rate`: Learning rate (default: 1e-4)
- `--epochs`: Number of epochs (default: 100)
- `--device`: Device to use (default: cuda:0)

### Training Output

- Checkpoints: `models/checkpoint_epoch_*.pt` (every 5 epochs)
- Best model: `models/best_model.pt` (lowest validation loss)

### Monitoring Training

Training logs show:
- Batch progress
- Epoch loss (train/val)
- Best model saves

## Evaluation

### Evaluate on Validation Set

```bash
python scripts/eval.py
```

### Custom Evaluation

```bash
python scripts/eval.py \
    --checkpoint models/best_model.pt \
    --sequences data/processed/sequences.pkl \
    --batch-size 64 \
    --device cuda
```

### Evaluation Output

- Metrics: `eval_metrics.json` (overall RMSE, per-step RMSE)
- Predictions: `eval_preds.npy`
- Targets: `eval_targets.npy`

### Metrics

- **Overall RMSE**: Root Mean Squared Error across all predictions
- **Per-step RMSE**: RMSE for each of 21 prediction steps

## Inference

### Generate Predictions

```bash
python scripts/infer.py
```

### Custom Inference

```bash
python scripts/infer.py \
    --checkpoint models/best_model.pt \
    --sequences data/processed/test_sequences.pkl \
    --output predictions.csv \
    --batch-size 64 \
    --device cuda
```

### Inference Output

CSV file with columns:
- `gameId`: Game ID
- `playId`: Play ID
- `nflId`: Player ID
- `step`: Prediction step (0-20)
- `x`: X coordinate
- `y`: Y coordinate

## Submission Generation

### Generate Kaggle Submission

```bash
python scripts/submit.py
```

### Custom Submission

```bash
python scripts/submit.py \
    --input data/test_sample/test_input.csv \
    --output submission.parquet
```

### Submission Format

Parquet file with columns:
- `row_id`: Format: `<game_id>_<play_id>_<nfl_id>_<frame_id>`
- `x`: X coordinate (float64)
- `y`: Y coordinate (float64)

**Note**: Only rows where `player_to_predict == True` are included.

## Configuration

### Config File

All hyperparameters are in `configs/default.py`:

```python
# Model architecture
HIDDEN_DIM = 256
GRAPH_LAYERS = 3
TEMPORAL_LAYERS = 4
HEADS = 8
DROPOUT = 0.1

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 100
GRAD_CLIP_NORM = 1.0

# Data
MAX_PLAYERS = 22
MAX_FRAMES = 100
RADIUS = 20.0
```

### Overriding Config

Option 1: Edit `configs/default.py` directly

Option 2: Override in script:
```python
import configs.default as config
config.BATCH_SIZE = 64
```

Option 3: Use CLI arguments (for training)

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Error**: `ModuleNotFoundError: No module named 'nfl_bdb'`

**Solution**: 
```bash
# Install in development mode
pip install -e .
```

#### 2. CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
- Reduce batch size: `--batch-size 16`
- Use CPU: `--device cpu`
- Clear GPU cache: Restart Python

#### 3. File Not Found

**Error**: `FileNotFoundError: sequences.pkl not found`

**Solution**: Run preprocessing first:
```bash
python scripts/build_sequences.py
```

#### 4. Shape Mismatch

**Error**: `RuntimeError: shape mismatch`

**Solution**: 
- Check data preprocessing
- Verify config matches model architecture
- Check dataset/collate functions

#### 5. NaN/Inf Values

**Error**: `RuntimeError: NaN or Inf detected`

**Solution**:
- Check input data for missing values
- Verify preprocessing steps
- Check feature engineering

### Getting Help

1. Check error messages carefully
2. Review relevant documentation
3. Check GitHub issues
4. Open a new issue with:
   - Error message
   - Steps to reproduce
   - Environment details

## Advanced Usage

### Custom Dataset

Create custom dataset class:

```python
from nfl_bdb.utils import NFLTrajectoryDataset

class CustomDataset(NFLTrajectoryDataset):
    def __getitem__(self, idx):
        # Custom implementation
        pass
```

### Custom Model

Create custom model:

```python
from nfl_bdb.models import STGNNRefine

class CustomModel(STGNNRefine):
    def forward(self, graph, initial_pos=None):
        # Custom forward pass
        pass
```

### Using Notebooks

Jupyter notebooks in `notebooks/` for interactive development:

```bash
jupyter notebook notebooks/
```

## Best Practices

1. **Always preprocess data first** before training
2. **Use validation set** to monitor training
3. **Save checkpoints regularly** for recovery
4. **Test inference** before generating submission
5. **Verify submission format** matches competition requirements

## Examples

### Complete Workflow

```bash
# 1. Preprocess
python scripts/build_sequences.py

# 2. Train
python scripts/train.py --epochs 50

# 3. Evaluate
python scripts/eval.py --checkpoint models/best_model.pt

# 4. Infer
python scripts/infer.py --checkpoint models/best_model.pt --output predictions.csv

# 5. Submit
python scripts/submit.py --input data/test_sample/test_input.csv --output submission.parquet
```

### Quick Test

```bash
# Small test run
python scripts/train.py --epochs 1 --batch-size 4
python scripts/eval.py --checkpoint models/best_model.pt
```

