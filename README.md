# NFL Big Data Bowl 2026 - Trajectory Prediction

Spatio-Temporal Graph Neural Network (STGNN) for predicting NFL player trajectories using graph attention networks and transformers.

## Overview

This project implements a deep learning solution for the NFL Big Data Bowl 2026 competition, predicting player trajectories using:

- **Spatio-Temporal Graph Neural Networks** with Graph Attention (GAT) layers
- **Transformer** layers for temporal modeling
- **Graph-based** representation of player interactions

## Features

- 🎯 **State-of-the-art architecture**: STGNN with GAT + Transformer
- 📊 **Complete pipeline**: Preprocessing → Training → Inference → Evaluation
- 🚀 **CLI interface**: Easy-to-use command-line tools
- 📝 **Well-documented**: Comprehensive documentation and examples
- 🧪 **Reproducible**: Configurable hyperparameters and reproducible results

## Installation

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended) or CPU

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd nfl-bdb
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e ".[dev]"
```

## Quick Start

### 1. Prepare Data

Place your training data in `data/train/`:
- `input_*.csv` files (input frames)
- `output_*.csv` files (output trajectories)

### 2. Preprocess Data

Build sequences from raw data:
```bash
python scripts/build_sequences.py
```

This creates `data/processed/sequences.pkl`.

### 3. Train Model

Train the model:
```bash
python scripts/train.py
```

Or with custom parameters:
```bash
python scripts/train.py --batch-size 64 --learning-rate 1e-4 --epochs 100
```

### 4. Evaluate Model

Evaluate on validation set:
```bash
python scripts/eval.py --checkpoint models/best_model.pt
```

### 5. Run Inference

Generate predictions:
```bash
python scripts/infer.py --checkpoint models/best_model.pt --output submission.csv
```

### 6. Generate Submission

Create Kaggle submission file:
```bash
python scripts/submit.py --input data/test_sample/test_input.csv --output submission.parquet
```

## Project Structure

```
nfl-bdb/
├── configs/              # Configuration files
│   └── default.py        # Main configuration
├── data/                 # Data directory (gitignored)
│   ├── train/           # Training CSVs
│   ├── processed/      # Processed sequences
│   └── test_sample/    # Test data
├── docs/                # Documentation
│   ├── _inventory.md    # Project inventory
│   ├── PROJECT_OVERVIEW.md
│   ├── DEVELOPMENT_GUIDE.md
│   └── USAGE.md
├── models/              # Model checkpoints (gitignored)
├── notebooks/           # Jupyter notebooks
├── scripts/             # CLI scripts
│   ├── train.py
│   ├── infer.py
│   ├── eval.py
│   ├── submit.py
│   └── build_sequences.py
├── src/                 # Source code
│   └── nfl_bdb/        # Main package
│       ├── models/     # Model architectures
│       ├── training/   # Training code
│       ├── inference/  # Inference code
│       ├── evaluation/ # Evaluation code
│       ├── preprocessing/ # Data preprocessing
│       └── utils/      # Utilities
├── tests/              # Tests
├── archive/            # Archived files
├── requirements.txt    # Python dependencies
├── pyproject.toml     # Project metadata
└── README.md          # This file
```

## Configuration

All hyperparameters are configured in `configs/default.py`:

- **Model**: Hidden dim, graph layers, temporal layers, heads, dropout
- **Training**: Batch size, learning rate, epochs, gradient clipping
- **Data**: Max players, max frames, graph radius

## Model Architecture

The model consists of:

1. **Node Encoder**: Linear → LayerNorm → GELU
2. **Graph Attention Layers** (3x): GATv2 with multi-head attention
3. **Temporal Transformer** (4x): Transformer encoder layers
4. **Refinement Block**: Feed-forward refinement
5. **Output Head**: Predicts velocities → positions

## Documentation

- [Project Overview](docs/PROJECT_OVERVIEW.md) - High-level project overview
- [Development Guide](docs/DEVELOPMENT_GUIDE.md) - Development setup and guidelines
- [Usage Guide](docs/USAGE.md) - Detailed usage instructions
- [Project Inventory](docs/_inventory.md) - Complete component inventory

## CLI Commands

### Training
```bash
python scripts/train.py [--batch-size SIZE] [--learning-rate LR] [--epochs N] [--device DEVICE]
```

### Inference
```bash
python scripts/infer.py [--checkpoint PATH] [--sequences PATH] [--output PATH] [--batch-size SIZE] [--device DEVICE]
```

### Evaluation
```bash
python scripts/eval.py [--checkpoint PATH] [--sequences PATH] [--batch-size SIZE] [--device DEVICE]
```

### Submission
```bash
python scripts/submit.py [--input PATH] [--output PATH]
```

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

Follow PEP 8 style guidelines.

## License

MIT License

## Acknowledgments

- NFL Big Data Bowl 2026 competition
- PyTorch team for the excellent framework

## Contact

For questions or issues, please open an issue on GitHub.

