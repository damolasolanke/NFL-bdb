# NFL Big Data Bowl 2026 - Project Inventory

**Generated:** 2025-01-XX  
**Project Root:** `~/nfl-bdb`  
**Purpose:** Trajectory prediction for NFL players using Spatio-Temporal Graph Neural Networks

---

## 1. Project Overview

This project implements a deep learning solution for predicting NFL player trajectories in the Big Data Bowl 2026 competition. The model uses:
- **Spatio-Temporal Graph Neural Networks (STGNN)** with Graph Attention (GAT) layers
- **Transformer** layers for temporal modeling
- **Graph-based** representation of player interactions

---

## 2. Directory Structure

```
nfl-bdb/
├── baseline/              # Baseline models (straight-line, evaluation utilities)
├── config.py             # Main configuration file (hyperparameters, paths)
├── data/                 # Data directory (gitignored)
│   ├── train/            # Training CSVs (input_*.csv, output_*.csv)
│   ├── test_sample/      # Test sample data
│   ├── processed/        # Processed sequences (sequences.pkl, test_sequences.pkl)
│   └── metadata/         # Kaggle evaluation server code
├── models/                # Saved model checkpoints (.pt files)
├── notebooks/            # Jupyter notebooks
├── outputs/              # Generated plots/images
├── preprocessing/        # Data preprocessing modules
├── scripts/              # Utility scripts
├── src/                  # Main source code
│   ├── eval/             # Evaluation code
│   ├── inference/        # Inference/prediction code
│   ├── models/           # Model architectures
│   ├── preprocessing/   # Additional preprocessing utilities
│   ├── training/         # Training scripts
│   └── utils/            # Utilities (datasets, collate, graph builder)
└── [various .md files]   # Documentation/fix guides
```

---

## 3. Core Components

### 3.1 Configuration (`config.py`)
**Status:** ✅ Canonical  
**Purpose:** Centralized configuration for all hyperparameters, paths, and training settings.

**Key Settings:**
- Model architecture: `HIDDEN_DIM=256`, `GRAPH_LAYERS=3`, `TEMPORAL_LAYERS=4`
- Training: `BATCH_SIZE=32`, `LEARNING_RATE=1e-4`, `EPOCHS=100`
- Data: `MAX_PLAYERS=22`, `MAX_FRAMES=100`, `RADIUS=20.0`

---

### 3.2 Data Preprocessing (`preprocessing/`)

**Status:** ✅ Canonical  
**Purpose:** Load, normalize, and sequence building for training data.

**Key Files:**
- `load.py` - Load input/output CSVs
- `normalize.py` - Normalize play direction (offense → right)
- `features.py` - Feature engineering
- `sequences.py` - Build sequences dictionary (input → target mapping)
- `save_sequences.py` - Save sequences to pickle

**Pipeline:**
1. Load all input/output CSVs
2. Normalize coordinates (play direction)
3. Add engineered features
4. Build sequences: `(game_id, play_id, nfl_id) → {input: Series, target: ndarray}`
5. Save to `data/processed/sequences.pkl`

---

### 3.3 Model Architecture (`src/models/`)

**Status:** ✅ Canonical  
**Purpose:** Neural network architectures.

**Files:**
- `stgnn_refine.py` - **MAIN MODEL** - STGNN with GAT + Transformer
  - Node encoder (Linear → LayerNorm → GELU)
  - Graph Attention Layers (GATv2)
  - Temporal Transformer (4 layers)
  - Output decoder (predicts velocities → positions)
- `stgnn_transformer.py` - Alternative/experimental model

**Model Flow:**
```
Input: GraphFeatures (node_feats, edge_index, edge_attr, batch)
  → Node Encoder
  → Graph Attention Layers (3x)
  → Temporal Transformer (4x)
  → Output Decoder
Output: [B, max_len, 2] velocities or positions
```

---

### 3.4 Training (`src/training/`)

**Status:** ✅ Canonical  
**Purpose:** Training loop, checkpointing, validation.

**Files:**
- `train.py` - **MAIN TRAINING SCRIPT**
  - Loads sequences from `data/processed/sequences.pkl`
  - Creates train/val split (90/10)
  - Training loop with AMP, gradient clipping
  - Saves checkpoints every 5 epochs
  - Saves best model based on validation loss
- `test_checkpoint.py` - Checkpoint loading tests
- `test_ckpt_load.py` - Additional checkpoint tests

**Training Process:**
1. Load sequences and input dataframe
2. Create `NFLTrajectoryDataset` instances
3. Initialize `STGNNRefine` model
4. Train with masked MSE loss (velocity prediction)
5. Validate and save best model

---

### 3.5 Inference (`src/inference/`)

**Status:** ✅ Canonical  
**Purpose:** Generate predictions for test data and Kaggle submission.

**Files:**
- `run_inference.py` - **MAIN INFERENCE SCRIPT**
  - Loads model checkpoint
  - Loads test sequences from `data/processed/test_sequences.pkl`
  - Creates `InferenceDataset`
  - Runs batched inference
  - Generates submission CSV
- `inference_dataset.py` - Dataset for inference (no targets)
- `predict.py` - Additional prediction utilities

**Inference Process:**
1. Load checkpoint and test sequences
2. Create `InferenceDataset` (matches training dataset contract)
3. Run model forward pass (with `initial_pos`)
4. Extract 21-step predictions
5. Format as submission CSV: `(gameId, playId, nflId, step, x, y)`

---

### 3.6 Evaluation (`src/eval/`)

**Status:** ✅ Canonical  
**Purpose:** Evaluate model on validation/test data.

**Files:**
- `run_eval.py` - **MAIN EVALUATION SCRIPT**
  - Loads sequences and model checkpoint
  - Creates `EvalDataset`
  - Computes RMSE (overall and per-step)
  - Saves metrics to `eval_metrics.json`
  - Saves predictions/targets to `.npy` files

**Metrics:**
- Overall RMSE
- Per-step RMSE (21 steps)

---

### 3.7 Utilities (`src/utils/`)

**Status:** ✅ Canonical  
**Purpose:** Shared utilities for datasets, collation, graph building.

**Files:**
- `datasets.py` - `NFLTrajectoryDataset` class (training dataset)
- `collate.py` - Collate function for batching
- `graph_builder.py` - Graph construction utilities (`build_graph`, `GraphFeatures`)

**Key Classes:**
- `NFLTrajectoryDataset` - PyTorch Dataset for training
- `GraphFeatures` - Container for graph data (node_feats, edge_index, edge_attr, batch)

---

### 3.8 Baseline Models (`baseline/`)

**Status:** ✅ Reference/Utility  
**Purpose:** Simple baseline models for comparison.

**Files:**
- `straight_line.py` - Straight-line trajectory baseline
- `evaluation.py` - Evaluation utilities
- `visualize.py` - Visualization utilities
- `utils.py` - Baseline utilities

---

### 3.9 Scripts (`scripts/`)

**Status:** ✅ Utility  
**Purpose:** Standalone utility scripts.

**Files:**
- `build_sequences.py` - CLI script to rebuild sequences.pkl
- `plot_eval.py` - Plot evaluation metrics

---

### 3.10 Notebooks (`notebooks/`)

**Status:** ⚠️ Mixed (some canonical, some experimental)  
**Purpose:** Exploratory analysis, training, debugging.

**Files:**
- `01_preprocessing_validation.ipynb` - Preprocessing validation
- `02_model_training.ipynb` - **MAIN TRAINING NOTEBOOK** (contains submission generation code)
- `02_model_training_backup.ipynb` - Backup version
- `02_model_training_recovered.ipynb` - Recovered version
- `Untitled.ipynb` - Untitled notebook

**Note:** The main notebook contains Kaggle submission generation code (parquet writer).

---

### 3.11 Root-Level Scripts

**Status:** ✅ Utility  
**Files:**
- `prepare_data.py` - Create data directory structure
- `validate_dataset_contract.py` - Validate inference dataset matches training contract

---

## 4. Data Files

### 4.1 Training Data (`data/train/`)
- `input_2023_w*.csv` - Input frames (18 weeks)
- `output_2023_w*.csv` - Output trajectories (18 weeks)

### 4.2 Processed Data (`data/processed/`)
- `sequences.pkl` - Training sequences dictionary
- `test_sequences.pkl` - Test sequences dictionary

### 4.3 Model Checkpoints (`models/`)
- `best_model.pt` - Best model (lowest validation loss)
- `stgnn_refine_best.pt` - Alternative best model checkpoint
- `checkpoint_epoch_*.pt` - Periodic checkpoints (every 5 epochs)

---

## 5. Documentation Files

**Status:** ⚠️ Historical/Fix Guides (can be archived)

**Files:**
- `AUDIT_REPORT.md` - Audit report
- `CONTRACT_VALIDATION_SUMMARY.md` - Contract validation summary
- `VALIDATION_COMPLETE.md` - Validation completion notice
- `VALIDATION_REPORT.md` - Validation report
- `SUBMISSION_FIX_GUIDE.md` - Submission fix guide
- `EVAL_*.md` - Various evaluation fix guides
- `INFERENCE_*.md` - Various inference fix guides

**Note:** These are historical documentation of fixes/debugging. Can be moved to `archive/` or `docs/history/`.

---

## 6. Canonical vs Experimental Files

### ✅ Canonical (Main Version)
- `config.py` - Main configuration
- `src/training/train.py` - Main training script
- `src/inference/run_inference.py` - Main inference script
- `src/eval/run_eval.py` - Main evaluation script
- `src/models/stgnn_refine.py` - Main model architecture
- `preprocessing/*.py` - Preprocessing pipeline
- `src/utils/*.py` - Core utilities
- `notebooks/02_model_training.ipynb` - Main training notebook

### ⚠️ Experimental/Backup
- `notebooks/02_model_training_backup.ipynb` - Backup
- `notebooks/02_model_training_recovered.ipynb` - Recovered version
- `notebooks/Untitled.ipynb` - Untitled notebook
- `src/models/stgnn_transformer.py` - Alternative model
- `src/training/test_*.py` - Test scripts

### 📦 Output Files (gitignored)
- `models/*.pt` - Model checkpoints
- `data/processed/*.pkl` - Processed data
- `eval_metrics.json`, `eval_preds.npy`, `eval_targets.npy` - Evaluation outputs
- `submission.csv`, `submission.zip` - Submission files
- `outputs/*.png` - Generated plots

---

## 7. Missing Components (To Add)

1. **Dependencies:** No `requirements.txt` or `pyproject.toml`
2. **CLI Entrypoints:** No standardized CLI commands (train/infer/eval)
3. **Tests:** No unit tests or integration tests
4. **Git Setup:** No `.gitignore`, no git repo initialized
5. **Documentation:** No README.md, no API docs
6. **Environment:** No `.env` or environment setup guide
7. **Config Management:** Single `config.py` (could use YAML/JSON)

---

## 8. Entry Points Summary

### Current Entry Points:
1. **Training:** `python src/training/train.py`
2. **Inference:** `python src/inference/run_inference.py`
3. **Evaluation:** `python src/eval/run_eval.py`
4. **Build Sequences:** `python scripts/build_sequences.py`
5. **Notebook:** `jupyter notebook notebooks/02_model_training.ipynb`

### Desired Entry Points (To Create):
1. `nfl-bdb train` - Training CLI
2. `nfl-bdb infer` - Inference CLI
3. `nfl-bdb eval` - Evaluation CLI
4. `nfl-bdb preprocess` - Preprocessing CLI

---

## 9. Key Dependencies (Inferred)

Based on code analysis:
- `torch` (PyTorch)
- `torch-geometric` (or custom GAT implementation)
- `pandas`
- `numpy`
- `tqdm`
- `pyarrow` (for parquet submission)
- `jupyter` (for notebooks)

---

## 10. Next Steps

1. ✅ **COMPLETE:** Project discovery and inventory
2. ⏭️ **NEXT:** Design target repo structure
3. ⏭️ **NEXT:** Refactor code into clean structure
4. ⏭️ **NEXT:** Add CLI entrypoints
5. ⏭️ **NEXT:** Add tests
6. ⏭️ **NEXT:** Add dependencies and config files
7. ⏭️ **NEXT:** Add documentation
8. ⏭️ **NEXT:** Setup Git and prepare GitHub push

---

**End of Inventory**

