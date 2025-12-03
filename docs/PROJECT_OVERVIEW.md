# NFL Big Data Bowl 2026 - Project Overview

## Introduction

This project implements a Spatio-Temporal Graph Neural Network (STGNN) for predicting NFL player trajectories in the Big Data Bowl 2026 competition. The model uses graph attention networks to capture spatial relationships between players and transformer layers to model temporal dynamics.

## Problem Statement

Given tracking data of NFL players before a pass is thrown, predict the trajectory of the targeted receiver for 21 frames after the ball is released.

## Solution Architecture

### Model Components

1. **Graph Construction**: Builds spatial graphs connecting players within a radius threshold
2. **Node Encoder**: Encodes player features (position, velocity, acceleration, etc.)
3. **Graph Attention Layers**: Captures spatial interactions between players
4. **Temporal Transformer**: Models temporal dynamics of trajectories
5. **Output Decoder**: Predicts velocities and integrates to positions

### Key Features

- **Spatial Modeling**: Graph attention captures player-to-player interactions
- **Temporal Modeling**: Transformer layers model trajectory evolution
- **Robust Preprocessing**: Normalized coordinates, feature engineering
- **Efficient Training**: Mixed precision, gradient clipping, checkpointing

## Data Pipeline

1. **Load**: Read input/output CSV files
2. **Normalize**: Standardize play direction (offense → right)
3. **Feature Engineering**: Add derived features (distances, angles, etc.)
4. **Sequence Building**: Extract (input_frame, target_trajectory) pairs
5. **Graph Construction**: Build spatial graphs for each sequence

## Training Process

1. **Data Loading**: Load sequences and input dataframes
2. **Dataset Creation**: Create train/val split (90/10)
3. **Model Initialization**: Initialize STGNN with config hyperparameters
4. **Training Loop**: 
   - Forward pass (predict velocities)
   - Loss computation (masked MSE)
   - Backward pass with gradient clipping
   - Checkpointing every N epochs
5. **Validation**: Evaluate on validation set, save best model

## Inference Process

1. **Load Model**: Load trained checkpoint
2. **Load Test Sequences**: Load test_sequences.pkl
3. **Graph Construction**: Build graphs for test data
4. **Forward Pass**: Predict trajectories
5. **Format Output**: Convert to submission format (CSV/Parquet)

## Evaluation Metrics

- **RMSE**: Root Mean Squared Error (overall and per-step)
- **Per-step Analysis**: RMSE for each of 21 prediction steps

## Technical Highlights

- **Zero PyG Dependencies**: Custom graph implementation
- **Strict Shape Contracts**: Guaranteed tensor shapes throughout pipeline
- **NaN/Inf Handling**: Robust handling of edge cases
- **Memory Efficient**: Optimized batching and data loading

## Project Status

✅ **Complete**: Full pipeline implemented and tested
✅ **Documented**: Comprehensive documentation
✅ **Reproducible**: Configurable and version-controlled

## Next Steps

- Hyperparameter tuning
- Model ensembling
- Advanced feature engineering
- Cross-validation strategies

