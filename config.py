"""
Configuration file for NFL Big Data Bowl 2026 project.

This file is a compatibility wrapper that imports from configs/default.py.
For new code, import directly from configs.default.
"""

# Import all config from the new location
from configs.default import *

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_SAMPLE_DIR = DATA_DIR / "test_sample"
METADATA_DIR = DATA_DIR / "metadata"
PROCESSED_DIR = DATA_DIR / "processed"
DATA_PATH = str(DATA_DIR)

# ============================================================================
# MODEL ARCHITECTURE HYPERPARAMETERS
# ============================================================================

# Graph structure
MAX_PLAYERS = 22  # Maximum number of players in a graph
RADIUS = 20.0  # Maximum distance (yards) for edge connections in spatial graph

# Node features
NODE_DIM = 13  # Input dimension for node features (after numeric filtering)
EDGE_DIM = 4  # Edge feature dimension (distance, angle, rel_vx, rel_vy)

# Model dimensions
HIDDEN_DIM = 256  # Hidden dimension throughout the model
GRAPH_LAYERS = 3  # Number of graph attention layers
TEMPORAL_LAYERS = 4  # Number of temporal transformer layers
HEADS = 8  # Number of attention heads
DROPOUT = 0.1  # Dropout rate

# Temporal sequence
MAX_FRAMES = 100  # Maximum trajectory length (frames)
MAX_TRAJECTORY_LENGTH = MAX_FRAMES  # Alias for compatibility

# ============================================================================
# TRAINING HYPERPARAMETERS
# ============================================================================

BATCH_SIZE = 32  # Batch size
LEARNING_RATE = 1e-4  # Learning rate
LR = LEARNING_RATE  # Alias
EPOCHS = 100  # Number of training epochs
GRAD_CLIP_NORM = 1.0  # Gradient clipping norm
WEIGHT_DECAY = 1e-5  # Weight decay for optimizer

# Training settings
USE_AMP = True  # Use Automatic Mixed Precision
DEVICE = "cuda:0"  # Default device (will auto-detect GPU1 if available)

# ============================================================================
# DATA PROCESSING
# ============================================================================

# Feature engineering
INCLUDE_SELF_LOOPS = True  # Include self-loops in graph construction

# ============================================================================
# LOGGING & CHECKPOINTING
# ============================================================================

LOG_INTERVAL = 10  # Log every N batches
CHECKPOINT_INTERVAL = 5  # Save checkpoint every N epochs
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# ============================================================================
# VALIDATION
# ============================================================================

VALIDATION_SPLIT = 0.1  # Fraction of data to use for validation
