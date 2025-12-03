"""
Tests for model architectures.
"""

import pytest
import torch
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nfl_bdb.models import STGNNRefine, GraphFeatures


def test_stgnn_refine_initialization():
    """Test STGNNRefine model initialization."""
    model = STGNNRefine(
        node_dim=13,
        hidden=256,
        graph_layers=3,
        temporal_layers=4,
        heads=8,
        max_len=100,
        dropout=0.1
    )
    assert model is not None
    assert model.hidden == 256
    assert model.max_len == 100


def test_stgnn_refine_forward_training():
    """Test STGNNRefine forward pass in training mode."""
    model = STGNNRefine(node_dim=13, hidden=64, max_len=100)
    model.eval()
    
    # Create dummy graph
    batch_size = 2
    num_nodes = 22
    num_edges = 100
    
    graph = GraphFeatures(
        node_features=torch.randn(batch_size * num_nodes, 13),
        edge_index=torch.randint(0, batch_size * num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 4),
        batch=torch.cat([torch.zeros(num_nodes), torch.ones(num_nodes)]).long()
    )
    
    # Forward pass (training mode - no initial_pos)
    with torch.no_grad():
        output = model(graph, initial_pos=None)
    
    assert output.shape == (batch_size, 100, 2)  # [B, max_len, 2]


def test_stgnn_refine_forward_inference():
    """Test STGNNRefine forward pass in inference mode."""
    model = STGNNRefine(node_dim=13, hidden=64, max_len=100)
    model.eval()
    
    # Create dummy graph
    batch_size = 2
    num_nodes = 22
    num_edges = 100
    
    graph = GraphFeatures(
        node_features=torch.randn(batch_size * num_nodes, 13),
        edge_index=torch.randint(0, batch_size * num_nodes, (2, num_edges)),
        edge_attr=torch.randn(num_edges, 4),
        batch=torch.cat([torch.zeros(num_nodes), torch.ones(num_nodes)]).long()
    )
    
    # Forward pass (inference mode - with initial_pos)
    initial_pos = torch.randn(batch_size, 2)  # [B, 2]
    
    with torch.no_grad():
        output = model(graph, initial_pos=initial_pos)
    
    assert output.shape == (batch_size, 100, 2)  # [B, max_len, 2]

