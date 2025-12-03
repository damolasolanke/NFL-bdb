"""
Tests for utility functions.
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from nfl_bdb.utils import GraphFeatures, build_graph


def test_graph_features():
    """Test GraphFeatures dataclass."""
    node_features = torch.randn(10, 13)
    edge_index = torch.randint(0, 10, (2, 20))
    edge_attr = torch.randn(20, 4)
    batch = torch.zeros(10, dtype=torch.long)
    
    graph = GraphFeatures(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch
    )
    
    assert graph.node_features.shape == (10, 13)
    assert graph.edge_index.shape == (2, 20)
    assert graph.edge_attr.shape == (20, 4)
    assert graph.batch.shape == (10,)
    assert graph.x.shape == (10, 13)  # Test property alias


def test_graph_features_to_device():
    """Test GraphFeatures.to() method."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    graph = GraphFeatures(
        node_features=torch.randn(10, 13),
        edge_index=torch.randint(0, 10, (2, 20)),
        edge_attr=torch.randn(20, 4),
        batch=torch.zeros(10, dtype=torch.long)
    )
    
    graph_gpu = graph.to(torch.device("cuda"))
    
    assert graph_gpu.node_features.device.type == "cuda"
    assert graph_gpu.edge_index.device.type == "cuda"
    assert graph_gpu.edge_attr.device.type == "cuda"
    assert graph_gpu.batch.device.type == "cuda"


def test_build_graph():
    """Test build_graph function."""
    input_row = {
        'x': 10.0,
        'y': 20.0,
        'vx': 1.0,
        'vy': 2.0,
        's': 2.24,
        'a': 0.5,
        'dist_ball': 5.0,
        'dx_ball': 3.0,
        'dy_ball': 4.0,
        'player_role_Targeted Receiver': 1.0,
        'player_role_Defensive Coverage': 0.0,
        'player_side_Offense': 1.0,
        'player_side_Defense': 0.0,
    }
    
    all_players = [input_row.copy() for _ in range(5)]
    
    graph = build_graph(
        input_row=input_row,
        all_players=all_players,
        max_distance=20.0,
        include_self_loops=True
    )
    
    assert isinstance(graph, GraphFeatures)
    assert graph.node_features.shape[0] == 5
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_attr.shape[1] == 4

