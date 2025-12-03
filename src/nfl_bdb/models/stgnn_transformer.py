"""
Spatial-Temporal Graph Neural Network + Refinement Variant

Includes:
- NodeEncoder
- EdgeEncoder
- GraphAttentionLayer
- TemporalTransformerEncoder
- RefinementBlock
- STGNNRefine  (required to load refine checkpoint)
- STGNNTransformer (your new model)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# =====================================================================
# Node Encoder
# =====================================================================
class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.encoder(x)


# =====================================================================
# Edge Encoder
# =====================================================================
class EdgeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, edge_attr):
        return self.encoder(edge_attr)


# =====================================================================
# Graph Attention Layer (GATv2-style)
# =====================================================================
class GraphAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)
        self.W_o = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def _scatter_sum(self, src, index, dim_size):
        out = torch.zeros(dim_size, *src.shape[1:], device=src.device)
        return out.index_add_(0, index, src)

    def forward(self, x, edge_index):
        N = x.shape[0]
        E = edge_index.shape[1]
        row, col = edge_index

        Q = self.W_q(x).view(N, self.num_heads, self.head_dim)
        K = self.W_k(x).view(N, self.num_heads, self.head_dim)
        V = self.W_v(x).view(N, self.num_heads, self.head_dim)

        Q_i = Q[row]
        K_j = K[col]
        V_j = V[col]

        scores = (Q_i * K_j).sum(-1) / math.sqrt(self.head_dim)
        attn = scores.softmax(dim=0).unsqueeze(-1)

        out = self._scatter_sum(attn * V_j, row, N)
        out = out.reshape(N, self.hidden_dim)
        out = self.W_o(out)

        return self.norm(x + out)


# =====================================================================
# Temporal Transformer Encoder
# =====================================================================
class TemporalTransformer(nn.Module):
    def __init__(self, dim, layers=4, heads=4, dropout=0.1, max_len=100):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x):
        return self.encoder(x + self.pos[:, : x.size(1), :])


# =====================================================================
# Refinement Block
# =====================================================================
class RefinementBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.refine = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return x + self.refine(x)


# =====================================================================
# RESTORED: STGNNRefine (needed for checkpoint)
# =====================================================================
class STGNNRefine(nn.Module):
    def __init__(
        self,
        node_dim=13,
        hidden=256,
        graph_layers=3,
        temporal_layers=4,
        heads=4,
        max_len=100,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden = hidden
        self.max_len = max_len

        self.node_encoder = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU()
        )

        self.gnn_layers = nn.ModuleList([
            GraphAttentionLayer(hidden, num_heads=heads, dropout=dropout)
            for _ in range(graph_layers)
        ])

        self.temporal = TemporalTransformer(
            dim=hidden,
            layers=temporal_layers,
            heads=heads,
            dropout=dropout,
            max_len=max_len
        )

        self.refine = RefinementBlock(hidden)

        self.head = nn.Sequential(
            nn.Linear(hidden, hidden * 2),
            nn.GELU(),
            nn.Linear(hidden * 2, 2),
        )

    def forward(self, graph, initial_pos=None):
        x = self.node_encoder(graph.x)

        for layer in self.gnn_layers:
            x = layer(x, graph.edge_index)

        batch_ids = graph.batch
        B = int(batch_ids.max().item()) + 1
        target_nodes = [ (batch_ids == b).nonzero()[0].item() for b in range(B) ]
        target_nodes = torch.tensor(target_nodes, device=x.device)

        z = x[target_nodes].unsqueeze(1).repeat(1, self.max_len, 1)

        z = self.temporal(z)
        z = self.refine(z)

        deltas = self.head(z)

        if initial_pos is not None:
            return deltas.cumsum(1) + initial_pos.unsqueeze(1)

        return deltas


# =====================================================================
# STGNNTransformer — your current model
# =====================================================================
class STGNNTransformer(nn.Module):
    def __init__(self, node_dim=13, edge_dim=4, hidden=256, heads=4):
        super().__init__()
        self.node_encoder = NodeEncoder(node_dim, hidden)
        self.edge_encoder = EdgeEncoder(edge_dim, hidden)
        self.gnn = GraphAttentionLayer(hidden, heads)
        self.temporal = TemporalTransformer(hidden)
        self.head = nn.Linear(hidden, 2)

    def forward(self, graph):
        x = self.node_encoder(graph.x)
        e = self.edge_encoder(graph.edge_attr)
        x = self.gnn(x, graph.edge_index)
        x = self.temporal(x.unsqueeze(0)).squeeze(0)
        return self.head(x)
