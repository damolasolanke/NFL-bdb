import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# -----------------------------
# Node Encoder (EXACT checkpoint architecture)
# -----------------------------
def build_node_encoder(input_dim, hidden_dim):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU()
    )


# -----------------------------
# Graph Attention Layer (GATv2)
# -----------------------------
class GATLayer(nn.Module):
    def __init__(self, hidden_dim, heads=4, dropout=0.1):
        super().__init__()
        self.hidden = hidden_dim
        self.heads = heads
        self.d_k = hidden_dim // heads

        self.Wq = nn.Linear(hidden_dim, hidden_dim)
        self.Wk = nn.Linear(hidden_dim, hidden_dim)
        self.Wv = nn.Linear(hidden_dim, hidden_dim)

        self.attn_dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        row, col = edge_index

        Q = self.Wq(x).view(N, self.heads, self.d_k)
        K = self.Wk(x).view(N, self.heads, self.d_k)
        V = self.Wv(x).view(N, self.heads, self.d_k)

        Q_i = Q[row]
        K_j = K[col]
        V_j = V[col]

        scores = (Q_i * K_j).sum(dim=-1) / math.sqrt(self.d_k)

        attn = torch.zeros_like(scores)
        for h in range(self.heads):
            attn[:, h] = torch.softmax(scores[:, h], dim=0)

        attn = self.attn_dropout(attn)

        messages = attn.unsqueeze(-1) * V_j

        out = torch.zeros(N, self.heads, self.d_k, device=x.device)
        out.index_add_(0, row, messages)

        out = out.reshape(N, self.hidden)
        out = self.out(out)

        return self.norm(x + out)


# -----------------------------
# Temporal Transformer (checkpoint-aligned)
# -----------------------------
class TemporalTransformer(nn.Module):
    def __init__(self, dim=256, layers=4, heads=4, dropout=0.1, max_len=100):
        super().__init__()

        self.pos = nn.Parameter(torch.zeros(max_len, dim))

        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=heads,
                dim_feedforward=dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True
            )
            for _ in range(layers)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, T, D = x.shape
        x = x + self.pos[:T]

        out = x
        for layer in self.layers:
            out = layer(out)

        return self.norm(out)


# -----------------------------
# Refinement block
# -----------------------------
class RefinementBlock(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        return self.norm(x + self.ff(x))


# -----------------------------
# STGNNRefine — FINAL VERSION
# -----------------------------
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

        # EXACT checkpoint architecture:
        self.node_encoder = build_node_encoder(node_dim, hidden)

        self.gnn_layers = nn.ModuleList([
            GATLayer(hidden, heads=heads, dropout=dropout)
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

        target_nodes = []
        for b in range(B):
            idx = (batch_ids == b).nonzero(as_tuple=False).flatten()
            target_nodes.append(idx[0])
        target_nodes = torch.stack(target_nodes)

        z = x[target_nodes]

        z = z.unsqueeze(1).repeat(1, self.max_len, 1)

        z = self.temporal(z)
        z = self.refine(z)

        vel = self.head(z)  # predicted velocities [B, T, 2]

        # training: return raw velocity predictions
        if initial_pos is None:
            return vel

        # inference: integrate velocities into positions
        # normalize shapes
        if initial_pos.ndim == 1:
            initial_pos = initial_pos.unsqueeze(0).unsqueeze(0)
        elif initial_pos.ndim == 2:
            initial_pos = initial_pos.unsqueeze(1)

        start = initial_pos[:, 0, :]     # [B, 2]
        pos = start.unsqueeze(1) + vel.cumsum(dim=1)
        return pos