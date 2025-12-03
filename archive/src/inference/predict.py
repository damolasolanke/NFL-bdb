import torch
import pandas as pd
from torch_geometric.data import Data, Batch
from pathlib import Path
import numpy as np

from src.models.stgnn_refine import STGNNRefine
import config


# ============================================================
# 1. BUILD GRAPH FROM A SINGLE PLAY
# ============================================================
def build_graph_from_sequence(seq):
    """
    seq: dict with keys:
        - 'node_features': [N, 13]
        - 'edge_index': [2, E]
        - 'initial_pos': [2] (x, y) of target player
        - 'target_idx': integer (which node is the target)
    """

    x = torch.tensor(seq["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(seq["edge_index"], dtype=torch.long)
    batch = torch.zeros(x.size(0), dtype=torch.long)

    return Data(x=x, edge_index=edge_index, batch=batch), torch.tensor(seq["initial_pos"], dtype=torch.float32)


# ============================================================
# 2. LOAD MODEL
# ============================================================
def load_model(ckpt_path, device="cuda"):
    print(f"[+] Loading model from: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model_state_dict"]

    model = STGNNRefine(
        node_dim=config.NODE_FEATURES,
        hidden=config.HIDDEN_DIM,
        graph_layers=config.GNN_LAYERS,
        temporal_layers=config.TEMPORAL_LAYERS,
        heads=config.NUM_HEADS,
        max_len=config.MAX_LEN,
        dropout=config.DROPOUT
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    print("[✓] Model loaded.")
    return model


# ============================================================
# 3. RUN INFERENCE ON ONE SEQUENCE
# ============================================================
@torch.no_grad()
def predict_sequence(model, seq, device="cuda"):
    graph, initial_pos = build_graph_from_sequence(seq)

    graph = graph.to(device)
    initial_pos = initial_pos.to(device)

    pred = model(graph, initial_pos=initial_pos)   # [T, 2]
    pred = pred.squeeze(0).cpu().numpy()
    return pred


# ============================================================
# 4. BATCH INFERENCE (FOR SUBMISSION)
# ============================================================
def predict_all(model, sequences, device="cuda"):
    predictions = {}

    for uid, seq in sequences.items():
        pred = predict_sequence(model, seq, device=device)
        predictions[uid] = pred.tolist()

    return predictions


# ============================================================
# 5. MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", type=str, default="models/stgnn_refine_best.pt")
    parser.add_argument("--out", type=str, default="preds.json")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load sequences.pkl
    seq_path = config.PROCESSED_DIR / "sequences.pkl"
    print(f"[+] Loading sequences: {seq_path}")
    sequences = torch.load(seq_path)

    # Load model
    model = load_model(args.ckpt, device=device)

    # Predict all
    preds = predict_all(model, sequences, device=device)

    # Save out
    import json
    with open(args.out, "w") as f:
        json.dump(preds, f)

    print(f"[✓] Saved predictions to: {args.out}")


if __name__ == "__main__":
    main()
