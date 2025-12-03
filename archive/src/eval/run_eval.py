import os
import json
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from src.models.stgnn_refine import STGNNRefine
from src.utils.graph_builder import build_graph, GraphFeatures


# -------------------------------------------------------------
# Dataset for evaluation (same as training, but only for eval)
# -------------------------------------------------------------
class EvalDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, max_players=22, max_T=22):
        self.seq_keys = list(sequences.keys())
        self.sequences = sequences
        self.max_players = max_players
        self.max_T = max_T  # 1 initial + 21 predictions

    def __len__(self):
        return len(self.seq_keys)

    def __getitem__(self, idx):
        key = self.seq_keys[idx]
        item = self.sequences[key]

        input_df = item["input"].copy()
        target = item["target"]  # shape (21, 2)

        # --- Build node features (numeric only) ---
        # Ensure input_df is a DataFrame during type filtering
        if isinstance(input_df, pd.Series):
            df_numeric = input_df.to_frame().T
        else:
            df_numeric = input_df
        
        numeric_cols = df_numeric.select_dtypes(
            include=["number", "bool"]
        ).columns.tolist()
        df_numeric = df_numeric[numeric_cols].fillna(0)
        
        # Ensure input_df is DataFrame for subsequent operations
        if isinstance(input_df, pd.Series):
            input_df = input_df.to_frame().T

        # Pad to max_players
        node_feats = np.zeros((self.max_players, df_numeric.shape[1]), dtype=np.float32)
        num_players = min(len(df_numeric), self.max_players)
        node_feats[:num_players] = df_numeric.values[:num_players]

        # Build graph - match training dataset call exactly
        all_players = input_df.to_dict("records")
        all_players = all_players[:num_players]
        
        # Get target player row (same as training: input_row.to_dict())
        player_row = input_df[input_df["player_to_predict"] == True].iloc[0]
        input_row = player_row.to_dict()
        
        # Convert target to tensor first
        target_tensor = torch.tensor(target, dtype=torch.float32)
        
        # Extract initial position sequence: first position + full future sequence length
        # Shape should be [T, 2] where T = pred_steps (21)
        # Based on user instruction: "first position from the sequence + the full future sequence length"
        # This means: initial_pos[0] = start, initial_pos[1:] = target[:T-1] to get [T, 2] total
        pred_steps = 21
        start_pos = torch.tensor([input_row["x"], input_row["y"]], dtype=torch.float32)  # [2]
        start_pos = start_pos.unsqueeze(0)  # [1, 2]
        
        # Construct initial_pos: [T, 2] = start_pos[0] + target[0:T-1]
        if target_tensor.shape[0] >= pred_steps - 1:
            initial_pos = torch.cat([start_pos, target_tensor[:pred_steps-1]], dim=0)  # [T, 2]
        else:
            # Pad if target is shorter
            remaining = pred_steps - 1 - target_tensor.shape[0]
            pad = target_tensor[-1:].repeat(remaining, 1) if target_tensor.shape[0] > 0 else start_pos.repeat(remaining, 1)
            initial_pos = torch.cat([start_pos, target_tensor, pad], dim=0)  # [T, 2]
        
        # Build graph exactly like training
        graph = build_graph(
            input_row=input_row,               # dict for target player
            all_players=all_players[:num_players],
            max_distance=20.0,
            include_self_loops=True
        )
        
        # Ensure tensors are safe for batching
        graph = graph.contiguous()
        
        # ---- Ensure target is always 21 steps (same as training) ----
        MAX_STEPS = 21
        num_steps = target_tensor.shape[0]
        if num_steps < MAX_STEPS:
            pad_amount = MAX_STEPS - num_steps
            pad = torch.zeros((pad_amount, 2), dtype=target_tensor.dtype)
            target_tensor = torch.cat([target_tensor, pad], dim=0)
        elif num_steps > MAX_STEPS:
            # Truncate extra steps (should be rare)
            target_tensor = target_tensor[:MAX_STEPS]
        # --------------------------------------------------------------

        return {
            "graph": graph,                    # full GraphFeatures object
            "initial_pos": initial_pos,        # shape (T, 2) where T=21
            "target": target_tensor,           # shape (21, 2)
        }


# -------------------------------------------------------------
# Collate Function
# -------------------------------------------------------------
def eval_collate(batch):
    graphs = [b["graph"] for b in batch]
    initial_pos = torch.stack([b["initial_pos"] for b in batch])  # [B, T, 2] where T=21
    # Do NOT squeeze - keep shape [B, T, 2]
    targets = torch.stack([b["target"] for b in batch])  # [B, 21, 2]
    
    # Batch graphs into a single GraphFeatures object
    B = len(graphs)
    N = graphs[0].node_features.shape[0]  # nodes per graph
    
    # Concatenate node features
    batched_node_features = torch.cat([g.node_features for g in graphs], dim=0)  # [B*N, F]
    
    # Batch edge indices with offsets
    batched_edge_index_list = []
    batched_edge_attr_list = []
    node_offset = 0
    
    for g in graphs:
        offset_edge_index = g.edge_index + node_offset
        batched_edge_index_list.append(offset_edge_index)
        batched_edge_attr_list.append(g.edge_attr)
        node_offset += N
    
    batched_edge_index = torch.cat(batched_edge_index_list, dim=1)  # [2, total_edges]
    batched_edge_attr = torch.cat(batched_edge_attr_list, dim=0)  # [total_edges, 4]
    
    # Create batch assignment vector
    batch_index = torch.cat([torch.full((N,), i, dtype=torch.long) for i in range(B)], dim=0)  # [B*N]
    
    # Create batched graph
    batched_graph = GraphFeatures(
        node_features=batched_node_features,
        edge_index=batched_edge_index,
        edge_attr=batched_edge_attr,
        batch=batch_index
    )
    
    return batched_graph, initial_pos, targets


# -------------------------------------------------------------
# RMSE Computation
# -------------------------------------------------------------
def compute_rmse(pred, target):
    mse = ((pred - target) ** 2).mean()
    return float(torch.sqrt(mse).item())


def compute_step_rmse(pred, target):
    rmse_per_step = []
    for t in range(21):
        mse_t = ((pred[:, t] - target[:, t]) ** 2).mean()
        rmse_per_step.append(float(torch.sqrt(mse_t).item()))
    return rmse_per_step


# -------------------------------------------------------------
# Evaluation Logic
# -------------------------------------------------------------
def run_eval(
    ckpt_path="models/best_model.pt",
    seq_path="data/processed/sequences.pkl",
    batch_size=64,
    device="cuda"
):
    print("[+] Loading sequences:", seq_path)
    sequences = pickle.load(open(seq_path, "rb"))

    print("[+] Loading model:", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location=device)

    model = STGNNRefine()
    # Handle both dict format and direct state_dict format
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    print("[+] Building EvalDataset...")
    ds = EvalDataset(sequences)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=eval_collate)

    all_preds = []
    all_targets = []

    print("[+] Running local evaluation...")
    with torch.no_grad():
        for graph, initial_pos, targets in tqdm(dl):
            graph = graph.to(device)
            initial_pos = initial_pos.to(device)  # [B, T, 2] where T=21
            
            # initial_pos is already [B, T, 2] - do NOT squeeze
            preds = model(graph, initial_pos)  # [B, max_len, 2] where max_len=100
            
            # Extract only first 21 steps (matching target shape) - use clone to ensure it's a new tensor
            preds = preds[:, :21, :].clone()  # [B, 21, 2]
            
            # Verify shapes match before appending
            if preds.shape[1] != 21:
                raise RuntimeError(f"Preds shape mismatch: expected [B, 21, 2], got {preds.shape}")
            if targets.shape[1] != 21:
                raise RuntimeError(f"Targets shape mismatch: expected [B, 21, 2], got {targets.shape}")

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    preds = torch.cat(all_preds, dim=0)  # [total_samples, 21, 2]
    targets = torch.cat(all_targets, dim=0)  # [total_samples, 21, 2]
    
    # Final shape verification
    assert preds.shape == targets.shape, f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}"

    # Compute metrics
    overall_rmse = compute_rmse(preds, targets)
    step_rmse = compute_step_rmse(preds, targets)

    print("\n[=== Evaluation Results ===]")
    print(f"Overall RMSE: {overall_rmse:.5f}")
    for i, r in enumerate(step_rmse):
        print(f"Step {i}: {r:.5f}")

    # Save metrics
    metrics = {
        "overall_rmse": overall_rmse,
        "step_rmse": step_rmse,
    }
    json.dump(metrics, open("eval_metrics.json", "w"), indent=4)
    print("[+] Saved eval_metrics.json")

    # Save predictions for further analysis
    np.save("eval_preds.npy", preds.numpy())
    np.save("eval_targets.npy", targets.numpy())
    print("[+] Saved eval_preds.npy and eval_targets.npy")

    return overall_rmse


if __name__ == "__main__":
    run_eval()
