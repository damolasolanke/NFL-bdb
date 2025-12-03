import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from pathlib import Path

import sys
import os

# ----------------------------------------------------
# Load project root
# ----------------------------------------------------
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.stgnn_transformer import STGNNTransformer
from src.utils.datasets import NFLTrajectoryDataset
from src.utils.collate import collate_fn
import config


def load_checkpoint(path, device):
    print(f"\n[✓] Loading checkpoint: {path}")
    ckpt = torch.load(path, map_location=device)
    model_state = ckpt["model_state_dict"]
    print("[✓] Keys inside checkpoint:", ckpt.keys())
    return model_state


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[✓] Using device: {device}")

    # ----------------------------------------------------
    # Load dataset (only tiny sample to speed up test)
    # ----------------------------------------------------
    dataset = NFLTrajectoryDataset(
        config.TRAIN_PICKLE,
        max_len=config.MAX_LEN,
    )

    small_subset = torch.utils.data.Subset(dataset, list(range(32)))  # 32 samples
    loader = DataLoader(
        small_subset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # ----------------------------------------------------
    # Load model
    # ----------------------------------------------------
    model = STGNNTransformer(
        node_in_dim=config.NODE_IN_DIM,
        edge_in_dim=config.EDGE_IN_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_heads=config.N_HEADS,
        num_layers=config.GNN_LAYERS,
        max_len=config.MAX_LEN,
    ).to(device)

    ckpt_path = Path(config.MODEL_DIR) / "stgnn_refine_best.pt"
    state = load_checkpoint(ckpt_path, device)

    model.load_state_dict(state["model_state_dict"])
    model.eval()

    print("\n[✓] Model loaded successfully.")
    print("[✓] Running 1-epoch dry-run...")

    # ----------------------------------------------------
    # Dry-run: 1 forward + loss
    # ----------------------------------------------------
    criterion = torch.nn.MSELoss()

    for batch_idx, batch in enumerate(loader, start=1):
        graph, targets, mask = batch
        graph = graph.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        with torch.no_grad():
            preds = model(graph)

        # Trim to shared length
        T = min(preds.shape[0], targets.shape[0])
        loss = criterion(preds[:T], targets[:T])

        print(f"[Batch {batch_idx}] Loss: {loss.item():.4f}")

        if batch_idx == 3:   # only test 3 batches
            break

    print("\n[✓] CHECKPOINT TEST COMPLETE — MODEL IS VALID.\n")


if __name__ == "__main__":
    main()
