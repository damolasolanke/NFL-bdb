import argparse
from pathlib import Path

import torch


def inspect_checkpoint(ckpt_path: Path):
    print(f"[+] Loading checkpoint from: {ckpt_path}")

    obj = torch.load(ckpt_path, map_location="cpu")
    print(f"[✓] Loaded object of type: {type(obj)}")

    # Case 1: standard training checkpoint dict
    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"[+] Top-level keys ({len(keys)}): {keys}")

        # If it has a model_state_dict, inspect it
        if "model_state_dict" in obj:
            state = obj["model_state_dict"]
            if isinstance(state, dict):
                print(f"[+] model_state_dict has {len(state)} tensors/entries.")
                print("[+] Sample of parameter names and shapes:")
                for i, (name, tensor) in enumerate(state.items()):
                    if hasattr(tensor, "shape"):
                        print(f"    - {name}: {tuple(tensor.shape)}")
                    else:
                        print(f"    - {name}: type={type(tensor)}")
                    if i >= 9:
                        break
            else:
                print(f"[!] model_state_dict is not a dict (type={type(state)})")

        # Optional: surface common metadata if present
        for meta_key in ["epoch", "val_rmse", "val_loss", "history"]:
            if meta_key in obj:
                print(f"[+] {meta_key}: {obj[meta_key]}")
    else:
        # Case 2: direct model object or something else
        print("[!] Checkpoint is not a dict; might be a raw model object.")
        attrs = [a for a in dir(obj) if not a.startswith("_")]
        print(f"[+] Available attributes on loaded object (truncated): {attrs[:20]}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect a PyTorch checkpoint (.pt) without needing the full training stack."
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint file (e.g., models/stgnn_refine_best.pt)",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).expanduser().resolve()
    if not ckpt_path.is_file():
        print(f"[!] Checkpoint file not found: {ckpt_path}")
        raise SystemExit(1)

    inspect_checkpoint(ckpt_path)


if __name__ == "__main__":
    main()
