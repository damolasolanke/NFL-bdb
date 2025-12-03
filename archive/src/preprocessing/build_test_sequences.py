import pandas as pd
import numpy as np
import pickle
from pathlib import Path

DATA_DIR = Path("data/test_sample")
OUT_PATH = Path("data/processed/test_sequences.pkl")

def load_data():
    test_df = pd.read_csv(DATA_DIR / "test_input.csv")
    meta_df = pd.read_csv(DATA_DIR / "test.csv")
    return test_df, meta_df

def build_sequences(test_df, meta_df):
    """
    Build test sequences that match the exact structure of training sequences.pkl:
    
    (game_id, play_id, nfl_id) → {
        "input": pandas.Series,
        "target": np.ndarray[T,2]  (None for test)
    }
    """

    sequences = {}

    # Group by player for final-frame extraction
    grouped = test_df.sort_values(["game_id", "play_id", "nfl_id", "frame_id"]) \
                     .groupby(["game_id", "play_id", "nfl_id"])

    for (gid, pid, nid), df in grouped:
        last_row = df.iloc[-1]

        key = (np.int64(gid), np.int64(pid), np.int64(nid))

        # No target trajectory exists for test set → use None
        sequences[key] = {
            "input": last_row,
            "target": None
        }

    return sequences

def main():
    print("[+] Loading test data…")
    test_df, meta_df = load_data()
    print(f"[✓] Loaded: test_input.csv ({len(test_df)} rows), test.csv ({len(meta_df)} rows)")

    print("[+] Building sequences…")
    sequences = build_sequences(test_df, meta_df)
    print(f"[✓] Built {len(sequences)} sequences")

    print("[+] Saving:", OUT_PATH)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(sequences, open(OUT_PATH, "wb"))

    print("[✓] Done — test_sequences.pkl created successfully.")

if __name__ == "__main__":
    main()
