import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

preds = np.load("eval_preds.npy")     # [N, T, 2]
targets = np.load("eval_targets.npy") # [N, T, 2]

diff = preds - targets                # [N, T, 2]
err = np.sqrt((diff ** 2).sum(axis=-1))  # [N, T], per-sample per-step error

per_step_rmse = err.mean(axis=0)      # [T]

plt.figure(figsize=(10,5))
plt.plot(per_step_rmse, marker='o')
plt.title("Per-step RMSE")
plt.xlabel("Timestep")
plt.ylabel("RMSE")
plt.grid(True)

# Create outputs directory if it doesn't exist
Path("outputs").mkdir(exist_ok=True)

# Save figure
plt.savefig("outputs/per_step_rmse.png", dpi=200)

