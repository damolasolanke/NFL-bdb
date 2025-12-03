# DIFF BLOCK: InferenceDataset to Match NFLTrajectoryDataset Contract

## CRITICAL MISMATCHES IDENTIFIED

### 1. MISSING KEY: "trajectory_length"
**Location:** Return dictionary (line ~200)

**Training Contract:**
```python
result = {
    ...
    "trajectory_length": T  # int - actual trajectory length
}
```

**Current Inference:**
```python
result = {
    ...
    # MISSING "trajectory_length"
}
```

**Fix:**
```diff
        result = {
            "node_feats": node_feats,  # [max_players, F]
            "edge_index": graph.edge_index,  # [2, E]
            "edge_attr": graph.edge_attr,  # [E, 4]
            "initial_pos": torch.tensor(padded_initial, dtype=torch.float32),  # [max_T, 2]
            "mask": torch.tensor(mask, dtype=torch.bool),  # [max_T-1]
            "key": key,
+           "trajectory_length": 22,  # For inference: 1 initial + 21 predictions
        }
```

---

### 2. GRAPH CONSTRUCTION: Should use `all_players[:N]` to match training exactly
**Location:** Graph building (line ~172-177)

**Training Contract:**
```python
graph = build_graph(
    input_row.to_dict(),
    all_players[:N],  # Use actual number of players (before padding)
    max_distance=self.radius,
    include_self_loops=True
)
```
Where N = max_players (after padding)

**Current Inference:**
```python
graph = build_graph(
    input_row.to_dict(),
    all_players,  # Use all actual players (before padding)
    max_distance=self.radius,
    include_self_loops=True
)
```

**Fix:**
```diff
        # ====================================================================
        # 4. Build graph (edge_index, edge_attr)
        # ====================================================================
-       # Use actual number of players (before padding) for graph construction
-       actual_num_players = len(all_players)
        graph = build_graph(
            input_row.to_dict(),
-           all_players,  # Use all actual players (before padding)
+           all_players[:N],  # Use actual number of players (before padding) - matches training
            max_distance=self.radius,
            include_self_loops=True
        )
```

**Note:** Since len(all_players) <= max_players and N = max_players after padding, `all_players[:N]` is equivalent to `all_players[:]`, but we match training exactly.

---

### 3. INITIAL_POS PADDING: Should match training padding pattern exactly
**Location:** Initial position extraction and padding (lines ~80-91)

**Training Contract:**
```python
# Extract initial position (first frame)
initial_pos = target_trajectory[0:1].copy()  # [1, 2] - first position

# Pad initial_pos to [max_T, 2]
padded_initial = np.zeros((self.max_trajectory_length, 2), dtype=np.float32)
padded_initial[0] = initial_pos[0]  # First position
# Repeat first position for padding (or use last valid)
if T > 1:
    padded_initial[1:T] = target_trajectory[1:T]
    padded_initial[T:] = target_trajectory[-1]  # Repeat last position
```

**Current Inference:**
```python
last_x = float(input_row.get('x', 0.0))
last_y = float(input_row.get('y', 0.0))
initial_pos = np.array([[last_x, last_y]], dtype=np.float32)  # [1, 2]

# Pad initial_pos to [max_T, 2] by repeating the last position
padded_initial = np.zeros((self.max_trajectory_length, 2), dtype=np.float32)
padded_initial[:] = initial_pos[0]  # Repeat last position
```

**Fix:**
```diff
        # ====================================================================
        # 1. Extract initial position (last observed frame)
        # ====================================================================
        # For inference, initial_pos is the last observed position
        # We need to predict 21 steps (t+1 to t+21), so we pad to max_trajectory_length
        last_x = float(input_row.get('x', 0.0))
        last_y = float(input_row.get('y', 0.0))
        initial_pos = np.array([[last_x, last_y]], dtype=np.float32)  # [1, 2]
        
-       # Pad initial_pos to [max_T, 2] by repeating the last position
+       # Pad initial_pos to [max_T, 2] - match training padding pattern exactly
        padded_initial = np.zeros((self.max_trajectory_length, 2), dtype=np.float32)
-       padded_initial[:] = initial_pos[0]  # Repeat last position
+       padded_initial[0] = initial_pos[0]  # First position
+       # For inference, we don't have a trajectory, so repeat the last observed position
+       # This matches training's pattern: padded_initial[T:] = target_trajectory[-1]
+       padded_initial[1:] = initial_pos[0]  # Repeat last position for all remaining frames
```

**Note:** This matches training's structure: `padded_initial[0] = initial_pos[0]`, then fill the rest. For inference, we fill with the same value since we don't have a trajectory.

---

### 4. MASK PADDING: Should match training padding pattern exactly
**Location:** Mask creation (line ~95)

**Training Contract:**
```python
# Create mask (all True since we truncated/padded)
mask = np.ones(T - 1, dtype=bool)  # [T-1]

# Pad mask to [max_T-1]
padded_mask = np.zeros(self.max_trajectory_length - 1, dtype=bool)
padded_mask[:T-1] = mask
```

**Current Inference:**
```python
# Create mask (all ones for inference - we predict all 21 steps)
# mask length is max_T - 1 (for steps 1 to max_T-1)
mask = np.ones(self.max_trajectory_length - 1, dtype=bool)  # [max_T-1]
```

**Fix:**
```diff
        # Create mask (all ones for inference - we predict all 21 steps)
-       # mask length is max_T - 1 (for steps 1 to max_T-1)
-       mask = np.ones(self.max_trajectory_length - 1, dtype=bool)  # [max_T-1]
+       # For inference: T = 22 (1 initial + 21 predictions)
+       T = 22
+       mask = np.ones(T - 1, dtype=bool)  # [21] - matches training pattern
+       
+       # Pad mask to [max_T-1] - match training padding pattern exactly
+       padded_mask = np.zeros(self.max_trajectory_length - 1, dtype=bool)
+       padded_mask[:T-1] = mask
```

And update the return statement:
```diff
        result = {
            "node_feats": node_feats,  # [max_players, F]
            "edge_index": graph.edge_index,  # [2, E]
            "edge_attr": graph.edge_attr,  # [E, 4]
            "initial_pos": torch.tensor(padded_initial, dtype=torch.float32),  # [max_T, 2]
-           "mask": torch.tensor(mask, dtype=torch.bool),  # [max_T-1]
+           "mask": torch.tensor(padded_mask, dtype=torch.bool),  # [max_T-1]
            "key": key,
            "trajectory_length": 22,  # For inference: 1 initial + 21 predictions
        }
```

---

## SUMMARY OF CHANGES

1. ✅ Add `"trajectory_length": 22` to return dictionary
2. ✅ Change `all_players` to `all_players[:N]` in build_graph call
3. ✅ Change `padded_initial[:] = initial_pos[0]` to `padded_initial[0] = initial_pos[0]; padded_initial[1:] = initial_pos[0]`
4. ✅ Change mask creation to match training pattern: create mask of length T-1, then pad to max_T-1

These changes ensure InferenceDataset EXACTLY mirrors NFLTrajectoryDataset structure, except for the absence of "target".



