# DeepMind-Grade Contract Validation: InferenceDataset vs NFLTrajectoryDataset

## ✅ EXTRACTED CONTRACT FROM NFLTrajectoryDataset.__getitem__()

### Return Dictionary Structure:
```python
{
    "node_feats": Tensor[max_players, F],      # dtype=torch.float32
    "edge_index": Tensor[2, E],                 # dtype=torch.long
    "edge_attr": Tensor[E, 4],                  # dtype=torch.float32
    "initial_pos": Tensor[max_trajectory_length, 2],  # dtype=torch.float32
    "target": Tensor[max_trajectory_length - 1, 2],  # dtype=torch.float32 (NOT in inference)
    "mask": Tensor[max_trajectory_length - 1],        # dtype=torch.bool
    "key": Tuple[int, int, int],
    "trajectory_length": int  # ⚠️ MISSING in inference
}
```

### Key Processing Steps (in order):

1. **Trajectory Extraction** (lines 82-108)
   - Extract target_trajectory from seq['target']
   - Validate shape [T, 2]
   - Truncate if T > max_trajectory_length
   - Extract initial_pos = target_trajectory[0:1]  # [1, 2]
   - Extract target = target_trajectory[1:]  # [T-1, 2]
   - Create mask = np.ones(T - 1, dtype=bool)  # [T-1]

2. **Graph Construction** (lines 110-137)
   - Load all players from all_input_df (last frame for each)
   - Ensure target player is first
   - Limit to max_players

3. **Node Feature Extraction** (lines 139-178)
   - Convert to DataFrame
   - select_dtypes(include=["number", "bool"])
   - Drop ["frame_id", "x", "y"]
   - Convert to numpy: node_feats_np = numeric_df.values  # [N, F]
   - **Pad to max_players**: if N < max_players, pad with zeros
   - **Set N = max_players** after padding
   - Convert to tensor: torch.tensor(node_feats_np, dtype=torch.float32)

4. **Graph Building** (lines 180-200)
   - **build_graph(input_row.to_dict(), all_players[:N], ...)** where N = max_players
   - Pad graph with self-loops for nodes N to max_players-1

5. **Trajectory Padding** (lines 202-221)
   - **initial_pos padding**:
     ```python
     padded_initial = np.zeros((max_trajectory_length, 2), dtype=np.float32)
     padded_initial[0] = initial_pos[0]
     if T > 1:
         padded_initial[1:T] = target_trajectory[1:T]
     padded_initial[T:] = target_trajectory[-1]  # Repeat last
     ```
   - **mask padding**:
     ```python
     padded_mask = np.zeros(max_trajectory_length - 1, dtype=bool)
     padded_mask[:T-1] = mask
     ```

6. **Return** (lines 223-249)
   - All tensors with exact shapes and dtypes
   - Includes "trajectory_length": T

---

## ❌ MISMATCHES FOUND IN InferenceDataset

### 1. MISSING KEY: "trajectory_length"
- **Training**: Returns `"trajectory_length": T` (int)
- **Inference**: Does NOT return this key
- **Impact**: May break code that expects this key (though inference_collate_fn doesn't use it)
- **Severity**: MEDIUM (should match for exact contract)

### 2. GRAPH CONSTRUCTION: Uses `all_players` instead of `all_players[:N]`
- **Training**: `build_graph(..., all_players[:N], ...)` where N = max_players
- **Inference**: `build_graph(..., all_players, ...)`
- **Analysis**: Functionally equivalent (since len(all_players) <= max_players), but not exact match
- **Severity**: LOW (functional, but should match exactly)

### 3. INITIAL_POS PADDING: Different pattern
- **Training**: 
  ```python
  padded_initial[0] = initial_pos[0]
  padded_initial[1:T] = target_trajectory[1:T]
  padded_initial[T:] = target_trajectory[-1]
  ```
- **Inference**: 
  ```python
  padded_initial[:] = initial_pos[0]  # All same value
  ```
- **Analysis**: For inference, we don't have trajectory, so repeating is correct, BUT structure should match
- **Severity**: MEDIUM (should match padding pattern)

### 4. MASK CREATION: Different approach
- **Training**: 
  ```python
  mask = np.ones(T - 1, dtype=bool)  # [T-1]
  padded_mask = np.zeros(max_trajectory_length - 1, dtype=bool)
  padded_mask[:T-1] = mask
  ```
- **Inference**: 
  ```python
  mask = np.ones(max_trajectory_length - 1, dtype=bool)  # [max_T-1] directly
  ```
- **Analysis**: Functionally same result, but structure doesn't match training
- **Severity**: MEDIUM (should match padding pattern)

---

## 📋 REQUIRED FIXES (See INFERENCE_DATASET_DIFF.md for exact diffs)

1. Add `"trajectory_length": 22` to return dictionary (22 = 1 initial + 21 predictions)
2. Change `all_players` to `all_players[:N]` in build_graph call
3. Change initial_pos padding to match training pattern: `padded_initial[0] = ...; padded_initial[1:] = ...`
4. Change mask creation to match training: create mask of length T-1, then pad to max_T-1

---

## ✅ VALIDATION COMPLETE

All mismatches identified. Ready for diff application.



