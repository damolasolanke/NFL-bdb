# DeepMind-Grade Contract Validation Report

## NFLTrajectoryDataset.__getitem__() EXACT CONTRACT

### Return Dictionary Keys:
1. `"node_feats"`: Tensor[max_players, F], dtype=torch.float32
2. `"edge_index"`: Tensor[2, E], dtype=torch.long (from graph.edge_index)
3. `"edge_attr"`: Tensor[E, 4], dtype=torch.float32 (from graph.edge_attr)
4. `"initial_pos"`: Tensor[max_trajectory_length, 2], dtype=torch.float32
5. `"target"`: Tensor[max_trajectory_length - 1, 2], dtype=torch.float32 ⚠️ (NOT in inference)
6. `"mask"`: Tensor[max_trajectory_length - 1], dtype=torch.bool
7. `"key"`: Tuple[int, int, int]
8. `"trajectory_length"`: int ⚠️ (MISSING in inference)

### Shape Contracts:
- `node_feats`: [max_players, F] where F = number of numeric features after dropping frame_id, x, y
- `edge_index`: [2, E] where E = number of edges (variable)
- `edge_attr`: [E, 4] where 4 = [distance, angle, rel_vx, rel_vy]
- `initial_pos`: [max_trajectory_length, 2] - PADDED to max_trajectory_length
- `mask`: [max_trajectory_length - 1] - PADDED to max_trajectory_length - 1
- `target`: [max_trajectory_length - 1, 2] - PADDED to max_trajectory_length - 1

### Padding Rules:
1. **Node features**: Padded to max_players with zeros if N < max_players
2. **Graph edges**: Self-loops added for padded nodes (indices N to max_players-1)
3. **initial_pos**: 
   - padded_initial[0] = initial_pos[0] (first position)
   - padded_initial[1:T] = target_trajectory[1:T] (if T > 1)
   - padded_initial[T:] = target_trajectory[-1] (repeat last position)
4. **mask**: 
   - padded_mask[:T-1] = mask (True values)
   - padded_mask[T-1:] = False (zeros for padding)

### Graph Construction:
- Uses `all_players[:N]` where N = max_players (after padding)
- Since len(all_players) <= max_players, this is equivalent to `all_players[:]`
- Graph is built BEFORE padding nodes to max_players
- Then graph is padded with self-loops for nodes N to max_players-1

### Numeric Column Selection:
- `select_dtypes(include=["number", "bool"])`
- Drops: `["frame_id", "x", "y"]`
- All remaining numeric columns become node features

### Dtype Requirements:
- `node_feats`: torch.float32
- `edge_index`: torch.long (from graph.edge_index)
- `edge_attr`: torch.float32 (from graph.edge_attr)
- `initial_pos`: torch.float32
- `mask`: torch.bool

---

## InferenceDataset.__getitem__() COMPARISON

### ✅ CORRECT:
1. Keys: node_feats, edge_index, edge_attr, initial_pos, mask, key
2. Shapes: All match training shapes
3. Dtypes: All match training dtypes
4. Node feature extraction: Identical (select_dtypes, drop frame_id/x/y)
5. Graph construction: Uses build_graph with same parameters
6. Graph padding: Adds self-loops for padded nodes (identical logic)

### ❌ MISMATCHES FOUND:

#### 1. MISSING KEY: "trajectory_length"
- Training returns: `"trajectory_length": T` where T is the actual trajectory length
- Inference returns: NOTHING
- Impact: Collate function may expect this key (though inference_collate_fn doesn't use it)

#### 2. GRAPH CONSTRUCTION: Uses `all_players` instead of `all_players[:N]`
- Training: `build_graph(..., all_players[:N], ...)` where N = max_players
- Inference: `build_graph(..., all_players, ...)`
- Analysis: Since len(all_players) <= max_players, these are equivalent, BUT for exact mirroring, should use `all_players[:N]` where N is set to max_players after padding

#### 3. INITIAL_POS PADDING: Different logic (but acceptable for inference)
- Training: padded_initial[0] = initial_pos[0], then fills with trajectory[1:T], then repeats last
- Inference: padded_initial[:] = initial_pos[0] (just repeats last observed position)
- Analysis: For inference, we don't have a trajectory, so this is acceptable, BUT the structure should match - we should still use the same padding pattern (just with repeated values)

#### 4. MASK CREATION: Different approach (but acceptable for inference)
- Training: Creates mask of length T-1, then pads with zeros to max_trajectory_length - 1
- Inference: Creates mask of length max_trajectory_length - 1 with all ones
- Analysis: For inference, all ones is correct, but the structure should match training's padding pattern

---

## REQUIRED FIXES



