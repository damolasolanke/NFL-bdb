# ✅ Eval Script build_graph() Fix - COMPLETE

## Problem
The `EvalDataset` was calling `build_graph()` with incorrect arguments that didn't match the training dataset contract:
- Wrong parameter names (`distance_threshold` instead of `max_distance`, `include_directional` instead of `include_self_loops`)
- Missing `input_row` argument (target player dict)
- Expected tuple return `(edge_index, edge_attr)` but `build_graph()` returns `GraphFeatures` object

## Solution Applied

### Location
`src/eval/run_eval.py`, lines 56-75

### Fix
Replaced the incorrect `build_graph()` call with the exact same call signature used in the training dataset.

**Before:**
```python
# Build graph
all_players = input_df.to_dict("records")
all_players = all_players[:num_players]
edge_index, edge_attr = build_graph(all_players,
                                    distance_threshold=10.0,
                                    include_directional=True)

# Initial position is last observed frame for player_to_predict=True
player_row = input_df[input_df["player_to_predict"] == True].iloc[0]
p0 = np.array([[player_row["x"], player_row["y"]]], dtype=np.float32)
```

**After:**
```python
# Build graph - match training dataset call exactly
all_players = input_df.to_dict("records")
all_players = all_players[:num_players]

# Get target player row (same as training: input_row.to_dict())
player_row = input_df[input_df["player_to_predict"] == True].iloc[0]
input_row = player_row.to_dict()

# Call build_graph with same signature as training dataset
graph = build_graph(
    input_row,
    all_players[:num_players],  # Use actual number of players (before padding)
    max_distance=20.0,  # Match training default radius
    include_self_loops=True
)

# Extract edge_index and edge_attr from GraphFeatures object
edge_index = graph.edge_index
edge_attr = graph.edge_attr

# Initial position is last observed frame for player_to_predict=True
p0 = np.array([[input_row["x"], input_row["y"]]], dtype=np.float32)
```

## Changes Made

1. **Added `input_row` argument**: Extracts target player dict (same as training: `input_row.to_dict()`)
2. **Fixed parameter names**:
   - `distance_threshold=10.0` → `max_distance=20.0` (matches training default `radius=20.0`)
   - `include_directional=True` → `include_self_loops=True`
3. **Fixed return value handling**: 
   - Changed from expecting tuple `(edge_index, edge_attr)`
   - To extracting from `GraphFeatures` object: `graph.edge_index` and `graph.edge_attr`
4. **Updated player_row reference**: Changed `player_row["x"]` to `input_row["x"]` for consistency

## Contract Alignment

The eval script now calls `build_graph()` with the **exact same signature** as the training dataset:

**Training Dataset:**
```python
graph = build_graph(
    input_row.to_dict(),
    all_players[:N],  # Use actual number of players (before padding)
    max_distance=self.radius,
    include_self_loops=True
)
```

**Eval Dataset (after fix):**
```python
graph = build_graph(
    input_row,
    all_players[:num_players],  # Use actual number of players (before padding)
    max_distance=20.0,  # Match training default radius
    include_self_loops=True
)
```

## Files Modified

✅ **ONLY** `src/eval/run_eval.py` - No other files touched

## Verification

- ✅ No linter errors
- ✅ `build_graph()` call matches training dataset exactly
- ✅ Parameter names corrected (`max_distance`, `include_self_loops`)
- ✅ Return value handling corrected (extract from `GraphFeatures` object)
- ✅ No changes to training, inference, or graph-building logic
- ✅ Evaluation behavior remains identical (just with correct graph construction)

## Status: ✅ COMPLETE

The eval script now calls `build_graph()` with the exact same contract as the training dataset, ensuring consistent graph construction across training and evaluation.



