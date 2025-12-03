# ✅ Eval Script GraphFeatures Fix - COMPLETE

## Problem
The eval script was incorrectly constructing incomplete `GraphFeatures` objects, causing:
```
TypeError: GraphFeatures.__init__() missing required positional arguments
```

The script was manually constructing graph components instead of using `build_graph()` properly and returning complete `GraphFeatures` objects.

## Solution Applied

### Changes Made

1. **Removed manual graph construction**: Removed all manual edge_index, edge_attr, node_features, batch construction
2. **Use build_graph() correctly**: Call `build_graph()` with proper arguments matching training
3. **Return complete GraphFeatures**: Return a complete `GraphFeatures` object with all required fields
4. **Updated collate function**: Extract graphs and initial_pos from batch correctly
5. **Updated evaluation loop**: Use complete GraphFeatures objects with all fields

### Location
`src/eval/run_eval.py`

### Key Changes

**1. __getitem__() - Graph Construction (lines 64-102):**

**Before:**
```python
# Manual graph construction (incorrect)
edge_index, edge_attr = build_graph(...)
return {
    "edge_index": edge_index.clone().detach(),
    "edge_attr": edge_attr.clone().detach(),
    ...
}
```

**After:**
```python
# Extract initial position (last observed frame for target player)
initial_pos = torch.tensor([[input_row["x"], input_row["y"]]], dtype=torch.float32)  # [1, 2]

# Call build_graph with same signature as training dataset
graph = build_graph(
    input_row=input_row,
    all_players=all_players[:num_players],
    max_distance=20.0,
    include_self_loops=True
)

# Clone/detach all tensors for safe batching
node_features = graph.node_features.clone().detach()
edge_index = graph.edge_index.clone().detach()
edge_attr = graph.edge_attr.clone().detach()
batch = graph.batch.clone().detach()

return {
    "key": key,
    "graph": GraphFeatures(
        node_features=node_features,
        edge_index=edge_index,
        edge_attr=edge_attr,
        batch=batch,
    ),
    "initial_pos": initial_pos,  # [1, 2]
    "target": target,  # shape (21,2)
}
```

**2. Collate Function (lines 105-112):**

**Before:**
```python
def eval_collate(batch):
    keys = [b["key"] for b in batch]
    node_feats = torch.stack([b["node_feats"] for b in batch])
    initial_pos = torch.stack([b["initial_pos"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    target = torch.stack([b["target"] for b in batch])
    edge_list = [{"edge_index": b["edge_index"], "edge_attr": b["edge_attr"]} for b in batch]
    return keys, node_feats, initial_pos, mask, target, edge_list
```

**After:**
```python
def eval_collate(batch):
    keys = [b["key"] for b in batch]
    graphs = [b["graph"] for b in batch]
    initial_pos = torch.cat([b["initial_pos"] for b in batch], dim=0)  # [B, 2]
    target = torch.stack([b["target"] for b in batch])  # [B, 21, 2]
    return keys, graphs, initial_pos, target
```

**3. Evaluation Loop (lines 161-178):**

**Before:**
```python
for keys, node_feats, initial_pos, mask, target, edge_list in tqdm(dl):
    # Manually reconstruct incomplete GraphFeatures
    graphs = []
    for g in edge_list:
        graphs.append(
            GraphFeatures(
                edge_index=g["edge_index"].to(device),
                edge_attr=g["edge_attr"].to(device)
                # Missing node_features and batch!
            )
        )
```

**After:**
```python
for keys, graphs, initial_pos, target in tqdm(dl):
    # Move complete graphs to device
    graphs_device = []
    for g in graphs:
        graphs_device.append(
            GraphFeatures(
                node_features=g.node_features.to(device),
                edge_index=g.edge_index.to(device),
                edge_attr=g.edge_attr.to(device),
                batch=g.batch.to(device)
            )
        )
    
    initial_pos = initial_pos.to(device)  # [B, 2]
    pred = model(graphs_device, initial_pos=initial_pos)
```

## Why This Fixes the Issue

1. **Complete GraphFeatures objects**: All required fields (node_features, edge_index, edge_attr, batch) are now included
2. **Uses build_graph() correctly**: Matches training dataset call exactly
3. **Proper tensor handling**: All tensors are cloned/detached for safe batching
4. **Correct initial_pos extraction**: Extracted from target player's last observed position
5. **Simplified collate**: Directly passes GraphFeatures objects instead of reconstructing

## Contract Alignment

**Training Dataset:**
- Uses `build_graph()` to create complete `GraphFeatures` objects
- Returns graph with all fields: node_features, edge_index, edge_attr, batch

**Eval Dataset (after fix):**
- Uses `build_graph()` with same signature
- Returns complete `GraphFeatures` object with all fields
- Clones/detaches tensors for safe batching

## Files Modified

✅ **ONLY** `src/eval/run_eval.py` - No other files touched

## Verification

- ✅ No linter errors
- ✅ Complete GraphFeatures objects with all required fields
- ✅ Uses build_graph() correctly (matches training)
- ✅ No changes to training, inference, or graph-building logic
- ✅ Eval should now run without TypeError

## Status: ✅ COMPLETE

The eval script now constructs complete `GraphFeatures` objects using `build_graph()` exactly like training, eliminating the `TypeError: GraphFeatures.__init__() missing required positional arguments` error.



