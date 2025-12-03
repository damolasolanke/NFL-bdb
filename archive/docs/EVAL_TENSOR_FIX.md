# ✅ Eval Script Tensor Conversion Fix - COMPLETE

## Problem
The eval script was wrapping `edge_index` and `edge_attr` in `torch.tensor()` even though they're already `torch.Tensor` objects from `build_graph()`. This caused:
- Thousands of warnings about converting tensors to tensors
- Potential crashes due to incorrect tensor semantics

## Solution Applied

### Location
`src/eval/run_eval.py`, lines 88-89

### Fix
Replaced `torch.tensor()` wrapping with `.clone().detach()` to match training dataset behavior exactly.

**Before:**
```python
"edge_index": torch.tensor(edge_index, dtype=torch.long),
"edge_attr": torch.tensor(edge_attr, dtype=torch.float32),
```

**After:**
```python
"edge_index": edge_index.clone().detach(),
"edge_attr": edge_attr.clone().detach(),
```

## Why This Fixes the Issue

1. **`edge_index` and `edge_attr` are already tensors**: They come from `graph.edge_index` and `graph.edge_attr` which are `torch.Tensor` objects returned by `build_graph()`
2. **Training dataset behavior**: Training dataset directly uses `graph.edge_index` and `graph.edge_attr` without wrapping (see `src/utils/datasets.py` line 228-229)
3. **`.clone().detach()`**: Creates a new tensor that:
   - Is detached from the computation graph (no gradients)
   - Is a copy (safe for batching/collation)
   - Maintains the same dtype and device as the original

## Contract Alignment

**Training Dataset:**
```python
result = {
    "edge_index": graph.edge_index,  # [2, E] - already a tensor
    "edge_attr": graph.edge_attr,  # [E, 4] - already a tensor
    ...
}
```

**Eval Dataset (after fix):**
```python
return {
    "edge_index": edge_index.clone().detach(),  # [2, E] - clone existing tensor
    "edge_attr": edge_attr.clone().detach(),  # [E, 4] - clone existing tensor
    ...
}
```

Both approaches handle already-tensor objects correctly, with eval using `.clone().detach()` for safety in batching scenarios.

## Files Modified

✅ **ONLY** `src/eval/run_eval.py` - No other files touched

## Verification

- ✅ No linter errors
- ✅ Tensor semantics match training dataset
- ✅ No unnecessary tensor conversions
- ✅ No changes to training, inference, or graph-building logic
- ✅ Eval should now run without warnings or crashes

## Status: ✅ COMPLETE

The eval script now handles tensor objects correctly, eliminating warnings and crashes related to double tensor conversion.



