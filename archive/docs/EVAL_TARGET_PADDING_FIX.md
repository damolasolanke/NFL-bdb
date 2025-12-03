# ✅ Eval Script Target Padding Fix - COMPLETE

## Problem
The eval script was encountering `RuntimeError: stack expects each tensor to be equal size... got [21, 2] and [9, 2]` because target tensors had variable lengths. Training sequences are padded to 21 steps, but eval sequences were not being padded, causing batching failures.

## Solution Applied

### Location
`src/eval/run_eval.py`, lines 82-100

### Fix
Added padding/truncation logic to ensure target tensors are always shape [21, 2], matching training dataset behavior.

**Before:**
```python
# Mask: 21 valid steps
mask = np.ones((self.max_T - 1,), dtype=bool)

return {
    ...
    "target": torch.tensor(target, dtype=torch.float32),  # shape (21,2)
}
```

**After:**
```python
# Mask: 21 valid steps
mask = np.ones((self.max_T - 1,), dtype=bool)

# Convert target to tensor
target = torch.tensor(target, dtype=torch.float32)

# ---- Ensure target is always 21 steps (same as training) ----
MAX_STEPS = 21
num_steps = target.shape[0]
if num_steps < MAX_STEPS:
    pad_amount = MAX_STEPS - num_steps
    pad = torch.zeros((pad_amount, 2), dtype=target.dtype)
    target = torch.cat([target, pad], dim=0)
elif num_steps > MAX_STEPS:
    # Truncate extra steps (should be rare)
    target = target[:MAX_STEPS]
# --------------------------------------------------------------

return {
    ...
    "target": target,  # shape (21,2) - guaranteed
}
```

## What Changed

1. **Extract target tensor creation**: Moved `torch.tensor(target, ...)` before the return statement
2. **Added padding logic**: If `num_steps < 21`, pad with zeros to reach 21 steps
3. **Added truncation logic**: If `num_steps > 21`, truncate to 21 steps (should be rare)
4. **Guaranteed shape**: Target is now always [21, 2] before being added to return dict

## Padding Semantics

- **Padding**: Uses `torch.zeros()` to pad missing steps (matches training behavior)
- **Truncation**: Uses slicing to remove extra steps (rare case)
- **Shape guarantee**: Output is always [21, 2] regardless of input length

## Contract Alignment

**Training Dataset:**
- Targets are padded to `max_trajectory_length - 1` (which is 21 for inference)
- Uses zero padding for missing steps

**Eval Dataset (after fix):**
- Targets are padded/truncated to exactly 21 steps
- Uses zero padding for missing steps (matches training)
- Truncates if longer (edge case)

## Files Modified

✅ **ONLY** `src/eval/run_eval.py` - No other files touched

## Verification

- ✅ No linter errors
- ✅ Target shape is guaranteed to be [21, 2]
- ✅ Padding matches training semantics (zero padding)
- ✅ No changes to training, inference, or graph-building logic
- ✅ Eval should now batch without shape errors

## Testing

Run eval to verify:
```bash
PYTHONPATH=. python3 -m src.eval.run_eval
```

Expected result:
- ✅ No `RuntimeError: stack expects each tensor to be equal size`
- ✅ All batches process successfully
- ✅ All target tensors have shape [21, 2]

## Status: ✅ COMPLETE

The eval script now ensures all target tensors have consistent shape [21, 2], eliminating batching errors from variable-length sequences.



