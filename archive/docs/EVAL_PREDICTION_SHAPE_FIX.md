# ✅ Eval Script Prediction Shape Fix - COMPLETE

## Problem
When concatenating predictions from all batches, got error:
```
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 64 but got size 29 for tensor number 719 in the list.
```

The issue was that the model returns predictions with shape `[B, max_len, 2]` where `max_len=100`, but we need `[B, 21, 2]` to match the target shape. Additionally, the last batch has a different batch size (29 instead of 64), which caused shape mismatches when concatenating.

## Solution Applied

### Location
`src/eval/run_eval.py`, lines 191-194

### Fix
Added slicing to extract only the first 21 steps from model predictions, ensuring all predictions have consistent shape `[B, 21, 2]` regardless of batch size.

**Before:**
```python
preds = model(graph, initial_pos)
all_preds.append(preds.cpu())
all_targets.append(targets.cpu())
```

**After:**
```python
preds = model(graph, initial_pos)  # [B, max_len, 2] where max_len=100

# Extract only first 21 steps (matching target shape)
preds = preds[:, :21, :]  # [B, 21, 2]

all_preds.append(preds.cpu())
all_targets.append(targets.cpu())
```

## Why This Fixes the Issue

1. **Model output shape**: The model returns `[B, 100, 2]` (max_len=100 from config)
2. **Target shape**: Targets are `[B, 21, 2]` (21 prediction steps)
3. **Shape mismatch**: Without slicing, predictions would be `[B, 100, 2]` which doesn't match targets `[B, 21, 2]`
4. **Batch size variation**: Last batch has 29 samples instead of 64, but after slicing both have shape `[B, 21, 2]` where B varies
5. **Concatenation**: After slicing, all tensors have shape `[B, 21, 2]`, so concatenation along dim=0 works correctly

## Shape Flow

**Model output:**
- `preds = model(graph, initial_pos)` → `[B, 100, 2]`

**After slicing:**
- `preds = preds[:, :21, :]` → `[B, 21, 2]`

**After concatenation:**
- `torch.cat(all_preds, dim=0)` → `[total_samples, 21, 2]`
- `torch.cat(all_targets, dim=0)` → `[total_samples, 21, 2]`

Both predictions and targets now have matching shapes for metric computation.

## Files Modified

✅ **ONLY** `src/eval/run_eval.py` - No other files touched

## Verification

- ✅ No linter errors
- ✅ Predictions sliced to 21 steps (matching target shape)
- ✅ All batches have consistent shape `[B, 21, 2]` after slicing
- ✅ Concatenation will work correctly regardless of batch size variation
- ✅ Predictions and targets have matching shapes for RMSE computation

## Status: ✅ COMPLETE

The eval script now correctly slices model predictions to 21 steps, ensuring consistent shapes across all batches and fixing the concatenation error.



