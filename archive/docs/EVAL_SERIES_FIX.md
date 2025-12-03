# ✅ Eval Script Series/DataFrame Fix - COMPLETE

## Problem
`AttributeError: 'Series' object has no attribute 'select_dtypes'`

The `EvalDataset.__getitem__()` method was calling `select_dtypes()` on `input_df`, which can be a `pd.Series` instead of a `pd.DataFrame`. Series objects don't have the `select_dtypes()` method.

## Solution Applied

### Location
`src/eval/run_eval.py`, lines 35-50

### Fix
Added defensive code to handle both Series and DataFrame cases:

```python
# --- Build node features (numeric only) ---
# Ensure input_df is a DataFrame during type filtering
if isinstance(input_df, pd.Series):
    df_numeric = input_df.to_frame().T
else:
    df_numeric = input_df

numeric_cols = df_numeric.select_dtypes(
    include=["number", "bool"]
).columns.tolist()
df_numeric = df_numeric[numeric_cols].fillna(0)

# Ensure input_df is DataFrame for subsequent operations
if isinstance(input_df, pd.Series):
    input_df = input_df.to_frame().T
```

## What Changed

**Before:**
```python
numeric_cols = input_df.select_dtypes(
    include=["number", "bool"]
).columns.tolist()
df_numeric = input_df[numeric_cols].fillna(0)
```

**After:**
```python
# Ensure input_df is a DataFrame during type filtering
if isinstance(input_df, pd.Series):
    df_numeric = input_df.to_frame().T
else:
    df_numeric = input_df

numeric_cols = df_numeric.select_dtypes(
    include=["number", "bool"]
).columns.tolist()
df_numeric = df_numeric[numeric_cols].fillna(0)

# Ensure input_df is DataFrame for subsequent operations
if isinstance(input_df, pd.Series):
    input_df = input_df.to_frame().T
```

## Why This Works

1. **Handles Series case**: If `input_df` is a Series, converts it to DataFrame using `to_frame().T` (transpose to get a single-row DataFrame)
2. **Preserves DataFrame case**: If `input_df` is already a DataFrame, uses it directly
3. **Ensures consistency**: Converts `input_df` itself to DataFrame so subsequent operations (line 53, 60) work correctly
4. **No behavioral changes**: The logic remains identical, just with defensive type handling

## Files Modified

✅ **ONLY** `src/eval/run_eval.py` - No other files touched

## Verification

- ✅ No linter errors
- ✅ Defensive type checking added
- ✅ All subsequent `input_df` operations will work (to_dict, indexing)
- ✅ Evaluation behavior remains identical
- ✅ No changes to training, inference, or graph-building logic

## Status: ✅ COMPLETE

The eval script now handles both Series and DataFrame inputs gracefully, fixing the `AttributeError` while maintaining identical evaluation behavior.



