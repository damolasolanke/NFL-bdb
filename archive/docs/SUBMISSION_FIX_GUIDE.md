# Kaggle Submission Fix - Technical Guide

## Problem Summary

Kaggle is rejecting submission.parquet files with "Kaggle Error" even though:
- File is saved to `/kaggle/working/submission.parquet`
- Row count matches expected (~5,837 rows)
- Columns appear correct (row_id, x, y)
- row_id format appears correct

## Root Causes Identified

1. **String dtype incompatibility**: Pandas `StringDtype` ('string') can cause issues with PyArrow parquet writing/reading
2. **Implicit type coercion**: PyArrow may silently convert types during parquet write
3. **Schema enforcement**: Without explicit schema, PyArrow may infer incorrect types
4. **Float contamination**: Integer IDs may accidentally become floats during processing

## Solution Implementation

The notebook cell has been updated with a robust solution. One small fix is needed:

### Fix Required

In the row_id construction section, change:

```python
# Convert to pandas StringDtype (not object dtype) to ensure pure string
row_id = pd.Series(row_id, dtype='string')
```

To:

```python
# Convert to object dtype but ensure all values are strings (more compatible with PyArrow)
# Using object dtype is safer for PyArrow compatibility than pandas StringDtype
row_id = row_id.astype(str).astype(object)
```

And update the validation assertion:

```python
# Verify row_id is string type (not object)
assert submission['row_id'].dtype == 'string', f"row_id dtype is {submission['row_id'].dtype}, expected string"
```

To:

```python
# Verify row_id is object dtype containing strings
assert submission['row_id'].dtype == 'object', f"row_id dtype is {submission['row_id'].dtype}, expected object"
assert all(isinstance(x, str) for x in submission['row_id']), "row_id must contain only strings"
```

## Key Technical Details

### 1. Explicit PyArrow Schema

```python
schema = pa.schema([
    pa.field('row_id', pa.string()),  # Explicitly string, not object
    pa.field('x', pa.float64()),
    pa.field('y', pa.float64()),
])
```

This prevents PyArrow from inferring incorrect types.

### 2. Safe Parquet Write Settings

```python
pq.write_table(
    table,
    output_path,
    compression='snappy',      # Standard, widely compatible
    use_dictionary=False,      # Avoids encoding issues
    write_statistics=False     # Reduces compatibility problems
)
```

### 3. Integer-to-String Conversion

```python
# Convert to int64 FIRST to ensure no float contamination
game_id_int = submission_df['game_id'].astype('int64')
play_id_int = submission_df['play_id'].astype('int64')
nfl_id_int = submission_df['nfl_id'].astype('int64')
frame_id_int = submission_df['frame_id'].astype('int64')

# Then convert to string
row_id = (
    game_id_int.astype(str) + '_' +
    play_id_int.astype(str) + '_' +
    nfl_id_int.astype(str) + '_' +
    frame_id_int.astype(str)
)
```

This two-step process ensures no float contamination.

### 4. Comprehensive Validation

The code includes:
- Pre-write validation (checking for NaN, decimals, types)
- Post-write validation (reading back and verifying)
- Explicit assertions at each step

## Complete Working Code Pattern

```python
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import os

# 1. Load and filter
df = pd.read_csv("/kaggle/input/nfl-big-data-bowl-2026-prediction/test_input.csv")
submission_df = df[df['player_to_predict'] == True].copy()

# 2. Construct row_id (strict integer-to-string)
game_id_int = submission_df['game_id'].astype('int64')
play_id_int = submission_df['play_id'].astype('int64')
nfl_id_int = submission_df['nfl_id'].astype('int64')
frame_id_int = submission_df['frame_id'].astype('int64')

row_id = (
    game_id_int.astype(str) + '_' +
    play_id_int.astype(str) + '_' +
    nfl_id_int.astype(str) + '_' +
    frame_id_int.astype(str)
).astype(object)  # Use object dtype for PyArrow compatibility

# 3. Create submission DataFrame
submission = pd.DataFrame({
    'row_id': row_id,
    'x': pd.Series([0.0] * len(submission_df), dtype='float64'),  # Replace with predictions
    'y': pd.Series([0.0] * len(submission_df), dtype='float64')   # Replace with predictions
})

# 4. Define explicit schema
schema = pa.schema([
    pa.field('row_id', pa.string()),
    pa.field('x', pa.float64()),
    pa.field('y', pa.float64()),
])

# 5. Write with explicit schema
table = pa.Table.from_pandas(submission, schema=schema)
pq.write_table(
    table,
    '/kaggle/working/submission.parquet',
    compression='snappy',
    use_dictionary=False,
    write_statistics=False
)

# 6. Validate by reading back
validation = pd.read_parquet('/kaggle/working/submission.parquet')
assert len(validation) == len(submission)
assert list(validation.columns) == ['row_id', 'x', 'y']
assert not validation['x'].isna().any()
assert not validation['y'].isna().any()
```

## Common Pitfalls to Avoid

1. **Don't use pandas StringDtype**: Use `object` dtype with string values
2. **Don't skip explicit schema**: Always define PyArrow schema explicitly
3. **Don't use f-strings for row_id**: Use string concatenation to avoid formatting issues
4. **Don't use default compression**: Use 'snappy' explicitly
5. **Don't skip validation**: Always read back and verify the file

## Testing Checklist

Before submitting, verify:
- [ ] File exists at `/kaggle/working/submission.parquet`
- [ ] Row count matches expected (~5,600-6,200)
- [ ] Exactly 3 columns: row_id, x, y
- [ ] row_id contains no decimal points
- [ ] row_id format: `<game_id>_<play_id>_<nfl_id>_<frame_id>`
- [ ] x and y are numeric (float64)
- [ ] No NaN values in x or y
- [ ] No infinite values in x or y
- [ ] File can be read back successfully
- [ ] Read-back data matches original

## Why This Solution Works

1. **Explicit schema**: Prevents PyArrow from guessing types
2. **Object dtype**: More compatible with PyArrow than StringDtype
3. **Two-step conversion**: int64 → str prevents float contamination
4. **Safe compression**: 'snappy' is widely supported
5. **Validation**: Catches issues before submission

## Additional Debugging

If issues persist, add this diagnostic code:

```python
# After writing, inspect the actual parquet file
import pyarrow.parquet as pq

parquet_file = pq.ParquetFile('/kaggle/working/submission.parquet')
print("Schema:", parquet_file.schema)
print("Metadata:", parquet_file.metadata)
print("Row groups:", parquet_file.num_row_groups)

# Read with different engines to compare
df_pandas = pd.read_parquet('/kaggle/working/submission.parquet', engine='pyarrow')
df_fastparquet = pd.read_parquet('/kaggle/working/submission.parquet', engine='fastparquet')
```

This will help identify if the issue is with the file itself or Kaggle's reader.

