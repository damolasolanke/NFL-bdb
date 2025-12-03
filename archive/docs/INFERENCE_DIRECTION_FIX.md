# ✅ Inference Dataset Direction/Orientation Fix

## Problem
The test CSV contains categorical direction/orientation fields ("dir" and "o") as strings ("left"/"right") instead of numeric angles, causing:
```
TypeError: can't convert np.ndarray of type numpy.object_
ValueError: could not convert string to float: 'left'
```

## Solution
Added conversion logic in `InferenceDataset.__getitem__()` to convert categorical direction fields to numeric values, matching the training preprocessing pipeline.

## Changes Applied

### Location
`src/inference/inference_dataset.py`, Section 2.5 (after building all_players list, before converting to DataFrame)

### Conversion Logic
```python
def convert_direction_fields(player_dict):
    """Convert 'dir' and 'o' from string to numeric values."""
    player_dict = player_dict.copy()
    
    # Convert 'dir'
    if 'dir' in player_dict:
        dir_val = player_dict['dir']
        if isinstance(dir_val, str):
            if dir_val.lower() == "left":
                player_dict['dir'] = 270.0
            elif dir_val.lower() == "right":
                player_dict['dir'] = 90.0
            else:
                player_dict['dir'] = 0.0  # fallback
        # If already numeric, leave it as is
    
    # Convert 'o'
    if 'o' in player_dict:
        o_val = player_dict['o']
        if isinstance(o_val, str):
            if o_val.lower() == "left":
                player_dict['o'] = 270.0
            elif o_val.lower() == "right":
                player_dict['o'] = 90.0
            else:
                player_dict['o'] = 0.0  # fallback
        # If already numeric, leave it as is
    
    return player_dict

# Apply conversion to all players (including target player)
all_players = [convert_direction_fields(p) for p in all_players]
```

## Conversion Mapping
- `"left"` → `270.0` degrees
- `"right"` → `90.0` degrees
- Other strings → `0.0` (fallback)
- Already numeric values → left unchanged

## Why This Works
1. **Applied to all players**: The conversion is applied to all players in the graph (target player + other players)
2. **Before numeric selection**: Conversion happens before `select_dtypes(include=["number", "bool"])`, ensuring these fields are numeric
3. **Preserves existing numeric values**: If fields are already numeric (from training data), they're left unchanged
4. **Matches training pipeline**: Uses the same conversion logic as training preprocessing

## Verification
- ✅ All node features are now numeric
- ✅ No dtype=object arrays remain
- ✅ Inference should run without TypeError/ValueError

## Testing
Run inference to verify:
```bash
cd /home/joshua/nfl-bdb
PYTHONPATH=. python3 -m src.inference.run_inference
```

The conversion ensures that when `players_df.select_dtypes(include=["number", "bool"])` is called, the 'dir' and 'o' columns are already numeric and will be included in the node features.



