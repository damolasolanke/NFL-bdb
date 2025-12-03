# DIAGNOSTIC AUDIT REPORT
## Project: NFL-BDB Model Training
## Date: Audit performed on current codebase

---

## CATEGORY 1: input_df MultiIndex DataFrame

**STATUS: FAIL**

**Issue:** The notebook `notebooks/02_model_training.ipynb` is essentially empty (contains only 18 characters total across all cells). There is no code that:
- Creates `input_df` with MultiIndex using `set_index(["game_id", "play_id", "nfl_id"])`
- Ensures the MultiIndex is preserved throughout the notebook

**Location:** `notebooks/02_model_training.ipynb` - No relevant code found (notebook is empty)

**Details:** 
- The notebook does not contain any code to create or manage `input_df`
- No `set_index()` call is present
- No verification of `input_df.index.names == ["game_id", "play_id", "nfl_id"]` exists

---

## CATEGORY 2: NFLTrajectoryDataset receives MultiIndex input_df

**STATUS: FAIL**

**Issue:** The notebook `notebooks/02_model_training.ipynb` is empty and contains no code that:
- Creates `NFLTrajectoryDataset` instances
- Passes `input_df` to the `all_input_df` parameter

**Location:** `notebooks/02_model_training.ipynb` - No relevant code found (notebook is empty)

**Details:**
- No `NFLTrajectoryDataset` instantiation found in the main notebook
- Cannot verify that MultiIndex `input_df` is being passed correctly
- Note: A backup notebook (`02_model_training_backup.ipynb`) shows `NFLTrajectoryDataset` usage, but the main notebook is empty

---

## CATEGORY 3: collate.py file and import

**STATUS: FAIL**

**Issue:** The required file `src/utils/collate.py` does NOT exist. The `collate_fn` function is located in `src/utils/datasets.py` instead.

**Location:** 
- Missing file: `src/utils/collate.py` (does not exist)
- Actual location: `src/utils/datasets.py` line 118

**Details:**
- `collate_fn` is defined in `datasets.py` at line 118, not in a separate `collate.py` file
- The function IS exported via `src/utils/__init__.py` (line 5), which allows imports like `from src.utils import collate_fn`
- However, the requirement specifies that `collate.py` must exist as a separate file
- The notebook is empty, so no import statement can be verified, but even if it existed, it could not import from `src.utils.collate` as that module does not exist

**Additional Note:** The backup notebook shows `from src.utils import NFLTrajectoryDataset, collate_fn`, which works via `__init__.py` but does not match the required import path `from src.utils.collate import collate_fn`

---

## CATEGORY 4: datasets.py uses MultiIndex .loc access

**STATUS: FAIL**

**Issue:** The `NFLTrajectoryDataset.__getitem__` method in `datasets.py` uses boolean filtering instead of MultiIndex `.loc` access.

**Location:** `src/utils/datasets.py` lines 66-70

**Details:**
- Lines 66-70 use boolean filtering:
  ```python
  play_mask = (
      (self.all_input_df['game_id'] == game_id) &
      (self.all_input_df['play_id'] == play_id)
  )
  play_data = self.all_input_df[play_mask]
  ```
- This approach assumes `game_id`, `play_id`, and `nfl_id` are regular columns, not index levels
- The required implementation should use: `df = self.input_df.loc[(game_id, play_id, nfl_id)]`
- Lines 74-75 also use column-based filtering: `play_data[play_data['nfl_id'] == nfl_id]`
- The code references raw columns `'game_id'`, `'play_id'`, `'nfl_id'` as if they are DataFrame columns rather than index levels

**Additional Issues:**
- Line 65: `game_id, play_id, _ = key` correctly unpacks the key tuple
- Lines 67-68: Accesses `self.all_input_df['game_id']` and `self.all_input_df['play_id']` as columns
- Line 74: Accesses `play_data['nfl_id']` as a column
- Line 75: Uses boolean filtering `play_data[play_data['nfl_id'] == nfl_id]`

---

## SUMMARY

**Total Categories Checked:** 4
**Categories Passing:** 0
**Categories Failing:** 4

### Critical Issues Found:

1. **Notebook is empty** - The main training notebook contains no executable code
2. **Missing collate.py** - Required file does not exist; `collate_fn` is in `datasets.py` instead
3. **Incorrect DataFrame access** - `datasets.py` uses column-based boolean filtering instead of MultiIndex `.loc` access
4. **No MultiIndex setup** - No code exists to create or verify the MultiIndex structure

---

## END OF REPORT



