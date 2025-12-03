# Development Guide

## Setup

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install in development mode
pip install -e ".[dev]"
```

### 2. Project Structure

```
src/nfl_bdb/
├── models/          # Model architectures
├── training/        # Training scripts
├── inference/       # Inference code
├── evaluation/      # Evaluation code
├── preprocessing/    # Data preprocessing
├── utils/          # Utilities (datasets, collate, graph builder)
└── cli.py          # CLI interface
```

## Code Organization

### Adding New Models

1. Create model file in `src/nfl_bdb/models/`
2. Implement `__init__` and `forward` methods
3. Add to `src/nfl_bdb/models/__init__.py`
4. Update config if needed

### Adding New Features

1. Add feature computation in `src/nfl_bdb/preprocessing/features.py`
2. Update node feature extraction in `src/nfl_bdb/utils/datasets.py`
3. Update `NODE_DIM` in config if feature count changes

### Adding New Metrics

1. Add metric computation in `src/nfl_bdb/evaluation/run_eval.py`
2. Update CLI if needed

## Development Workflow

### 1. Make Changes

- Edit code in `src/nfl_bdb/`
- Update tests in `tests/`
- Update documentation as needed

### 2. Test Changes

```bash
# Run tests
pytest tests/

# Run specific test
pytest tests/test_models.py

# Run with coverage
pytest --cov=nfl_bdb tests/
```

### 3. Check Imports

```bash
# Verify imports work
python -c "from nfl_bdb.models import STGNNRefine"
```

### 4. Run Training (Quick Test)

```bash
# Small test run
python scripts/train.py --epochs 1 --batch-size 4
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Document functions with docstrings
- Keep functions focused and small

## Debugging

### Common Issues

1. **Import Errors**: Check `sys.path` includes project root
2. **Shape Mismatches**: Check tensor shapes in dataset/collate
3. **NaN/Inf**: Check input data preprocessing
4. **CUDA Errors**: Verify device availability

### Debugging Tools

```python
# Add debug prints
print(f"Shape: {tensor.shape}")

# Check for NaN/Inf
assert not torch.isnan(tensor).any()
assert not torch.isinf(tensor).any()

# Verify device
print(f"Device: {tensor.device}")
```

## Testing

### Writing Tests

Create test files in `tests/`:

```python
import pytest
from nfl_bdb.models import STGNNRefine

def test_model_initialization():
    model = STGNNRefine()
    assert model is not None
```

### Running Tests

```bash
# All tests
pytest

# Specific file
pytest tests/test_models.py

# With verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Configuration Management

### Adding New Config

1. Add to `configs/default.py`
2. Document in docstring
3. Use in code: `config.NEW_PARAM`

### Overriding Config

```python
# In script
import configs.default as config
config.BATCH_SIZE = 64
```

## Git Workflow

1. Create feature branch: `git checkout -b feature-name`
2. Make changes and commit
3. Push and create PR
4. Review and merge

## Documentation

### Updating Docs

- Update README.md for user-facing changes
- Update docs/*.md for detailed documentation
- Add docstrings to new functions/classes

### Building Docs

Documentation is in Markdown format. No build step required.

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create git tag
4. Push to GitHub

## Getting Help

- Check existing documentation
- Review code comments
- Open an issue on GitHub

