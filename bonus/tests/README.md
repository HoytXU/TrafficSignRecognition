# Testing Guide

This directory contains tests for the Bonus Level implementation.

## Test Structure

```
tests/
├── __init__.py
├── conftest.py           # Pytest fixtures and configuration
├── test_dataset.py       # Tests for dataset module
├── test_models.py        # Tests for model definitions
├── test_config.py        # Tests for configuration
└── test_training.py      # Integration tests for training
```

## Running Tests

### Install Dependencies

```bash
pip install pytest pytest-cov
```

### Run All Tests

```bash
# From project root
pytest bonus/tests/

# From bonus directory
pytest tests/
```

### Run Specific Test File

```bash
pytest bonus/tests/test_models.py
```

### Run with Coverage

```bash
pytest bonus/tests/ --cov=bonus/core --cov=bonus/training --cov=bonus/evaluation --cov-report=html
```

### Run Verbose Output

```bash
pytest bonus/tests/ -v
```

## Test Categories

### Unit Tests
- **test_dataset.py**: Tests dataset loading, ROI cropping, and data access
- **test_models.py**: Tests model creation, forward passes, and architecture
- **test_config.py**: Tests configuration values and paths

### Integration Tests
- **test_training.py**: Tests training workflow and model training steps

## Writing New Tests

1. Create test file: `test_<module_name>.py`
2. Import the module to test
3. Create test class: `class Test<ModuleName>`
4. Write test methods: `def test_<functionality>(self):`
5. Use fixtures from `conftest.py` for common setup

Example:
```python
import pytest
from core.models import get_model

class TestModels:
    def test_get_model(self):
        model = get_model('lenet', num_classes=43)
        assert model is not None
```

## Continuous Integration

Tests can be integrated into CI/CD pipelines:

```yaml
# Example GitHub Actions
- name: Run tests
  run: pytest bonus/tests/ -v
```

