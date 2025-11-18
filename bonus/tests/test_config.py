"""
Tests for config module.
"""

import pytest
import os
import sys

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.config import (
    DATASET_PATH, TRAIN_CSV, TEST_CSV, META_PATH,
    CHECKPOINT_DIR, LOGS_DIR,
    DEFAULT_CONFIG, MODEL_CONFIGS, ENSEMBLE_WEIGHTS, AVAILABLE_MODELS
)


class TestConfig:
    """Test cases for configuration."""
    
    def test_paths_exist(self):
        """Test that all path variables are strings."""
        assert isinstance(DATASET_PATH, str)
        assert isinstance(TRAIN_CSV, str)
        assert isinstance(TEST_CSV, str)
        assert isinstance(META_PATH, str)
        assert isinstance(CHECKPOINT_DIR, str)
        assert isinstance(LOGS_DIR, str)
    
    def test_default_config(self):
        """Test default configuration values."""
        assert 'epoch' in DEFAULT_CONFIG
        assert 'lr' in DEFAULT_CONFIG
        assert 'weight_decay' in DEFAULT_CONFIG
        assert 'batch_size' in DEFAULT_CONFIG
        assert 'num_classes' in DEFAULT_CONFIG
        assert DEFAULT_CONFIG['num_classes'] == 43
    
    def test_model_configs(self):
        """Test model configurations."""
        assert len(MODEL_CONFIGS) > 0
        for model_name, config in MODEL_CONFIGS.items():
            assert 'lr' in config
            assert isinstance(config['lr'], float)
    
    def test_ensemble_weights(self):
        """Test ensemble weights."""
        assert len(ENSEMBLE_WEIGHTS) > 0
        for model_name, weight in ENSEMBLE_WEIGHTS.items():
            assert isinstance(weight, float)
            assert 0 <= weight <= 1
    
    def test_available_models(self):
        """Test available models list."""
        assert len(AVAILABLE_MODELS) > 0
        assert isinstance(AVAILABLE_MODELS, list)
        assert 'lenet' in AVAILABLE_MODELS
        assert 'resnet18' in AVAILABLE_MODELS
    
    def test_ensemble_weights_match_models(self):
        """Test that ensemble weights exist for all available models."""
        for model in AVAILABLE_MODELS:
            assert model in ENSEMBLE_WEIGHTS, f"Missing weight for {model}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

