"""
Integration tests for training module.
"""

import pytest
import os
import sys
import torch
from unittest.mock import patch, MagicMock

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.dataset import GTSRBDataset
from core.models import get_model
from core.config import DEFAULT_CONFIG


class TestTraining:
    """Test cases for training functionality."""
    
    @patch('training.train.wandb')
    def test_training_imports(self, mock_wandb):
        """Test that training module imports correctly."""
        try:
            import training.train
            assert True
        except ImportError as e:
            pytest.fail(f"Training module import failed: {e}")
    
    def test_get_transforms(self):
        """Test transform creation."""
        from training.train import get_transforms
        
        transform = get_transforms(augment=False)
        assert transform is not None
        
        transform_aug = get_transforms(augment=True)
        assert transform_aug is not None
    
    def test_model_training_step(self):
        """Test a single training step."""
        model = get_model('lenet', num_classes=43, pretrained=False)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Create dummy data
        x = torch.randn(2, 3, 224, 224)
        y = torch.randint(0, 43, (2,))
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0
        assert output.shape == (2, 43)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

