"""
Tests for models module.
"""

import pytest
import torch
import os
import sys

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.models import get_model, LeNet, MY_NET, ResidualBlock


class TestModels:
    """Test cases for model definitions."""
    
    def test_get_model_lenet(self):
        """Test getting LeNet model."""
        model = get_model('lenet', num_classes=43, pretrained=False)
        assert isinstance(model, LeNet)
        assert model.fc3.out_features == 43
    
    def test_get_model_my_net(self):
        """Test getting MY_NET model."""
        model = get_model('my_net', num_classes=43, pretrained=False)
        assert isinstance(model, MY_NET)
        assert model.dense[4].out_features == 43
    
    def test_get_model_resnet18(self):
        """Test getting ResNet18 model."""
        model = get_model('resnet18', num_classes=43, pretrained=False)
        assert model.fc.out_features == 43
    
    def test_get_model_vgg16(self):
        """Test getting VGG16 model."""
        model = get_model('vgg16', num_classes=43, pretrained=False)
        assert model.classifier[6].out_features == 43
    
    def test_get_model_alexnet(self):
        """Test getting AlexNet model."""
        model = get_model('alexnet', num_classes=43, pretrained=False)
        assert model.classifier[6].out_features == 43
    
    def test_get_model_invalid(self):
        """Test getting invalid model raises error."""
        with pytest.raises(ValueError):
            get_model('invalid_model', num_classes=43)
    
    def test_model_forward_pass(self):
        """Test model forward pass."""
        model = get_model('lenet', num_classes=43, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 43)
    
    def test_residual_block(self):
        """Test ResidualBlock forward pass."""
        block = ResidualBlock(64)
        x = torch.randn(1, 64, 32, 32)
        output = block(x)
        assert output.shape == x.shape
    
    def test_model_num_classes_adaptation(self):
        """Test that models adapt to different num_classes."""
        model = get_model('lenet', num_classes=10, pretrained=False)
        assert model.fc3.out_features == 10
        
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

