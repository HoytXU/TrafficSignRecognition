"""
Model Definitions

This module contains custom neural network architectures for traffic sign recognition.
Also provides utilities to load pre-trained models from torchvision.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Add nets directory to path for custom networks
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nets_dir = os.path.join(bonus_dir, "nets")
if nets_dir not in sys.path:
    sys.path.insert(0, nets_dir)

# Import custom networks
try:
    from lenet import LeNet as LeNetOriginal
    from my_net import MY_NET as MY_NETOriginal, ResidualBlock as ResidualBlockOriginal
except ImportError:
    # Fallback: define here if import fails
    LeNetOriginal = None
    MY_NETOriginal = None
    ResidualBlockOriginal = None


# Use original implementations from nets/ but wrap to support num_classes parameter
if LeNetOriginal is not None:
    # Wrap LeNet to support num_classes
    class LeNet(nn.Module):
        def __init__(self, num_classes=43):
            super().__init__()
            base_model = LeNetOriginal()
            # Replace final layer
            self.conv1 = base_model.conv1
            self.conv2 = base_model.conv2
            self.fc1 = base_model.fc1
            self.fc2 = base_model.fc2
            self.fc3 = nn.Linear(84, num_classes)
        
        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x
    
    # Wrap MY_NET to support num_classes
    class MY_NET(nn.Module):
        def __init__(self, num_classes=43):
            super().__init__()
            base_model = MY_NETOriginal()
            # Copy all layers except final dense layer
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.res1 = base_model.res1
            self.conv2 = base_model.conv2
            self.bn2 = base_model.bn2
            self.res2 = base_model.res2
            self.conv3 = base_model.conv3
            self.bn3 = base_model.bn3
            self.res3 = base_model.res3
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            # Replace final layer
            self.dense = nn.Sequential(
                base_model.dense[0],  # Linear(28*28*256, 256)
                base_model.dense[1],  # BatchNorm1d(256)
                base_model.dense[2],  # ReLU
                base_model.dense[3],  # Dropout(0.5)
                nn.Linear(256, num_classes)  # Adapt to num_classes
            )
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.res1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.res2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.res3(x)
            x = x.view(-1, 28 * 28 * 256)
            x = self.dense(x)
            return x
    
    ResidualBlock = ResidualBlockOriginal
else:
    # Fallback definitions (same as in nets/)
    class ResidualBlock(nn.Module):
        """Residual block with skip connection."""
        def __init__(self, channels):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            y = self.conv1(x)
            y = self.bn1(y)
            y = F.relu(y)
            y = self.conv2(y)
            y = self.bn2(y)
            return F.relu(x + y)

    class LeNet(nn.Module):
        """LeNet architecture for traffic sign recognition."""
        def __init__(self, num_classes=43):
            super(LeNet, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
            self.conv2 = nn.Conv2d(6, 16, kernel_size=5, padding=2)
            self.fc1 = nn.Linear(16 * 56 * 56, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, num_classes)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    class MY_NET(nn.Module):
        """Custom CNN with residual blocks."""
        def __init__(self, num_classes=43):
            super(MY_NET, self).__init__()
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal_(self.conv1.weight)
            self.bn1 = nn.BatchNorm2d(64)
            self.res1 = ResidualBlock(64)
            self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal_(self.conv2.weight)
            self.bn2 = nn.BatchNorm2d(128)
            self.res2 = ResidualBlock(128)
            self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            nn.init.xavier_normal_(self.conv3.weight)
            self.bn3 = nn.BatchNorm2d(256)
            self.res3 = ResidualBlock(256)
            self.relu = nn.ReLU()
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.dense = nn.Sequential(
                nn.Linear(28 * 28 * 256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(p=0.5),
                nn.Linear(256, num_classes)  # Adapt to num_classes
            )
            nn.init.xavier_normal_(self.dense[0].weight)
            nn.init.xavier_normal_(self.dense[4].weight)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.res1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.res2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.res3(x)
            x = x.view(-1, 28 * 28 * 256)
            x = self.dense(x)
            return x


def get_model(model_name, num_classes=43, pretrained=True):
    """
    Get a model by name.
    
    Args:
        model_name: Name of the model ('lenet', 'resnet18', 'vgg16', etc.)
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights (for torchvision models)
        
    Returns:
        torch.nn.Module: Model instance
    """
    if model_name == 'lenet':
        model = LeNet(num_classes=num_classes)
        return model
    elif model_name == 'my_net':
        model = MY_NET(num_classes=num_classes)
        return model
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == 'alexnet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == 'squeezenet1_0':
        model = models.squeezenet1_0(pretrained=pretrained)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
        return model
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=pretrained)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unknown model: {model_name}. Available: {['lenet', 'resnet18', 'vgg16', 'alexnet', 'squeezenet1_0', 'vit_b_16', 'my_net']}")

