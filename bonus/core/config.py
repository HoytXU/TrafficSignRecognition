"""
Configuration Module

Centralized configuration for hyperparameters, paths, and model settings.
"""

import os

# Get project root directory
current_file_path = os.path.abspath(__file__)
bonus_directory = os.path.dirname(os.path.dirname(current_file_path))
project_root = os.path.dirname(bonus_directory)

# Dataset paths (GTSRB - German Traffic Sign Recognition Benchmark)
DATASET_PATH = os.path.join(project_root, "datasets", "dataset2")
TRAIN_CSV = os.path.join(DATASET_PATH, "Train.csv")
TEST_CSV = os.path.join(DATASET_PATH, "Test.csv")
META_PATH = os.path.join(DATASET_PATH, "Meta")

# Model checkpoint paths
CHECKPOINT_DIR = os.path.join(bonus_directory, "checkpoints")
LOGS_DIR = os.path.join(bonus_directory, "logs")
FAIL_EXAMPLES_DIR = os.path.join(bonus_directory, "fail_example")

# Default hyperparameters
DEFAULT_CONFIG = {
    'epoch': 10,
    'lr': 0.001,
    'weight_decay': 0.001,
    'batch_size': 128,
    'num_classes': 43,  # GTSRB has 43 classes
    'image_size': 224,
}

# Model configurations
MODEL_CONFIGS = {
    # 'lenet': {'lr': 0.001},
    # 'resnet18': {'lr': 0.001},
    'vgg16': {'lr': 0.001},
    'alexnet': {'lr': 0.001},
    'squeezenet1_0': {'lr': 0.001},
    'vit_b_16': {'lr': 0.0001},  # Vision Transformer needs lower LR
    'my_net': {'lr': 0.001},
}

# Ensemble model weights (based on individual model accuracies)
ENSEMBLE_WEIGHTS = {
    'alexnet': 0.9854,
    'lenet': 0.8856,
    'my_net': 0.9736,
    'resnet18': 0.9759,
    'squeezenet1_0': 0.9696,
    'vgg16': 0.9675,
    'vit_b_16': 0.9869,
}

# Available models
AVAILABLE_MODELS = list(MODEL_CONFIGS.keys())

