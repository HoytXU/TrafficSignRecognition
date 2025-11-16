# Bonus Level - Deep Learning Implementation

This section implements deep learning approaches for traffic sign recognition using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset.

## Project Structure

```
bonus/
├── core/                    # Core modules
│   ├── dataset.py           # GTSRB dataset loader
│   ├── models.py            # Model definitions (LeNet, Custom Net, etc.)
│   └── config.py            # Configuration and paths
│
├── training/                # Training scripts
│   └── train.py             # Main training script
│
├── evaluation/              # Evaluation and analysis
│   ├── ensemble.py          # Ensemble evaluation (weighted voting)
│   └── failure_analysis.py  # Error analysis and visualization
│
├── utils/                   # Utility scripts
│   ├── preprocessing.py     # Preprocessing visualization
│   └── model_info.py        # Model architecture visualization
│
├── nets/                    # Custom network definitions
│   ├── lenet.py             # LeNet architecture
│   └── my_net.py            # Custom CNN with residual blocks
│
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs
├── fail_example/            # Failure case visualizations
│
├── scripts/                 # Batch training scripts
│   ├── train_all.sh          # Linux/Mac batch training
│   └── train_all.bat         # Windows batch training
└── visualization.ipynb       # Jupyter notebook for analysis
```

## Quick Start

### 1. Train a Single Model

```bash
python bonus/training/train.py --model resnet18 --epoch 10 --lr 0.001 --batch_size 128
```

### 2. Train All Models (Batch)

**Linux/Mac:**
```bash
bash bonus/scripts/train_all.sh
```

**Windows:**
```cmd
bonus\scripts\train_all.bat
```

### 3. Evaluate Ensemble

```bash
python bonus/evaluation/ensemble.py
```

### 4. Analyze Failures

```bash
python bonus/evaluation/failure_analysis.py
```

## Available Models

- **LeNet**: Classic CNN architecture
- **ResNet18**: Residual network (pretrained)
- **VGG16**: Deep CNN (pretrained)
- **AlexNet**: Early deep CNN (pretrained)
- **SqueezeNet**: Lightweight CNN (pretrained)
- **ViT-B/16**: Vision Transformer (pretrained)
- **MY_NET**: Custom CNN with residual blocks

## Key Features

### Core Modules

- **`core/dataset.py`**: Handles GTSRB dataset loading with ROI cropping
- **`core/models.py`**: Unified model loading interface
- **`core/config.py`**: Centralized configuration

### Training

- Supports multiple architectures
- Configurable hyperparameters
- WandB integration for logging
- Automatic checkpoint saving

### Evaluation

- **Ensemble**: Weighted voting from multiple models
- **Failure Analysis**: Visualizes misclassified samples
- Comprehensive metrics (accuracy, F1-score, confusion matrix)

## Configuration

Edit `core/config.py` to change:
- Dataset paths
- Default hyperparameters
- Model weights for ensemble
- Checkpoint locations

## Dataset

**GTSRB Dataset** (German Traffic Sign Recognition Benchmark):
- 43 classes
- Pre-split into Train/Test sets
- Includes ROI coordinates for cropping
- Meta images for each class

## Testing

Run tests to verify the implementation:

```bash
# Install test dependencies
pip install -r bonus/requirements-test.txt

# Run all tests
pytest bonus/tests/

# Run with coverage
pytest bonus/tests/ --cov=bonus/core --cov=bonus/training --cov=bonus/evaluation

# Or use the test runner script
bash bonus/scripts/run_tests.sh
# Windows: bonus\scripts\run_tests.bat
```

See `bonus/tests/README.md` for detailed testing documentation.

## Results

The ensemble approach combines predictions from all models using weighted voting, typically achieving **>98% accuracy** on the test set.

