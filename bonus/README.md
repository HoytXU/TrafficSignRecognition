# Bonus Level - Deep Learning Implementation

This section implements deep learning approaches for traffic sign recognition using the **GTSRB (German Traffic Sign Recognition Benchmark)** dataset. The implementation supports multiple CNN architectures and provides a complete training pipeline with experiment tracking.

## Project Structure

```
bonus/
├── core/                    # Core modules
│   ├── dataset.py           # GTSRB dataset loader with ROI cropping
│   ├── models.py            # Unified model loading interface
│   └── config.py            # Configuration and paths
│
├── training/                # Training scripts
│   └── train.py             # Main training script with WandB integration
│
├── utils/                   # Utility scripts
│   ├── preprocessing.py     # Preprocessing visualization
│   └── model_info.py        # Model architecture visualization
│
├── nets/                    # Custom network definitions
│   ├── lenet.py             # LeNet architecture
│   └── my_net.py            # Custom CNN with residual blocks
│
├── analysis/                # Model analysis and comparison
│   └── compare_all_models.py # Compare trained models
│
├── scripts/                 # Batch training and utility scripts
│   ├── train_all.sh          # Linux/Mac batch training
│   ├── train_all.bat         # Windows batch training
│   ├── train_all_models.py   # Python batch training script
│   ├── setup_bonus.sh        # Environment setup script
│   ├── run_tests.sh          # Test runner (Linux/Mac)
│   └── run_tests.bat         # Test runner (Windows)
│
├── checkpoints/             # Saved model weights
├── logs/                    # Training logs
├── tests/                   # Unit and integration tests
└── visualization.ipynb      # Jupyter notebook for model visualization
```

## Quick Start

### 1. Setup Environment

See [SETUP.md](SETUP.md) for detailed setup instructions, or use the quick setup script:

```bash
bash bonus/scripts/setup_bonus.sh
```

### 2. Train a Single Model

```bash
python bonus/training/train.py --model resnet18 --epoch 10 --lr 0.001 --batch_size 128
```

### 3. Train All Models (Batch)

**Python (recommended):**
```bash
python bonus/scripts/train_all_models.py --epoch 5 
```

**Linux/Mac:**
```bash
bash bonus/scripts/train_all.sh
```

**Windows:**
```cmd
bonus\scripts\train_all.bat
```

### 4. Compare Model Performance

After training, compare all models:

```bash
python bonus/analysis/compare_all_models.py
```

This generates:
- `bonus/analysis/model_comparison.png` - Visual comparison chart
- `bonus/analysis/model_comparison_summary.json` - Detailed metrics

## Available Models

The implementation supports the following architectures:

- **LeNet**: Classic CNN architecture (LeCun et al., 1998)
- **ResNet18**: Residual network with pretrained ImageNet weights
- **VGG16**: Deep CNN with pretrained ImageNet weights
- **AlexNet**: Early deep CNN with pretrained ImageNet weights
- **SqueezeNet**: Lightweight CNN with pretrained ImageNet weights
- **ViT-B/16**: Vision Transformer with pretrained ImageNet weights
- **MY_NET**: Custom CNN with residual blocks and batch normalization

## Key Features

### Core Modules

- **`core/dataset.py`**: 
  - Handles GTSRB dataset loading
  - Automatic ROI (Region of Interest) cropping from CSV metadata
  - Support for custom transforms and data augmentation
  - Efficient data loading with PyTorch DataLoader

- **`core/models.py`**: 
  - Unified interface for loading all model architectures
  - Automatic model initialization with correct parameters
  - Support for pretrained weights and custom architectures

- **`core/config.py`**: 
  - Centralized configuration management
  - Dataset paths, hyperparameters, and model settings
  - Ensemble weights for model combination

### Training Features

- **Multiple Architectures**: Support for 7 different CNN architectures
- **Configurable Hyperparameters**: Learning rate, batch size, epochs, weight decay
- **WandB Integration**: Automatic experiment tracking and visualization
- **Checkpoint Management**: Automatic saving of best models
- **Progress Tracking**: Real-time training progress with tqdm
- **GPU Support**: Automatic CUDA detection and usage

### Analysis Tools

- **Model Comparison**: Compare accuracy, training time, and model size
- **Visualization Notebook**: Feature maps, Grad-CAM, and attention visualizations
- **Training Logs**: Detailed logs for each training run

## Configuration

Edit `core/config.py` to customize:

- **Dataset paths**: `DATASET_PATH`, `TRAIN_CSV`, `TEST_CSV`
- **Default hyperparameters**: Epochs, learning rate, batch size, weight decay
- **Model-specific learning rates**: Per-model learning rate configurations
- **Checkpoint and log directories**: Where models and logs are saved

## Dataset

**GTSRB Dataset** (German Traffic Sign Recognition Benchmark):
- **43 classes** of German traffic signs
- Pre-split into Train/Test sets
- Includes ROI coordinates for automatic cropping
- Meta images for each class (reference images)
- Dataset path: `datasets/dataset2/`

Expected structure:
```
datasets/dataset2/
├── Train.csv          # Training set with ROI coordinates
├── Test.csv           # Test set with ROI coordinates
├── Meta/              # Class reference images (0.png to 42.png)
├── Train/             # Training images organized by class
└── Test/              # Test images
```

## Training Workflow

### Single Model Training

```bash
# Basic training
python bonus/training/train.py --model resnet18

# With custom parameters
python bonus/training/train.py \
    --model vit_b_16 \
    --epoch 20 \
    --lr 0.0001 \
    --batch_size 64 \
    --weight_decay 0.0001
```

### Batch Training

Train all models sequentially:

```bash
python bonus/scripts/train_all_models.py --epoch 5 
```

This will:
1. Train each model with optimal hyperparameters
2. Save checkpoints to `bonus/checkpoints/`
3. Generate training logs in `bonus/logs/`

### WandB Integration

The training script automatically logs to WandB (Weights & Biases) if configured:

1. Create account at https://wandb.ai
2. Login: `wandb login`
3. Training metrics will be logged automatically

To disable WandB: `export WANDB_MODE=disabled`

## Testing

Run tests to verify the implementation:

```bash
# Install test dependencies
pip install -r bonus/requirements-test.txt

# Run all tests
pytest bonus/tests/

# Run with coverage
pytest bonus/tests/ --cov=bonus/core --cov=bonus/training --cov-report=html

# Or use the test runner script
bash bonus/scripts/run_tests.sh --coverage
```

See `bonus/tests/README.md` for detailed testing documentation.

## Model Performance

Typical performance on GTSRB test set (10 epochs, default hyperparameters):

| Model | Accuracy | Notes |
|-------|----------|-------|
| ViT-B/16 | ~98.7% | Best performance, requires more memory |
| AlexNet | ~98.5% | Good balance of speed and accuracy |
| ResNet18 | ~97.6% | Reliable and fast |
| MY_NET | ~97.4% | Custom architecture |
| VGG16 | ~96.8% | Deep architecture |
| SqueezeNet | ~97.0% | Lightweight model |
| LeNet | ~88.6% | Classic architecture |

*Note: Results may vary based on hyperparameters and training duration*

## Visualization

The `visualization.ipynb` notebook provides interactive visualizations:

- **Feature Maps**: Visualize intermediate CNN layer activations
- **Grad-CAM**: Highlight important image regions for predictions
- **Attention Maps**: Visualize Vision Transformer attention patterns

Open in Jupyter:
```bash
jupyter notebook bonus/visualization.ipynb
```

## Troubleshooting

### Common Issues

**Import Errors:**
```bash
# Make sure you're in the project root
cd /home/paradox/TrafficSignRecongnition
export PYTHONPATH="${PYTHONPATH}:$(pwd)/bonus"
```

**CUDA/GPU Issues:**
```python
import torch
print(torch.cuda.is_available())  # Should print True if GPU available
```

**Dataset Not Found:**
- Verify dataset is in `datasets/dataset2/`
- Check that `Train.csv` and `Test.csv` exist
- Verify paths in `bonus/core/config.py`

**Out of Memory:**
- Reduce batch size: `--batch_size 64`
- Use smaller model: `--model lenet`
- Enable gradient checkpointing for large models

## Requirements

See `requirements-bonus.txt` for complete dependency list. Key packages:

- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- wandb >= 0.13.0 (for experiment tracking)
- tqdm >= 4.64.0 (for progress bars)
- Pillow >= 9.0.0 (for image processing)

## Documentation

- **[SETUP.md](SETUP.md)**: Detailed environment setup guide
- **[README_TRAINING.md](README_TRAINING.md)**: Training workflow guide
- **[TESTING.md](TESTING.md)**: Testing documentation
- **[tests/README.md](tests/README.md)**: Test suite documentation

## License

This implementation is part of the Traffic Sign Recognition project.

## Contributing

When contributing to this project:
1. Follow the existing code structure
2. Add tests for new features
3. Update documentation as needed
4. Ensure all tests pass before submitting
