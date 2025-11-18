# Traffic Sign Recognition Project

Final Project for NUS Summer Camp - A comprehensive implementation of traffic sign recognition using classical computer vision and deep learning approaches.

## Project Overview

This project implements traffic sign recognition through three progressive difficulty levels, demonstrating the evolution from classical computer vision to modern deep learning:

- **Beginner Level**: Traditional feature extraction (HOG) + SVM classifier
- **Expert Level**: Advanced classical CV with multiple preprocessing, feature extraction, and classification methods
- **Bonus Level**: Deep learning with CNN architectures and transfer learning

## Project Structure

```
TrafficSignRecongnition/
├── beginner/              # Beginner level: HOG + SVM
│   ├── starter.py         # Complete implementation script
│   └── details.md         # Detailed explanation of concepts
│
├── expert/                # Expert level: Advanced classical CV
│   ├── concepts.ipynb     # Visualizations and concept explanations
│   └── task.ipynb         # Implementation with multiple methods
│
├── bonus/                 # Bonus level: Deep learning
│   ├── core/              # Core modules (dataset, models, config)
│   ├── training/          # Training scripts
│   ├── nets/              # Custom network definitions
│   ├── analysis/          # Model comparison and analysis
│   ├── scripts/           # Batch training scripts
│   ├── tests/             # Unit and integration tests
│   ├── visualization.ipynb # Model visualization notebook
│   ├── README.md          # Detailed bonus level documentation
│   └── SETUP.md           # Setup guide for bonus level
│
├── datasets/
│   ├── dataset1/          # Used by beginner/expert (58 classes)
│   └── dataset2/          # GTSRB dataset for bonus (43 classes)
│
├── requirements.txt       # Base dependencies
└── README.md              # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip or conda package manager

### Installation

1. **Clone the repository** (if applicable):
```bash
git clone <repository-url>
cd TrafficSignRecongnition
```

2. **Install base dependencies**:
```bash
pip install -r requirements.txt
```

This installs:
- numpy, scikit-learn, scikit-image
- opencv-python
- pandas, matplotlib
- jupyter/notebook

3. **For Bonus Level (Deep Learning)**, install additional dependencies:
```bash
# See bonus/SETUP.md for detailed instructions
pip install torch torchvision torchaudio wandb tqdm Pillow
```

## Level 1: Beginner - HOG + SVM

**Goal**: Learn basic computer vision pipeline using traditional methods.

### Approach
- **Feature Extraction**: HOG (Histogram of Oriented Gradients)
- **Classifier**: SVM (Support Vector Machine)
- **Dataset**: 5,998 images, 58 classes

### Key Concepts
- Image preprocessing (resize, grayscale)
- Feature extraction from images
- Train/test split
- Model training and evaluation

### Run Beginner Level

```bash
python beginner/starter.py
```

**Expected Output**: 
- Loads 5,998 images
- Extracts HOG features (128-D vectors)
- Trains SVM classifier
- Achieves ~90-95% accuracy

### What You'll Learn
- How to load and preprocess images
- Understanding HOG feature extraction
- Training and evaluating classifiers
- Basic machine learning workflow

**See `beginner/details.md` for detailed explanations.**

---

## Level 2: Expert - Advanced Classical CV

**Goal**: Explore multiple preprocessing, feature extraction, and classification methods.

### Approach
- **Preprocessing Methods**: Simple, Blur, Histogram Equalization, Advanced
- **Feature Extraction**: HOG, LBP (Local Binary Pattern), Color features
- **Classifiers**: SVM, Random Forest, k-NN, Decision Tree, Naive Bayes, MLP
- **Dataset**: Same as beginner (5,998 images, 58 classes)

### Key Concepts
- Comparing different preprocessing techniques
- Multiple feature extraction methods
- Classifier comparison and selection
- Dimensionality reduction (PCA)
- Model interpretability

### Run Expert Level

Open the Jupyter notebooks:

```bash
jupyter notebook expert/concepts.ipynb  # Visualizations and explanations
jupyter notebook expert/task.ipynb     # Implementation
```

**What You'll Learn**
- Impact of preprocessing on model performance
- Trade-offs between different feature extraction methods
- How different classifiers work
- Classical computer vision pipeline optimization

---

## Level 3: Bonus - Deep Learning

**Goal**: Implement modern deep learning approaches using CNNs and transfer learning.

### Approach
- **Models**: LeNet, ResNet18, VGG16, AlexNet, SqueezeNet, Vision Transformer, Custom Net
- **Dataset**: GTSRB (German Traffic Sign Recognition Benchmark) - 43 classes
- **Features**: Transfer learning, data augmentation, ensemble methods

### Key Concepts
- Convolutional Neural Networks (CNNs)
- Transfer learning with pretrained models
- Data augmentation and preprocessing
- Model training and checkpointing
- Experiment tracking with WandB

### Run Bonus Level

**Quick Start**:
```bash
# Train a single model
python bonus/training/train.py --model resnet18 --epoch 10

# Train all models
python bonus/scripts/train_all_models.py --epoch 5

# Compare models
python bonus/analysis/compare_all_models.py
```

**Expected Performance**: 
- Individual models: 88-99% accuracy
- Best model (ViT-B/16): ~98.7% accuracy

**See `bonus/README.md` and `bonus/SETUP.md` for detailed documentation.**

### What You'll Learn
- Deep learning architecture design
- Transfer learning strategies
- Training optimization techniques
- Model evaluation and comparison
- Visualization of learned features

---

## Dataset Information

### Dataset 1 (Beginner/Expert)
- **Source**: Traffic sign images
- **Size**: 5,998 images
- **Classes**: 58 different traffic sign types
- **Format**: PNG files named `XXX_yyyy.png` where `XXX` is class ID (000-057)
- **Location**: `datasets/dataset1/`

### Dataset 2 (Bonus)
- **Source**: GTSRB (German Traffic Sign Recognition Benchmark)
- **Size**: Pre-split train/test sets
- **Classes**: 43 German traffic sign classes
- **Features**: Includes ROI (Region of Interest) coordinates for cropping
- **Location**: `datasets/dataset2/`
- **Structure**: 
  - `Train.csv` / `Test.csv` with ROI metadata
  - `Meta/` directory with class reference images
  - Organized train/test directories

---

## Comparison: Classical CV vs Deep Learning

| Aspect | Beginner/Expert (Classical) | Bonus (Deep Learning) |
|--------|----------------------------|----------------------|
| **Approach** | Handcrafted features + shallow learning | End-to-end learning |
| **Features** | HOG, LBP (explicit) | Learned automatically |
| **Preprocessing** | Manual (resize, grayscale, etc.) | Data augmentation |
| **Classifier** | SVM, Random Forest, etc. | CNN architectures |
| **Accuracy** | ~90-95% | ~88-99% |
| **Interpretability** | High (understandable features) | Lower (black box) |
| **Training Time** | Minutes | Hours (depending on hardware) |
| **Data Requirements** | Moderate | Large (benefits from pretraining) |

---

## Results Summary

### Beginner Level
- **Method**: HOG + SVM
- **Accuracy**: ~90-95%
- **Training Time**: < 5 minutes
- **Key Insight**: Traditional features work well for structured problems

### Expert Level
- **Best Combination**: Varies by preprocessing/feature/classifier combination
- **Accuracy**: ~90-96% (depending on method)
- **Key Insight**: Preprocessing and feature selection significantly impact performance

### Bonus Level
- **Best Model**: Vision Transformer (ViT-B/16)
- **Accuracy**: ~98.7%
- **Training Time**: 1-2 hours per model (on GPU)
- **Key Insight**: Deep learning achieves highest accuracy with sufficient data

---

## Requirements by Level

### All Levels
```bash
pip install -r requirements.txt
```

### Bonus Level (Additional)
- PyTorch >= 1.12.0
- torchvision >= 0.13.0
- wandb >= 0.13.0 (for experiment tracking)
- CUDA-capable GPU (recommended, but CPU works)

See `bonus/requirements-bonus.txt` for complete list.

---

## Getting Started Guide

### For Beginners
1. Start with `beginner/starter.py`
2. Read `beginner/details.md` for explanations
3. Understand each step: loading → preprocessing → features → training

### For Intermediate Users
1. Complete beginner level first
2. Explore `expert/concepts.ipynb` for visualizations
3. Experiment with different combinations in `expert/task.ipynb`

### For Advanced Users
1. Review beginner/expert levels for context
2. Follow `bonus/SETUP.md` for environment setup
3. Train models and compare performance
4. Explore `bonus/visualization.ipynb` for model insights

---

## Project Highlights

- **Progressive Difficulty**: Three levels from basic to advanced
- **Multiple Approaches**: Classical CV and deep learning
- **Comprehensive Documentation**: Detailed explanations at each level
- **Production-Ready Code**: Well-structured, tested, and documented
- **Visualizations**: Jupyter notebooks for understanding concepts
- **Experiment Tracking**: WandB integration for bonus level

---

## Additional Resources

- **Presentation**: See `assets/slides/talk.pdf` for project overview
- **Project PDF**: See `TrafficSignProject.pdf` for detailed specifications
- **WandB Dashboard**: [View training metrics](https://wandb.ai/irides_paradox/Traffic%20Sign%20Recongnition) (for bonus level)

---

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Make sure you're in the project root
cd TrafficSignRecongnition
pip install -r requirements.txt
```

**Dataset Not Found**:
- Verify datasets are in `datasets/dataset1/` and `datasets/dataset2/`
- Check file paths in scripts

**CUDA/GPU Issues (Bonus Level)**:
- See `bonus/SETUP.md` for GPU setup
- Training works on CPU but is slower

**For detailed troubleshooting**, see:
- `bonus/SETUP.md` - Bonus level setup and troubleshooting
- `bonus/README.md` - Bonus level documentation

---

## Contributing

When contributing to this project:
1. Follow the existing code structure
2. Add tests for new features (bonus level)
3. Update documentation as needed
4. Ensure all tests pass before submitting

---

## License

This project is part of the NUS Summer Camp final project.

---

## Acknowledgments

- GTSRB dataset providers
- PyTorch and scikit-learn communities
- NUS Summer Camp instructors

---

## Next Steps

1. **Start with Beginner Level**: Run `python beginner/starter.py`
2. **Explore Expert Level**: Open `expert/task.ipynb` in Jupyter
3. **Try Bonus Level**: Follow `bonus/SETUP.md` and train models
4. **Compare Results**: Understand trade-offs between approaches
5. **Experiment**: Modify parameters and see how performance changes

**Happy Learning! 🚦**

