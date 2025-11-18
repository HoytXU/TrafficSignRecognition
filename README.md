# Traffic Sign Recognition Project

This is the final Project for NUS Summer Camp, a comprehensive implementation of traffic sign recognition using classical computer vision and deep learning approaches.

## Quick Links

**Jupyter Notebooks** (View online, no setup required)
- [Vision before Neural Network era](https://nbviewer.org/github/HoytXU/TrafficSignRecongnition/blob/master/expert/concepts.ipynb) - These are our exploration for classical computer vision concepts, with visualizations
- [Deep Learning Model Visualization](https://nbviewer.org/github/HoytXU/TrafficSignRecongnition/blob/master/bonus/visualization.ipynb) - We visualized how convolutional neural networks process images. Also explore ViT's (Vision Transformer) feature extraction process.

**Datasets and Models**
If you want to replicate the work, you can get the dataset and pretrained checkpoints from here.
- [Hugging Face Dataset](https://huggingface.co/datasets/IridesParadox/TrafficSignRecognition) - Download Dataset 1 (5,998 images, 58 classes) and Dataset 2 (GTSRB, 43 classes)

Put the dataset under `datasets/dataset1` and `datasets/dataset2`.

- [Pre-trained Model Checkpoints](https://huggingface.co/IridesParadox/TrafficSignRecognition_checkpoints) - Trained model weights for all architectures.

Put the checkpoints under `bonus/checkpoints`.

**Documentation**
- [Presentation Slides](https://github.com/HoytXU/TrafficSignRecongnition/blob/master/assets/slides/talk.pdf) - Project overview, methodology, and results

---

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

**Prerequisites**: Python 3.8+, pip/conda

```bash
git clone <repository-url> && cd TrafficSignRecongnition
pip install -r requirements.txt
pip install torch torchvision torchaudio wandb tqdm Pillow  # For bonus level
```

## Level 1: Beginner - HOG + SVM

**HOG feature extraction + SVM classifier** | 5,998 images, 58 classes | ~90-95% accuracy

```bash
python beginner/starter.py
```

See `beginner/details.md` for detailed explanations.

---

## Level 2: Expert - Advanced Classical CV

**Preprocessing**: Simple, Blur, Histogram Equalization, Advanced | **Features**: HOG, LBP, Color features | **Classifiers**: SVM, Random Forest, k-NN, Decision Tree, Naive Bayes, MLP

```bash
jupyter notebook expert/concepts.ipynb  # Visualizations
jupyter notebook expert/task.ipynb     # Implementation
```

---

## Level 3: Bonus - Deep Learning

**Models**: LeNet, ResNet18, VGG16, AlexNet, SqueezeNet, Vision Transformer, Custom Net | **Dataset**: GTSRB (43 classes) | **Best Performance**: ViT-B/16 ~98.7% accuracy

```bash
python bonus/training/train.py --model resnet18 --epoch 10  # Single model
python bonus/scripts/train_all_models.py --epoch 5          # All models
python bonus/analysis/compare_all_models.py                 # Compare results
```

See `bonus/README.md` and `bonus/SETUP.md` for detailed documentation.

---

## Datasets

**Dataset 1** (Beginner/Expert): 5,998 images, 58 classes | `datasets/dataset1/` | Format: `XXX_yyyy.png` where `XXX` is class ID  
**Dataset 2** (Bonus): GTSRB, 43 classes, pre-split train/test with ROI coordinates | `datasets/dataset2/`

![Dataset Analysis](bonus/analysis/data_analysis.png)

## Model Comparison

We evaluated 7 deep learning architectures on the GTSRB dataset (43 classes). All models were trained with transfer learning (ImageNet pretrained weights except LeNet and MY_NET), optimized hyperparameters, and evaluated on the test set.

**Performance Summary** (sorted by test accuracy):
- **ViT-B/16** (Vision Transformer): ~98.7% accuracy | Best overall performance, transformer-based architecture with self-attention mechanisms, requires more GPU memory (~330M parameters)
- **AlexNet**: ~98.5% accuracy | Excellent speed-accuracy tradeoff, lightweight (~60M parameters), fast inference
- **ResNet18**: ~97.6% accuracy | Reliable residual learning, good generalization, moderate size (~11M parameters)
- **MY_NET** (Custom CNN): ~97.4% accuracy | Custom architecture with residual blocks and batch normalization, designed specifically for traffic signs (~2M parameters)
- **SqueezeNet**: ~97.0% accuracy | Extremely lightweight (~1.2M parameters), suitable for edge devices, competitive performance
- **VGG16**: ~96.8% accuracy | Deep architecture (~138M parameters), strong feature extraction but slower inference
- **LeNet**: ~88.6% accuracy | Classic CNN architecture, baseline comparison, limited capacity (~60K parameters)

**Key Metrics Compared**: Test Accuracy, F1-score (macro), Training Convergence (epochs), Train/Test Gap (generalization), Model Size, Inference Speed

**Training Configuration**: Batch size 128, Adam optimizer, learning rate 0.001-0.0001 (model-specific), 10-20 epochs, data augmentation (rotation, translation, color jitter), ROI cropping from GTSRB metadata

**Findings**: Vision Transformer achieves highest accuracy through global attention mechanisms, while AlexNet provides best efficiency-accuracy balance. Transfer learning significantly improves performance (10-15% accuracy gain) compared to training from scratch. All models converge within 10-15 epochs with proper hyperparameter tuning.

![Model Comparison](bonus/analysis/model_comparison.png)

*Run `python bonus/analysis/compare_all_models.py` to regenerate comparison charts and detailed metrics JSON.*

---

## Requirements

```bash
pip install -r requirements.txt                    # Base dependencies
pip install torch torchvision torchaudio wandb tqdm Pillow  # Bonus level
```

