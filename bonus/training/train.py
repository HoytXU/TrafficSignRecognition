"""
Training Script for Traffic Sign Recognition

Trains deep learning models on GTSRB dataset with configurable hyperparameters.
Supports multiple architectures: LeNet, ResNet18, VGG16, AlexNet, SqueezeNet, ViT, Custom Net.
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import wandb
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Import from core modules
import sys
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)
from core.dataset import GTSRBDataset
from core.models import get_model
from core.config import DATASET_PATH, TRAIN_CSV, TEST_CSV, CHECKPOINT_DIR, DEFAULT_CONFIG


def get_transforms(augment=False):
    """
    Get data transformation pipeline.
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        transforms.Compose: Transformation pipeline
    """
    transform_list = [
        transforms.Resize((224, 224)),
    ]
    
    if augment:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ])
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transforms.Compose(transform_list)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    for images, labels in tqdm(dataloader, desc="Training", leave=False):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += images.size(0)

    epoch_loss = running_loss / total_samples
    epoch_accuracy = correct_predictions / total_samples
    
    return epoch_loss, epoch_accuracy


def evaluate(model, dataloader, device):
    """
    Evaluate model on test set.
    
    Returns:
        tuple: (accuracy, f1_score, predictions, true_labels, confusion_matrix)
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Testing", leave=False):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    cm = confusion_matrix(true_labels, predictions)
    
    return accuracy, f1_macro, predictions, true_labels, cm


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train traffic sign recognition model')
    parser.add_argument('--epoch', type=int, default=DEFAULT_CONFIG['epoch'], help='Number of epochs')
    parser.add_argument('--lr', type=float, default=DEFAULT_CONFIG['lr'], help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=DEFAULT_CONFIG['weight_decay'], help='Weight decay')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_CONFIG['batch_size'], help='Batch size')
    parser.add_argument('--model', type=str, default='resnet18', 
                       choices=['lenet', 'resnet18', 'vgg16', 'alexnet', 'squeezenet1_0', 'vit_b_16', 'my_net'],
                       help='Model architecture')
    parser.add_argument('--augment', action='store_true', help='Use data augmentation')
    parser.add_argument('--use_wandb', action='store_true', help='Log to Weights & Biases')
    args = parser.parse_args()

    # Initialize WandB if requested
    if args.use_wandb:
        run_name = f"{args.model}-epoch{args.epoch}-lr{args.lr}-bs{args.batch_size}-wd{args.weight_decay}"
        wandb.init(project="Traffic Sign Recognition", name=run_name)
        wandb.config.update(args)

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Create datasets and dataloaders
    train_transform = get_transforms(augment=args.augment)
    test_transform = get_transforms(augment=False)
    
    train_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TRAIN_CSV, transform=train_transform)
    test_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TEST_CSV, transform=test_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = get_model(args.model, num_classes=DEFAULT_CONFIG['num_classes'], pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_accuracy = 0.0
    best_f1_score = 0.0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 
                                   f"{args.model}_epoch{args.epoch}_lr{args.lr}_wd{args.weight_decay}.pt")

    for epoch in range(args.epoch):
        # Train
        train_loss, train_accuracy = train_epoch(model, train_dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{args.epoch} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.4f}")

        # Evaluate
        test_accuracy, f1_macro, predictions, true_labels, cm = evaluate(model, test_dataloader, device)
        print(f"Test Accuracy: {test_accuracy:.4f} - Test F1-score (macro): {f1_macro:.4f}")

        # Save best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_f1_score = f1_macro
            torch.save(model.state_dict(), checkpoint_path)
            print(f"✓ Saved best model (Accuracy: {best_accuracy:.4f})")

        # Log to WandB
        if args.use_wandb:
            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Training Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Test F1-score (macro)": f1_macro
            })
            wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=predictions,
                class_names=np.arange(len(test_dataset.classes)),
                title="Confusion Matrix"
            )})

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best Accuracy: {best_accuracy:.4f}")
    print(f"Best F1-score: {best_f1_score:.4f}")
    print(f"Model saved to: {checkpoint_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

