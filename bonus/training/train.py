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
from datetime import datetime
import time

# Import from core modules
import sys
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)
from core.dataset import GTSRBDataset
from core.models import get_model
from core.config import DATASET_PATH, TRAIN_CSV, TEST_CSV, CHECKPOINT_DIR, DEFAULT_CONFIG


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    GRAY = '\033[90m'


def print_header(text, char='='):
    """Print a formatted header."""
    width = 70
    print(f"\n{Colors.HEADER}{Colors.BOLD}{char * width}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text.center(width)}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}{char * width}{Colors.ENDC}\n")


def print_info(text):
    """Print info message."""
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def print_success(text):
    """Print success message."""
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def format_metric(name, value, color=None):
    """Format a metric for display."""
    if color is None:
        color = Colors.OKCYAN
    return f"{color}{name}: {Colors.BOLD}{value:.4f}{Colors.ENDC}"


def print_epoch_summary(epoch, total_epochs, train_loss, train_acc, test_acc, test_f1, 
                        best_acc, epoch_time, is_best=False):
    """Print formatted epoch summary."""
    progress = (epoch + 1) / total_epochs * 100
    bar_length = 30
    filled = int(bar_length * progress / 100)
    bar = '█' * filled + '░' * (bar_length - filled)
    
    print(f"\n{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
    print(f"{Colors.BOLD}Epoch {epoch+1}/{total_epochs} [{bar}] {progress:.1f}%{Colors.ENDC}")
    print(f"{Colors.BOLD}{'─' * 70}{Colors.ENDC}")
    
    print(f"  {format_metric('Train Loss', train_loss, Colors.FAIL)}")
    print(f"  {format_metric('Train Acc', train_acc, Colors.OKGREEN)}")
    print(f"  {format_metric('Test Acc', test_acc, Colors.OKBLUE)}")
    print(f"  {format_metric('Test F1', test_f1, Colors.OKBLUE)}")
    print(f"  {format_metric('Best Acc', best_acc, Colors.WARNING)}")
    print(f"  {Colors.GRAY}Time: {epoch_time:.2f}s{Colors.ENDC}")
    
    if is_best:
        print(f"  {Colors.OKGREEN}{Colors.BOLD}★ New best model saved!{Colors.ENDC}")


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


def train_epoch(model, dataloader, criterion, optimizer, device, epoch=None, total_epochs=None):
    """
    Train for one epoch.
    
    Returns:
        tuple: (average_loss, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # Create progress bar with dynamic metrics
    epoch_prefix = f"Epoch {epoch+1}/{total_epochs} - " if epoch is not None else ""
    pbar = tqdm(dataloader, desc=f"{epoch_prefix}Training", 
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += batch_size
        
        # Update progress bar with real-time metrics
        current_loss = running_loss / total_samples
        current_acc = correct_predictions / total_samples
        pbar.set_postfix({
            'loss': f'{current_loss:.4f}',
            'acc': f'{current_acc:.4f}'
        })

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
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Evaluating", 
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]')
        
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar with real-time accuracy
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            current_acc = correct / total if total > 0 else 0.0
            pbar.set_postfix({'acc': f'{current_acc:.4f}'})

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

    # Print startup header
    print_header("Traffic Sign Recognition - Training", '═')
    print_info(f"Model: {Colors.BOLD}{args.model.upper()}{Colors.ENDC}")
    print_info(f"Epochs: {Colors.BOLD}{args.epoch}{Colors.ENDC}")
    print_info(f"Learning Rate: {Colors.BOLD}{args.lr}{Colors.ENDC}")
    print_info(f"Batch Size: {Colors.BOLD}{args.batch_size}{Colors.ENDC}")
    print_info(f"Weight Decay: {Colors.BOLD}{args.weight_decay}{Colors.ENDC}")
    print_info(f"Data Augmentation: {Colors.BOLD}{'Enabled' if args.augment else 'Disabled'}{Colors.ENDC}")
    print_info(f"WandB Logging: {Colors.BOLD}{'Enabled' if args.use_wandb else 'Disabled'}{Colors.ENDC}")

    # Initialize WandB if requested
    if args.use_wandb:
        run_name = f"{args.model}-epoch{args.epoch}-lr{args.lr}-bs{args.batch_size}-wd{args.weight_decay}"
        wandb.init(project="Traffic Sign Recognition", name=run_name)
        wandb.config.update(args)
        print_success(f"WandB initialized: {run_name}")

    # Create directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    print_info(f"Checkpoint directory: {CHECKPOINT_DIR}")

    # Create datasets and dataloaders
    print_info("Loading datasets...")
    train_transform = get_transforms(augment=args.augment)
    test_transform = get_transforms(augment=False)
    
    train_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TRAIN_CSV, transform=train_transform)
    test_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TEST_CSV, transform=test_transform)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    print_success(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print_success(f"Number of classes: {DEFAULT_CONFIG['num_classes']}")

    # Initialize model
    print_info(f"Initializing {args.model} model...")
    model = get_model(args.model, num_classes=DEFAULT_CONFIG['num_classes'], pretrained=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_success(f"Device: {Colors.BOLD}{device}{Colors.ENDC}")
    print_success(f"Total parameters: {Colors.BOLD}{total_params:,}{Colors.ENDC}")
    print_success(f"Trainable parameters: {Colors.BOLD}{trainable_params:,}{Colors.ENDC}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    best_accuracy = 0.0
    best_f1_score = 0.0
    best_epoch = 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 
                                   f"{args.model}_epoch{args.epoch}_lr{args.lr}_wd{args.weight_decay}.pt")
    
    # Track training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'test_acc': [],
        'test_f1': [],
        'epoch_times': []
    }
    
    start_time = time.time()
    print_header("Starting Training", '─')

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_accuracy = train_epoch(
            model, train_dataloader, criterion, optimizer, device, 
            epoch=epoch, total_epochs=args.epoch
        )

        # Evaluate
        test_accuracy, f1_macro, predictions, true_labels, cm = evaluate(model, test_dataloader, device)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        # Track history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_accuracy)
        history['test_acc'].append(test_accuracy)
        history['test_f1'].append(f1_macro)
        history['epoch_times'].append(epoch_time)
        
        # Check if best model
        is_best = test_accuracy > best_accuracy
        if is_best:
            best_accuracy = test_accuracy
            best_f1_score = f1_macro
            best_epoch = epoch + 1
            torch.save(model.state_dict(), checkpoint_path)

        # Print epoch summary
        print_epoch_summary(
            epoch, args.epoch, train_loss, train_accuracy, 
            test_accuracy, f1_macro, best_accuracy, epoch_time, is_best
        )

        # Log to WandB
        if args.use_wandb:
            wandb.log({
                "Epoch": epoch + 1,
                "Training Loss": train_loss,
                "Training Accuracy": train_accuracy,
                "Test Accuracy": test_accuracy,
                "Test F1-score (macro)": f1_macro,
                "Epoch Time": epoch_time
            })
            wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(
                probs=None,
                y_true=true_labels,
                preds=predictions,
                class_names=np.arange(len(test_dataset.classes)),
                title="Confusion Matrix"
            )})

    # Final summary
    total_time = time.time() - start_time
    avg_epoch_time = np.mean(history['epoch_times'])
    
    print_header("Training Complete!", '═')
    print(f"\n{Colors.BOLD}Summary:{Colors.ENDC}")
    print(f"  {format_metric('Best Accuracy', best_accuracy, Colors.OKGREEN)} (Epoch {best_epoch})")
    print(f"  {format_metric('Best F1-score', best_f1_score, Colors.OKGREEN)}")
    print(f"  {format_metric('Final Train Loss', history['train_loss'][-1], Colors.FAIL)}")
    print(f"  {format_metric('Final Train Acc', history['train_acc'][-1], Colors.OKCYAN)}")
    print(f"  {format_metric('Final Test Acc', history['test_acc'][-1], Colors.OKBLUE)}")
    print(f"\n{Colors.BOLD}Timing:{Colors.ENDC}")
    print(f"  {Colors.GRAY}Total Time: {total_time/60:.2f} minutes ({total_time:.2f}s){Colors.ENDC}")
    print(f"  {Colors.GRAY}Average Epoch Time: {avg_epoch_time:.2f}s{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Model Saved:{Colors.ENDC}")
    print(f"  {Colors.OKCYAN}{checkpoint_path}{Colors.ENDC}")
    
    # Show improvement metrics
    if len(history['test_acc']) > 1:
        improvement = history['test_acc'][-1] - history['test_acc'][0]
        print(f"\n{Colors.BOLD}Improvement:{Colors.ENDC}")
        if improvement > 0:
            print(f"  {Colors.OKGREEN}Test Accuracy improved by {improvement:.4f} ({improvement*100:.2f}%){Colors.ENDC}")
        else:
            print(f"  {Colors.WARNING}Test Accuracy changed by {improvement:.4f} ({improvement*100:.2f}%){Colors.ENDC}")
    
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'═' * 70}{Colors.ENDC}\n")


if __name__ == "__main__":
    main()

