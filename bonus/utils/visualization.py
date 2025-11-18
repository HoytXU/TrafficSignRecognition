"""
Visualization utilities for training feedback.

Provides real-time and post-training visualizations including:
- Loss and accuracy curves
- Confusion matrices
- Sample predictions
- Per-class accuracy
- Learning rate schedules
"""

import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix
import seaborn as sns
from typing import List, Tuple, Optional, Dict
from collections import defaultdict


class TrainingVisualizer:
    """Manages all training visualizations and feedback."""
    
    def __init__(self, output_dir: str, model_name: str, num_classes: int = 43):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualization files
            model_name: Name of the model being trained
            num_classes: Number of classes in the dataset
        """
        self.output_dir = output_dir
        self.model_name = model_name
        self.num_classes = num_classes
        os.makedirs(output_dir, exist_ok=True)
        
        # Track training history
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
        self.test_f1_scores = []
        self.learning_rates = []
        self.epochs = []
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def update_epoch(self, epoch: int, train_loss: float, train_acc: float, 
                     test_acc: float, test_f1: float, lr: float):
        """Update training metrics for current epoch."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.test_accuracies.append(test_acc)
        self.test_f1_scores.append(test_f1)
        self.learning_rates.append(lr)
    
    def plot_training_curves(self, save_path: Optional[str] = None, show: bool = False):
        """
        Plot training and validation curves (loss, accuracy, F1-score).
        
        Args:
            save_path: Path to save the figure (default: auto-generated)
            show: Whether to display the plot
        """
        if not self.epochs:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Training Progress - {self.model_name}', fontsize=16, fontweight='bold')
        
        # Loss curve
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
        axes[0, 0].set_xlabel('Epoch', fontsize=12)
        axes[0, 0].set_ylabel('Loss', fontsize=12)
        axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].legend()
        
        # Accuracy curves
        axes[0, 1].plot(self.epochs, self.train_accuracies, 'g-o', label='Train Accuracy', 
                        linewidth=2, markersize=6)
        axes[0, 1].plot(self.epochs, self.test_accuracies, 'r-o', label='Test Accuracy', 
                        linewidth=2, markersize=6)
        axes[0, 1].set_xlabel('Epoch', fontsize=12)
        axes[0, 1].set_ylabel('Accuracy', fontsize=12)
        axes[0, 1].set_title('Accuracy Curves', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].legend()
        axes[0, 1].set_ylim([0, 1])
        
        # F1-score curve
        axes[1, 0].plot(self.epochs, self.test_f1_scores, 'm-o', label='Test F1-score', 
                        linewidth=2, markersize=6)
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('F1-score', fontsize=12)
        axes[1, 0].set_title('F1-score (Macro)', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].legend()
        axes[1, 0].set_ylim([0, 1])
        
        # Learning rate curve
        axes[1, 1].plot(self.epochs, self.learning_rates, 'c-o', label='Learning Rate', 
                        linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Epoch', fontsize=12)
        axes[1, 1].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_training_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Saved training curves to: {save_path}")
    
    def plot_confusion_matrix(self, y_true: List[int], y_pred: List[int], 
                              class_names: Optional[List[str]] = None,
                              save_path: Optional[str] = None, show: bool = False):
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # Absolute values
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                    xticklabels=class_names if class_names else range(self.num_classes),
                    yticklabels=class_names if class_names else range(self.num_classes),
                    cbar_kws={'label': 'Count'})
        axes[0].set_title('Confusion Matrix (Absolute)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted Label', fontsize=12)
        axes[0].set_ylabel('True Label', fontsize=12)
        
        # Normalized values
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
                    xticklabels=class_names if class_names else range(self.num_classes),
                    yticklabels=class_names if class_names else range(self.num_classes),
                    cbar_kws={'label': 'Normalized'})
        axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Predicted Label', fontsize=12)
        axes[1].set_ylabel('True Label', fontsize=12)
        
        plt.suptitle(f'Confusion Matrix - {self.model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_confusion_matrix.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Saved confusion matrix to: {save_path}")
    
    def plot_per_class_accuracy(self, y_true: List[int], y_pred: List[int],
                                class_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None, show: bool = False):
        """
        Plot per-class accuracy.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Optional list of class names
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)
        
        # Sort by accuracy
        sorted_indices = np.argsort(per_class_acc)
        sorted_acc = per_class_acc[sorted_indices]
        sorted_classes = [class_names[i] if class_names else f'Class {i}' 
                         for i in sorted_indices]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = ['green' if acc > 0.8 else 'orange' if acc > 0.5 else 'red' 
                 for acc in sorted_acc]
        bars = ax.barh(range(len(sorted_acc)), sorted_acc, color=colors, alpha=0.7)
        
        ax.set_yticks(range(len(sorted_acc)))
        ax.set_yticklabels(sorted_classes, fontsize=8)
        ax.set_xlabel('Accuracy', fontsize=12)
        ax.set_ylabel('Class', fontsize=12)
        ax.set_title(f'Per-Class Accuracy - {self.model_name}', fontsize=14, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, sorted_acc)):
            ax.text(acc + 0.01, i, f'{acc:.3f}', va='center', fontsize=8)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_per_class_accuracy.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Saved per-class accuracy to: {save_path}")
    
    def plot_sample_predictions(self, model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
                                device: torch.device, num_samples: int = 16,
                                class_names: Optional[List[str]] = None,
                                save_path: Optional[str] = None, show: bool = False):
        """
        Visualize sample predictions from the model.
        
        Args:
            model: Trained model
            dataloader: DataLoader for samples
            device: Device to run inference on
            num_samples: Number of samples to visualize
            class_names: Optional list of class names
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        model.eval()
        
        # Get a batch of samples
        images, labels = next(iter(dataloader))
        images = images.to(device)
        labels = labels.to(device)
        
        with torch.no_grad():
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidences = probabilities.max(dim=1)[0]
        
        # Convert to numpy
        images_np = images.cpu().numpy()
        labels_np = labels.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        confidences_np = confidences.cpu().numpy()
        
        # Denormalize images (assuming normalization was mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        images_np = images_np * 0.5 + 0.5
        images_np = np.clip(images_np, 0, 1)
        images_np = np.transpose(images_np, (0, 2, 3, 1))
        
        # Create grid
        num_samples = min(num_samples, len(images))
        cols = 4
        rows = (num_samples + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        fig.suptitle(f'Sample Predictions - {self.model_name}', fontsize=16, fontweight='bold')
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx in range(num_samples):
            row = idx // cols
            col = idx % cols
            ax = axes[row, col]
            
            # Display image
            ax.imshow(images_np[idx])
            ax.axis('off')
            
            # Get labels
            true_label = labels_np[idx]
            pred_label = predicted_np[idx]
            confidence = confidences_np[idx]
            
            true_name = class_names[true_label] if class_names else f'Class {true_label}'
            pred_name = class_names[pred_label] if class_names else f'Class {pred_label}'
            
            # Color based on correctness
            color = 'green' if true_label == pred_label else 'red'
            title = f'True: {true_name}\nPred: {pred_name}\nConf: {confidence:.2f}'
            
            ax.set_title(title, fontsize=10, color=color, fontweight='bold')
        
        # Hide empty subplots
        for idx in range(num_samples, rows * cols):
            row = idx // cols
            col = idx % cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_sample_predictions.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Saved sample predictions to: {save_path}")
    
    def plot_training_summary(self, best_accuracy: float, best_f1: float,
                             save_path: Optional[str] = None, show: bool = False):
        """
        Create a summary visualization with key metrics.
        
        Args:
            best_accuracy: Best test accuracy achieved
            best_f1: Best F1-score achieved
            save_path: Path to save the figure
            show: Whether to display the plot
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('off')
        
        # Create summary text
        summary_text = f"""
        Training Summary - {self.model_name}
        
        {'='*50}
        
        Best Test Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)
        Best F1-score (Macro): {best_f1:.4f}
        
        Final Training Loss: {self.train_losses[-1]:.4f}
        Final Training Accuracy: {self.train_accuracies[-1]:.4f}
        Final Test Accuracy: {self.test_accuracies[-1]:.4f}
        
        Total Epochs: {len(self.epochs)}
        Final Learning Rate: {self.learning_rates[-1]:.6f}
        
        {'='*50}
        """
        
        ax.text(0.5, 0.5, summary_text, transform=ax.transAxes,
                fontsize=14, verticalalignment='center', horizontalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_summary.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        if show:
            plt.show()
        else:
            plt.close()
        
        print(f"✓ Saved training summary to: {save_path}")
    
    def save_training_history(self, save_path: Optional[str] = None):
        """Save training history to a text file."""
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'{self.model_name}_training_history.txt')
        
        with open(save_path, 'w') as f:
            f.write(f"Training History - {self.model_name}\n")
            f.write("="*60 + "\n\n")
            f.write(f"{'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Test Acc':<12} {'Test F1':<12} {'LR':<12}\n")
            f.write("-"*60 + "\n")
            
            for epoch, loss, train_acc, test_acc, test_f1, lr in zip(
                self.epochs, self.train_losses, self.train_accuracies,
                self.test_accuracies, self.test_f1_scores, self.learning_rates
            ):
                f.write(f"{epoch:<8} {loss:<12.4f} {train_acc:<12.4f} {test_acc:<12.4f} "
                       f"{test_f1:<12.4f} {lr:<12.6f}\n")
        
        print(f"✓ Saved training history to: {save_path}")



