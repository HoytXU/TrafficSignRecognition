"""
Model Information Utilities

Visualizes and saves model architectures for documentation.
"""

import os
import torch
from core.models import get_model
from core.config import CHECKPOINT_DIR, DEFAULT_CONFIG, AVAILABLE_MODELS


def save_model_structure(model_name, output_dir=None):
    """
    Save model architecture to text file.
    
    Args:
        model_name: Name of the model
        output_dir: Directory to save structure file (default: nets/)
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "nets")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create model
    model = get_model(model_name, num_classes=DEFAULT_CONFIG['num_classes'], pretrained=False)
    
    # Save structure
    output_path = os.path.join(output_dir, f"{model_name}_structure.txt")
    with open(output_path, "w") as f:
        f.write(f"Model Architecture: {model_name}\n")
        f.write("=" * 60 + "\n\n")
        f.write(str(model))
        f.write("\n\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
        f.write(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")
    
    print(f"✓ Saved {model_name} structure to {output_path}")
    return output_path


def print_model_summary(model_name):
    """
    Print model architecture summary.
    
    Args:
        model_name: Name of the model
    """
    model = get_model(model_name, num_classes=DEFAULT_CONFIG['num_classes'], pretrained=False)
    
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    print(model)
    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"{'='*60}\n")


def save_all_model_structures(output_dir=None):
    """Save structures for all available models."""
    print("Saving model structures...")
    for model_name in AVAILABLE_MODELS:
        try:
            save_model_structure(model_name, output_dir)
        except Exception as e:
            print(f"⚠ Error saving {model_name}: {e}")


if __name__ == "__main__":
    # Save all model structures
    save_all_model_structures()

