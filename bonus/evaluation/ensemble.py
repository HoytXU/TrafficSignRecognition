"""
Ensemble Evaluation Script

Evaluates multiple models using weighted voting ensemble.
Combines predictions from multiple trained models for better accuracy.
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

# Import from core modules
import sys
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)
from core.dataset import GTSRBDataset
from core.models import get_model
from core.config import DATASET_PATH, TEST_CSV, CHECKPOINT_DIR, DEFAULT_CONFIG, ENSEMBLE_WEIGHTS, AVAILABLE_MODELS


def load_model(model_name, checkpoint_path, device, num_classes=43):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_name: Name of the model
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        num_classes: Number of classes
        
    Returns:
        torch.nn.Module: Loaded model in evaluation mode
    """
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def ensemble_predict(models, model_weights, dataloader, device, num_classes=43):
    """
    Perform ensemble prediction using weighted voting.
    
    Args:
        models: List of trained models
        model_weights: List of weights for each model
        dataloader: DataLoader for test data
        device: Device to run inference on
        num_classes: Number of classes
        
    Returns:
        tuple: (predictions, true_labels)
    """
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Ensemble Prediction"):
            images = images.to(device)
            labels = labels.to(device)

            # Get predictions from each model
            model_probabilities = []
            for model, weight in zip(models, model_weights):
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)[:, :num_classes]
                model_probabilities.append(probabilities.cpu().numpy() * weight)

            # Weighted voting
            ensemble_probabilities = np.sum(model_probabilities, axis=0)
            ensemble_probabilities /= np.sum(ensemble_probabilities, axis=1, keepdims=True)

            # Get predictions
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)

            predictions.extend(ensemble_predictions)
            true_labels.extend(labels.cpu().numpy())

    return np.array(predictions), np.array(true_labels)


def main():
    """Main ensemble evaluation function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get checkpoint paths
    checkpoint_paths = [
        os.path.join(CHECKPOINT_DIR, f"{model}_epoch10_lr0.001_wd0.001.pt")
        for model in AVAILABLE_MODELS
    ]
    # Special case for ViT (different learning rate)
    checkpoint_paths[AVAILABLE_MODELS.index('vit_b_16')] = os.path.join(
        CHECKPOINT_DIR, "vit_b_16_epoch10_lr0.0001_wd0.001.pt"
    )
    
    # Get model weights
    model_weights = [ENSEMBLE_WEIGHTS[model] for model in AVAILABLE_MODELS]

    # Load all models
    print("Loading models...")
    models = []
    for model_name, checkpoint_path in zip(AVAILABLE_MODELS, checkpoint_paths):
        if os.path.exists(checkpoint_path):
            model = load_model(model_name, checkpoint_path, device, DEFAULT_CONFIG['num_classes'])
            models.append(model)
            print(f"✓ Loaded {model_name}")
        else:
            print(f"⚠ Checkpoint not found: {checkpoint_path}")

    if len(models) == 0:
        print("Error: No models loaded!")
        return

    # Create test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TEST_CSV, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Perform ensemble prediction
    print("\nPerforming ensemble prediction...")
    predictions, true_labels = ensemble_predict(
        models, model_weights[:len(models)], test_dataloader, device, DEFAULT_CONFIG['num_classes']
    )

    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')

    print(f"\n{'='*60}")
    print(f"Ensemble Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-score (macro): {f1_macro:.4f}")
    print(f"  Models used: {len(models)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

