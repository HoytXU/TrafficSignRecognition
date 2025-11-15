"""
Failure Analysis Script

Analyzes misclassified samples and saves visualizations comparing
true labels vs predicted labels for error analysis.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
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
from core.config import (DATASET_PATH, TEST_CSV, META_PATH, FAIL_EXAMPLES_DIR, 
                        CHECKPOINT_DIR, DEFAULT_CONFIG, ENSEMBLE_WEIGHTS, AVAILABLE_MODELS)


def load_model(model_name, checkpoint_path, device, num_classes=43):
    """Load a trained model from checkpoint."""
    model = get_model(model_name, num_classes=num_classes, pretrained=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)
    return model


def ensemble_predict(models, model_weights, dataloader, device, num_classes=43):
    """
    Perform ensemble prediction and return failed indices.
    
    Returns:
        tuple: (predictions, true_labels, failed_indices)
    """
    predictions = []
    true_labels = []
    failed_indices = []

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Predicting")):
            images = images.to(device)
            labels = labels.to(device)

            # Ensemble prediction
            model_probabilities = []
            for model, weight in zip(models, model_weights):
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)[:, :num_classes]
                model_probabilities.append(probabilities.cpu().numpy() * weight)

            ensemble_probabilities = np.sum(model_probabilities, axis=0)
            ensemble_probabilities /= np.sum(ensemble_probabilities, axis=1, keepdims=True)
            ensemble_predictions = np.argmax(ensemble_probabilities, axis=1)

            # Track predictions and failures
            predictions.extend(ensemble_predictions)
            true_labels.extend(labels.cpu().numpy())
            
            # Track failed indices
            batch_failed = [
                batch_idx * dataloader.batch_size + idx 
                for idx, (true, pred) in enumerate(zip(labels.cpu().numpy(), ensemble_predictions))
                if true != pred
            ]
            failed_indices.extend(batch_failed)

    return np.array(predictions), np.array(true_labels), failed_indices


def visualize_failures(test_dataset, predictions, failed_indices, meta_images, output_dir):
    """
    Visualize failed predictions.
    
    Creates comparison images showing:
    - Original image
    - Preprocessed image
    - True label (from meta)
    - Predicted label (from meta)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nSaving {len(failed_indices)} failure examples...")
    
    for idx in tqdm(failed_indices, desc="Saving failures"):
        image, true_label = test_dataset[idx]
        pred_label = predictions[idx]
        
        # Convert tensor to PIL for visualization
        image_pil = transforms.ToPILImage()(image)
        original_image = test_dataset.get_original_image(idx)
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title(f"Original Image\n{os.path.basename(test_dataset.data[idx][0])}")
        axes[0, 0].axis('off')
        
        # Preprocessed image
        axes[0, 1].imshow(image_pil)
        axes[0, 1].set_title(f"Preprocessed Image")
        axes[0, 1].axis('off')
        
        # True label
        true_image = meta_images[true_label]
        axes[1, 0].imshow(true_image)
        axes[1, 0].set_title(f"True Label: {true_label}")
        axes[1, 0].axis('off')
        
        # Predicted label
        pred_image = meta_images[pred_label]
        axes[1, 1].imshow(pred_image)
        axes[1, 1].set_title(f"Predicted Label: {pred_label}")
        axes[1, 1].axis('off')
        
        # Save
        plt.savefig(os.path.join(output_dir, f"fail_{idx}.png"))
        plt.close()


def main():
    """Main failure analysis function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Setup output directory
    output_dir = os.path.join(FAIL_EXAMPLES_DIR, "ensemble_failures")
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models (same as ensemble.py)
    checkpoint_paths = [
        os.path.join(CHECKPOINT_DIR, f"{model}_epoch10_lr0.001_wd0.001.pt")
        for model in AVAILABLE_MODELS
    ]
    checkpoint_paths[AVAILABLE_MODELS.index('vit_b_16')] = os.path.join(
        CHECKPOINT_DIR, "vit_b_16_epoch10_lr0.0001_wd0.001.pt"
    )
    
    model_weights = [ENSEMBLE_WEIGHTS[model] for model in AVAILABLE_MODELS]
    
    print("Loading models...")
    models = []
    for model_name, checkpoint_path in zip(AVAILABLE_MODELS, checkpoint_paths):
        if os.path.exists(checkpoint_path):
            model = load_model(model_name, checkpoint_path, device, DEFAULT_CONFIG['num_classes'])
            models.append(model)
            print(f"✓ Loaded {model_name}")
    
    if len(models) == 0:
        print("Error: No models loaded!")
        return
    
    # Create test dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    test_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TEST_CSV, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    # Load meta images (class reference images)
    print("Loading meta images...")
    meta_images = []
    for i in range(len(test_dataset.classes)):
        image_path = os.path.join(META_PATH, f"{i}.png")
        if os.path.exists(image_path):
            meta_images.append(plt.imread(image_path))
        else:
            print(f"⚠ Meta image not found: {image_path}")
    
    # Perform prediction and get failures
    print("\nPerforming ensemble prediction...")
    predictions, true_labels, failed_indices = ensemble_predict(
        models, model_weights[:len(models)], test_dataloader, device, DEFAULT_CONFIG['num_classes']
    )
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    print(f"\n{'='*60}")
    print(f"Ensemble Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-score: {f1_macro:.4f}")
    print(f"  Failures: {len(failed_indices)} / {len(true_labels)}")
    print(f"{'='*60}")
    
    # Visualize failures
    if len(failed_indices) > 0:
        visualize_failures(test_dataset, predictions, failed_indices, meta_images, output_dir)
        print(f"\n✓ Failure examples saved to: {output_dir}")
    else:
        print("\n✓ Perfect accuracy! No failures to visualize.")


if __name__ == "__main__":
    main()

