"""
Preprocessing Visualization Utilities

Visualizes data preprocessing effects for understanding transformations.
"""

import os
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Import from core modules
import sys
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)
from core.dataset import GTSRBDataset
from core.config import DATASET_PATH, TEST_CSV


def plot_preprocessing_comparison(original_img, preprocessed_img, save_path=None):
    """
    Plot comparison between original and preprocessed images.
    
    Args:
        original_img: PIL Image - original image
        preprocessed_img: PIL Image - preprocessed image
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(preprocessed_img)
    axes[1].set_title('Preprocessed Image', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_preprocessing(sample_index=0, save_path=None):
    """
    Visualize preprocessing on a sample image.
    
    Args:
        sample_index: Index of sample to visualize
        save_path: Optional path to save the figure
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Load dataset
    test_dataset = GTSRBDataset(folder_path=DATASET_PATH, csv_file=TEST_CSV, transform=transform)
    
    # Get original and preprocessed images
    original_image = test_dataset.get_original_image(sample_index)
    preprocessed_tensor, _ = test_dataset[sample_index]
    
    # Convert tensor back to PIL for visualization
    preprocessed_image = transforms.ToPILImage()(preprocessed_tensor)
    
    # Plot comparison
    plot_preprocessing_comparison(original_image, preprocessed_image, save_path)


if __name__ == "__main__":
    # Example usage
    output_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    save_path = os.path.join(output_dir, "preprocessing_example.png")
    visualize_preprocessing(sample_index=0, save_path=save_path)

