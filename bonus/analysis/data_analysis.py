"""
Data Analysis Script for GTSRB Dataset

Performs comprehensive analysis of the dataset including:
- Dataset statistics
- Class distribution
- Image statistics
- Visualization
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
from tqdm import tqdm

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.config import DATASET_PATH, TRAIN_CSV, TEST_CSV, META_PATH
from core.dataset import GTSRBDataset


def analyze_dataset():
    """Perform comprehensive dataset analysis."""
    
    print("="*70)
    print("GTSRB Dataset Analysis")
    print("="*70)
    
    # Load CSV files
    print("\n1. Loading dataset files...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    print(f"   ✓ Train CSV loaded: {len(train_df)} samples")
    print(f"   ✓ Test CSV loaded: {len(test_df)} samples")
    
    # Basic statistics
    print("\n2. Dataset Statistics:")
    print(f"   Total samples: {len(train_df) + len(test_df):,}")
    print(f"   Training samples: {len(train_df):,} ({len(train_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    print(f"   Test samples: {len(test_df):,} ({len(test_df)/(len(train_df)+len(test_df))*100:.1f}%)")
    print(f"   Number of classes: {len(set(train_df['ClassId'].unique()) | set(test_df['ClassId'].unique()))}")
    
    # Class distribution
    print("\n3. Class Distribution Analysis:")
    train_classes = Counter(train_df['ClassId'])
    test_classes = Counter(test_df['ClassId'])
    all_classes = sorted(set(train_df['ClassId'].unique()) | set(test_df['ClassId'].unique()))
    
    print(f"   Classes in training set: {len(train_classes)}")
    print(f"   Classes in test set: {len(test_classes)}")
    print(f"   Total unique classes: {len(all_classes)}")
    
    # Class statistics
    train_counts = [train_classes.get(c, 0) for c in all_classes]
    test_counts = [test_classes.get(c, 0) for c in all_classes]
    
    print(f"\n   Class distribution statistics:")
    print(f"   - Training: min={min(train_counts)}, max={max(train_counts)}, "
          f"mean={np.mean(train_counts):.1f}, median={np.median(train_counts):.1f}")
    print(f"   - Test: min={min(test_counts)}, max={max(test_counts)}, "
          f"mean={np.mean(test_counts):.1f}, median={np.median(test_counts):.1f}")
    
    # ROI statistics
    print("\n4. ROI (Region of Interest) Statistics:")
    roi_widths = train_df['Roi.X2'] - train_df['Roi.X1']
    roi_heights = train_df['Roi.Y2'] - train_df['Roi.Y1']
    roi_areas = roi_widths * roi_heights
    
    print(f"   ROI Width:  min={roi_widths.min()}, max={roi_widths.max()}, "
          f"mean={roi_widths.mean():.1f}, median={roi_widths.median():.1f}")
    print(f"   ROI Height: min={roi_heights.min()}, max={roi_heights.max()}, "
          f"mean={roi_heights.mean():.1f}, median={roi_heights.median():.1f}")
    print(f"   ROI Area:   min={roi_areas.min()}, max={roi_areas.max()}, "
          f"mean={roi_areas.mean():.1f}, median={roi_areas.median():.1f}")
    
    # Image statistics (sample a subset for speed)
    print("\n5. Image Statistics (sampling 1000 images):")
    sample_size = min(1000, len(train_df))
    sample_indices = np.random.choice(len(train_df), sample_size, replace=False)
    
    widths, heights, areas = [], [], []
    for idx in tqdm(sample_indices, desc="   Processing images"):
        img_path = os.path.join(DATASET_PATH, train_df.iloc[idx]['Path'])
        try:
            img = Image.open(img_path)
            widths.append(img.size[0])
            heights.append(img.size[1])
            areas.append(img.size[0] * img.size[1])
        except:
            continue
    
    if widths:
        print(f"   Image Width:  min={min(widths)}, max={max(widths)}, "
              f"mean={np.mean(widths):.1f}, median={np.median(widths):.1f}")
        print(f"   Image Height: min={min(heights)}, max={max(heights)}, "
              f"mean={np.mean(heights):.1f}, median={np.median(heights):.1f}")
        print(f"   Image Area:   min={min(areas)}, max={max(areas)}, "
              f"mean={np.mean(areas):.1f}, median={np.median(areas):.1f}")
    
    # Create visualizations
    print("\n6. Generating visualizations...")
    create_visualizations(train_df, test_df, train_classes, test_classes, all_classes, 
                         train_counts, test_counts, roi_widths, roi_heights)
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    
    return {
        'train_samples': len(train_df),
        'test_samples': len(test_df),
        'num_classes': len(all_classes),
        'train_class_dist': train_classes,
        'test_class_dist': test_classes,
        'roi_stats': {
            'width': {'min': roi_widths.min(), 'max': roi_widths.max(), 
                     'mean': roi_widths.mean(), 'median': roi_widths.median()},
            'height': {'min': roi_heights.min(), 'max': roi_heights.max(), 
                      'mean': roi_heights.mean(), 'median': roi_heights.median()},
        }
    }


def create_visualizations(train_df, test_df, train_classes, test_classes, all_classes,
                         train_counts, test_counts, roi_widths, roi_heights):
    """Create visualization plots."""
    
    # Set style
    plt.style.use('seaborn-v0_8')
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Class distribution comparison
    ax1 = plt.subplot(3, 2, 1)
    x_pos = np.arange(len(all_classes))
    width = 0.35
    ax1.bar(x_pos - width/2, train_counts, width, label='Train', alpha=0.8)
    ax1.bar(x_pos + width/2, test_counts, width, label='Test', alpha=0.8)
    ax1.set_xlabel('Class ID')
    ax1.set_ylabel('Number of Samples')
    ax1.set_title('Class Distribution: Train vs Test')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Training set class distribution (sorted)
    ax2 = plt.subplot(3, 2, 2)
    sorted_train = sorted(train_classes.items())
    classes_sorted, counts_sorted = zip(*sorted_train)
    ax2.bar(range(len(classes_sorted)), counts_sorted, alpha=0.8, color='steelblue')
    ax2.set_xlabel('Class ID')
    ax2.set_ylabel('Number of Samples')
    ax2.set_title('Training Set Class Distribution')
    ax2.grid(True, alpha=0.3)
    
    # 3. ROI size distribution
    ax3 = plt.subplot(3, 2, 3)
    ax3.scatter(roi_widths, roi_heights, alpha=0.3, s=10)
    ax3.set_xlabel('ROI Width (pixels)')
    ax3.set_ylabel('ROI Height (pixels)')
    ax3.set_title('ROI Size Distribution')
    ax3.grid(True, alpha=0.3)
    
    # 4. ROI area distribution
    ax4 = plt.subplot(3, 2, 4)
    roi_areas = roi_widths * roi_heights
    ax4.hist(roi_areas, bins=50, alpha=0.7, edgecolor='black')
    ax4.set_xlabel('ROI Area (pixels²)')
    ax4.set_ylabel('Frequency')
    ax4.set_title('ROI Area Distribution')
    ax4.grid(True, alpha=0.3)
    
    # 5. Class imbalance visualization
    ax5 = plt.subplot(3, 2, 5)
    imbalance_ratio = [train_counts[i] / max(train_counts) for i in range(len(train_counts))]
    ax5.bar(range(len(all_classes)), imbalance_ratio, alpha=0.8, color='coral')
    ax5.set_xlabel('Class ID')
    ax5.set_ylabel('Normalized Count (relative to max)')
    ax5.set_title('Class Imbalance Ratio')
    ax5.grid(True, alpha=0.3)
    
    # 6. Sample count comparison
    ax6 = plt.subplot(3, 2, 6)
    total_counts = [train_counts[i] + test_counts[i] for i in range(len(all_classes))]
    ax6.bar(range(len(all_classes)), total_counts, alpha=0.8, color='green')
    ax6.set_xlabel('Class ID')
    ax6.set_ylabel('Total Samples')
    ax6.set_title('Total Samples per Class (Train + Test)')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.join(bonus_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "data_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved visualization to: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    analysis_results = analyze_dataset()
    
    # Print summary
    print("\n📊 Summary:")
    print(f"   • Dataset is {'balanced' if np.std(list(analysis_results['train_class_dist'].values())) < np.mean(list(analysis_results['train_class_dist'].values())) * 0.5 else 'imbalanced'}")
    print(f"   • Average samples per class: {np.mean(list(analysis_results['train_class_dist'].values())):.1f}")
    print(f"   • ROI sizes vary significantly: {analysis_results['roi_stats']['width']['min']}-{analysis_results['roi_stats']['width']['max']} pixels")

