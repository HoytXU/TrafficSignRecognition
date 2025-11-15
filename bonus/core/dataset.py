"""
Dataset Module for GTSRB (German Traffic Sign Recognition Benchmark)

This module provides the MyDataset class for loading and preprocessing
the GTSRB dataset with ROI (Region of Interest) cropping.
"""

import os
import csv
from PIL import Image
from torch.utils.data import Dataset


class GTSRBDataset(Dataset):
    """
    Dataset class for GTSRB (German Traffic Sign Recognition Benchmark).
    
    Loads images from CSV files that specify:
    - Image paths
    - ROI coordinates (Region of Interest)
    - Class labels
    
    Args:
        folder_path: Path to dataset folder containing images
        csv_file: Path to CSV file with image metadata
        transform: Optional transform to apply to images
    """
    
    def __init__(self, folder_path, csv_file, transform=None):
        self.folder_path = folder_path
        self.csv_file = csv_file
        self.transform = transform
        self.data = []
        self.classes = []

        # Load data from CSV
        with open(csv_file, 'r') as file:
            csv_reader = csv.DictReader(file)
            for row in csv_reader:
                image_path = os.path.join(folder_path, row['Path'])
                x1, y1, x2, y2 = int(row['Roi.X1']), int(row['Roi.Y1']), int(row['Roi.X2']), int(row['Roi.Y2'])
                class_id = int(row['ClassId'])

                self.data.append((image_path, x1, y1, x2, y2, class_id))
                if class_id not in self.classes:
                    self.classes.append(class_id)
        
        # Sort classes for consistency
        self.classes.sort()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """
        Get a sample from the dataset.
        
        Args:
            index: Index of the sample
            
        Returns:
            tuple: (preprocessed_image, class_id)
        """
        image_path, x1, y1, x2, y2, class_id = self.data[index]

        # Load and crop image using ROI
        image = Image.open(image_path)
        cropped_image = image.crop((x1, y1, x2, y2))

        # Apply transforms if defined
        if self.transform is not None:
            cropped_image = self.transform(cropped_image)

        return cropped_image, class_id
    
    def get_original_image(self, index):
        """
        Get the original (uncropped) image.
        
        Args:
            index: Index of the sample
            
        Returns:
            PIL.Image: Original image
        """
        image_path, x1, y1, x2, y2, class_id = self.data[index]
        image = Image.open(image_path)
        return image

