"""
Tests for dataset module.
"""

import pytest
import os
import sys
from unittest.mock import patch, mock_open
import csv

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)

from core.dataset import GTSRBDataset


class TestGTSRBDataset:
    """Test cases for GTSRBDataset."""
    
    def test_dataset_initialization(self, tmp_path):
        """Test dataset initialization with mock CSV."""
        # Create mock CSV file
        csv_file = tmp_path / "test.csv"
        csv_content = """Path,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId
Test/00000.png,10,20,50,60,0
Test/00001.png,15,25,55,65,1"""
        
        with open(csv_file, 'w') as f:
            f.write(csv_content)
        
        # Create mock image directory
        img_dir = tmp_path / "Test"
        img_dir.mkdir()
        
        # Create dummy images (we'll mock PIL Image.open)
        (img_dir / "00000.png").touch()
        (img_dir / "00001.png").touch()
        
        # Mock PIL Image
        with patch('PIL.Image.open') as mock_open_img:
            mock_img = mock_open_img.return_value
            mock_img.crop.return_value = mock_img
            
            dataset = GTSRBDataset(
                folder_path=str(tmp_path),
                csv_file=str(csv_file),
                transform=None
            )
            
            assert len(dataset) == 2
            assert len(dataset.classes) == 2
            assert 0 in dataset.classes
            assert 1 in dataset.classes
    
    def test_dataset_getitem(self, tmp_path):
        """Test dataset __getitem__ method."""
        csv_file = tmp_path / "test.csv"
        csv_content = """Path,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId
Test/00000.png,10,20,50,60,0"""
        
        with open(csv_file, 'w') as f:
            f.write(csv_content)
        
        img_dir = tmp_path / "Test"
        img_dir.mkdir()
        (img_dir / "00000.png").touch()
        
        with patch('PIL.Image.open') as mock_open_img:
            mock_img = mock_open_img.return_value
            mock_img.crop.return_value = mock_img
            
            dataset = GTSRBDataset(
                folder_path=str(tmp_path),
                csv_file=str(csv_file),
                transform=None
            )
            
            image, class_id = dataset[0]
            assert class_id == 0
    
    def test_dataset_get_original_image(self, tmp_path):
        """Test get_original_image method."""
        csv_file = tmp_path / "test.csv"
        csv_content = """Path,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId
Test/00000.png,10,20,50,60,0"""
        
        with open(csv_file, 'w') as f:
            f.write(csv_content)
        
        img_dir = tmp_path / "Test"
        img_dir.mkdir()
        (img_dir / "00000.png").touch()
        
        with patch('PIL.Image.open') as mock_open_img:
            mock_img = mock_open_img.return_value
            
            dataset = GTSRBDataset(
                folder_path=str(tmp_path),
                csv_file=str(csv_file),
                transform=None
            )
            
            original = dataset.get_original_image(0)
            mock_open_img.assert_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

