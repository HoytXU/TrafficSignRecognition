"""
Pytest configuration and fixtures.
"""

import pytest
import os
import sys
import tempfile
import shutil

# Add bonus directory to path
bonus_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, bonus_dir)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a sample image path."""
    img_path = os.path.join(temp_dir, "test_image.png")
    # Create empty file (actual image creation would require PIL)
    with open(img_path, 'w') as f:
        f.write("")
    return img_path


@pytest.fixture
def mock_dataset_config(temp_dir):
    """Create mock dataset configuration."""
    csv_file = os.path.join(temp_dir, "test.csv")
    img_dir = os.path.join(temp_dir, "Test")
    os.makedirs(img_dir, exist_ok=True)
    
    # Create sample CSV
    csv_content = """Path,Roi.X1,Roi.Y1,Roi.X2,Roi.Y2,ClassId
Test/00000.png,10,20,50,60,0
Test/00001.png,15,25,55,65,1"""
    
    with open(csv_file, 'w') as f:
        f.write(csv_content)
    
    return {
        'folder_path': temp_dir,
        'csv_file': csv_file,
        'img_dir': img_dir
    }

