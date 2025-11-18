#!/bin/bash
# Setup script for Bonus Environment
# This script sets up the Python environment and installs all required dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root directory (two levels up from bonus/scripts/)
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BONUS_DIR="$(dirname "$SCRIPT_DIR")"
PROJECT_ROOT="$(dirname "$BONUS_DIR")"

echo -e "${GREEN}=== Bonus Environment Setup ===${NC}"
echo "Project root: $PROJECT_ROOT"
echo "Bonus directory: $BONUS_DIR"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created!${NC}"
else
    echo -e "${GREEN}Virtual environment already exists.${NC}"
fi

# Activate venv
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip --quiet

# Install base dependencies
if [ -f "requirements.txt" ]; then
    echo -e "${YELLOW}Installing base dependencies from requirements.txt...${NC}"
    pip install -r requirements.txt --quiet
    echo -e "${GREEN}Base dependencies installed!${NC}"
else
    echo -e "${RED}Warning: requirements.txt not found!${NC}"
fi

# Install PyTorch
echo -e "${YELLOW}Installing PyTorch...${NC}"
echo "Installing CPU version. For GPU support, see SETUP.md"
pip install torch torchvision torchaudio --quiet
echo -e "${GREEN}PyTorch installed!${NC}"

# Install bonus-specific dependencies
echo -e "${YELLOW}Installing bonus-specific dependencies...${NC}"
pip install wandb tqdm Pillow --quiet
echo -e "${GREEN}Bonus dependencies installed!${NC}"

# Install test dependencies
if [ -f "$BONUS_DIR/requirements-test.txt" ]; then
    echo -e "${YELLOW}Installing test dependencies...${NC}"
    pip install -r "$BONUS_DIR/requirements-test.txt" --quiet
    echo -e "${GREEN}Test dependencies installed!${NC}"
fi

# Create required directories
echo -e "${YELLOW}Creating required directories...${NC}"
mkdir -p "$BONUS_DIR/checkpoints"
mkdir -p "$BONUS_DIR/logs"
echo -e "${GREEN}Directories created!${NC}"

# Verify installation
echo -e "${YELLOW}Verifying installation...${NC}"
python -c "import torch; import torchvision; import wandb; print('✓ All dependencies installed successfully!')" 2>/dev/null && \
    echo -e "${GREEN}Verification passed!${NC}" || \
    echo -e "${RED}Warning: Some dependencies may not be installed correctly${NC}"

echo ""
echo -e "${GREEN}=== Setup Complete! ===${NC}"
echo ""
echo "To activate the environment in the future, run:"
echo -e "${YELLOW}  source venv/bin/activate${NC}"
echo ""
echo "Next steps:"
echo "  1. Verify dataset is in datasets/dataset2/"
echo "  2. Train a model: python bonus/training/train.py --model resnet18 --epoch 10"
echo "  3. Run tests: pytest bonus/tests/"
echo ""
echo "For detailed setup instructions, see: bonus/SETUP.md"

