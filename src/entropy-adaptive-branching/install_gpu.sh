#!/bin/bash
# Installation script for Entropy-Adaptive Branching (GPU-enabled)

echo "======================================================================"
echo "Installing Entropy-Adaptive Branching (GPU-enabled with CUDA 12.1)"
echo "======================================================================"
echo ""

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Check NVIDIA GPU
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null
then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    echo ""
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Install PyTorch with CUDA 12.1 support
echo "Step 1: Installing PyTorch with CUDA 12.1 support..."
echo "This will take a few minutes..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
echo ""
echo "Step 2: Installing other dependencies..."
pip3 install transformers>=4.30.0 numpy>=1.24.0 tqdm>=4.65.0

# Install optional dependencies
echo ""
echo "Step 3: Installing optional dependencies..."
pip3 install matplotlib>=3.7.0 seaborn>=0.12.0 pytest>=7.3.0 scikit-learn>=1.3.0

# Install the package
echo ""
echo "Step 4: Installing entropy-adaptive-branching package..."
pip3 install -e .

# Verify CUDA is working
echo ""
echo "======================================================================"
echo "Verifying CUDA setup..."
echo "======================================================================"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'Number of GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('WARNING: CUDA not available! Will fall back to CPU.')
"

echo ""
echo "======================================================================"
echo "Installation complete!"
echo "======================================================================"
echo ""
echo "Next steps:"
echo "  1. Test with GPU: python3 quickstart.py"
echo "  2. Run examples: python3 examples/basic_usage.py"
echo "  3. Read docs: cat README.md"
echo ""
echo "Note: First run will download GPT-2 model (~500MB)"
echo ""