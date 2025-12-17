#!/bin/bash
# RunPod Setup Script
# Run this on your RunPod pod to setup the environment

echo "=========================================="
echo "EdgeMoE RunPod Setup"
echo "=========================================="

# Update system
echo "1. Updating system..."
apt-get update -qq

# Install dependencies
echo "2. Installing Python packages..."
pip install -q transformers>=4.55.0 peft>=0.7.0 datasets tqdm psutil accelerate

# Verify GPU
echo "3. Checking GPU..."
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"NOT FOUND\"}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check directories
echo "4. Checking directories..."
if [ -d "/workspace/edgemoe/data/processed" ]; then
    echo "✓ Data directory found"
    echo "  Files:"
    find /workspace/edgemoe/data/processed -name "*.jsonl" | wc -l
else
    echo "✗ Data directory not found!"
    echo "  Please upload data to /workspace/edgemoe/data/processed/"
fi

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Train router:"
echo "   python runpod/train_router_runpod.py --data_dir ./data/processed --output_dir ./models/router_real --epochs 5 --batch_size 32"
echo ""
echo "2. Train experts:"
echo "   python runpod/train_expert_runpod.py --expert_id 0 --domain code --data_dir ./data/processed/code --output_dir ./models/experts/expert_0_code --epochs 3"
