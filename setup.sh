#!/usr/bin/env bash
# GeoSemTwin-X setup for Ubuntu 22.04/24.04 (native or WSL2) with ROS 2 already installed.
# Creates .venv that can also see ROS 2's rclpy (system site packages).
set -e
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv python3-pip
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install --upgrade pip wheel
# PyTorch with CUDA 12.6 (RTX 3500 Ada = sm_89). Change cu126 if your driver needs another build.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
export SAM2_BUILD_CUDA=0          # SAM 2 works without its optional CUDA extension
pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
echo
echo "Next: log in to Hugging Face with the account that has access to staerrobotics/warehouses:"
echo "   source .venv/bin/activate && hf auth login      (older versions: huggingface-cli login)"
echo "Then: python scripts/run_all.py check"
