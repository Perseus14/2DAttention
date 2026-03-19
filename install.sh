#!/bin/bash

# Exit on error
set -e

echo "Checking for Python 3.12..."
if ! command -v python3.12 &> /dev/null; then
    echo "Python 3.12 not found. Installing..."
    sudo apt-get update
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
    sudo apt-get install -y python3.12 python3.12-virtualenv python3.12-dev
else
    echo "Python 3.12 is already installed."
fi

echo "Checking for NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Installing ubuntu-drivers-common..."
    sudo apt-get update
    sudo apt-get install -y ubuntu-drivers-common
    echo "Autoinstalling recommended NVIDIA drivers..."
    sudo ubuntu-drivers autoinstall
    echo "WARNING: You must REBOOT your system after this script finishes for the NVIDIA drivers to take effect."
else
    echo "NVIDIA drivers (nvidia-smi) are already installed."
fi

echo "Checking for virtual environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment 'venv' with Python 3.12..."
    virtualenv venv -p /usr/bin/python3.12
else
    echo "Virtual environment 'venv' already exists."
fi

echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Upgrade pip and install the requirements listed in the nanogpt_2d.py docstring
pip install --upgrade pip
pip install torch numpy tiktoken datasets transformers wandb tqdm

echo "Installation complete! To activate the environment manually, run: source venv/bin/activate"
