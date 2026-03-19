#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source venv/bin/activate

# Auto-detect number of GPUs
if command -v nvidia-smi &> /dev/null; then
    NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
else
    NUM_GPUS=0
fi

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Detected $NUM_GPUS GPUs. Starting multi-GPU training with torchrun..."
    # Pass all extra shell arguments to the training script
    torchrun --standalone --nproc_per_node=$NUM_GPUS nanogpt_2d.py train "$@"
else
    echo "Detected $NUM_GPUS GPU(s). Starting single-GPU training..."
    # Pass all extra shell arguments to the training script
    python nanogpt_2d.py train "$@"
fi
