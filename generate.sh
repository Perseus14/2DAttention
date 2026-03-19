#!/bin/bash
set -e

if [ ! -d "venv" ]; then
    echo "Virtual environment not found. Please run ./install.sh first."
    exit 1
fi

source venv/bin/activate

echo "Generating text..."
# Pass all extra shell arguments to the generation script
python nanogpt_2d.py generate "$@"
