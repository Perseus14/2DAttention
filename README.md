# NanoGPT 2D Attention

A single-file implementation of a Transformer (NanoGPT style) equipped with **2D Attention Heads**, based on the concepts explored in Percepta's *"Can LLMs Be Computers?"* (March 2026).

This repository is designed to be fully self-contained in `nanogpt_2d.py`, covering the entire end-to-end pipeline from data preparation to distributed multi-GPU training, and culminating in a highly-optimized generation loop.

## Key Features

- **2D Causal Self-Attention**: Modifies standard attention to operate on 2D embeddings per head (`head_dim = 2`).
- **Convex Hull KV Cache**: Implements an amortized $O(1)$ KV cache insert by maintaining an online 2-D convex hull of past tokens. When generating, this replaces the typical $O(N)$ linear KV cache scan with a single $O(\log N)$ or fast matrix-vector dot product over the hull boundary vertices!
- **End-to-End Single File**: Data downloading, GPT-2 BPE tokenization, Distributed Data Parallel (DDP) training, and text generation all live inside one script.
- **Multiprocessing Optimized Data Pipeline**: The `prepare` command dynamically detects CPU cores to perform batched memory-mapped `.bin` generation incredibly fast.
- **Modern Hardware Setup**: Fully supports `bfloat16`, `torch.compile`, and `torchrun` for bleeding-edge Ampere/A100 optimization out-of-the-box.

## Usage

This repository includes helpful bash scripts that automatically handle python virtual environments (`venv`) and CLI arguments for you.

### 0. Installation
Run the install script to automatically install Python 3.12 (if missing), set up the `venv`, and install all PyTorch/CUDA dependencies.
```bash
./install.sh
```

### 1. Data Preparation
Download the OpenWebText dataset from HuggingFace, tokenize it via GPT-2 BPE, and compile it into extremely fast memory-mapped `.bin` files automatically.

```bash
# Prepare the default OpenWebText dataset
./prepare.sh
```

You can also natively prepare other popular datasets like **TinyStories** or **FineWeb-edu** directly:

```bash
# Prepare TinyStories
./prepare.sh --dataset tinystories

# Prepare FineWeb-edu (10BT sample)
./prepare.sh --dataset fineweb_edu
```

### 2. Training
Train from scratch (or resume from a checkpoint) using the prepared binary mapped files. The training script automatically detects if you have multiple GPUs and launches `torchrun` for Distributed Data Parallel, otherwise falls back to a single GPU.

```bash
./train.sh
```

### Training Configuration

You can seamlessly override any model or training parameters via command line flags. For instance, here is how you would train a **~7 Million parameter 2D-Attention model** using your prepared TinyStories data, while cleanly logging it to Weights & Biases:

```bash
./train.sh \
    --run_id tinystories_7M_run \
    --data_dir data/tinystories \
    --n_layer 19 \
    --n_head 48 \
    --n_embd 96 \
    --batch_size 16 \
    --grad_accum 32 \
    --learning_rate 8e-4 \
    --min_lr 8e-5 \
    --max_iters 10000 \
    --lr_decay_iters 10000 \
    --warmup_iters 200 \
    --eval_interval 250 \
    --eval_iters 50 \
    --log_interval 10 \
    --weight_decay 0.1 \
    --wandb_log
```

### 3. Generation
Generate text using the fully-trained model and the hardware-accelerated Hull KV Cache. You must point `--out_dir` to the exact folder where your checkpoint (`ckpt.pt` or `ckpt_best.pt`) is saved.

```bash
./generate.sh --out_dir out-owt-2d/tinystories_7M_run --prompt "Once upon a time," --use_hull
```

## Requirements

Ensure you have the latest stable versions of:
- `torch` (>= 2.1 recommended for optimal `torch.compile` speed)
- `numpy`
- `tiktoken`
- `datasets`
- `transformers`
- `tqdm`
- `wandb` (optional, for logging)
