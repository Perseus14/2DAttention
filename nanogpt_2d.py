"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  NanoGPT · 2D Attention Heads · OpenWebText  ·  Single-file Edition        ║
║                                                                              ║
║  Based on Percepta "Can LLMs Be Computers?" (March 2026)                    ║
║  Inspired by Karpathy's nanoGPT (github.com/karpathy/nanoGPT)              ║
║                                                                              ║
║  SECTIONS                                                                    ║
║    1. Config & CLI                                                           ║
║    2. Data pipeline  (download + tokenise OpenWebText → .bin files)         ║
║    3. Model          (2D attention + HullKVCache)                            ║
║    4. Training loop  (DDP, AMP, grad-accum, cosine LR, wandb, ckpt)        ║
║    5. Inference      (HullKVCache decode, top-k / temperature)              ║
║    6. Main entry-point                                                       ║
║                                                                              ║
║  USAGE                                                                       ║
║    # Step 1 – prepare data (one-time, ~1 hr, ~54 GB disk)                  ║
║    python nanogpt_2d.py prepare [--dataset tinystories]                     ║
║                                                                              ║
║    # Step 2 – train  (single GPU)                                           ║
║    python nanogpt_2d_owt.py train                                           ║
║                                                                              ║
║    # Step 2 – train  (multi-GPU, 8 GPUs)                                   ║
║    torchrun --standalone --nproc_per_node=8 nanogpt_2d_owt.py train        ║
║                                                                              ║
║    # Step 3 – resume from checkpoint                                        ║
║    python nanogpt_2d_owt.py train --init_from=resume                       ║
║                                                                              ║
║    # Step 4 – generate text                                                 ║
║    python nanogpt_2d_owt.py generate --prompt="The universe is"            ║
║                                                                              ║
║  DEPENDENCIES                                                                ║
║    pip install torch numpy tiktoken datasets transformers wandb tqdm        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ──────────────────────────────────────────────────────────────────────────────
# Stdlib
# ──────────────────────────────────────────────────────────────────────────────
import argparse
import inspect
import math
import os
import sys
import time
import pickle
from contextlib import nullcontext
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Third-party (lazy imports where heavy)
# ──────────────────────────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CONFIG & CLI
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    """Model hyper-parameters.
    CRITICAL invariant:  n_embd // n_head == 2  (2D attention heads).
    """
    # Architecture  (Percepta defaults – tiny but correct)
    n_embd:     int   = 36       # d_model
    n_head:     int   = 18       # head_dim = n_embd / n_head = 2
    n_layer:    int   = 7
    block_size: int   = 1024     # context window (tokens)
    vocab_size: int   = 50304    # GPT-2 BPE vocab (padded to nice multiple)
    dropout:    float = 0.0
    bias:       bool  = False
    use_rope:   bool  = False
    rope_base:  float = 10000.0
    activation: str   = 'relu2'
    norm_type:  str   = 'rmsnorm'
    qk_norm:    bool  = True
    logit_soft_cap: float = 30.0

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_embd // self.n_head == 2, (
            f"head_dim must equal 2 for 2D-attention.  "
            f"Set n_head = n_embd // 2 = {self.n_embd // 2}."
        )


@dataclass
class TrainConfig:
    # ── I/O ──────────────────────────────────────────────────────────────────
    out_dir:    str  = "out-owt-2d"
    data_dir:   str  = "data/openwebtext"
    init_from:  str  = "scratch"      # "scratch" | "resume"
    always_save_checkpoint: bool = True

    # ── logging ──────────────────────────────────────────────────────────────
    log_interval:  int  = 10
    eval_interval: int  = 500
    eval_iters:    int  = 200
    eval_only:     bool = False
    wandb_log:     bool = False
    wandb_project: str  = "nanogpt-2d"
    wandb_run_name: str = "run"

    # ── data ─────────────────────────────────────────────────────────────────
    batch_size:                  int = 12    # micro-batch per GPU
    gradient_accumulation_steps: int = 40   # effective batch ≈ batch*accum*n_gpu

    # ── optimiser ────────────────────────────────────────────────────────────
    learning_rate:  float = 6e-4
    max_iters:      int   = 600_000
    weight_decay:   float = 1e-1
    beta1:          float = 0.9
    beta2:          float = 0.95
    grad_clip:      float = 1.0
    use_muon:       bool  = False
    muon_lr:        float = 0.02
    muon_momentum:  float = 0.95

    # ── LR schedule (cosine with warmup) ─────────────────────────────────────
    decay_lr:      bool  = True
    warmup_iters:  int   = 2_000
    lr_decay_iters: int  = 600_000
    min_lr:        float = 6e-5

    # ── system ───────────────────────────────────────────────────────────────
    device:   str  = "cuda"          # "cpu" | "cuda" | "cuda:0"
    dtype:    str  = "bfloat16"      # "float32" | "bfloat16" | "float16"
    compile:  bool = True            # torch.compile (PyTorch ≥ 2.0)
    seed:     int  = 1337


@dataclass
class GenerateConfig:
    out_dir:      str   = "out-owt-2d"
    prompt:       str   = "The"
    num_samples:  int   = 3
    max_new_tokens: int = 200
    temperature:  float = 0.8
    top_k:        int   = 200
    use_hull:     bool  = True       # False → standard greedy fallback
    device:       str   = "cuda"
    dtype:        str   = "bfloat16"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA PIPELINE  –  OpenWebText → train.bin / val.bin
# ══════════════════════════════════════════════════════════════════════════════

def _write_chunk(args):
    """Worker process for parallel writing of tokenised chunks to memmap."""
    import numpy as np
    out_path, ds, start_row, end_row, dest_idx, total_tokens = args
    mmap = np.memmap(out_path, dtype=np.uint16, mode="r+", shape=(total_tokens,))
    idx = dest_idx
    shard_size = 1024
    for batch_idx in range(start_row, end_row, shard_size):
        batch_end = min(batch_idx + shard_size, end_row)
        batch = ds[batch_idx:batch_end]
        if len(batch["ids"]) > 0:
            arr_batch = np.concatenate(batch["ids"])
            mmap[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
    mmap.flush()

def prepare_dataset(dataset_name: str = "openwebtext", data_dir: Optional[str] = None, num_proc: Optional[int] = None) -> None:
    """
    Download dataset (OpenWebText or TinyStories) from HuggingFace Hub, tokenise with GPT-2 BPE,
    and write uint16 memory-mapped binary files:
        <data_dir>/train.bin
        <data_dir>/val.bin

    This is a one-time operation (~1 hr on fast hardware, ~54 GB disk for OpenWebText).
    Subsequent runs are skipped if the .bin files already exist.
    """
    import os
    if num_proc is None:
        # Default to all available CPU cores, fallback to 8 if detection fails
        num_proc = os.cpu_count() or 8

    if data_dir is None:
        data_dir = f"data/{dataset_name}"

    import tiktoken
    from datasets import load_dataset
    import multiprocessing as mp
    from tqdm import tqdm

    train_bin = Path(data_dir) / "train.bin"
    val_bin   = Path(data_dir) / "val.bin"

    if train_bin.exists() and val_bin.exists():
        print(f"[prepare] {train_bin} and {val_bin} already exist – skipping.")
        return

    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    if dataset_name == "fineweb_edu":
        hf_path, hf_name = "HuggingFaceFW/fineweb-edu", "sample-10BT"
    elif dataset_name == "tinystories":
        hf_path, hf_name = "roneneldan/TinyStories", None
    else:
        hf_path, hf_name = "openwebtext", None

    print(f"[prepare] Downloading {hf_path} from HuggingFace Hub …")
    dataset = load_dataset(hf_path, name=hf_name, num_proc=num_proc)

    if "validation" in dataset:
        split = dataset
        split["val"] = split.pop("validation")
    else:
        # 99.95% train / 0.05% val  (mirrors nanoGPT)
        split = dataset["train"].train_test_split(test_size=0.0005, seed=2357)
        split["val"] = split.pop("test")

    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens["<|endoftext|>"]

    def tokenise(example):
        ids = enc.encode_ordinary(example["text"])
        ids.append(eot)
        return {"ids": ids, "len": len(ids)}

    print("[prepare] Tokenising …")
    tokenised = split.map(
        tokenise,
        remove_columns=["text"],
        desc="tokenising",
        num_proc=num_proc,
    )

    for name, ds in tokenised.items():
        total_tokens = sum(ds["len"])
        out_path = Path(data_dir) / f"{name}.bin"
        
        print(f"[prepare] Pre-allocating {out_path} ({total_tokens:,} tokens) …")
        arr = np.memmap(out_path, dtype=np.uint16, mode="w+", shape=(total_tokens,))
        arr.flush()
        del arr  # Release so workers can open it

        workers = min(num_proc, len(ds))
        if workers == 0:
            continue

        print(f"[prepare] Writing {name} with {workers} workers in parallel …")
        lens = np.array(ds["len"], dtype=np.uint64)
        
        chunk_args = []
        step = math.ceil(len(ds) / workers)
        dest_idx = 0
        for i in range(workers):
            start_row = i * step
            end_row = min((i + 1) * step, len(ds))
            if start_row >= end_row:
                break
            
            chunk_args.append((out_path, ds, start_row, end_row, dest_idx, total_tokens))
            dest_idx += np.sum(lens[start_row:end_row])

        with mp.Pool(workers) as pool:
            list(tqdm(pool.imap(_write_chunk, chunk_args), total=len(chunk_args), desc=f"Writing {name}"))
            
        print(f"[prepare] ✓ {out_path}")

    # save meta so we can restore vocab_size later
    meta = {"vocab_size": enc.n_vocab}
    with open(Path(data_dir) / "meta.pkl", "wb") as f:
        pickle.dump(meta, f)

    print("[prepare] Done.")


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader  (memory-mapped, no workers needed)
# ──────────────────────────────────────────────────────────────────────────────

class OpenWebTextLoader:
    """
    Infinite streaming loader over a memory-mapped .bin file.
    Each call to next_batch() returns (x, y) uint16 tensors of shape
    (batch_size, block_size).  The file is re-mmapped on each call so
    that the OS can page it in/out freely – identical to nanoGPT's approach.
    """

    def __init__(
        self,
        split: str,
        data_dir: str,
        batch_size: int,
        block_size: int,
        device: str,
        rank: int = 0,
        world_size: int = 1,
    ):
        assert split in ("train", "val")
        self.path       = Path(data_dir) / f"{split}.bin"
        self.batch_size = batch_size
        self.block_size = block_size
        self.device     = device
        self.rank       = rank
        self.world_size = world_size

        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.path} not found.  "
                f"Run:  python {__file__} prepare"
            )

    def next_batch(self):
        data = np.memmap(self.path, dtype=np.uint16, mode="r")
        # each GPU draws from a different part of the file to avoid overlap
        ix = torch.randint(
            len(data) - self.block_size,
            (self.batch_size,),
        )
        x = torch.stack([
            torch.from_numpy(data[i     : i + self.block_size].astype(np.int64))
            for i in ix
        ])
        y = torch.stack([
            torch.from_numpy(data[i + 1 : i + 1 + self.block_size].astype(np.int64))
            for i in ix
        ])
        if "cuda" in self.device:
            x = x.pin_memory().to(self.device, non_blocking=True)
            y = y.pin_memory().to(self.device, non_blocking=True)
        else:
            x, y = x.to(self.device), y.to(self.device)
        return x, y


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL  –  2D Attention Heads + HullKVCache
# ══════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────────
# 3a. Convex-hull KV cache (inference only)
# ──────────────────────────────────────────────────────────────────────────────

class ConvexHull2D:
    """
    Online 2-D convex hull — amortised O(1) insert, O(|hull|) query.

    Key insight
    ───────────
    We maintain upper and lower chains as lists sorted by x-coordinate
    (ties broken by y).  Each chain satisfies a turn invariant:
      • upper chain: every consecutive triple makes a RIGHT turn (cross < 0)
      • lower chain: every consecutive triple makes a LEFT  turn (cross > 0)

    When a new point p arrives:
      1. Use bisect to find its position in the x-sorted chain — O(log n).
      2. Replace the sub-sequence around that position that violates the
         turn invariant — each replaced point is evicted permanently, so
         across all inserts the total evictions ≤ total inserts → O(1) amortised.

    This avoids re-sorting the full accumulated list on every insert.
    The old _rebuild() was O(n log n) per insert → O(n² log n) over a
    context window.  This implementation is O(n log n) total.

    Chains store (x, y, token_idx) triples.
    We use parallel xs_* lists of just the x-values so bisect works on
    a plain list of floats without a key= (which bisect does not support).
    """

    def __init__(self):
        import bisect as _bisect
        self._bisect = _bisect
        # Upper hull: sorted left-to-right, right-turn invariant (cross <= 0)
        self._upper:    list = []
        self._upper_xs: list = []   # x-coords only, for bisect
        # Lower hull: sorted left-to-right, left-turn invariant (cross >= 0)
        self._lower:    list = []
        self._lower_xs: list = []

    @staticmethod
    def _cross(O, A, B) -> float:
        """Signed area of OAB. Positive = left turn, negative = right turn."""
        return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0])

    def _insert_chain(self, chain, xs, p, upper: bool) -> None:
        """
        Insert point p = (x, y, token_idx) into one sorted hull chain.

        upper=True  keeps right turns (cross <= 0): upper convex hull.
        upper=False keeps left  turns (cross >= 0): lower convex hull.

        Complexity: O(log n) for the bisect + O(evictions) for the trim.
        Each point is evicted at most once from each chain across all
        inserts, so evictions amortise to O(1) per insert.
        """
        x, y, _ = p
        i = self._bisect.bisect_left(xs, x)

        # ── same-x: keep only the extreme point for this chain ────────────
        # Upper hull wants max-y; lower hull wants min-y at the same x.
        j = i
        while j < len(chain) and chain[j][0] == x:
            ey = chain[j][1]
            dominated = (y <= ey) if upper else (y >= ey)
            if dominated:
                return      # existing point is at least as good; discard p
            # p beats the existing point at this x: remove the inferior one
            chain.pop(j)
            xs.pop(j)
            # don't advance j; check again in case of duplicates
        i = j  # insertion point may have shifted

        # ── trim right: evict chain[i], chain[i+1], … while they become
        #    interior after p is inserted just before them ─────────────────
        while i < len(chain) - 1:
            # Would p -> chain[i] -> chain[i+1] violate the invariant?
            c = self._cross(p, chain[i], chain[i + 1])
            bad = (c >= 0) if upper else (c <= 0)
            if bad:
                chain.pop(i)
                xs.pop(i)
            else:
                break

        # ── trim left: evict chain[i-1], chain[i-2], … similarly ─────────
        while i >= 2:
            c = self._cross(chain[i - 2], chain[i - 1], p)
            bad = (c >= 0) if upper else (c <= 0)
            if bad:
                chain.pop(i - 1)
                xs.pop(i - 1)
                i -= 1
            else:
                break

        # ── interior check: if p lies strictly inside the existing hull,
        #    skip it (it cannot be a hull vertex) ──────────────────────────
        left_ok  = i == 0 or (
            (self._cross(chain[i - 1], p, chain[i]) < 0) if upper else
            (self._cross(chain[i - 1], p, chain[i]) > 0)
        ) if i < len(chain) else True
        if not left_ok:
            return

        chain.insert(i, p)
        xs.insert(i, x)

    # ── public API ────────────────────────────────────────────────────────────

    def insert_batch(self, keys_np, start_token_idx: int) -> None:
        """
        Add a batch of 2-D key vectors.
        keys_np          : float32 numpy array, shape (N, 2)
        start_token_idx  : token index of keys_np[0]

        Each call is O(k * log n + evictions) amortised.
        Since we insert exactly one key per decode step (k=1), this is
        O(log n) per step with O(1) amortised eviction cost.
        """
        for i, (x, y) in enumerate(keys_np):
            p = (float(x), float(y), start_token_idx + i)
            self._insert_chain(self._upper, self._upper_xs, p, upper=True)
            self._insert_chain(self._lower, self._lower_xs, p, upper=False)

    def argmax_dot_batch(self, dirs_np) -> "np.ndarray":
        """
        For each direction vector in dirs_np (shape M x 2), return the token
        index of the hull vertex maximising dot(vertex, direction).

        One numpy matmul over the hull (which is small) instead of any
        Python loops.  O(|hull| * M) arithmetic.
        Returns int64 array of shape (M,).
        """
        hull = self._upper + self._lower
        if not hull:
            import numpy as _np
            return _np.zeros(len(dirs_np), dtype=_np.int64)

        import numpy as _np
        hull_pts = _np.array([[p[0], p[1]] for p in hull], dtype=_np.float32)
        hull_idx = _np.array([p[2]         for p in hull], dtype=_np.int64)

        dots   = dirs_np @ hull_pts.T   # (M, 2) @ (2, H) -> (M, H)
        best_h = dots.argmax(axis=1)    # (M,)
        return hull_idx[best_h]         # (M,)

class HullKVCache:
    """
    GPU-resident KV cache with CPU-side convex-hull key management.

    Design decisions to eliminate CPU/GPU sync bottlenecks:
    ─────────────────────────────────────────────────────────
    1. Values stay on GPU the whole time.
       v_buf: (max_len, n_head, head_dim) pre-allocated on the target device.
       No .cpu() / .to(device) round-trips during decode.

    2. Keys are transferred to CPU in ONE batched copy per step.
       We call k_t.to('cpu', non_blocking=True) once, then pass the
       resulting numpy array to all hulls – no per-head .item() calls.

    3. Queries are transferred to CPU in ONE batched copy per step.
       q_np: (n_head, 2) numpy.  Each hull's argmax_dot_batch() does a
       small matrix multiply on CPU rather than 2*n_head .item() calls.

    4. Value lookup uses advanced indexing on the GPU tensor.
       The winning token indices come back as a numpy int array; we
       convert to a single LongTensor and index v_buf in one GPU op.

    Net result: 2 D→H transfers (k, q) and 1 H→D index per step,
    replacing 126+ blocking .item() syncs in the old implementation.
    """

    def __init__(self, n_head: int, head_dim: int, max_len: int, device: torch.device):
        self.n_head   = n_head
        self.head_dim = head_dim
        self.device   = device
        self._t       = 0

        # Pre-allocate value buffer on the target device
        self.v_buf = torch.empty(max_len, n_head, head_dim, device=device)
        # Pre-allocate head index range — reused every query() call, never reallocated
        self.h_range = torch.arange(n_head, device=device)
        # One hull per head
        self.hulls: list[ConvexHull2D] = [ConvexHull2D() for _ in range(n_head)]

    def append(self, k_t: torch.Tensor, v_t: torch.Tensor):
        """
        k_t : (n_head, 2)   – on GPU
        v_t : (n_head, head_dim) – on GPU

        ONE non-blocking D→H copy for keys; values stay on GPU.
        """
        # Store value directly in pre-allocated GPU buffer
        self.v_buf[self._t] = v_t.detach()

        # Single batched transfer: (n_head, 2) → numpy, no per-element .item()
        k_np = k_t.detach().float().cpu().numpy()   # one D→H copy, (n_head, 2)

        # Feed each head's hull (a tiny CPU loop over n_head rows)
        for h, hull in enumerate(self.hulls):
            hull.insert_batch(k_np[h:h+1], self._t)   # shape (1,2)

        self._t += 1

    def query(self, q_t: torch.Tensor) -> torch.Tensor:
        """
        q_t : (n_head, 2) – on GPU
        Returns : (n_head, head_dim) – on GPU

        ONE D→H copy for queries; result assembled with one GPU index op.
        """
        # Single batched transfer: (n_head, 2) → numpy
        q_np = q_t.detach().float().cpu().numpy()    # (n_head, 2)

        # Each hull returns the best token index for its head's direction
        best_token = np.stack([
            hull.argmax_dot_batch(q_np[h:h+1])       # (1,) int array
            for h, hull in enumerate(self.hulls)
        ], axis=0).squeeze(1)                         # (n_head,)

        # Convert to LongTensor and index v_buf in a single GPU op
        idx = torch.from_numpy(best_token).long().to(self.device)  # (n_head,)
        return self.v_buf[idx, self.h_range, :]       # (n_head, head_dim)




# ============================================================================
# Normalization
# ============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization. Faster than LayerNorm, no bias."""
    def __init__(self, ndim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class LayerNorm(nn.Module):
    """LayerNorm with optional bias (for GPT-2 compat)."""
    def __init__(self, ndim, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

def build_norm(ndim, config):
    """Create the appropriate normalization layer."""
    if config.norm_type == 'rmsnorm':
        return RMSNorm(ndim)
    return LayerNorm(ndim, bias=config.bias)

# ============================================================================
# Rotary Position Embeddings (RoPE)
# ============================================================================

def precompute_rope_cache(seq_len, head_dim, base=10000.0, device=None):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    theta = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, theta)
    return freqs.cos(), freqs.sin()

def apply_rope(x, cos, sin):
    T = x.size(2)
    half = x.size(3) // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_t = cos[:T].unsqueeze(0).unsqueeze(0)
    sin_t = sin[:T].unsqueeze(0).unsqueeze(0)
    return torch.cat([
        x1 * cos_t - x2 * sin_t,
        x2 * cos_t + x1 * sin_t,
    ], dim=-1)

# ──────────────────────────────────────────────────────────────────────────────
# 3b. 2-D Causal Self-Attention
# ──────────────────────────────────────────────────────────────────────────────

class TwoDCausalSelfAttention(nn.Module):
    """
    Drop-in replacement for nanoGPT's CausalSelfAttention.

    * head_dim is forced to 2 by GPTConfig.
    * Training  → standard flash-attention (unchanged from nanoGPT).
    * Inference → HullKVCache for O(log n) per decode step.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.n_embd // config.n_head   # == 2
        self.dropout  = config.dropout

        self.c_attn  = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(config.n_embd, config.n_embd,     bias=config.bias)

        self.qk_norm = config.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        else:
            self.q_norm = nn.Identity()
            self.k_norm = nn.Identity()

        self.attn_dropout  = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask for training
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
            .view(1, 1, config.block_size, config.block_size),
        )

    # ── training forward (full sequence) ─────────────────────────────────────

    def forward(self, x: torch.Tensor, rope_cos=None, rope_sin=None) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # Flash-attention (PyTorch ≥ 2.0) or manual fallback
        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale
            att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))

    # ── single-step inference with hull cache ─────────────────────────────────

    def forward_with_hull(
        self,
        x_t: torch.Tensor,    # (1, 1, n_embd)
        cache: HullKVCache,
        rope_cos=None,
        rope_sin=None,
    ) -> torch.Tensor:
        """One autoregressive decode step using HullKVCache."""
        assert x_t.shape[:2] == (1, 1)
        C = self.n_embd

        q, k, v = self.c_attn(x_t).split(C, dim=2)  # each (1,1,C)

        q_h = q.view(1, 1, self.n_head, self.head_dim).transpose(1, 2)
        k_h = k.view(1, 1, self.n_head, self.head_dim).transpose(1, 2)
        v_h = v.view(1, 1, self.n_head, self.head_dim).transpose(1, 2)

        q_h = self.q_norm(q_h)
        k_h = self.k_norm(k_h)

        if rope_cos is not None and rope_sin is not None:
            q_h = apply_rope(q_h, rope_cos, rope_sin)
            k_h = apply_rope(k_h, rope_cos, rope_sin)

        q_h = q_h.transpose(1, 2).squeeze(0).squeeze(0)
        k_h = k_h.transpose(1, 2).squeeze(0).squeeze(0)
        v_h = v_h.transpose(1, 2).squeeze(0).squeeze(0)

        cache.append(k_h, v_h)   # one D→H transfer (keys only)
        agg = cache.query(q_h)   # one D→H transfer (queries) + one GPU index
        y   = agg.view(1, C)     # (1, C)  – stays on GPU
        y   = self.resid_dropout(self.c_proj(y))
        return y.unsqueeze(0)    # (1, 1, C)


# ──────────────────────────────────────────────────────────────────────────────
# 3c. MLP, Block, GPT
# ──────────────────────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.activation = config.activation
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        if self.activation == 'relu2':
            x = F.relu(x).square()
        else:
            x = F.gelu(x)
        return self.dropout(self.c_proj(x))


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = build_norm(config.n_embd, config)
        self.attn = TwoDCausalSelfAttention(config)
        self.ln_2 = build_norm(config.n_embd, config)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

    def forward_with_hull(self, x_t: torch.Tensor, cache: HullKVCache) -> torch.Tensor:
        attn_out = self.attn.forward_with_hull(self.ln_1(x_t), cache)
        x_t = x_t + attn_out
        x_t = x_t + self.mlp(self.ln_2(x_t))
        return x_t


class NanoGPT2D(nn.Module):
    """
    nanoGPT with 2-D attention heads.

    Training:   model(idx, targets)  →  (logits, loss)
    Inference:  model.generate(idx, …, use_hull=True)
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte  = nn.Embedding(config.vocab_size, config.n_embd),
            wpe  = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h    = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying (GPT-2 paper)
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[model] NanoGPT2D  params={n_params:,}  "
              f"head_dim={config.n_embd // config.n_head}  "
              f"n_layer={config.n_layer}")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ── training forward ──────────────────────────────────────────────────────

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        B, T = idx.shape
        assert T <= self.config.block_size, \
            f"Sequence length {T} > block_size {self.config.block_size}"

        device = idx.device
        tok_emb = self.transformer.wte(idx)

        if self.config.use_rope:
            x = self.transformer.drop(tok_emb)
            rope_cos, rope_sin = self.rope_cos, self.rope_sin
        else:
            pos = torch.arange(T, device=device).unsqueeze(0)
            pos_emb = self.transformer.wpe(pos)
            x = self.transformer.drop(tok_emb + pos_emb)
            rope_cos, rope_sin = None, None

        for block in self.transformer.h:
            x = block(x, rope_cos, rope_sin)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        if self.config.logit_soft_cap > 0:
            cap = self.config.logit_soft_cap
            logits = cap * torch.tanh(logits / cap)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ── hull-accelerated generate (inference) ─────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        idx:            torch.Tensor,
        max_new_tokens: int,
        temperature:    float = 1.0,
        top_k:          Optional[int] = None,
        use_hull:       bool = True,
    ) -> torch.Tensor:
        """
        Autoregressive text generation.

        use_hull=True  → HullKVCache  (O(log n) per step, Percepta method)
        use_hull=False → standard nanoGPT forward  (reference / baseline)
        """
        if not use_hull:
            return self._generate_standard(idx, max_new_tokens, temperature, top_k)

        self.eval()
        device = idx.device
        cfg    = self.config

        # Build one GPU-resident cache per layer (pre-allocated value buffers)
        head_dim = cfg.n_embd // cfg.n_head
        caches = [
            HullKVCache(cfg.n_head, head_dim, cfg.block_size, device)
            for _ in self.transformer.h
        ]

        rope_cos = self.rope_cos if cfg.use_rope else None
        rope_sin = self.rope_sin if cfg.use_rope else None

        # ── warm-up: encode the prompt into caches token by token ────────────
        for t in range(idx.size(1)):
            tok = idx[:, t : t + 1]                        # (1,1)
            tok_emb = self.transformer.wte(tok)
            if cfg.use_rope:
                x_t = self.transformer.drop(tok_emb)
                r_cos = rope_cos[t:t+1] if rope_cos is not None else None
                r_sin = rope_sin[t:t+1] if rope_sin is not None else None
            else:
                pos = torch.tensor([[t]], device=device)
                x_t = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
                r_cos, r_sin = None, None

            for i, block in enumerate(self.transformer.h):
                x_t = block.forward_with_hull(x_t, caches[i], r_cos, r_sin)

        # ── decode loop ──────────────────────────────────────────────────────
        generated = idx.clone()
        cur_len   = idx.size(1)

        for _ in range(max_new_tokens):
            if cur_len >= cfg.block_size:
                break

            last = generated[:, -1:]                       # (1,1)
            tok_emb = self.transformer.wte(last)
            if cfg.use_rope:
                x_t = self.transformer.drop(tok_emb)
                r_cos = rope_cos[cur_len-1 : cur_len] if rope_cos is not None else None
                r_sin = rope_sin[cur_len-1 : cur_len] if rope_sin is not None else None
            else:
                pos  = torch.tensor([[cur_len - 1]], device=device)
                x_t  = self.transformer.drop(tok_emb + self.transformer.wpe(pos))
                r_cos, r_sin = None, None
                
            for i, block in enumerate(self.transformer.h):
                x_t = block.forward_with_hull(x_t, caches[i], r_cos, r_sin)

            x_t    = self.transformer.ln_f(x_t)
            logits = self.lm_head(x_t[:, -1, :])
            if cfg.logit_soft_cap > 0:
                cap = cfg.logit_soft_cap
                logits = cap * torch.tanh(logits / cap)
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_t], dim=1)
            cur_len  += 1

        return generated

    def _generate_standard(self, idx, max_new_tokens, temperature, top_k):
        """Vanilla nanoGPT generate (baseline / reference)."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs  = F.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)
            idx    = torch.cat([idx, next_t], dim=1)
        return idx

    # ── optimizer factory (mirrors nanoGPT) ──────────────────────────────────

    def configure_optimizers(self, cfg: TrainConfig, device_type: str):
        """AdamW with weight-decay applied only to 2-D+ parameters."""
        decay_params    = [p for p in self.parameters() if p.requires_grad and p.dim() >= 2]
        no_decay_params = [p for p in self.parameters() if p.requires_grad and p.dim() < 2]

        groups = [
            {"params": decay_params,    "weight_decay": cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        # fused AdamW available in PyTorch ≥ 2.0 on CUDA
        use_fused = device_type == "cuda" and "fused" in inspect.signature(
            torch.optim.AdamW
        ).parameters
        optimizer = torch.optim.AdamW(
            groups,
            lr=cfg.learning_rate,
            betas=(cfg.beta1, cfg.beta2),
            fused=use_fused if use_fused else False,
        )
        return optimizer


# ══════════════════════════════════════════════════════════════════════════════
# 4.  TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

def get_lr(step: int, cfg: TrainConfig) -> float:
    """Cosine decay with linear warm-up (nanoGPT schedule)."""
    if not cfg.decay_lr:
        return cfg.learning_rate
    if step < cfg.warmup_iters:
        return cfg.learning_rate * step / cfg.warmup_iters
    if step > cfg.lr_decay_iters:
        return cfg.min_lr
    progress = (step - cfg.warmup_iters) / max(1, cfg.lr_decay_iters - cfg.warmup_iters)
    coeff    = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def estimate_loss(
    model,
    train_loader: OpenWebTextLoader,
    val_loader:   OpenWebTextLoader,
    eval_iters:   int,
    ctx,
) -> dict:
    model.eval()
    out = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = loader.next_batch()
            with ctx:
                _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(model_cfg: GPTConfig, train_cfg: TrainConfig):
    # ── DDP setup ────────────────────────────────────────────────────────────
    ddp         = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        init_process_group(backend="nccl")
        rank       = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device     = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        master     = (rank == 0)
    else:
        rank = local_rank = 0
        world_size = 1
        device     = train_cfg.device
        master     = True

    device_type = "cuda" if "cuda" in device else "cpu"
    torch.manual_seed(train_cfg.seed + rank)

    # ── AMP context ──────────────────────────────────────────────────────────
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16":  torch.float16}[train_cfg.dtype]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda" else nullcontext()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(train_cfg.dtype == "float16"))

    # ── output dir ───────────────────────────────────────────────────────────
    if master:
        Path(train_cfg.out_dir).mkdir(parents=True, exist_ok=True)

    # ── data loaders ─────────────────────────────────────────────────────────
    train_loader = OpenWebTextLoader(
        "train", train_cfg.data_dir, train_cfg.batch_size,
        model_cfg.block_size, device, rank, world_size,
    )
    val_loader = OpenWebTextLoader(
        "val", train_cfg.data_dir, train_cfg.batch_size,
        model_cfg.block_size, device, rank, world_size,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    iter_num   = 0
    best_val   = float("inf")

    ckpt_path = Path(train_cfg.out_dir) / "ckpt.pt"

    if train_cfg.init_from == "resume" and ckpt_path.exists():
        print(f"[train] Resuming from {ckpt_path} …")
        ckpt     = torch.load(ckpt_path, map_location=device)
        model_cfg = GPTConfig(**ckpt["model_cfg"])
        model    = NanoGPT2D(model_cfg).to(device)
        model.load_state_dict(ckpt["model"])
        iter_num = ckpt.get("iter_num", 0)
        best_val = ckpt.get("best_val", float("inf"))
        print(f"[train] Resumed at iter {iter_num}, best_val={best_val:.4f}")
    else:
        print("[train] Training from scratch …")
        model = NanoGPT2D(model_cfg).to(device)

    # optional torch.compile
    if train_cfg.compile and hasattr(torch, "compile"):
        print("[train] Compiling model with torch.compile …")
        model = torch.compile(model)

    raw_model = model  # keep ref to unwrapped model

    if ddp:
        model = DDP(model, device_ids=[local_rank])
        raw_model = model.module

    optimizer = raw_model.configure_optimizers(train_cfg, device_type)

    if train_cfg.init_from == "resume" and ckpt_path.exists() and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    # ── wandb ─────────────────────────────────────────────────────────────────
    if train_cfg.wandb_log and master:
        import wandb
        wandb.init(
            project=train_cfg.wandb_project,
            name=train_cfg.wandb_run_name,
            id=train_cfg.wandb_run_name,
            resume="allow",
            config={**asdict(model_cfg), **asdict(train_cfg)},
        )

    # ── training loop ─────────────────────────────────────────────────────────
    model.train()
    t0          = time.time()
    local_step  = 0          # steps since last log
    running_mfu = -1.0

    X, Y = train_loader.next_batch()   # pre-fetch first batch

    while iter_num <= train_cfg.max_iters:

        # ── LR update ─────────────────────────────────────────────────────
        lr = get_lr(iter_num, train_cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ── eval + checkpoint ─────────────────────────────────────────────
        if iter_num % train_cfg.eval_interval == 0 and master:
            losses = estimate_loss(model, train_loader, val_loader,
                                   train_cfg.eval_iters, ctx)
            print(f"[eval]  iter={iter_num:6d}  "
                  f"train={losses['train']:.4f}  val={losses['val']:.4f}")

            if train_cfg.wandb_log:
                import wandb
                wandb.log({"iter": iter_num, "lr": lr, **losses})

            is_best = losses["val"] < best_val
            if is_best:
                best_val = losses["val"]

            if train_cfg.always_save_checkpoint or is_best:
                ckpt = {
                    "model":     raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(train_cfg),
                    "iter_num":  iter_num,
                    "best_val":  best_val,
                }
                torch.save(ckpt, ckpt_path)
                if is_best:
                    torch.save(ckpt, Path(train_cfg.out_dir) / "ckpt_best.pt")
                print(f"[ckpt]  saved → {ckpt_path}  (best_val={best_val:.4f})")

        if train_cfg.eval_only:
            break

        # ── gradient-accumulation forward/backward ─────────────────────────
        for micro_step in range(train_cfg.gradient_accumulation_steps):
            if ddp:
                # only sync on the last micro-step
                model.require_backward_grad_sync = (
                    micro_step == train_cfg.gradient_accumulation_steps - 1
                )
            with ctx:
                _, loss = model(X, Y)
                # scale loss for gradient accumulation
                loss = loss / train_cfg.gradient_accumulation_steps

            # pre-fetch next batch while GPU is busy
            X, Y = train_loader.next_batch()

            scaler.scale(loss).backward()

        # ── gradient clip + optimiser step ────────────────────────────────
        if train_cfg.grad_clip != 0.0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # ── logging ───────────────────────────────────────────────────────
        if iter_num % train_cfg.log_interval == 0 and master:
            t1        = time.time()
            dt        = t1 - t0
            t0        = t1
            lossf     = loss.item() * train_cfg.gradient_accumulation_steps
            print(f"[train] iter={iter_num:6d}  loss={lossf:.4f}  "
                  f"lr={lr:.2e}  dt={dt*1000:.0f}ms")

        iter_num  += 1
        local_step += 1

    if ddp:
        destroy_process_group()


# ══════════════════════════════════════════════════════════════════════════════
# 5.  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def generate_text(gen_cfg: GenerateConfig):
    import tiktoken

    device_type = "cuda" if "cuda" in gen_cfg.device else "cpu"
    ptdtype = {"float32": torch.float32,
               "bfloat16": torch.bfloat16,
               "float16":  torch.float16}[gen_cfg.dtype]
    ctx = (
        torch.amp.autocast(device_type=device_type, dtype=ptdtype)
        if device_type == "cuda" else nullcontext()
    )

    # ── load checkpoint ───────────────────────────────────────────────────────
    ckpt_path = Path(gen_cfg.out_dir) / "ckpt_best.pt"
    if not ckpt_path.exists():
        ckpt_path = Path(gen_cfg.out_dir) / "ckpt.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"No checkpoint found in {gen_cfg.out_dir}. "
            "Run training first."
        )

    print(f"[generate] Loading checkpoint from {ckpt_path} …")
    ckpt      = torch.load(ckpt_path, map_location=gen_cfg.device)
    model_cfg = GPTConfig(**ckpt["model_cfg"])
    model     = NanoGPT2D(model_cfg)
    # Strip wrapper prefixes anchored to the START of each key.
    # torch.compile adds "_orig_mod.", DDP adds "module." — always as a
    # true prefix, never mid-key.  Using .replace() is unsafe because it
    # corrupts any layer genuinely named e.g. "transformer.module_norm.weight".
    def _strip_prefixes(k: str) -> str:
        for prefix in ("_orig_mod.", "module."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k
    state = {_strip_prefixes(k): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    model.eval()
    model.to(gen_cfg.device)

    # ── tokenise prompt ───────────────────────────────────────────────────────
    enc    = tiktoken.get_encoding("gpt2")
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<|endoftext|>"}),
                                    dtype=torch.long, device=gen_cfg.device).unsqueeze(0)
    decode = lambda ids: enc.decode(ids.tolist())

    print(f"\n[generate] Prompt : {gen_cfg.prompt!r}")
    print(f"[generate] Hull   : {gen_cfg.use_hull}")
    print("─" * 60)

    for i in range(gen_cfg.num_samples):
        idx = encode(gen_cfg.prompt)
        t0  = time.perf_counter()
        with ctx:
            out = model.generate(
                idx,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_k=gen_cfg.top_k,
                use_hull=gen_cfg.use_hull,
            )
        dt   = time.perf_counter() - t0
        text = decode(out[0])
        tps  = gen_cfg.max_new_tokens / dt
        print(f"\n--- Sample {i+1}  ({tps:.0f} tok/s) ---")
        print(text)
        print()

    print("─" * 60)


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN ENTRY-POINT
# ══════════════════════════════════════════════════════════════════════════════

def build_cli() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="NanoGPT 2D Attention – single-file edition",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── prepare ───────────────────────────────────────────────────────────────
    pp = sub.add_parser("prepare", help="Download + tokenise a dataset (OpenWebText, TinyStories, FineWeb-edu)")
    pp.add_argument("--dataset",   choices=["openwebtext", "tinystories", "fineweb_edu"], default="openwebtext")
    pp.add_argument("--data_dir",  default=None, help="Overrides default data_dir (data/{dataset})")
    pp.add_argument("--num_proc",  type=int, default=None)

    # ── train ─────────────────────────────────────────────────────────────────
    tp = sub.add_parser("train", help="Train the model")
    # model
    tp.add_argument("--n_embd",      type=int,   default=36)
    tp.add_argument("--n_head",      type=int,   default=18)
    tp.add_argument("--n_layer",     type=int,   default=7)
    tp.add_argument("--block_size",  type=int,   default=1024)
    tp.add_argument("--vocab_size",  type=int,   default=50304)
    tp.add_argument("--dropout",     type=float, default=0.0)
    # train
    tp.add_argument("--out_dir",     default="out-owt-2d")
    tp.add_argument("--data_dir",    default="data/openwebtext")
    tp.add_argument("--init_from",   default="scratch", choices=["scratch", "resume"])
    tp.add_argument("--batch_size",  type=int,   default=12)
    tp.add_argument("--grad_accum",  type=int,   default=40, dest="gradient_accumulation_steps")
    tp.add_argument("--max_iters",   type=int,   default=600_000)
    tp.add_argument("--learning_rate", type=float, default=6e-4)
    tp.add_argument("--min_lr",        type=float, default=6e-5)
    tp.add_argument("--warmup_iters",  type=int,   default=2_000)
    tp.add_argument("--lr_decay_iters", type=int,   default=600_000)
    tp.add_argument("--weight_decay",  type=float, default=1e-1)
    tp.add_argument("--use_rope",      action="store_true")
    tp.add_argument("--use_muon",      action="store_true")
    tp.add_argument("--eval_interval", type=int, default=500)
    tp.add_argument("--eval_iters",    type=int, default=200)
    tp.add_argument("--log_interval",  type=int, default=10)
    tp.add_argument("--device",      default="cuda")
    tp.add_argument("--dtype",       default="bfloat16")
    tp.add_argument("--compile",     action="store_true", default=True)
    tp.add_argument("--no_compile",  action="store_false", dest="compile")
    tp.add_argument("--wandb_log",   action="store_true")
    tp.add_argument("--wandb_project", default="nanogpt-2d")
    tp.add_argument("--run_id",      default="run", help="Unique ID for this run (used for wandb name and subfolder in out_dir)")
    tp.add_argument("--eval_only",   action="store_true")

    # ── generate ──────────────────────────────────────────────────────────────
    gp = sub.add_parser("generate", help="Generate text from a checkpoint")
    gp.add_argument("--out_dir",         default="out-owt-2d")
    gp.add_argument("--prompt",          default="The")
    gp.add_argument("--num_samples",     type=int,   default=3)
    gp.add_argument("--max_new_tokens",  type=int,   default=200)
    gp.add_argument("--temperature",     type=float, default=0.8)
    gp.add_argument("--top_k",           type=int,   default=200)
    gp.add_argument("--no_hull",         action="store_false", dest="use_hull",
                    help="Use standard generate instead of HullKVCache")
    gp.add_argument("--device",          default="cuda")
    gp.add_argument("--dtype",           default="bfloat16")

    return p


def main():
    parser = build_cli()
    args   = parser.parse_args()

    if args.command == "prepare":
        prepare_dataset(args.dataset, args.data_dir, args.num_proc)

    elif args.command == "train":
        model_cfg = GPTConfig(
            n_embd     = args.n_embd,
            n_head     = args.n_head,
            n_layer    = args.n_layer,
            block_size = args.block_size,
            vocab_size = args.vocab_size,
            dropout    = args.dropout,
            use_rope   = getattr(args, "use_rope", False),
        )
        import os
        train_cfg = TrainConfig(
            out_dir                    = os.path.join(args.out_dir, args.run_id) if args.run_id != "run" else args.out_dir,
            data_dir                   = args.data_dir,
            init_from                  = args.init_from,
            batch_size                 = args.batch_size,
            gradient_accumulation_steps= args.gradient_accumulation_steps,
            max_iters                  = args.max_iters,
            learning_rate              = args.learning_rate,
            min_lr                     = args.min_lr,
            warmup_iters               = args.warmup_iters,
            lr_decay_iters             = args.lr_decay_iters,
            weight_decay               = args.weight_decay,
            use_muon                   = getattr(args, "use_muon", False),
            eval_interval              = args.eval_interval,
            eval_iters                 = args.eval_iters,
            log_interval               = args.log_interval,
            device                     = args.device,
            dtype                      = args.dtype,
            compile                    = args.compile,
            wandb_log                  = args.wandb_log,
            wandb_project              = args.wandb_project,
            wandb_run_name             = args.run_id,
            eval_only                  = args.eval_only,
        )
        train(model_cfg, train_cfg)

    elif args.command == "generate":
        gen_cfg = GenerateConfig(
            out_dir        = args.out_dir,
            prompt         = args.prompt,
            num_samples    = args.num_samples,
            max_new_tokens = args.max_new_tokens,
            temperature    = args.temperature,
            top_k          = args.top_k,
            use_hull       = args.use_hull,
            device         = args.device,
            dtype          = args.dtype,
        )
        generate_text(gen_cfg)




# ============================================================================
# Muon + AdamW Combined Optimizer
# ============================================================================

class MuonAdamW(torch.optim.Optimizer):
    def __init__(self, muon_params, adam_params,
                 muon_lr=0.02, muon_momentum=0.95, muon_weight_decay=0.01,
                 muon_nesterov=True, muon_ns_steps=5,
                 adam_betas=(0.9, 0.95), adam_eps=1e-8):

        all_groups = []

        # Muon param group
        muon_group = dict(
            params=list(muon_params),
            lr=muon_lr,
            base_lr=muon_lr,
            momentum=muon_momentum,
            weight_decay=muon_weight_decay,
            nesterov=muon_nesterov,
            ns_steps=muon_ns_steps,
            is_muon=True,
        )
        all_groups.append(muon_group)

        # AdamW param groups
        for g in adam_params:
            g['is_muon'] = False
            g['base_lr'] = g.get('lr', 6e-4)
            g['betas'] = adam_betas
            g['eps'] = adam_eps
            all_groups.append(g)

        defaults = dict(lr=muon_lr, weight_decay=0.0, is_muon=False)
        super().__init__(all_groups, defaults)

    @staticmethod
    @torch.no_grad()
    def _newton_schulz(G, steps=5):
        a, b, c = (3.4445, -4.7750, 2.0315)
        X = G.bfloat16()
        transposed = False
        if X.size(-2) > X.size(-1):
            X = X.mT
            transposed = True
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
        for _ in range(steps):
            A = X @ X.mT
            B = b * A + c * A @ A
            X = a * X + B @ X
        if transposed:
            X = X.mT
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get('is_muon', False):
                self._muon_step(group)
            else:
                self._adam_step(group)

        return loss

    def _muon_step(self, group):
        lr = group['lr']
        wd = group['weight_decay']
        beta = group['momentum']
        nesterov = group['nesterov']
        ns_steps = group['ns_steps']

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad
            state = self.state[p]

            if len(state) == 0:
                state['momentum_buffer'] = torch.zeros_like(grad)

            buf = state['momentum_buffer']
            buf.lerp_(grad, 1 - beta)

            if nesterov:
                update = grad.lerp(buf, beta)
            else:
                update = buf

            update = self._newton_schulz(update, steps=ns_steps)
            scale = (p.size(0) / p.size(1)) ** 0.5

            if wd > 0:
                p.mul_(1 - lr * wd)
            p.add_(update.to(p.dtype), alpha=-lr * scale)

    def _adam_step(self, group):
        lr = group['lr']
        wd = group.get('weight_decay', 0.0)
        beta1, beta2 = group['betas']
        eps = group['eps']

        for p in group['params']:
            if p.grad is None:
                continue
            grad = p.grad.float()
            state = self.state[p]

            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(grad)
                state['exp_avg_sq'] = torch.zeros_like(grad)

            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            exp_avg.lerp_(grad, 1 - beta1)
            exp_avg_sq.lerp_(grad.square(), 1 - beta2)

            step = state['step']
            bc1 = 1 - beta1 ** step
            bc2 = 1 - beta2 ** step

            if wd > 0:
                p.mul_(1 - lr * wd)

            step_size = lr / bc1
            denom = (exp_avg_sq / bc2).sqrt().add_(eps)
            update = exp_avg / denom
            p.add_(update.to(p.dtype), alpha=-step_size)

if __name__ == "__main__":
    main()
