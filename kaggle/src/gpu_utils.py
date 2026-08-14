# =============================================================================
# GPU utilities - guarantee EVERY available GPU is used
#
# `device_map="auto"` (accelerate) silently places the whole model on GPU 0 when
# the weights fit in one GPU's memory, leaving the other T4 idle. We instead
# build an EXPLICIT round-robin device_map that shards the decoder layer stack
# across ALL CUDA devices, so both T4s are always used during rollouts/training.
# =============================================================================
import math
from collections import Counter
from typing import Any, Dict, Optional

import torch

from transformers import AutoConfig


def round_robin_device_map(
    num_layers: int,
    num_gpus: int,
) -> Dict[str, int]:
    """Shard `model.layers.*` round-robin across all GPUs; embed pules/norm/lm_head land on GPU 0."""
    if num_gpus <= 0:
        raise RuntimeError("CUDA GPU is required. CPU-only computation is not allowed.")
    dm: Dict[str, int] = {}
    dm["model.embed_tokens"] = 0
    for i in range(num_layers):
        dm[f"model.layers.{i}"] = i % num_gpus
    dm["model.norm"] = (num_layers - 1) % num_gpus
    dm["lm_head"] = (num_layers - 1) % num_gpus
    return dm


def build_multi_gpu_device_map(
    model_name: str,
    cache_dir: Optional[str] = None,
    num_gpus: Optional[int] = None,
) -> Dict[str, int]:
    """Explicit device map using ALL GPUs (not just GPU 0)."""
    num_gpus = num_gpus or torch.cuda.device_count()
    cfg = AutoConfig.from_pretrained(model_name, cache_dir=cache_dir)
    num_layers = getattr(cfg, "num_hidden_layers", 0) or getattr(cfg, "num_layers", 0)
    if num_layers <= 0:
        # Unknown arch -> fall back to auto (still honours any multi-GPU we did).
        raise RuntimeError(f"Could not determine num_hidden_layers for {model_name}")
    dm = round_robin_device_map(num_layers, num_gpus)
    counts = Counter(dm.values())
    print(
        f"[GPU] Sharding {model_name}: {num_layers} layers across "
        f"{num_gpus} GPUs -> {dict(sorted(counts.items()))} modules per GPU"
    )
    return dm