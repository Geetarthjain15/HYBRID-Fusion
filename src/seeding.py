"""
Reproducibility helpers.

AUDIT FINDING 3: the original train.py seeded only the validation-split
permutation. DataLoader shuffling, all four augmentation transforms,
Dropout(0.3) and the fusion head's weight init were unseeded, so two runs of
identical code produced different models. No baseline-vs-improved comparison
is separable from that noise.

This module is a pure addition: it fixes WHICH random draws occur. It does not
change the experiment, the data, the architecture or the objective.
"""

import os
import random

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> dict:
    """Seed every RNG the training path touches. Returns what was set."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # cudnn.benchmark picks the fastest algorithm per shape; harmless for
        # reproducibility of *results* at fixed shapes, but not bitwise exact.
        torch.backends.cudnn.benchmark = True

    return {
        "seed": seed,
        "deterministic": deterministic,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
    }


def seed_worker(worker_id: int):
    """DataLoader worker init: give each worker a distinct, derived seed."""
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def environment_fingerprint() -> dict:
    """Everything a future researcher needs to reproduce the numbers."""
    import platform
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
    }
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except Exception:
        info["transformers"] = None
    try:
        import torchvision
        info["torchvision"] = torchvision.__version__
    except Exception:
        info["torchvision"] = None
    return info
