"""Shared device detection, precision selection, and OOM-wrapping utilities.

Device priority: cuda:0 → mps → cpu

MPS notes:
- On first import, if the selected device is MPS, this module sets
  PYTORCH_ENABLE_MPS_FALLBACK=1 so ops without MPS kernels silently
  fall back to CPU rather than raising.
- fp16 is used on both CUDA and MPS. If an individual model module
  determines fp16 is numerically unstable on MPS it may switch to fp32
  and document this in its own docstring.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

import torch

from .errors import GpuMemoryError, ModelInferenceError


def _detect_device() -> torch.device:
    """Return the best available device: cuda:0 → mps → cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _maybe_enable_mps_fallback(device: torch.device) -> None:
    """If device is MPS, set PYTORCH_ENABLE_MPS_FALLBACK=1 (no-op if already set)."""
    if device.type == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


# Run detection once at import time and set the fallback env var if needed.
_DEVICE = _detect_device()
_maybe_enable_mps_fallback(_DEVICE)


def get_device() -> torch.device:
    """Return the best available device (cuda:0, mps, or cpu)."""
    return _DEVICE


def get_precision(device: torch.device) -> str:
    """Return preferred precision string for the given device.

    Returns:
        "fp16" for cuda and mps, "fp32" for cpu.
    """
    if device.type in ("cuda", "mps"):
        return "fp16"
    return "fp32"


def get_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


def empty_cache(device: torch.device) -> None:
    """Drain the device memory cache, safe on all backends."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        # torch.mps.empty_cache() available since torch 2.0
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
    # cpu: no-op


@contextmanager
def inference_guard():
    """Map device OOM → GpuMemoryError; other runtime errors → ModelInferenceError."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise GpuMemoryError(f"CUDA out of memory during inference: {exc}") from exc
    except (RuntimeError, Exception) as exc:
        err_str = str(exc).lower()
        # CUDA OOM expressed as RuntimeError
        if "out of memory" in err_str and ("cuda" in err_str or "mps" in err_str):
            empty_cache(_DEVICE)
            raise GpuMemoryError(f"GPU memory error during inference: {exc}") from exc
        # MPS allocator errors
        if "mps" in err_str and ("memory" in err_str or "out of memory" in err_str):
            empty_cache(_DEVICE)
            raise GpuMemoryError(f"MPS memory error during inference: {exc}") from exc
        # Legacy CUDA OOM string match
        if "cuda" in err_str and "memory" in err_str:
            empty_cache(_DEVICE)
            raise GpuMemoryError(f"GPU memory error during inference: {exc}") from exc
        raise ModelInferenceError(f"Inference failed: {exc}") from exc


def vram_used_gib() -> float:
    """Return current peak VRAM/unified-memory allocated in GiB (0.0 on CPU/MPS)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    # MPS does not expose a reliable peak-memory API; return 0.0
    return 0.0


def reset_peak_vram() -> None:
    """Reset CUDA peak memory stats. No-op on MPS/CPU."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
