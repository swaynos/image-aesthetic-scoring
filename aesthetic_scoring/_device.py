"""Shared CUDA detection, precision selection, and OOM-wrapping utilities."""

import contextlib
from contextlib import contextmanager
from typing import Optional

import torch

from .errors import GpuMemoryError, ModelInferenceError


def get_device() -> torch.device:
    """Return the best available device (cuda:0 or cpu)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def get_precision(device: torch.device) -> str:
    """Return preferred precision string for the given device."""
    if device.type == "cuda":
        return "fp16"
    return "fp32"


def get_dtype(precision: str) -> torch.dtype:
    """Convert precision string to torch dtype."""
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[precision]


@contextmanager
def inference_guard():
    """Context manager that maps CUDA OOM -> GpuMemoryError, other runtime errors -> ModelInferenceError."""
    try:
        yield
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        raise GpuMemoryError(f"CUDA out of memory during inference: {exc}") from exc
    except (RuntimeError, Exception) as exc:
        err_str = str(exc).lower()
        if "out of memory" in err_str or "cuda" in err_str and "memory" in err_str:
            torch.cuda.empty_cache()
            raise GpuMemoryError(f"GPU memory error during inference: {exc}") from exc
        raise ModelInferenceError(f"Inference failed: {exc}") from exc


def vram_used_gib() -> float:
    """Return current peak VRAM allocated in GiB (0.0 on CPU)."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


def reset_peak_vram():
    """Reset the CUDA peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
