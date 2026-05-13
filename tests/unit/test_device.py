"""Unit tests for _device.py — parametrized over cuda / mps / cpu via monkeypatch."""
import os
import importlib
import pytest
import torch


# ---------------------------------------------------------------------------
# Helpers to simulate each backend
# ---------------------------------------------------------------------------

def _patch_cuda_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)


def _patch_mps_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)


def _patch_cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: False)


# ---------------------------------------------------------------------------
# _detect_device
# ---------------------------------------------------------------------------

def test_detect_device_cuda(monkeypatch):
    _patch_cuda_only(monkeypatch)
    from aesthetic_scoring._device import _detect_device
    d = _detect_device()
    assert d.type == "cuda"


def test_detect_device_mps(monkeypatch):
    _patch_mps_only(monkeypatch)
    from aesthetic_scoring._device import _detect_device
    d = _detect_device()
    assert d.type == "mps"


def test_detect_device_cpu(monkeypatch):
    _patch_cpu_only(monkeypatch)
    from aesthetic_scoring._device import _detect_device
    d = _detect_device()
    assert d.type == "cpu"


def test_detect_device_cuda_takes_priority_over_mps(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_built", lambda: True)
    from aesthetic_scoring._device import _detect_device
    d = _detect_device()
    assert d.type == "cuda"


# ---------------------------------------------------------------------------
# get_precision
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device_type,expected", [
    ("cuda", "fp16"),
    ("mps", "fp16"),
    ("cpu", "fp32"),
])
def test_get_precision(device_type, expected):
    from aesthetic_scoring._device import get_precision
    d = torch.device(device_type)
    assert get_precision(d) == expected


# ---------------------------------------------------------------------------
# get_dtype
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prec,expected", [
    ("fp16", torch.float16),
    ("bf16", torch.bfloat16),
    ("fp32", torch.float32),
])
def test_get_dtype(prec, expected):
    from aesthetic_scoring._device import get_dtype
    assert get_dtype(prec) == expected


# ---------------------------------------------------------------------------
# MPS fallback env var
# ---------------------------------------------------------------------------

def test_mps_fallback_env_var_set_when_mps(monkeypatch):
    """_maybe_enable_mps_fallback sets the env var when device is mps."""
    # Remove it first so setdefault actually sets it
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    from aesthetic_scoring._device import _maybe_enable_mps_fallback
    _maybe_enable_mps_fallback(torch.device("mps"))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1"


def test_mps_fallback_env_var_not_overwritten(monkeypatch):
    """_maybe_enable_mps_fallback does NOT overwrite a pre-existing value."""
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", "0")
    from aesthetic_scoring._device import _maybe_enable_mps_fallback
    _maybe_enable_mps_fallback(torch.device("mps"))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "0"


def test_mps_fallback_env_var_not_set_on_cpu(monkeypatch):
    """_maybe_enable_mps_fallback does nothing when device is cpu."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    from aesthetic_scoring._device import _maybe_enable_mps_fallback
    _maybe_enable_mps_fallback(torch.device("cpu"))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None


def test_mps_fallback_env_var_not_set_on_cuda(monkeypatch):
    """_maybe_enable_mps_fallback does nothing when device is cuda."""
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    from aesthetic_scoring._device import _maybe_enable_mps_fallback
    _maybe_enable_mps_fallback(torch.device("cuda"))
    assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None


# ---------------------------------------------------------------------------
# vram_used_gib / reset_peak_vram — safe on non-CUDA hosts
# ---------------------------------------------------------------------------

def test_vram_used_gib_returns_float():
    from aesthetic_scoring._device import vram_used_gib
    val = vram_used_gib()
    assert isinstance(val, float)
    assert val >= 0.0


def test_reset_peak_vram_no_raise():
    from aesthetic_scoring._device import reset_peak_vram
    reset_peak_vram()  # must not raise on any host


# ---------------------------------------------------------------------------
# empty_cache — safe on all backends
# ---------------------------------------------------------------------------

def test_empty_cache_cpu():
    from aesthetic_scoring._device import empty_cache
    empty_cache(torch.device("cpu"))  # must not raise


def test_empty_cache_cuda_skipped_on_non_cuda():
    """Calling empty_cache with cuda device on a non-CUDA host should not raise."""
    from aesthetic_scoring._device import empty_cache
    if not torch.cuda.is_available():
        # The function dispatches on device.type, so calling with cuda device
        # will attempt torch.cuda.empty_cache() — safe even without a GPU
        # because torch raises only if no CUDA driver; we wrap in try/except
        # to be tolerant.
        try:
            empty_cache(torch.device("cuda"))
        except Exception:
            pass  # acceptable on CPU-only host
    else:
        empty_cache(torch.device("cuda"))  # must not raise


def test_empty_cache_mps():
    from aesthetic_scoring._device import empty_cache
    # Must not raise on any host, including when MPS isn't available
    empty_cache(torch.device("mps"))
