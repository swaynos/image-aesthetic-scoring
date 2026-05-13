"""GPU smoke test for LAION-Aesthetics v2.5.

Skipped automatically when neither CUDA nor MPS is available.
"""
import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"

# Skip entire module if no GPU backend is available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="No GPU backend (CUDA or MPS) available",
)


def test_laion_smoke():
    from aesthetic_scoring import score_laion
    from aesthetic_scoring.laion import unload, _load
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib, get_device

    is_cuda = torch.cuda.is_available()
    is_mps = (not is_cuda) and torch.backends.mps.is_available()

    # Pre-warm: trigger weight download/load before the timed section
    # (spec excludes weight download from latency budget)
    _load()

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_laion(PHOTO_A)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    # Basic correctness
    assert isinstance(result.aesthetic_score, float), "aesthetic_score must be float"
    assert result.aesthetic_score > 0, "aesthetic_score must be positive"
    assert result.score_scale == "1-10"
    assert result.latency_ms > 0
    assert result.model_name != ""
    assert result.model_version != ""
    assert result.image_id != ""

    # JSON-serializable
    json.dumps(dataclasses.asdict(result))

    # Backend-specific assertions
    if is_cuda:
        assert elapsed < 30, f"LAION smoke took {elapsed:.1f}s, expected < 30s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB ceiling"
    elif is_mps:
        assert elapsed < 90, f"LAION smoke took {elapsed:.1f}s, expected < 90s (MPS)"
        assert result.device == "mps", f"Expected device='mps', got '{result.device}'"
        assert result.precision == "fp16", f"Expected precision='fp16' on MPS, got '{result.precision}'"

    print(f"\nLAION score: {result.aesthetic_score:.4f}, latency: {result.latency_ms:.1f}ms, "
          f"device: {result.device}, VRAM/peak: {peak_gib:.2f} GiB")
