"""GPU smoke test for FGAesQ.

Skipped automatically when neither CUDA nor MPS is available.
"""
import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="No GPU backend (CUDA or MPS) available",
)


def test_fgaesq_smoke():
    from aesthetic_scoring import score_fgaesq
    from aesthetic_scoring.fgaesq import unload, _load
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()
    is_mps = (not is_cuda) and torch.backends.mps.is_available()

    # Pre-warm: trigger weight download/load before the timed section
    _load()

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_fgaesq(PHOTO_A)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    assert isinstance(result.technical_score, float)
    assert isinstance(result.aesthetic_score, float)
    assert isinstance(result.subscores, dict)
    assert result.aesthetic_score > 0
    assert result.latency_ms > 0

    json.dumps(dataclasses.asdict(result))

    if is_cuda:
        assert elapsed < 60, f"FGAesQ smoke took {elapsed:.1f}s, expected < 60s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"
    elif is_mps:
        assert elapsed < 180, f"FGAesQ smoke took {elapsed:.1f}s, expected < 180s (MPS)"
        assert result.device == "mps", f"Expected device='mps', got '{result.device}'"
        # FGAesQ runs fp32; precision field should reflect that
        assert result.precision == "fp32", f"FGAesQ expected fp32, got '{result.precision}'"

    print(f"\nFGAesQ aesthetic: {result.aesthetic_score:.4f}, technical: {result.technical_score:.4f}, "
          f"device: {result.device}, VRAM/peak: {peak_gib:.2f} GiB")
