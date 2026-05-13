"""GPU smoke test for PickScore.

Skipped automatically when neither CUDA nor MPS is available.
"""
import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"
PHOTO_B = "tests/fixtures/photo_b.jpg"
PROMPT = "a vibrant colorful gradient abstract image"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="No GPU backend (CUDA or MPS) available",
)


def test_pickscore_smoke():
    from aesthetic_scoring import score_pickscore
    from aesthetic_scoring.pickscore import unload, _load
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()
    is_mps = (not is_cuda) and torch.backends.mps.is_available()

    # Pre-warm: trigger weight download/load before the timed section
    _load()

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_pickscore([PHOTO_A, PHOTO_B], PROMPT)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    # Correctness
    assert len(result.scores) == 2
    assert len(result.probabilities) == 2
    assert len(result.ranked_image_ids) == 2

    prob_sum = sum(result.probabilities)
    assert abs(prob_sum - 1.0) < 1e-3, f"probabilities sum {prob_sum:.6f} not ~1.0"

    assert result.latency_ms > 0
    assert result.prompt == PROMPT

    json.dumps(dataclasses.asdict(result))

    if is_cuda:
        assert elapsed < 60, f"PickScore smoke took {elapsed:.1f}s, expected < 60s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"
    elif is_mps:
        assert elapsed < 180, f"PickScore smoke took {elapsed:.1f}s, expected < 180s (MPS)"
        assert result.device == "mps", f"Expected device='mps', got '{result.device}'"
        assert result.precision == "fp16", f"Expected precision='fp16' on MPS, got '{result.precision}'"

    print(f"\nPickScore probs: {result.probabilities}, ranked: {result.ranked_image_ids}, "
          f"device: {result.device}, VRAM/peak: {peak_gib:.2f} GiB")
