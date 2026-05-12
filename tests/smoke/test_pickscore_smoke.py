"""GPU smoke test for PickScore."""
import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"
PHOTO_B = "tests/fixtures/photo_b.jpg"
PROMPT = "a vibrant colorful gradient abstract image"


def test_pickscore_smoke():
    from aesthetic_scoring import score_pickscore
    from aesthetic_scoring.pickscore import unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_pickscore([PHOTO_A, PHOTO_B], PROMPT)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    # Assertions
    assert len(result.scores) == 2
    assert len(result.probabilities) == 2
    assert len(result.ranked_image_ids) == 2

    prob_sum = sum(result.probabilities)
    assert abs(prob_sum - 1.0) < 1e-3, f"probabilities sum {prob_sum:.6f} not ~1.0"

    assert result.latency_ms > 0
    assert result.prompt == PROMPT

    # JSON-serializable
    json.dumps(dataclasses.asdict(result))

    assert elapsed < 60, f"PickScore smoke took {elapsed:.1f}s, expected < 60s"

    if torch.cuda.is_available():
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"

    print(f"\nPickScore probs: {result.probabilities}, ranked: {result.ranked_image_ids}, VRAM: {peak_gib:.2f} GiB")
