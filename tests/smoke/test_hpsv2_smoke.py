"""GPU smoke test for HPSv2."""
import dataclasses
import json
import time
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"
PROMPT = "a vibrant colorful gradient abstract image"


def test_hpsv2_smoke():
    from aesthetic_scoring import score_hpsv2
    from aesthetic_scoring.hpsv2 import unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_hpsv2(PHOTO_A, PROMPT)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    assert isinstance(result.preference_score, float)
    assert result.preference_score != 0.0
    assert result.latency_ms > 0
    assert result.prompt == PROMPT

    json.dumps(dataclasses.asdict(result))

    assert elapsed < 60, f"HPSv2 smoke took {elapsed:.1f}s, expected < 60s"

    if torch.cuda.is_available():
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"

    print(f"\nHPSv2 preference_score: {result.preference_score:.4f}, latency: {result.latency_ms:.1f}ms, VRAM: {peak_gib:.2f} GiB")
