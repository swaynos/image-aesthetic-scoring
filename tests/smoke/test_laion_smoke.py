"""GPU smoke test for LAION-Aesthetics v2.5."""
import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"


def test_laion_smoke():
    from aesthetic_scoring import score_laion
    from aesthetic_scoring.laion import unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_laion(PHOTO_A)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    # Unload after test to free VRAM for subsequent tests
    unload()

    # Assertions
    assert isinstance(result.aesthetic_score, float), "aesthetic_score must be float"
    assert result.aesthetic_score > 0, "aesthetic_score must be positive"
    assert result.score_scale == "1-10"
    assert result.latency_ms > 0
    assert result.model_name != ""
    assert result.model_version != ""
    assert result.image_id != ""

    # JSON-serializable
    json.dumps(dataclasses.asdict(result))

    # Latency under 30 s (weight download excluded by the lazy load before timer)
    assert elapsed < 30, f"LAION smoke took {elapsed:.1f}s, expected < 30s"

    # VRAM ceiling
    if torch.cuda.is_available():
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB ceiling"

    print(f"\nLAION score: {result.aesthetic_score:.4f}, latency: {result.latency_ms:.1f}ms, VRAM: {peak_gib:.2f} GiB")
