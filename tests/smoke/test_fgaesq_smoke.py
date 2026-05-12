"""GPU smoke test for FGAesQ."""
import dataclasses
import json
import time
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"


def test_fgaesq_smoke():
    from aesthetic_scoring import score_fgaesq
    from aesthetic_scoring.fgaesq import unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

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

    assert elapsed < 60, f"FGAesQ smoke took {elapsed:.1f}s, expected < 60s"

    if torch.cuda.is_available():
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"

    print(f"\nFGAesQ aesthetic: {result.aesthetic_score:.4f}, technical: {result.technical_score:.4f}, VRAM: {peak_gib:.2f} GiB")
