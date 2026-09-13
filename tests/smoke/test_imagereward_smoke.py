"""GPU smoke test for ImageReward (optional dependency).

Skipped when no GPU backend is available, or when the optional ImageReward
package is absent or not importable (e.g. an install whose transformers
version is incompatible with its vendored BLIP code).
"""
import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"
PHOTO_B = "tests/fixtures/photo_b.jpg"
PROMPT = "a vibrant colorful gradient abstract image"


def _imagereward_usable() -> bool:
    """True only if the optional ImageReward package actually imports.

    A find_spec check is not enough: an installed but broken ImageReward is
    present on disk yet raises at import time. Such an install must skip, not
    fail — the smoke test's contract is "run only when the model is usable".
    """
    try:
        import ImageReward  # noqa: F401
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available()
        and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
        reason="No GPU backend (CUDA or MPS) available",
    ),
    pytest.mark.skipif(
        not _imagereward_usable(),
        reason="ImageReward optional dependency absent or not importable",
    ),
]


def test_imagereward_smoke():
    from aesthetic_scoring import score_imagereward
    from aesthetic_scoring.imagereward import unload, _load
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()
    is_mps = (not is_cuda) and torch.backends.mps.is_available()

    # Pre-warm: trigger weight download/load before the timed section
    _load()

    reset_peak_vram()
    t0 = time.perf_counter()

    result = score_imagereward([PHOTO_A, PHOTO_B], PROMPT)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    # Correctness
    assert len(result.scores) == 2
    assert len(result.ranked_image_ids) == 2
    assert all(isinstance(s, float) for s in result.scores)
    assert set(result.ranked_image_ids) == {"photo_a.jpg", "photo_b.jpg"}
    assert result.latency_ms > 0
    assert result.prompt == PROMPT

    json.dumps(dataclasses.asdict(result))

    if is_cuda:
        assert elapsed < 60, f"ImageReward smoke took {elapsed:.1f}s, expected < 60s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"
    elif is_mps:
        assert elapsed < 180, f"ImageReward smoke took {elapsed:.1f}s, expected < 180s (MPS)"
        assert result.device == "mps", f"Expected device='mps', got '{result.device}'"

    print(f"\nImageReward scores: {result.scores}, ranked: {result.ranked_image_ids}, "
          f"device: {result.device}, VRAM/peak: {peak_gib:.2f} GiB")
