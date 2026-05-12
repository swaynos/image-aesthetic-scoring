"""Unit tests for laion module — no real model loaded."""
import pytest
from unittest.mock import MagicMock, patch
import torch

from aesthetic_scoring.errors import ModelLoadError, GpuMemoryError
from aesthetic_scoring.types import LaionScoreResult


def _make_fake_result():
    return LaionScoreResult(
        image_id="photo_a.jpg",
        model_name="laion-aesthetics-v2.5",
        model_version="sac+logos+ava1-l14-linearMSE",
        latency_ms=50.0,
        device="cuda:0",
        precision="fp16",
        aesthetic_score=6.5,
        score_scale="1-10",
    )


def test_score_laion_missing_file(tmp_path):
    from aesthetic_scoring.laion import score_laion
    with pytest.raises(FileNotFoundError):
        score_laion(str(tmp_path / "nonexistent.jpg"))


def test_score_laion_wrong_type():
    from aesthetic_scoring.laion import score_laion
    with pytest.raises(TypeError):
        score_laion(123)


def test_score_laion_returns_correct_schema(tmp_path, monkeypatch):
    from aesthetic_scoring import laion as laion_mod

    # Create a real temp image
    from PIL import Image
    img_path = tmp_path / "test.jpg"
    Image.new("RGB", (64, 64), color=(100, 150, 200)).save(str(img_path))

    monkeypatch.setattr(laion_mod, "score_laion", lambda p: _make_fake_result())
    result = laion_mod.score_laion(str(img_path))

    assert isinstance(result, LaionScoreResult)
    assert isinstance(result.aesthetic_score, float)
    assert result.score_scale == "1-10"
    assert result.latency_ms > 0


def test_unload_is_noop_when_not_loaded():
    from aesthetic_scoring.laion import unload
    unload()  # Should not raise
    unload()  # Idempotent
