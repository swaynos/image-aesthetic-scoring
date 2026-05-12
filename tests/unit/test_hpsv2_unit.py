"""Unit tests for hpsv2 module — no real model loaded."""
import pytest
from aesthetic_scoring.types import HPSv2ScoreResult


def _make_fake_result():
    return HPSv2ScoreResult(
        image_id="photo_a.jpg",
        model_name="hpsv2",
        model_version="HPS_v2.1",
        latency_ms=60.0,
        device="cuda:0",
        precision="fp16",
        prompt="a test prompt",
        preference_score=0.275,
    )


def test_score_hpsv2_missing_file(tmp_path):
    from aesthetic_scoring.hpsv2 import score_hpsv2
    with pytest.raises(FileNotFoundError):
        score_hpsv2(str(tmp_path / "nope.jpg"), "test")


def test_score_hpsv2_wrong_type_path():
    from aesthetic_scoring.hpsv2 import score_hpsv2
    with pytest.raises(TypeError):
        score_hpsv2(None, "test")


def test_score_hpsv2_wrong_type_prompt(tmp_path):
    from aesthetic_scoring.hpsv2 import score_hpsv2
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64)).save(str(img))
    with pytest.raises(TypeError):
        score_hpsv2(str(img), 42)


def test_score_hpsv2_schema(monkeypatch):
    from aesthetic_scoring import hpsv2 as mod
    monkeypatch.setattr(mod, "score_hpsv2", lambda p, q: _make_fake_result())
    result = mod.score_hpsv2("a.jpg", "test")
    assert isinstance(result, HPSv2ScoreResult)
    assert isinstance(result.preference_score, float)


def test_unload_noop():
    from aesthetic_scoring.hpsv2 import unload
    unload()
    unload()
