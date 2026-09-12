"""Unit tests for imagereward module — no real model loaded."""
import pytest
from aesthetic_scoring.types import ImageRewardScoreResult


def _make_fake_result():
    return ImageRewardScoreResult(
        image_id="photo_a.jpg",
        model_name="imagereward",
        model_version="ImageReward-v1.0",
        latency_ms=70.0,
        device="cuda:0",
        precision="fp16",
        prompt="a test prompt",
        scores=[0.4, 0.1],
        ranked_image_ids=["photo_a.jpg", "photo_b.jpg"],
    )


def test_score_imagereward_missing_file(tmp_path):
    from aesthetic_scoring.imagereward import score_imagereward
    with pytest.raises(FileNotFoundError):
        score_imagereward([str(tmp_path / "nope.jpg")], "test")


def test_score_imagereward_empty_list():
    from aesthetic_scoring.imagereward import score_imagereward
    with pytest.raises(ValueError):
        score_imagereward([], "test")


def test_score_imagereward_non_list_paths():
    from aesthetic_scoring.imagereward import score_imagereward
    with pytest.raises(ValueError):
        score_imagereward("not_a_list", "test")


def test_score_imagereward_wrong_type_element():
    from aesthetic_scoring.imagereward import score_imagereward
    with pytest.raises(TypeError):
        score_imagereward([123], "test")


def test_score_imagereward_empty_prompt():
    from aesthetic_scoring.imagereward import score_imagereward
    with pytest.raises(ValueError):
        score_imagereward(["a.jpg"], "   ")


def test_score_imagereward_wrong_type_prompt():
    from aesthetic_scoring.imagereward import score_imagereward
    with pytest.raises(ValueError):
        score_imagereward(["a.jpg"], 123)


def test_score_imagereward_schema(monkeypatch):
    from aesthetic_scoring import imagereward as mod
    monkeypatch.setattr(mod, "score_imagereward", lambda paths, prompt: _make_fake_result())
    result = mod.score_imagereward(["a.jpg", "b.jpg"], "test")
    assert isinstance(result, ImageRewardScoreResult)
    assert len(result.scores) == len(result.ranked_image_ids)


def test_unload_noop():
    from aesthetic_scoring.imagereward import unload
    unload()
    unload()
