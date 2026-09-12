"""Unit tests for clipscore module — no real model loaded."""
import pytest
from aesthetic_scoring.types import CLIPScoreResult


def _make_fake_result():
    return CLIPScoreResult(
        image_id="photo_a.jpg",
        model_name="clipscore",
        model_version="ViT-B-32-openai",
        latency_ms=40.0,
        device="cuda:0",
        precision="fp16",
        prompt="a test prompt",
        scores=[0.31, 0.22],
        ranked_image_ids=["photo_a.jpg", "photo_b.jpg"],
    )


def test_score_clipscore_missing_file(tmp_path):
    from aesthetic_scoring.clipscore import score_clipscore
    with pytest.raises(FileNotFoundError):
        score_clipscore([str(tmp_path / "nope.jpg")], "test")


def test_score_clipscore_empty_list():
    from aesthetic_scoring.clipscore import score_clipscore
    with pytest.raises(ValueError):
        score_clipscore([], "test")


def test_score_clipscore_non_list_paths():
    from aesthetic_scoring.clipscore import score_clipscore
    with pytest.raises(ValueError):
        score_clipscore("not_a_list", "test")


def test_score_clipscore_wrong_type_element():
    from aesthetic_scoring.clipscore import score_clipscore
    with pytest.raises(TypeError):
        score_clipscore([123], "test")


def test_score_clipscore_empty_prompt():
    from aesthetic_scoring.clipscore import score_clipscore
    with pytest.raises(ValueError):
        score_clipscore(["a.jpg"], "   ")


def test_score_clipscore_wrong_type_prompt():
    from aesthetic_scoring.clipscore import score_clipscore
    with pytest.raises(ValueError):
        score_clipscore(["a.jpg"], 123)


def test_score_clipscore_schema(monkeypatch):
    from aesthetic_scoring import clipscore as mod
    monkeypatch.setattr(mod, "score_clipscore", lambda paths, prompt: _make_fake_result())
    result = mod.score_clipscore(["a.jpg", "b.jpg"], "test")
    assert isinstance(result, CLIPScoreResult)
    assert len(result.scores) == len(result.ranked_image_ids)


def test_unload_noop():
    from aesthetic_scoring.clipscore import unload
    unload()
    unload()
