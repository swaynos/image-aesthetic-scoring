"""Unit tests for pickscore module — no real model loaded."""
import pytest
from aesthetic_scoring.types import PickScoreResult


def _make_fake_result():
    return PickScoreResult(
        image_id="photo_a.jpg",
        model_name="pickscore",
        model_version="PickScore_v1",
        latency_ms=80.0,
        device="cuda:0",
        precision="fp16",
        prompt="a test prompt",
        scores=[1.2, 0.8],
        probabilities=[0.6, 0.4],
        ranked_image_ids=["photo_a.jpg", "photo_b.jpg"],
    )


def test_score_pickscore_missing_file(tmp_path):
    from aesthetic_scoring.pickscore import score_pickscore
    with pytest.raises(FileNotFoundError):
        score_pickscore([str(tmp_path / "nope.jpg")], "test")


def test_score_pickscore_empty_list():
    from aesthetic_scoring.pickscore import score_pickscore
    with pytest.raises(ValueError):
        score_pickscore([], "test")


def test_score_pickscore_wrong_type_paths():
    from aesthetic_scoring.pickscore import score_pickscore
    with pytest.raises(TypeError):
        score_pickscore("not_a_list", "test")


def test_score_pickscore_wrong_type_prompt(tmp_path):
    from aesthetic_scoring.pickscore import score_pickscore
    from PIL import Image
    img = tmp_path / "a.jpg"
    Image.new("RGB", (64, 64)).save(str(img))
    with pytest.raises(TypeError):
        score_pickscore([str(img)], 123)


def test_score_pickscore_schema(monkeypatch):
    from aesthetic_scoring import pickscore as ps_mod
    monkeypatch.setattr(ps_mod, "score_pickscore", lambda paths, prompt: _make_fake_result())
    result = ps_mod.score_pickscore(["a.jpg"], "test")
    assert isinstance(result, PickScoreResult)
    assert len(result.scores) == len(result.probabilities)
    assert len(result.ranked_image_ids) == 2


def test_unload_noop():
    from aesthetic_scoring.pickscore import unload
    unload()
    unload()
