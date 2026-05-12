"""Unit tests for fgaesq module — no real model loaded."""
import pytest
import dataclasses
import json
from aesthetic_scoring.types import FGAesQScoreResult


def _make_fake_result():
    return FGAesQScoreResult(
        image_id="photo_a.jpg",
        model_name="fgaesq",
        model_version="FGAesQ-v1.0",
        latency_ms=90.0,
        device="cuda:0",
        precision="fp32",
        technical_score=3.2,
        aesthetic_score=6.9,
        subscores={"bin_1": 0.01, "bin_2": 0.05, "raw_score": 6.9},
    )


def test_score_fgaesq_missing_file(tmp_path):
    from aesthetic_scoring.fgaesq import score_fgaesq
    with pytest.raises(FileNotFoundError):
        score_fgaesq(str(tmp_path / "nope.jpg"))


def test_score_fgaesq_wrong_type():
    from aesthetic_scoring.fgaesq import score_fgaesq
    with pytest.raises(TypeError):
        score_fgaesq(42)


def test_score_fgaesq_schema(monkeypatch):
    from aesthetic_scoring import fgaesq as mod
    monkeypatch.setattr(mod, "score_fgaesq", lambda p: _make_fake_result())
    result = mod.score_fgaesq("a.jpg")
    assert isinstance(result, FGAesQScoreResult)
    assert isinstance(result.technical_score, float)
    assert isinstance(result.aesthetic_score, float)
    assert isinstance(result.subscores, dict)


def test_fgaesq_result_json_serializable():
    r = _make_fake_result()
    d = dataclasses.asdict(r)
    json.dumps(d)  # must not raise


def test_unload_noop():
    from aesthetic_scoring.fgaesq import unload
    unload()
    unload()
