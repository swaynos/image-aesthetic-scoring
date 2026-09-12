"""Unit tests for the public multi-model scoring suite."""
import dataclasses
import json

import pytest

from aesthetic_scoring import score_images
from aesthetic_scoring.types import BaseScoreResult, ImageScoreReport


def _result(image_id="image.png"):
    return BaseScoreResult(
        image_id=image_id,
        model_name="test",
        model_version="1",
        latency_ms=1.0,
        device="cpu",
        precision="fp32",
    )


def test_prompt_models_require_an_evaluation_prompt(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not read by validation")

    with pytest.raises(ValueError, match="evaluation_prompt"):
        score_images([str(image)], models=["pickscore"])


def test_unknown_model_is_rejected(tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not read by validation")

    with pytest.raises(ValueError, match="Unknown model"):
        score_images([str(image)], models=["unknown"])


def test_runs_selected_models_sequentially_and_serializes(monkeypatch, tmp_path):
    image = tmp_path / "image.png"
    image.write_bytes(b"not read by test doubles")
    calls = []

    def score_laion(path):
        calls.append(("laion", path))
        return _result()

    def score_pickscore(paths, prompt):
        calls.append(("pickscore", paths, prompt))
        return _result()

    monkeypatch.setattr("aesthetic_scoring.suite.score_laion", score_laion)
    monkeypatch.setattr("aesthetic_scoring.suite.score_pickscore", score_pickscore)
    monkeypatch.setattr("aesthetic_scoring.suite.unload_laion", lambda: calls.append(("unload_laion",)))
    monkeypatch.setattr("aesthetic_scoring.suite.unload_pickscore", lambda: calls.append(("unload_pickscore",)))

    report = score_images(
        [str(image)],
        evaluation_prompt="a clear portrait",
        models=["laion", "pickscore"],
    )

    assert isinstance(report, ImageScoreReport)
    assert report.evaluation_prompt == "a clear portrait"
    assert list(report.results) == ["laion", "pickscore"]
    assert calls == [
        ("laion", str(image)),
        ("unload_laion",),
        ("pickscore", [str(image)], "a clear portrait"),
        ("unload_pickscore",),
    ]
    json.dumps(dataclasses.asdict(report))
