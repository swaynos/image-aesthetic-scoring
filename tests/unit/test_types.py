"""Unit tests: result type schema and JSON-serializability."""
import dataclasses
import json
import pytest

from aesthetic_scoring.types import (
    BaseScoreResult,
    LaionScoreResult,
    PickScoreResult,
    HPSv2ScoreResult,
    FGAesQScoreResult,
)


def make_base(**kwargs):
    defaults = dict(
        image_id="img.jpg",
        model_name="test-model",
        model_version="v1",
        latency_ms=12.5,
        device="cuda:0",
        precision="fp16",
    )
    defaults.update(kwargs)
    return defaults


def test_laion_result_schema():
    r = LaionScoreResult(**make_base(), aesthetic_score=7.2, score_scale="1-10")
    assert r.aesthetic_score == 7.2
    assert r.score_scale == "1-10"
    d = dataclasses.asdict(r)
    json.dumps(d)  # must not raise


def test_pickscore_result_schema():
    r = PickScoreResult(
        **make_base(),
        prompt="a cat",
        scores=[0.8, 0.2],
        probabilities=[0.7, 0.3],
        ranked_image_ids=["a.jpg", "b.jpg"],
    )
    assert len(r.scores) == 2
    assert len(r.probabilities) == 2
    assert len(r.ranked_image_ids) == 2
    d = dataclasses.asdict(r)
    json.dumps(d)


def test_hpsv2_result_schema():
    r = HPSv2ScoreResult(**make_base(), prompt="a cat", preference_score=0.27)
    assert r.preference_score == 0.27
    d = dataclasses.asdict(r)
    json.dumps(d)


def test_fgaesq_result_schema():
    r = FGAesQScoreResult(
        **make_base(),
        technical_score=3.1,
        aesthetic_score=6.8,
        subscores={"bin_1": 0.01, "raw_score": 6.8},
    )
    assert r.technical_score == 3.1
    assert r.aesthetic_score == 6.8
    assert isinstance(r.subscores, dict)
    d = dataclasses.asdict(r)
    json.dumps(d)


def test_base_fields_present():
    fields = {f.name for f in dataclasses.fields(BaseScoreResult)}
    assert fields == {"image_id", "model_name", "model_version", "latency_ms", "device", "precision"}
