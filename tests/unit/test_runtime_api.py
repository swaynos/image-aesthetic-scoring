"""Unit tests for score_reference_comparison runtime API."""
import dataclasses
import json
import math
import pytest
from aesthetic_scoring import score_reference_comparison
from aesthetic_scoring.types import ReferenceComparisonResult
from tests.conftest import PHOTO_A, PHOTO_B

REF   = PHOTO_A
DERIV = PHOTO_B


def test_returns_result_type():
    r = score_reference_comparison(REF, DERIV)
    assert isinstance(r, ReferenceComparisonResult)


def test_required_fields_present():
    r = score_reference_comparison(REF, DERIV)
    assert r.quality_score is not None
    assert r.divergence_score is not None
    assert r.artifact_score is not None
    assert r.temporal_consistency_score is not None
    assert isinstance(r.regional_breakdown, dict)
    assert r.feature_version != ""
    assert r.model_version != ""


def test_scores_bounded():
    r = score_reference_comparison(REF, DERIV)
    for field in ["quality_score", "divergence_score",
                  "artifact_score", "temporal_consistency_score"]:
        val = getattr(r, field)
        assert 0.0 <= val <= 1.0, f"{field}={val} out of [0,1]"


def test_json_serializable():
    r = score_reference_comparison(REF, DERIV)
    d = dataclasses.asdict(r)
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    cleaned = {
        k: ({kk: clean(vv) for kk, vv in v.items()} if isinstance(v, dict) else clean(v))
        for k, v in d.items()
    }
    json.dumps(cleaned)


def test_reference_not_found():
    with pytest.raises(FileNotFoundError):
        score_reference_comparison("/nonexistent/ref.png", DERIV)


def test_derivative_not_found():
    with pytest.raises(FileNotFoundError):
        score_reference_comparison(REF, "/nonexistent/deriv.png")


def test_type_error_reference():
    with pytest.raises(TypeError):
        score_reference_comparison(123, DERIV)


def test_type_error_derivative():
    with pytest.raises(TypeError):
        score_reference_comparison(REF, 456)


def test_mask_policy_none():
    r = score_reference_comparison(REF, DERIV, mask_policy="none")
    assert r.mask_policy == "none"


def test_whole_image_mask_policy():
    r = score_reference_comparison(REF, DERIV, mask_policy="whole_image")
    assert r.mask_policy == "whole_image"
    assert 0.0 <= r.quality_score <= 1.0


def test_latency_positive():
    r = score_reference_comparison(REF, DERIV)
    assert r.latency_ms > 0.0


def test_derivative_id_and_reference_id():
    r = score_reference_comparison(REF, DERIV)
    assert r.derivative_id == "photo_b.jpg"
    assert r.reference_id  == "photo_a.jpg"


def test_identity_has_low_divergence():
    """Comparing an image against itself should show near-zero divergence."""
    r = score_reference_comparison(REF, REF)
    assert r.divergence_score < 0.2
