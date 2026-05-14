"""
intent_api.py — score_reference_comparison runtime API.

Measures the structural and perceptual divergence between a reference image and
a derivative image.  The API is general-purpose: the reference/derivative pair
can be anything — an iterative edit and its source, a generative model output
and its conditioning image, an encode/decode round-trip, an upscale vs.
original, a style transfer vs. content source, a compressed variant vs.
original, a render frame vs. a golden frame, and so on.

Pass ``model_path`` to use a trained ridge model from aesthetic-model-training;
omit it to use the feature-derived passthrough scorer.
"""

from __future__ import annotations

import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Optional

from .types import ReferenceComparisonResult

_MODEL_CACHE: Optional[dict] = None

MODEL_VERSION = "baseline-1.0"
FEATURE_VERSION = "1.0.0"

_TARGET_KEYS = [
    "quality_target",
    "divergence_target",
    "artifact_target",
    "outside_intent_drift_target",
    "inside_intent_quality_target",
    "temporal_consistency_target",
]


def _load_model(model_path: Optional[str]) -> dict:
    global _MODEL_CACHE
    if model_path is None:
        return {"type": "passthrough"}
    if _MODEL_CACHE is not None and _MODEL_CACHE.get("_path") == model_path:
        return _MODEL_CACHE
    with open(model_path, "r", encoding="utf-8") as f:
        _MODEL_CACHE = json.load(f)
        _MODEL_CACHE["_path"] = model_path
    return _MODEL_CACHE


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    if not math.isfinite(v):
        return 0.5
    return max(lo, min(hi, v))


def _passthrough(fv_dict: dict) -> dict:
    """Derive comparison scores directly from feature values without a trained model."""
    def g(key):
        v = fv_dict.get(key)
        return 0.0 if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)

    ssim     = g("ssim")
    er       = g("edge_retention")
    hfr      = g("hf_retention")
    lpips    = g("lpips_proxy")
    banding  = g("banding")
    t_worst  = g("tile_worst")
    t_hr     = g("tile_high_risk_count")
    t_ssim   = g("temporal_ssim")
    out_ssim = g("outside_ssim")
    out_rgb  = g("outside_rgb_l1_delta")
    in_ssim  = g("inside_ssim")
    in_er    = g("inside_edge_retention")

    quality    = _clamp(0.5 * _clamp(ssim) + 0.25 * _clamp(er / 1.5) + 0.25 * _clamp(hfr / 1.5))
    divergence = _clamp(
        (1.0 - quality) * 0.6
        + _clamp(t_worst / 0.5) * 0.25
        + _clamp(banding * 10.0) * 0.15
    )
    artifact = _clamp(0.6 * _clamp(lpips) + 0.4 * _clamp(t_hr / max(1.0, 100.0)))

    if math.isfinite(fv_dict.get("outside_ssim", float("nan"))):
        outside_drift = _clamp(0.5 * (1.0 - _clamp(out_ssim)) + 0.5 * _clamp(out_rgb / 0.2))
    else:
        outside_drift = 0.0

    if math.isfinite(fv_dict.get("inside_ssim", float("nan"))):
        inside_quality = _clamp(0.6 * _clamp(in_ssim) + 0.4 * _clamp(in_er / 1.5))
    else:
        inside_quality = quality

    temporal_consistency = _clamp(t_ssim) if math.isfinite(
        fv_dict.get("temporal_ssim", float("nan"))
    ) else 0.5

    return {
        "quality_target":               quality,
        "divergence_target":            divergence,
        "artifact_target":              artifact,
        "outside_intent_drift_target":  outside_drift,
        "inside_intent_quality_target": inside_quality,
        "temporal_consistency_target":  temporal_consistency,
    }


def _predict(model: dict, fv_dict: dict) -> dict:
    if model.get("type") == "passthrough":
        return _passthrough(fv_dict)
    weights       = model["weights"]
    intercepts    = model["intercepts"]
    feature_names = model["feature_names"]
    x = [fv_dict.get(fn, 0.0) for fn in feature_names]
    x = [0.0 if (not isinstance(v, (int, float)) or not math.isfinite(v)) else v for v in x]
    preds = {}
    for target in _TARGET_KEYS:
        w   = weights.get(target, {})
        b   = intercepts.get(target, 0.5)
        val = b + sum(w.get(fn, 0.0) * xi for fn, xi in zip(feature_names, x))
        preds[target] = max(0.0, min(1.0, val))
    return preds


def score_reference_comparison(
    reference_path: str,
    derivative_path: str,
    prior_reference_path: Optional[str] = None,
    intent_mask_path: Optional[str] = None,
    subject_mask_path: Optional[str] = None,
    mask_policy: str = "none",
    model_path: Optional[str] = None,
) -> ReferenceComparisonResult:
    """Compare a derivative image against a reference and return divergence scores.

    The reference/derivative pair can be anything: iterative edit vs. source,
    generative output vs. conditioning image, encode/decode round-trip,
    upscale vs. original, style transfer vs. content source, compressed variant
    vs. original, render frame vs. golden frame, etc.

    Args:
        reference_path: Path to the reference (baseline/source) image.
        derivative_path: Path to the derivative image to score.
        prior_reference_path: Optional path to an earlier reference in the same
            intent group, used to compute temporal consistency metrics (e.g. an
            earlier step in a sequence, a prior same-prompt generation).
        intent_mask_path: Optional PNG mask (0/255) defining a region of
            interest for mask-aware metrics (``custom_intent`` policy).
        subject_mask_path: Optional PNG subject mask for ``subject`` or
            ``background`` policy.
        mask_policy: One of whole_image|subject|background|custom_intent|none.
        model_path: Optional path to a ``baseline_model.json`` from
            aesthetic-model-training.  If omitted, scores are derived from
            features directly (passthrough mode).

    Returns:
        ReferenceComparisonResult with all scoring fields.
    """
    if not isinstance(reference_path, str):
        raise TypeError(f"reference_path must be str, got {type(reference_path).__name__}")
    if not isinstance(derivative_path, str):
        raise TypeError(f"derivative_path must be str, got {type(derivative_path).__name__}")
    if not Path(reference_path).exists():
        raise FileNotFoundError(f"reference_path not found: {reference_path}")
    if not Path(derivative_path).exists():
        raise FileNotFoundError(f"derivative_path not found: {derivative_path}")

    t0 = time.perf_counter()

    from .features import extract_reference_features

    fv = extract_reference_features(
        reference_path=reference_path,
        derivative_path=derivative_path,
        prior_reference_path=prior_reference_path,
        intent_mask_path=intent_mask_path,
        subject_mask_path=subject_mask_path,
        mask_policy=mask_policy,
    )

    preds = _predict(_load_model(model_path), dataclasses.asdict(fv))

    regional = {
        "inside_ssim":              fv.inside_ssim,
        "inside_lpips_proxy":       fv.inside_lpips_proxy,
        "outside_ssim":             fv.outside_ssim,
        "outside_rgb_l1_delta":     fv.outside_rgb_l1_delta,
        "boundary_seam_intensity":  fv.boundary_seam_intensity,
        "boundary_edge_continuity": fv.boundary_edge_continuity,
        "tile_worst":               fv.tile_worst,
        "tile_high_risk_count":     float(fv.tile_high_risk_count),
        "banding":                  fv.banding,
    }

    return ReferenceComparisonResult(
        derivative_id=Path(derivative_path).name,
        reference_id=Path(reference_path).name,
        mask_policy=mask_policy,
        quality_score=preds.get("quality_target", 0.5),
        divergence_score=preds.get("divergence_target", 0.5),
        artifact_score=preds.get("artifact_target", 0.5),
        temporal_consistency_score=preds.get("temporal_consistency_target", 0.5),
        regional_breakdown=regional,
        feature_version=FEATURE_VERSION,
        model_version=MODEL_VERSION,
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        device="cpu",
    )
