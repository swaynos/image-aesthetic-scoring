"""
features.py — Deterministic feature extraction for reference-vs-derivative image comparison.

All computation is CPU / fp32 / numpy.  No model weights required.
NaN/inf values are preserved (not replaced); callers should handle them.

Proxies used (no heavy neural net):
  ssim         — skimage structural_similarity (multichannel)
  lpips_proxy  — weighted combo: (1 - ssim) * 0.5 + mean_abs_grad_diff * 0.5
  banding      — std of row-mean differences (detects horizontal banding artefacts)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from .intent import (
    FEATURE_VERSION,
    extract_boundary_ring,
    resolve_masks,
    tile_grid,
)
from .types import RegionalFeatureVector

# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_rgb_f32(path: str) -> np.ndarray:
    """Load image as float32 in [0, 1] shape (H, W, 3)."""
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def _resize_to_match(arr: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Resize arr to match target's (H, W) using PIL LANCZOS."""
    th, tw = target.shape[:2]
    ah, aw = arr.shape[:2]
    if (ah, aw) == (th, tw):
        return arr
    pil = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    pil = pil.resize((tw, th), Image.LANCZOS)
    return np.array(pil, dtype=np.float32) / 255.0


def _to_gray(arr: np.ndarray) -> np.ndarray:
    """(H,W,3) float32 → (H,W) float32 luminance."""
    return 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level metric primitives
# ─────────────────────────────────────────────────────────────────────────────

def _ssim(ref: np.ndarray, deriv: np.ndarray) -> float:
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ref, deriv, data_range=1.0, channel_axis=2))


def _sobel_edges(gray: np.ndarray) -> np.ndarray:
    from scipy.ndimage import sobel
    sx = sobel(gray, axis=0)
    sy = sobel(gray, axis=1)
    return np.sqrt(sx ** 2 + sy ** 2)


def _edge_retention(ref_gray: np.ndarray, deriv_gray: np.ndarray) -> float:
    ref_e  = _sobel_edges(ref_gray)
    deriv_e = _sobel_edges(deriv_gray)
    denom = float(np.sum(ref_e ** 2)) + 1e-8
    return float(np.sum(deriv_e ** 2)) / denom


def _hf_retention(ref: np.ndarray, deriv: np.ndarray) -> float:
    from scipy.ndimage import gaussian_filter
    ref_gray   = _to_gray(ref)
    deriv_gray = _to_gray(deriv)
    ref_hf   = ref_gray   - gaussian_filter(ref_gray,   sigma=2.0)
    deriv_hf = deriv_gray - gaussian_filter(deriv_gray, sigma=2.0)
    denom = float(np.mean(np.abs(ref_hf))) + 1e-8
    return float(np.mean(np.abs(deriv_hf))) / denom


def _grad_ratio(ref: np.ndarray, deriv: np.ndarray) -> float:
    ref_grad   = float(np.mean(_sobel_edges(_to_gray(ref))))   + 1e-8
    deriv_grad = float(np.mean(_sobel_edges(_to_gray(deriv))))
    return deriv_grad / ref_grad


def _banding(deriv: np.ndarray) -> float:
    """Detect horizontal banding: std of row-mean luminance differences."""
    row_means = _to_gray(deriv).mean(axis=1)
    return float(np.std(np.diff(row_means)))


def _rgb_l1_delta(ref: np.ndarray, deriv: np.ndarray) -> float:
    return float(np.mean(np.abs(deriv - ref)))


def _lpips_proxy(ssim_val: float, ref: np.ndarray, deriv: np.ndarray) -> float:
    ref_grad   = _sobel_edges(_to_gray(ref))
    deriv_grad = _sobel_edges(_to_gray(deriv))
    grad_diff  = float(np.mean(np.abs(deriv_grad - ref_grad)))
    return 0.5 * (1.0 - ssim_val) + 0.5 * grad_diff


# ─────────────────────────────────────────────────────────────────────────────
# Masked variants
# ─────────────────────────────────────────────────────────────────────────────

def _masked_ssim(ref: np.ndarray, deriv: np.ndarray, mask: np.ndarray) -> float:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return float("nan")
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    ref_crop   = ref[r0:r1,   c0:c1]
    deriv_crop = deriv[r0:r1, c0:c1]
    if ref_crop.shape[0] < 7 or ref_crop.shape[1] < 7:
        return float(1.0 - np.mean(np.abs(ref_crop - deriv_crop)))
    from skimage.metrics import structural_similarity
    return float(structural_similarity(ref_crop, deriv_crop, data_range=1.0, channel_axis=2))


def _masked_metric(ref: np.ndarray, deriv: np.ndarray, mask: np.ndarray, fn):
    if mask is None or not mask.any():
        return float("nan")
    return fn(ref[mask], deriv[mask])


# ─────────────────────────────────────────────────────────────────────────────
# Tile metrics
# ─────────────────────────────────────────────────────────────────────────────

_TILE_HIGH_RISK_THRESHOLD = 0.15


def _tile_metrics(ref: np.ndarray, deriv: np.ndarray, tile_size: int = 64):
    h, w = ref.shape[:2]
    tile_scores = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            r = ref[y : y + tile_size,   x : x + tile_size]
            d = deriv[y : y + tile_size, x : x + tile_size]
            tile_scores.append(float(np.mean(np.abs(r - d))))
    if not tile_scores:
        return 0.0, 0.0, 0.0, 0
    arr      = np.array(tile_scores)
    high_risk = int(np.sum(arr > _TILE_HIGH_RISK_THRESHOLD))
    return float(arr.mean()), float(arr.max()), float(arr.var()), high_risk


# ─────────────────────────────────────────────────────────────────────────────
# Boundary metrics
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_metrics(ref: np.ndarray, deriv: np.ndarray, mask: Optional[np.ndarray]) -> tuple:
    if mask is None or not mask.any():
        return 0.0, 1.0
    ring = extract_boundary_ring(mask, ring_width=4)
    if not ring.any():
        return 0.0, 1.0
    seam         = float(np.mean(np.abs(deriv[ring] - ref[ring])))
    ref_ring_e   = float(np.mean(_sobel_edges(_to_gray(ref))[ring]))
    deriv_ring_e = float(np.mean(_sobel_edges(_to_gray(deriv))[ring]))
    continuity   = deriv_ring_e / (ref_ring_e + 1e-8)
    return seam, continuity


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_reference_features(
    reference_path: str,
    derivative_path: str,
    source_id: str = "",
    step_index: int = 0,
    intent_state: str = "",
    prior_reference_path: Optional[str] = None,
    intent_mask_path: Optional[str] = None,
    subject_mask_path: Optional[str] = None,
    mask_policy: str = "none",
) -> RegionalFeatureVector:
    """Extract all deterministic features for one reference→derivative image pair.

    The function is fully general: ``reference_path`` is any baseline image,
    ``derivative_path`` is any image derived from it (edit, generation, encode/
    decode, upscale, style transfer, render frame, etc.).

    Args:
        reference_path: Path to the reference (source/baseline) image.
        derivative_path: Path to the derivative image to compare against it.
        source_id: Caller-defined identifier for the sequence this pair belongs to.
        step_index: Caller-defined position in the sequence (e.g. edit pass,
            frame number, variant index).
        intent_state: Caller-defined group/category label (e.g. prompt variant,
            style bucket, A/B group).
        prior_reference_path: Optional path to an earlier reference in the same
            intent group, used to compute temporal/cross-step consistency metrics.
        intent_mask_path: Optional PNG mask for custom_intent policy.
        subject_mask_path: Optional PNG mask for subject/background policy.
        mask_policy: One of whole_image|subject|background|custom_intent|none.

    Returns:
        RegionalFeatureVector with all 24+ metrics.  NaN is preserved for
        metrics that require unavailable inputs (e.g. mask metrics without a
        mask, temporal metrics without a prior reference).
    """
    ref   = _load_rgb_f32(reference_path)
    deriv = _load_rgb_f32(derivative_path)
    deriv = _resize_to_match(deriv, ref)

    h, w = ref.shape[:2]
    inside_mask, outside_mask = resolve_masks(
        h, w, mask_policy, intent_mask_path, subject_mask_path
    )

    # ── global ────────────────────────────────────────────────────────────────
    ssim_val  = _ssim(ref, deriv)
    lp        = _lpips_proxy(ssim_val, ref, deriv)
    rgb_l1    = _rgb_l1_delta(ref, deriv)
    ref_gray   = _to_gray(ref)
    deriv_gray = _to_gray(deriv)
    er        = _edge_retention(ref_gray, deriv_gray)
    hfr       = _hf_retention(ref, deriv)
    gr        = _grad_ratio(ref, deriv)
    band      = _banding(deriv)

    # ── inside / outside ──────────────────────────────────────────────────────
    if inside_mask is not None:
        in_ssim = _masked_ssim(ref, deriv, inside_mask)
        in_lp   = _lpips_proxy(in_ssim if math.isfinite(in_ssim) else ssim_val, ref, deriv)
        in_rgb  = _masked_metric(ref, deriv, inside_mask,
                                 lambda a, b: float(np.mean(np.abs(a - b))))
        in_er   = _edge_retention(ref_gray * inside_mask, deriv_gray * inside_mask)
    else:
        in_ssim = in_lp = in_rgb = in_er = float("nan")

    if outside_mask is not None and outside_mask.any():
        out_ssim = _masked_ssim(ref, deriv, outside_mask)
        out_lp   = _lpips_proxy(out_ssim if math.isfinite(out_ssim) else ssim_val, ref, deriv)
        out_rgb  = _masked_metric(ref, deriv, outside_mask,
                                  lambda a, b: float(np.mean(np.abs(a - b))))
        out_er   = _edge_retention(ref_gray * outside_mask, deriv_gray * outside_mask)
    else:
        out_ssim = out_lp = out_rgb = out_er = float("nan")

    # ── boundary ──────────────────────────────────────────────────────────────
    b_seam, b_cont = _boundary_metrics(ref, deriv, inside_mask)

    # ── tile ──────────────────────────────────────────────────────────────────
    t_mean, t_worst, t_var, t_hr = _tile_metrics(ref, deriv)

    # ── temporal ──────────────────────────────────────────────────────────────
    if prior_reference_path is not None:
        prior  = _load_rgb_f32(prior_reference_path)
        prior  = _resize_to_match(prior, ref)
        t_ssim = _ssim(prior, deriv)
        t_lp   = _lpips_proxy(t_ssim, prior, deriv)
        t_rgb  = _rgb_l1_delta(prior, deriv)
    else:
        t_ssim = t_lp = t_rgb = float("nan")

    # ── finiteness ────────────────────────────────────────────────────────────
    all_vals = [
        ssim_val, lp, rgb_l1, er, hfr, gr, band,
        in_ssim, in_lp, in_rgb, in_er,
        out_ssim, out_lp, out_rgb, out_er,
        b_seam, b_cont, t_mean, t_worst, t_var,
        t_ssim, t_lp, t_rgb,
    ]
    finite = all(math.isfinite(v) for v in all_vals)

    return RegionalFeatureVector(
        source_id=source_id,
        step_index=step_index,
        intent_state=intent_state,
        ssim=ssim_val,
        lpips_proxy=lp,
        rgb_l1_delta=rgb_l1,
        edge_retention=er,
        hf_retention=hfr,
        grad_ratio=gr,
        banding=band,
        inside_ssim=in_ssim,
        inside_lpips_proxy=in_lp,
        inside_rgb_l1_delta=in_rgb,
        inside_edge_retention=in_er,
        outside_ssim=out_ssim,
        outside_lpips_proxy=out_lp,
        outside_rgb_l1_delta=out_rgb,
        outside_edge_retention=out_er,
        boundary_seam_intensity=b_seam,
        boundary_edge_continuity=b_cont,
        tile_mean=t_mean,
        tile_worst=t_worst,
        tile_variance=t_var,
        tile_high_risk_count=t_hr,
        temporal_ssim=t_ssim,
        temporal_lpips_proxy=t_lp,
        temporal_rgb_l1_delta=t_rgb,
        feature_version=FEATURE_VERSION,
        mask_policy=mask_policy,
        finite=finite,
    )
