"""Typed result dataclasses returned by each scoring function."""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class BaseScoreResult:
    image_id: str
    model_name: str
    model_version: str
    latency_ms: float
    device: str
    precision: str


@dataclass
class LaionScoreResult(BaseScoreResult):
    aesthetic_score: float
    score_scale: str


@dataclass
class PickScoreResult(BaseScoreResult):
    prompt: str
    scores: List[float]           # raw logits per image, input order
    probabilities: List[float]    # softmax over scores, input order
    ranked_image_ids: List[str]   # image_ids sorted by probability desc


@dataclass
class HPSv2ScoreResult(BaseScoreResult):
    prompt: str
    preference_score: float


@dataclass
class FGAesQScoreResult(BaseScoreResult):
    technical_score: float
    aesthetic_score: float
    subscores: Dict[str, float]


@dataclass
class RegionalFeatureVector:
    """Features extracted from a reference→derivative image pair.

    Produced by ``extract_reference_features``; consumed by
    ``score_reference_comparison`` and the aesthetic-model-training pipeline.

    ``step_index`` and ``intent_state`` are caller-defined identifiers with no
    fixed semantics.  Common uses: step_index as a sequence position (edit
    pass, render frame, variant number), intent_state as a group/category
    label (prompt variant, style bucket, A/B group).
    """
    # caller-defined identifiers
    source_id: str
    step_index: int
    intent_state: str

    # global metrics
    ssim: float
    lpips_proxy: float
    rgb_l1_delta: float
    edge_retention: float
    hf_retention: float
    grad_ratio: float
    banding: float

    # inside-intent metrics (NaN when no mask)
    inside_ssim: float
    inside_lpips_proxy: float
    inside_rgb_l1_delta: float
    inside_edge_retention: float

    # outside-intent metrics (NaN when no mask)
    outside_ssim: float
    outside_lpips_proxy: float
    outside_rgb_l1_delta: float
    outside_edge_retention: float

    # boundary metrics
    boundary_seam_intensity: float
    boundary_edge_continuity: float

    # tile metrics
    tile_mean: float
    tile_worst: float
    tile_variance: float
    tile_high_risk_count: int

    # temporal metrics (NaN when no prior reference supplied)
    temporal_ssim: float
    temporal_lpips_proxy: float
    temporal_rgb_l1_delta: float

    # feature metadata
    feature_version: str
    mask_policy: str
    finite: bool


@dataclass
class ReferenceComparisonResult:
    """Result of ``score_reference_comparison``.

    Measures the structural and perceptual divergence between a reference image
    and a derivative image (e.g. an edit, a generation, an encode/decode round-
    trip, an upscale, a style transfer, a compression artefact, a render frame).
    """
    derivative_id: str
    reference_id: str
    mask_policy: str

    # primary divergence scores — all in [0, 1]
    quality_score: float           # structural fidelity; higher = closer to reference
    divergence_score: float        # overall divergence; higher = more different
    artifact_score: float          # perceptual/technical artefact severity
    temporal_consistency_score: float  # stability vs prior reference; higher = more stable

    regional_breakdown: Dict[str, float]

    feature_version: str
    model_version: str
    latency_ms: float
    device: str
