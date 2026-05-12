"""Typed result dataclasses returned by each scoring function."""

from dataclasses import dataclass
from typing import Dict, List


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
