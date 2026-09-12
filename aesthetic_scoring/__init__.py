"""GPU-backed image scoring suite for aesthetics, preferences, and alignment."""

from .errors import GpuMemoryError, ModelInferenceError, ModelLoadError
from .types import (
    BaseScoreResult,
    FGAesQScoreResult,
    HPSv2ScoreResult,
    LaionScoreResult,
    PickScoreResult,
    CLIPScoreResult,
    ImageRewardScoreResult,
    ImageScoreReport,
)
from .laion import score_laion
from .pickscore import score_pickscore
from .hpsv2 import score_hpsv2
from .fgaesq import score_fgaesq
from .clipscore import score_clipscore
from .imagereward import score_imagereward
from .suite import score_images

__all__ = [
    "score_laion",
    "score_pickscore",
    "score_hpsv2",
    "score_fgaesq",
    "score_clipscore",
    "score_imagereward",
    "score_images",
    # result types
    "BaseScoreResult",
    "LaionScoreResult",
    "PickScoreResult",
    "HPSv2ScoreResult",
    "FGAesQScoreResult",
    "CLIPScoreResult",
    "ImageRewardScoreResult",
    "ImageScoreReport",
    # exceptions
    "GpuMemoryError",
    "ModelInferenceError",
    "ModelLoadError",
]
