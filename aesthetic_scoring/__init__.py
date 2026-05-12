"""
aesthetic_scoring — GPU-backed image aesthetic scoring library.

Models:
  - LAION-Aesthetics v2.5  → score_laion
  - PickScore              → score_pickscore
  - HPSv2                  → score_hpsv2
  - FGAesQ                 → score_fgaesq
"""

from .errors import GpuMemoryError, ModelInferenceError, ModelLoadError
from .types import (
    BaseScoreResult,
    FGAesQScoreResult,
    HPSv2ScoreResult,
    LaionScoreResult,
    PickScoreResult,
)
from .laion import score_laion
from .pickscore import score_pickscore
from .hpsv2 import score_hpsv2
from .fgaesq import score_fgaesq

__all__ = [
    # scoring functions
    "score_laion",
    "score_pickscore",
    "score_hpsv2",
    "score_fgaesq",
    # result types
    "BaseScoreResult",
    "LaionScoreResult",
    "PickScoreResult",
    "HPSv2ScoreResult",
    "FGAesQScoreResult",
    # exceptions
    "GpuMemoryError",
    "ModelInferenceError",
    "ModelLoadError",
]
