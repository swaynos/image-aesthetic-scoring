"""
aesthetic_scoring — GPU-backed image aesthetic and comparison scoring library.

Models (v1):
  - LAION-Aesthetics v2.5  → score_laion
  - PickScore              → score_pickscore
  - HPSv2                  → score_hpsv2
  - FGAesQ                 → score_fgaesq

v2:
  - score_reference_comparison
"""

from .errors import GpuMemoryError, ModelInferenceError, ModelLoadError
from .types import (
    BaseScoreResult,
    FGAesQScoreResult,
    HPSv2ScoreResult,
    LaionScoreResult,
    PickScoreResult,
    ReferenceComparisonResult,
)
from .laion import score_laion
from .pickscore import score_pickscore
from .hpsv2 import score_hpsv2
from .fgaesq import score_fgaesq
from .intent_api import score_reference_comparison

__all__ = [
    # v1 scoring functions
    "score_laion",
    "score_pickscore",
    "score_hpsv2",
    "score_fgaesq",
    # v2 scoring function
    "score_reference_comparison",
    # result types
    "BaseScoreResult",
    "LaionScoreResult",
    "PickScoreResult",
    "HPSv2ScoreResult",
    "FGAesQScoreResult",
    "ReferenceComparisonResult",
    # exceptions
    "GpuMemoryError",
    "ModelInferenceError",
    "ModelLoadError",
]
