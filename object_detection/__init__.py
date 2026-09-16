"""Local object-detection APIs for image evaluation."""

from .owl import OWLV2_BASE, OWLVIT_BASE, compare_owl_models, detect_owl, unload
from .types import Detection, OwlDetectionResult

__all__ = [
    "Detection",
    "OwlDetectionResult",
    "OWLVIT_BASE",
    "OWLV2_BASE",
    "detect_owl",
    "compare_owl_models",
    "unload",
]
