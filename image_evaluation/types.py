"""Types that preserve raw model evidence for later policy evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from aesthetic_scoring.types import ImageScoreReport
from object_detection.types import OwlDetectionResult


@dataclass
class ImageEvaluationReport:
    image_id: str
    detections: list[OwlDetectionResult]
    scores: ImageScoreReport
    status: str

    def to_dict(self) -> dict[str, object]:
        """Return evidence that callers can persist before applying a policy."""

        return {
            "image_id": self.image_id,
            "detections": [result.to_dict() for result in self.detections],
            "scores": asdict(self.scores),
            "status": self.status,
        }
