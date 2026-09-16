"""JSON-safe result types for text-conditioned object detection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Detection:
    """One text-conditioned detection in source-image pixel coordinates."""

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "score": self.score,
            "box_xyxy": list(self.box_xyxy),
        }


@dataclass
class OwlDetectionResult:
    """Raw evidence returned by one OWL-family model for one image."""

    image_id: str
    model_id: str
    latency_ms: float
    device: str
    precision: str
    queries: list[str]
    image_width: int
    image_height: int
    detections: list[Detection]

    def to_dict(self) -> dict[str, object]:
        return {
            "image_id": self.image_id,
            "model_id": self.model_id,
            "latency_ms": self.latency_ms,
            "device": self.device,
            "precision": self.precision,
            "queries": self.queries,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "detections": [detection.to_dict() for detection in self.detections],
        }
