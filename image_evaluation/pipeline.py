"""Run raw detection and scoring without assigning a combined score."""

from __future__ import annotations

from aesthetic_scoring import score_images
from object_detection import compare_owl_models

from .types import ImageEvaluationReport


def evaluate_image(
    image_path: str,
    *,
    detection_queries: list[str],
    evaluation_prompt: str | None = None,
    scoring_models: list[str] | None = None,
) -> ImageEvaluationReport:
    """Return detector and scorer evidence for a later, explicit policy."""

    detections = compare_owl_models(image_path, detection_queries)
    scores = score_images(
        [image_path], evaluation_prompt=evaluation_prompt, models=scoring_models
    )
    return ImageEvaluationReport(
        image_id=detections[0].image_id,
        detections=detections,
        scores=scores,
        status="complete",
    )
