"""Run raw detection and scoring without assigning a combined score."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

from aesthetic_scoring import score_images
from aesthetic_scoring.suite import DEFAULT_MODELS, PROMPT_MODELS
from object_detection import compare_owl_models

from .types import ImageEvaluationReport


def _validate_scoring_args(
    image_path: str,
    evaluation_prompt: str | None,
    scoring_models: Sequence[str] | None,
) -> None:
    if not isinstance(image_path, str):
        raise TypeError("image_path must be str")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    selected = tuple(DEFAULT_MODELS) if scoring_models is None else tuple(dict.fromkeys(scoring_models))
    unknown = set(selected).difference(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")
    if PROMPT_MODELS.intersection(selected) and (
        not isinstance(evaluation_prompt, str) or not evaluation_prompt.strip()
    ):
        raise ValueError("evaluation_prompt is required for prompt-conditioned models")


def evaluate_image(
    image_path: str,
    detection_queries: list[str],
    *,
    evaluation_prompt: str | None = None,
    scoring_models: list[str] | None = None,
    **detection_kwargs: Any,
) -> ImageEvaluationReport:
    """Return detector and scorer evidence for a later, explicit policy."""

    _validate_scoring_args(image_path, evaluation_prompt, scoring_models)
    detections = compare_owl_models(image_path, detection_queries, **detection_kwargs)
    scores = score_images(
        [image_path], evaluation_prompt=evaluation_prompt, models=scoring_models
    )
    return ImageEvaluationReport(
        image_id=Path(image_path).name,
        detections=detections,
        scores=scores,
        status="complete",
    )
