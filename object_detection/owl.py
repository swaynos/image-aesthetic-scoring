"""Raw OWL-ViT and OWLv2 text-conditioned object detection."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torchvision.ops import nms

from .types import Detection, OwlDetectionResult

OWLVIT_BASE = "google/owlvit-base-patch32"
OWLV2_BASE = "google/owlv2-base-patch16-ensemble"
SUPPORTED_MODELS = frozenset({OWLVIT_BASE, OWLV2_BASE})

_model: Any | None = None
_processor: Any | None = None
_model_id: str | None = None
_device: torch.device | None = None
_precision: str | None = None


def _get_device() -> torch.device:
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def _get_dtype(device: torch.device) -> tuple[str, torch.dtype]:
    if device.type == "cuda":
        return "fp16", torch.float16
    return "fp32", torch.float32


def _load(model_id: str) -> tuple[Any, Any, torch.device, str, torch.dtype]:
    global _model, _processor, _model_id, _device, _precision
    if _model is not None and _model_id == model_id:
        assert _processor is not None and _device is not None and _precision is not None
        return _model, _processor, _device, _precision, _get_dtype(_device)[1]
    unload()

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    device = _get_device()
    precision, dtype = _get_dtype(device)
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).eval().to(device)
    if device.type == "cuda":
        model = model.to(dtype)

    _model = model
    _processor = processor
    _model_id = model_id
    _device = device
    _precision = precision
    return model, processor, device, precision, dtype


def unload() -> None:
    """Release the currently loaded detector and its CUDA cache."""

    global _model, _processor, _model_id, _device, _precision
    _model = None
    _processor = None
    _model_id = None
    _device = None
    _precision = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _validate(image_path: str, queries: list[str], model: str) -> list[str]:
    if not isinstance(image_path, str):
        raise TypeError("image_path must be str")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    if model not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported OWL model: {model}")
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a non-empty list[str]")
    cleaned = [" ".join(query.split()) for query in queries if isinstance(query, str) and query.strip()]
    if len(cleaned) != len(queries):
        raise ValueError("queries must contain only non-empty strings")
    return cleaned


def _nms_detections(
    result: dict[str, Any], queries: list[str], threshold: float, nms_threshold: float, max_detections: int
) -> list[Detection]:
    boxes = result["boxes"].detach().to(device="cpu", dtype=torch.float32)
    scores = result["scores"].detach().to(device="cpu", dtype=torch.float32)
    labels = result.get("text_labels") or [queries[int(index)] for index in result["labels"]]
    grouped: dict[str, list[int]] = {}
    for index, (label, score) in enumerate(zip(labels, scores.tolist(), strict=True)):
        if score >= threshold:
            grouped.setdefault(str(label), []).append(index)

    kept: list[Detection] = []
    for label, indexes in grouped.items():
        selected = nms(boxes[indexes], scores[indexes], nms_threshold).tolist()
        for index in selected:
            box = boxes[indexes[index]].tolist()
            kept.append(Detection(label=label, score=float(scores[indexes[index]]), box_xyxy=tuple(box)))
    kept.sort(key=lambda detection: (-detection.score, detection.label, detection.box_xyxy))
    return kept[:max_detections]


def detect_owl(
    image_path: str,
    queries: list[str],
    model: str = OWLV2_BASE,
    *,
    threshold: float = 0.1,
    nms_threshold: float = 0.3,
    max_detections: int = 100,
) -> OwlDetectionResult:
    """Detect text queries in one local image with an OWL-family model."""

    cleaned_queries = _validate(image_path, queries, model)
    if not 0.0 <= threshold <= 1.0 or not 0.0 <= nms_threshold <= 1.0:
        raise ValueError("threshold and nms_threshold must be between 0 and 1")
    if max_detections < 1:
        raise ValueError("max_detections must be positive")

    detector, processor, device, precision, dtype = _load(model)
    image = Image.open(image_path).convert("RGB")
    started = time.perf_counter()
    inputs = processor(text=[cleaned_queries], images=image, return_tensors="pt")
    inputs = {
        name: value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device)
        for name, value in inputs.items()
    }
    with torch.inference_mode():
        outputs = detector(**inputs)
    results = processor.post_process_grounded_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)], device=device),
        threshold=0.0,
        text_labels=[cleaned_queries],
    )
    detections = _nms_detections(results[0], cleaned_queries, threshold, nms_threshold, max_detections)
    return OwlDetectionResult(
        image_id=Path(image_path).name,
        model_id=model,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        device=str(device),
        precision=precision,
        queries=cleaned_queries,
        image_width=image.width,
        image_height=image.height,
        detections=detections,
    )


def compare_owl_models(image_path: str, queries: list[str], **kwargs: Any) -> list[OwlDetectionResult]:
    """Run OWL-ViT then OWLv2, releasing the first model before the second."""

    results = []
    for model in (OWLVIT_BASE, OWLV2_BASE):
        try:
            results.append(detect_owl(image_path, queries, model, **kwargs))
        finally:
            unload()
    return results
