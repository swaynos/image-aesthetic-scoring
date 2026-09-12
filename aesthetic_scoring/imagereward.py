"""Optional ImageReward prompt-conditioned human-preference scorer."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import torch

from ._device import empty_cache, get_device, get_precision, inference_guard
from .errors import ModelLoadError
from .types import ImageRewardScoreResult

_model = None
_device: Optional[torch.device] = None

MODEL_NAME = "imagereward"
MODEL_VERSION = "ImageReward-v1.0"


def _load():
    global _model, _device
    if _model is not None:
        return _model, _device
    try:
        import ImageReward as reward

        device = get_device()
        model = reward.load(MODEL_VERSION)
        if hasattr(model, "to"):
            model = model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        _model = model
        _device = device
    except ImportError as exc:
        raise ModelLoadError(
            "ImageReward is optional. Install it with: pip install -e '.[imagereward]'"
        ) from exc
    except Exception as exc:
        raise ModelLoadError(f"Failed to load ImageReward model: {exc}") from exc
    return _model, _device


def unload() -> None:
    global _model, _device
    device = _device
    _model = _device = None
    empty_cache(device or get_device())


def score_imagereward(image_paths: List[str], prompt: str) -> ImageRewardScoreResult:
    """Score images against an ordinary-language prompt with ImageReward."""
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("image_paths must be a non-empty list[str]")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    for path in image_paths:
        if not isinstance(path, str):
            raise TypeError("Each image_path must be str")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    model, device = _load()
    with inference_guard():
        started = time.perf_counter()
        with torch.no_grad():
            scores = model.score(prompt, image_paths)
        latency_ms = (time.perf_counter() - started) * 1000.0

    values = [float(value) for value in scores]
    image_ids = [Path(path).name for path in image_paths]
    ranked = [image_id for _, image_id in sorted(zip(values, image_ids), reverse=True)]
    return ImageRewardScoreResult(
        image_id=image_ids[0], model_name=MODEL_NAME, model_version=MODEL_VERSION,
        latency_ms=latency_ms, device=str(device), precision=get_precision(device), prompt=prompt,
        scores=values, ranked_image_ids=ranked,
    )
