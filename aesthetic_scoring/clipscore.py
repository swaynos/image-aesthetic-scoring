"""CLIPScore-style prompt-alignment scorer using CLIP ViT-B/32."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image

from ._device import empty_cache, get_device, get_dtype, get_precision, inference_guard
from .errors import ModelLoadError
from .types import CLIPScoreResult

_model = None
_preprocess = None
_tokenizer = None
_device: Optional[torch.device] = None
_precision: Optional[str] = None

MODEL_NAME = "clipscore"
MODEL_VERSION = "ViT-B-32-openai"
MAX_EDGE = 1024


def _load():
    global _model, _preprocess, _tokenizer, _device, _precision
    if _model is not None:
        return _model, _preprocess, _tokenizer, _device, _precision
    try:
        import open_clip

        device = get_device()
        precision = get_precision(device)
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai", device=device
        )
        if device.type in ("cuda", "mps"):
            model = model.to(get_dtype(precision))
        _model = model.eval()
        _preprocess = preprocess
        _tokenizer = open_clip.get_tokenizer("ViT-B-32")
        _device = device
        _precision = precision
    except Exception as exc:
        raise ModelLoadError(f"Failed to load CLIPScore model: {exc}") from exc
    return _model, _preprocess, _tokenizer, _device, _precision


def unload() -> None:
    global _model, _preprocess, _tokenizer, _device, _precision
    device = _device
    _model = _preprocess = _tokenizer = _device = _precision = None
    empty_cache(device or get_device())


def score_clipscore(image_paths: List[str], prompt: str) -> CLIPScoreResult:
    """Return raw image-text cosine similarities for an ordinary-language prompt."""
    if not isinstance(image_paths, list) or not image_paths:
        raise ValueError("image_paths must be a non-empty list[str]")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    for path in image_paths:
        if not isinstance(path, str):
            raise TypeError("Each image_path must be str")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

    model, preprocess, tokenizer, device, precision = _load()
    with inference_guard():
        started = time.perf_counter()
        images = []
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            if max(image.size) > MAX_EDGE:
                scale = MAX_EDGE / max(image.size)
                image = image.resize((int(image.width * scale), int(image.height * scale)), Image.LANCZOS)
            images.append(preprocess(image))
        image_tensor = torch.stack(images).to(device)
        if device.type in ("cuda", "mps"):
            image_tensor = image_tensor.to(get_dtype(precision))
        text_tokens = tokenizer([prompt]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            text_features = model.encode_text(text_tokens)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scores = (image_features @ text_features.T).squeeze(1)
        latency_ms = (time.perf_counter() - started) * 1000.0

    values = scores.float().cpu().tolist()
    image_ids = [Path(path).name for path in image_paths]
    ranked = [image_id for _, image_id in sorted(zip(values, image_ids), reverse=True)]
    return CLIPScoreResult(
        image_id=image_ids[0], model_name=MODEL_NAME, model_version=MODEL_VERSION,
        latency_ms=latency_ms, device=str(device), precision=precision, prompt=prompt,
        scores=values, ranked_image_ids=ranked,
    )
