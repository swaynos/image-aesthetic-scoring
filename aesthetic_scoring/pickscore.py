"""
PickScore scorer.

Model: CLIP-H/14 fine-tuned on Pick-a-Pic v1 (yuvalkirstain/PickScore_v1).
Uses HuggingFace CLIPModel + AutoProcessor.

Precision default: fp16 on CUDA and MPS; fp32 on CPU.
Max input edge: 1024 px.
Typical memory: ~3.5 GB fp16 (CUDA/MPS unified).

MPS notes:
- PYTORCH_ENABLE_MPS_FALLBACK=1 is set automatically by _device.py.
- pixel_values are cast to fp16 on both CUDA and MPS.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from ._device import get_device, get_precision, get_dtype, inference_guard, empty_cache
from .errors import ModelLoadError
from .types import PickScoreResult

# ── module-level lazy state ────────────────────────────────────────────────────
_model = None
_processor = None
_device: Optional[torch.device] = None
_precision: Optional[str] = None

MAX_EDGE = 1024
MODEL_NAME = "pickscore"
MODEL_VERSION = "PickScore_v1"
PROCESSOR_ID = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MODEL_ID = "yuvalkirstain/PickScore_v1"


def _load():
    global _model, _processor, _device, _precision

    if _model is not None:
        return _model, _processor, _device, _precision

    try:
        from transformers import AutoProcessor, CLIPModel

        dev = get_device()
        prec = get_precision(dev)
        dtype = get_dtype(prec)

        processor = AutoProcessor.from_pretrained(PROCESSOR_ID)
        model = CLIPModel.from_pretrained(MODEL_ID)
        model = model.eval().to(dev)
        if dev.type in ("cuda", "mps"):
            model = model.to(dtype)

        _model = model
        _processor = processor
        _device = dev
        _precision = prec

    except Exception as exc:
        raise ModelLoadError(f"Failed to load PickScore model: {exc}") from exc

    return _model, _processor, _device, _precision


def unload() -> None:
    global _model, _processor, _device, _precision
    dev = _device
    _model = None
    _processor = None
    _device = None
    _precision = None
    if dev is not None:
        empty_cache(dev)
    else:
        from ._device import _DEVICE
        empty_cache(_DEVICE)


def score_pickscore(image_paths: List[str], prompt: str) -> PickScoreResult:
    """Score a list of images against a prompt using PickScore.

    Args:
        image_paths: List of local image file paths (>=1).
        prompt: Text prompt used for comparison.

    Returns:
        PickScoreResult with per-image logits, softmax probabilities, and ranking.

    Raises:
        TypeError / ValueError: on bad argument types.
        FileNotFoundError: if any image_path does not exist.
        GpuMemoryError: on CUDA OOM.
        ModelInferenceError: on other inference failures.
        ModelLoadError: if model cannot be loaded.
    """
    if not isinstance(image_paths, list):
        raise TypeError(f"image_paths must be list[str], got {type(image_paths).__name__}")
    if len(image_paths) == 0:
        raise ValueError("image_paths must contain at least one path")
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    for p in image_paths:
        if not isinstance(p, str):
            raise TypeError(f"Each image_path must be str, got {type(p).__name__}")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

    image_id = Path(image_paths[0]).name
    model, processor, device, precision = _load()

    with inference_guard():
        t0 = time.perf_counter()

        images = []
        for p in image_paths:
            img = Image.open(p).convert("RGB")
            w, h = img.size
            if max(w, h) > MAX_EDGE:
                scale = MAX_EDGE / max(w, h)
                img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            images.append(img)

        image_inputs = processor(
            images=images,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs = processor(
            text=prompt,
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        if device.type in ("cuda", "mps"):
            dtype = get_dtype(precision)
            if "pixel_values" in image_inputs:
                image_inputs["pixel_values"] = image_inputs["pixel_values"].to(dtype)

        with torch.no_grad():
            # Use vision_model + visual_projection (CLIPModel pattern)
            vision_out = model.vision_model(**image_inputs)
            image_embs = model.visual_projection(vision_out.pooler_output)
            image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

            text_out = model.text_model(**text_inputs)
            text_embs = model.text_projection(text_out.pooler_output)
            text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

            scores = model.logit_scale.exp() * (text_embs @ image_embs.T)
            scores = scores[0]  # shape [n_images]
            probs = F.softmax(scores, dim=-1)

        latency_ms = (time.perf_counter() - t0) * 1000.0

    scores_list = scores.float().cpu().tolist()
    probs_list = probs.float().cpu().tolist()

    image_ids = [Path(p).name for p in image_paths]
    ranked = sorted(zip(probs_list, image_ids), key=lambda x: x[0], reverse=True)
    ranked_image_ids = [iid for _, iid in ranked]

    return PickScoreResult(
        image_id=image_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
        device=str(device),
        precision=precision,
        prompt=prompt,
        scores=scores_list,
        probabilities=probs_list,
        ranked_image_ids=ranked_image_ids,
    )
