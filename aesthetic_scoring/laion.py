"""
LAION-Aesthetics v2.5 scorer.

Model: CLIP ViT-L/14 (openai) + linear MLP head.
Checkpoint: sac+logos+ava1-l14-linearMSE.pth from
  https://github.com/christophschuhmann/improved-aesthetic-predictor
  (downloaded directly, no HF token required)

Precision default: fp16 on CUDA.
Max input edge: 1024 px.
Typical VRAM: ~1.5 GB fp16.
"""

from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image

from ._device import get_device, get_precision, get_dtype, inference_guard
from .errors import ModelLoadError
from .types import LaionScoreResult

# ── module-level lazy state ────────────────────────────────────────────────────
_model: Optional[Tuple] = None   # (clip_model, mlp)
_processor = None
_device: Optional[torch.device] = None
_precision: Optional[str] = None

MAX_EDGE = 1024
MODEL_NAME = "laion-aesthetics-v2.5"
MODEL_VERSION = "sac+logos+ava1-l14-linearMSE"
WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "aesthetic_scoring", "laion")


class _AestheticMLP(nn.Module):
    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


def _load():
    global _model, _processor, _device, _precision

    if _model is not None:
        return _model, _processor, _device, _precision

    try:
        import open_clip

        dev = get_device()
        prec = get_precision(dev)
        dtype = get_dtype(prec)

        # Load CLIP ViT-L/14
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained="openai"
        )
        clip_model = clip_model.to(dev)
        if dev.type == "cuda":
            clip_model = clip_model.to(dtype)
        clip_model.eval()

        # Download MLP head weights (public GitHub, no auth required)
        os.makedirs(CACHE_DIR, exist_ok=True)
        ckpt_path = os.path.join(CACHE_DIR, "sac+logos+ava1-l14-linearMSE.pth")
        if not os.path.exists(ckpt_path):
            urllib.request.urlretrieve(WEIGHTS_URL, ckpt_path)

        mlp = _AestheticMLP(input_size=768)
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        mlp.load_state_dict(state)
        mlp = mlp.to(dev)
        if dev.type == "cuda":
            mlp = mlp.to(dtype)
        mlp.eval()

        _model = (clip_model, mlp)
        _processor = preprocess
        _device = dev
        _precision = prec

    except Exception as exc:
        raise ModelLoadError(f"Failed to load LAION aesthetics model: {exc}") from exc

    return _model, _processor, _device, _precision


def unload() -> None:
    """Dereference the cached model and free GPU memory."""
    global _model, _processor, _device, _precision
    _model = None
    _processor = None
    _device = None
    _precision = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def score_laion(image_path: str) -> LaionScoreResult:
    """Score an image with LAION-Aesthetics v2.5.

    Args:
        image_path: Path to a local image file.

    Returns:
        LaionScoreResult with aesthetic_score in ~1-10 range.

    Raises:
        TypeError: if image_path is not a string.
        FileNotFoundError: if image_path does not exist.
        GpuMemoryError: on CUDA OOM.
        ModelInferenceError: on other inference failures.
        ModelLoadError: if model cannot be loaded.
    """
    if not isinstance(image_path, str):
        raise TypeError(f"image_path must be str, got {type(image_path).__name__}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_id = Path(image_path).name
    (clip_model, mlp), preprocess, device, precision = _load()

    with inference_guard():
        t0 = time.perf_counter()

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_EDGE:
            scale = MAX_EDGE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        tensor = preprocess(img).unsqueeze(0).to(device)
        if device.type == "cuda":
            tensor = tensor.to(get_dtype(precision))

        with torch.no_grad():
            image_features = clip_model.encode_image(tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            # Keep same dtype as MLP (fp16 on CUDA); cast output to fp32 for final value
            score_raw = mlp(image_features).squeeze().float().item()

        latency_ms = (time.perf_counter() - t0) * 1000.0

    return LaionScoreResult(
        image_id=image_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
        device=str(device),
        precision=precision,
        aesthetic_score=float(score_raw),
        score_scale="1-10",
    )
