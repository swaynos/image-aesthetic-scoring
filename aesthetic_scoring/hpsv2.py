"""
HPSv2 scorer.

Implements HPSv2.1 inference directly using open_clip + huggingface_hub,
without using the `hpsv2` PyPI package internals (which loads full model into CPU RAM).

Model: ViT-H-14 (CLIP-H) fine-tuned checkpoint from xswu/HPSv2 (HPS_v2.1_compressed.pt).
The model is loaded to the active device directly to minimise peak CPU RAM usage.

Precision default: fp16 on CUDA and MPS; fp32 on CPU.
Max input edge: 1024 px.
Typical memory: ~3.5 GB fp16 (CUDA/MPS unified).

MPS notes:
- PYTORCH_ENABLE_MPS_FALLBACK=1 is set automatically by _device.py.
- torch.autocast is NOT used on MPS (not supported for all ops); the model is
  cast to fp16 at load time instead.
- Image tensors are explicitly cast to fp16 before inference on MPS.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from ._device import get_device, get_precision, get_dtype, inference_guard, empty_cache
from .errors import ModelLoadError
from .types import HPSv2ScoreResult

# ── module-level lazy state ────────────────────────────────────────────────────
_model = None
_preprocess = None
_tokenizer = None
_device: Optional[torch.device] = None
_precision: Optional[str] = None

MAX_EDGE = 1024
MODEL_NAME = "hpsv2"
MODEL_VERSION = "HPS_v2.1"
HF_REPO_ID = "xswu/HPSv2"
HF_FILENAME = "HPS_v2.1_compressed.pt"


def _load():
    global _model, _preprocess, _tokenizer, _device, _precision

    if _model is not None:
        return _model, _preprocess, _tokenizer, _device, _precision

    try:
        import open_clip
        from huggingface_hub import hf_hub_download

        dev = get_device()
        prec = get_precision(dev)
        dtype = get_dtype(prec)

        # Download checkpoint
        ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

        # Create model skeleton on CPU first (meta device trick is not available
        # in open_clip, so we do: create on CPU, load weights, move to GPU)
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            "ViT-H-14",
            pretrained=None,   # no pretrained download — we load weights manually
            precision="fp32",  # create in fp32 first
            device="cpu",
            jit=False,
            force_quick_gelu=False,
            output_dict=True,
        )

        # Load weights directly to CPU then move to GPU
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict, strict=False)
        model = model.to(dev)
        if dev.type in ("cuda", "mps"):
            model = model.to(dtype)
        model.eval()

        tokenizer = open_clip.get_tokenizer("ViT-H-14")

        _model = model
        _preprocess = preprocess_val
        _tokenizer = tokenizer
        _device = dev
        _precision = prec

    except Exception as exc:
        raise ModelLoadError(f"Failed to load HPSv2 model: {exc}") from exc

    return _model, _preprocess, _tokenizer, _device, _precision


def unload() -> None:
    global _model, _preprocess, _tokenizer, _device, _precision
    dev = _device
    _model = None
    _preprocess = None
    _tokenizer = None
    _device = None
    _precision = None
    if dev is not None:
        empty_cache(dev)
    else:
        from ._device import _DEVICE
        empty_cache(_DEVICE)


def score_hpsv2(image_path: str, prompt: str) -> HPSv2ScoreResult:
    """Score an image+prompt pair with HPSv2.

    Args:
        image_path: Path to a local image file.
        prompt: Text prompt describing the image intent.

    Returns:
        HPSv2ScoreResult with preference_score.

    Raises:
        TypeError: on bad argument types.
        FileNotFoundError: if image_path does not exist.
        GpuMemoryError: on CUDA OOM.
        ModelInferenceError: on other inference failures.
        ModelLoadError: if model cannot be loaded.
    """
    if not isinstance(image_path, str):
        raise TypeError(f"image_path must be str, got {type(image_path).__name__}")
    if not isinstance(prompt, str):
        raise TypeError(f"prompt must be str, got {type(prompt).__name__}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_id = Path(image_path).name
    model, preprocess, tokenizer, device, precision = _load()

    with inference_guard():
        t0 = time.perf_counter()

        img = Image.open(image_path).convert("RGB")
        w, h = img.size
        if max(w, h) > MAX_EDGE:
            scale = MAX_EDGE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        image_tensor = preprocess(img).unsqueeze(0).to(device)
        if device.type in ("cuda", "mps"):
            image_tensor = image_tensor.to(get_dtype(precision))

        text_tokens = tokenizer([prompt]).to(device)

        with torch.no_grad():
            # autocast is CUDA-only; on MPS the model is already in fp16 from load time
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                    outputs = model(image_tensor, text_tokens)
            else:
                outputs = model(image_tensor, text_tokens)
            image_features = outputs["image_features"]
            text_features = outputs["text_features"]
            logits = image_features @ text_features.T
            score_val = float(torch.diagonal(logits).cpu().float().item())

        latency_ms = (time.perf_counter() - t0) * 1000.0

    return HPSv2ScoreResult(
        image_id=image_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
        device=str(device),
        precision=precision,
        prompt=prompt,
        preference_score=score_val,
    )
