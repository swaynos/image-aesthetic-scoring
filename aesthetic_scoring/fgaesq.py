"""
FGAesQ scorer.

Source: github.com/yzc-ippl/FG-IAA (commit 4bfd40fff7d935de1a613e3650815e0bb7a952e2)
Weights: huggingface.co/yzc002/FGAesQ

This module inlines the FGAesQ model and DiffToken preprocessor from the
upstream source so the package has no external source-code dependency.

Precision default: fp32 (model uses fp32 internally via CLIP float()).
Max input edge: applied via DiffToken's ensure_large_image_size (2048 default upstream),
  we cap at 1024 for the 6 GB VRAM budget.
Typical VRAM: ~2-3 GB.

subscores keys returned: {"dist_score": float, "raw_score": float}
  dist_score = weighted mean of 10-bin distribution
  raw_score  = direct model output (same as aesthetic_score but kept for transparency)
"""

from __future__ import annotations

import math
import os
import random
import time
from contextlib import contextmanager
from functools import reduce
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from ._device import get_device, inference_guard
from .errors import ModelLoadError
from .types import FGAesQScoreResult

# ── module-level lazy state ────────────────────────────────────────────────────
_model: Optional[nn.Module] = None
_preprocessor = None
_device: Optional[torch.device] = None
_precision: str = "fp32"

MAX_EDGE = 1024
MODEL_NAME = "fgaesq"
MODEL_VERSION = "FGAesQ-v1.0"
HF_REPO_ID = "yzc002/FGAesQ"
HF_FILENAME = "FGAesQ.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Inlined data_utils (subset needed for inference)
# ─────────────────────────────────────────────────────────────────────────────

def _add_padding(img: torch.Tensor, patch_size: int) -> torch.Tensor:
    _, height, width = img.size()
    pad_w = (patch_size - width % patch_size) % patch_size
    pad_h = (patch_size - height % patch_size) % patch_size
    return nn.ZeroPad2d((0, pad_w, 0, pad_h))(img)


def _image_to_patches(image: torch.Tensor, patch_size: int, stride: Optional[int] = None):
    if stride is None:
        stride = patch_size
    _, height, width = image.size()
    patches = []
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patches.append(image[:, y:y + patch_size, x:x + patch_size])
        if width % patch_size != 0:
            patches.append(image[:, y:y + patch_size, width - patch_size:width])
    return patches


def _lcm(arr):
    return reduce(lambda x, y: (x * y) // math.gcd(x, y), arr)


def _padding(x: torch.Tensor, max_seq_len: int) -> torch.Tensor:
    n, c, W, H = x.size()
    pad = torch.zeros(max_seq_len, c, W, H, dtype=x.dtype, device=x.device)
    x = torch.cat([x, pad], dim=0)[:max_seq_len]
    return x


def _pad_or_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    _, height, width = image.shape
    crop_w = width % size or 1
    crop_h = height % size or 1
    pad_w = (size - width % size) % size or 1
    pad_h = (size - height % size) % size or 1
    if pad_h * pad_w > crop_h * crop_w:
        # crop
        return image[:, :height - crop_h, :width - crop_w]
    else:
        return _add_padding(image, size)


def _create_binary_mask(
    image_size: Tuple[int, int, int],
    patch_size: int,
    selected_indices,
    n_coarse_cols: int,
) -> torch.Tensor:
    """Build a boolean mask on the fine-scale grid selecting `selected_indices` coarse patches.

    Each coarse-grid index maps to a patch_size x patch_size block in the fine
    mask using the correct 2-D mapping:

        fine_row = (coarse_idx // n_coarse_cols) * patch_size
        fine_col = (coarse_idx  % n_coarse_cols) * patch_size

    The previous implementation used ``starts = idx * patch_size`` and derived
    rows/cols by dividing by the fine-grid width, which produces wrong
    coordinates for patch_size > 1 on non-square grids.
    """
    _, height, width = image_size
    mask = torch.zeros((height, width), dtype=torch.bool)
    for idx in sorted(selected_indices):
        r = (idx // n_coarse_cols) * patch_size
        c = (idx  % n_coarse_cols) * patch_size
        mask[r:r + patch_size, c:c + patch_size] = True
    return mask


def _interpolate_pos_embed(pos_embed: torch.Tensor, size: Tuple[int, int], offset: float, dim: int):
    w0, h0 = size
    w0, h0 = w0 + offset, h0 + offset
    N = pos_embed.shape[1] - 1
    pos_embed = pos_embed.float()
    class_pe = pos_embed[:, 0]
    patch_pe = pos_embed[:, 1:]
    sqrt_N = math.sqrt(N)
    sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
    patch_pe = F.interpolate(
        patch_pe.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
        scale_factor=(sx, sy),
        mode="bicubic",
        align_corners=False,
    )
    patch_pe = patch_pe.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pe, class_pe


# ─────────────────────────────────────────────────────────────────────────────
# Inlined FGAesQ model
# ─────────────────────────────────────────────────────────────────────────────

class _FGAesQModel(nn.Module):
    def __init__(self):
        super().__init__()

        import clip as _clip
        self.clip_model, _ = _clip.load("ViT-B/16", device="cpu")
        self.clip_model = self.clip_model.float()

        self.scales = 2
        self.max_seq_len = 512

        for param in self.clip_model.parameters():
            param.requires_grad = False
        for param in self.clip_model.visual.parameters():
            param.requires_grad = True

        self.ln_pre = self.clip_model.visual.ln_pre
        self.transformer = self.clip_model.visual.transformer
        self.ln_post = self.clip_model.visual.ln_post
        self.proj = self.clip_model.visual.proj

        self.scale_embedding = nn.Parameter(torch.randn(1, 2, 768) * 0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, 768))

        self.iqa_head = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 64),  nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 10),
        )

    def forward(self, patches, pos_embeds, masks):
        tokens = self._patches_to_tokens(patches, masks)
        features = self._extract_features(tokens, pos_embeds, masks)
        dist_logits = self.iqa_head(features)
        score_weights = torch.arange(1, 11, dtype=torch.float32, device=dist_logits.device)
        dist_probs = F.softmax(dist_logits, dim=1)
        scores = torch.sum(dist_probs * score_weights.unsqueeze(0), dim=1, keepdim=True)
        return scores, dist_probs

    def _patches_to_tokens(self, patches, masks):
        batch, seq_len, c, h, w = patches.shape
        device = patches.device

        actual = patches[:, 1:, :, :, :]
        actual_seq = actual.shape[1]
        flat = actual.reshape(-1, c, h, w)
        emb = self.clip_model.visual.conv1(flat)
        emb = emb.flatten(2).transpose(1, 2).squeeze(1)
        emb = emb.reshape(batch, actual_seq, 768)

        if masks is not None:
            pad_mask = (masks == 9).int()
            masks_copy = masks.clone()
            masks_copy[masks_copy == 9] = 0
            seq_length = emb.shape[1]
            mask_tok = self.mask_token.expand(batch, seq_length, -1)
            m = pad_mask[:, 1:].unsqueeze(-1).type_as(mask_tok)
            emb = emb * (1.0 - m) + mask_tok * m
            masks = masks_copy

        emb = self._add_scale_embed(emb, masks)
        cls = self.clip_model.visual.class_embedding.expand(batch, 1, -1)
        return torch.cat((cls, emb), dim=1)

    def _add_scale_embed(self, emb, masks):
        se = self.scale_embedding.detach().cpu().numpy()[0]
        m = masks[:, 1:].cpu().long()  # np.take requires integer indices
        se_t = torch.tensor(np.take(se, m.numpy(), axis=0)).to(emb.device)
        return emb + se_t

    def _extract_features(self, tokens, pos_embeds, masks):
        x = tokens + pos_embeds
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        cls = x[:, 0, :]
        cls = self.ln_post(cls)
        return cls @ self.proj


# ─────────────────────────────────────────────────────────────────────────────
# Inlined DiffToken preprocessor (single-image inference path only)
# ─────────────────────────────────────────────────────────────────────────────

class _DiffToken:
    PATCH_SIZE = 16
    MAX_SEQ_LEN = 512   # hidden_size = 511 + 1 (cls)
    INTERP_OFFSET = 0.1

    def __init__(self, clip_model):
        self.clip_model = clip_model
        self.pos_embeds: dict = {}
        self.scaled_patchsizes = [16, 32]

        self.clip_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ])

        pe = clip_model.visual.positional_embedding.clone().detach().cpu()
        if pe.dim() == 2:
            pe = pe.unsqueeze(0)
        self.pos_embed = pe
        self._hidden_size = self.MAX_SEQ_LEN  # 512

    def process_image(self, image_pil: Image.Image) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image_pil = self._ensure_large(image_pil)
        tensor = self.clip_transform(image_pil)
        patches, pos_embed, mask = self._prepare_patches(tensor)
        return patches, pos_embed, mask

    def _ensure_large(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        n_w = math.ceil(w / self.PATCH_SIZE)
        n_h = math.ceil(h / self.PATCH_SIZE)
        if n_w * n_h <= self._hidden_size:
            target = self._hidden_size * 1.2
            scale = max(math.sqrt(target / (n_w * n_h)), 1.0)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        return img

    def _prepare_patches(self, image: torch.Tensor):
        _, H, W = image.size()
        n_w = math.ceil(W / self.PATCH_SIZE)
        n_h = math.ceil(H / self.PATCH_SIZE)
        if n_h * n_w <= self._hidden_size:
            return self._process_small(image)
        else:
            return self._process_large_multiscale(image)

    def _process_small(self, image: torch.Tensor):
        image = _add_padding(image, self.PATCH_SIZE)
        _, H, W = image.size()
        n_w = math.ceil(W / self.PATCH_SIZE)
        n_h = math.ceil(H / self.PATCH_SIZE)
        size = (n_h, n_w)

        patches = _image_to_patches(image, self.PATCH_SIZE, self.PATCH_SIZE)
        inp = torch.stack(patches)
        inp = torch.cat((torch.zeros(1, *inp.shape[1:]), inp), dim=0)

        pos = self._get_pos_embed(size)
        pos = pos.squeeze(0)

        if inp.shape[0] < self._hidden_size:
            pad_area = self._hidden_size - inp.shape[0]
            mask = torch.cat([torch.zeros(inp.shape[0]), torch.full((pad_area,), 9.0)])
            inp = _padding(inp, self._hidden_size)
            pos = _padding(pos.unsqueeze(-1).unsqueeze(-1), self._hidden_size).squeeze(-1).squeeze(-1)
        elif inp.shape[0] > self._hidden_size:
            inp, pos = self._drop_tokens(inp, pos)
            mask = torch.zeros(self._hidden_size)
        else:
            mask = torch.zeros(self._hidden_size)

        return inp, pos, mask

    def _process_large_multiscale(self, image: torch.Tensor):
        lcm_val = _lcm(self.scaled_patchsizes)
        image = _pad_or_crop(image, lcm_val)

        coarse_size = self.scaled_patchsizes[-1]
        coarse_patches = _image_to_patches(image, coarse_size, coarse_size)

        importance_threshold = 0.5
        n_fine = int(self._hidden_size * importance_threshold / 4)

        # Random selection (patch_selection='random' for inference)
        fine_indices = random.sample(range(len(coarse_patches)), min(n_fine, len(coarse_patches)))
        coarse_indices = [i for i in range(len(coarse_patches)) if i not in fine_indices]

        fine_patches = []
        for i in sorted(fine_indices):
            fp = _image_to_patches(coarse_patches[i], self.PATCH_SIZE, self.PATCH_SIZE)
            fine_patches.extend(fp)

        coarse_fine = []
        for i in sorted(coarse_indices):
            cp = F.interpolate(
                coarse_patches[i].unsqueeze(0),
                size=(self.scaled_patchsizes[0], self.scaled_patchsizes[0]),
                mode="bicubic",
            ).squeeze(0)
            sp = _image_to_patches(cp, self.PATCH_SIZE, self.PATCH_SIZE)
            coarse_fine.extend(sp)

        mask_ms = [0] * len(coarse_fine) + [1] * len(fine_patches)
        final = coarse_fine + fine_patches
        final_t = torch.stack(final)

        # Positional embeddings
        _, H, W = image.shape
        n_patch_row = H // coarse_size
        n_patch_col = W // coarse_size

        masks_for_pe = []
        for i, indices in enumerate([coarse_indices, fine_indices]):
            ps = self.scaled_patchsizes[i]
            p = ps // self.PATCH_SIZE
            sz = (3, p * n_patch_row, p * n_patch_col)
            m = _create_binary_mask(sz, p, indices, n_coarse_cols=n_patch_col)
            masks_for_pe.append(m)

        pos_embeds = self._prepare_multiscale_pe(masks_for_pe).squeeze(0)

        final_t = torch.cat((torch.zeros(1, *final_t.shape[1:]), final_t), dim=0)
        mask_ms.insert(0, 0)

        if final_t.shape[0] < self._hidden_size:
            pad_area = self._hidden_size - final_t.shape[0]
            mask_ms += [9] * pad_area
            final_t = _padding(final_t, self._hidden_size)
            pos_embeds = _padding(
                pos_embeds.unsqueeze(-1).unsqueeze(-1), self._hidden_size
            ).squeeze(-1).squeeze(-1)
            mask = torch.tensor(mask_ms, dtype=torch.float32)
        elif final_t.shape[0] > self._hidden_size:
            final_t, pos_embeds, mask_ms_t = self._drop_tokens(final_t, pos_embeds, torch.tensor(mask_ms))
            mask = mask_ms_t.float()
        else:
            mask = torch.tensor(mask_ms, dtype=torch.float32)

        return final_t, pos_embeds, mask

    def _get_pos_embed(self, size: Tuple[int, int]) -> torch.Tensor:
        if size not in self.pos_embeds:
            patch_pe, cls_pe = _interpolate_pos_embed(
                self.pos_embed, size, self.INTERP_OFFSET, self.pos_embed.shape[-1]
            )
            self.pos_embeds[size] = torch.cat((cls_pe.unsqueeze(0), patch_pe), dim=1)
        return self.pos_embeds[size]

    def _prepare_multiscale_pe(self, masks) -> torch.Tensor:
        class_pe = None
        fine_pes = []
        for fine_mask in masks:
            sz = fine_mask.size()
            if sz not in self.pos_embeds:
                patch_pe, cls_pe = _interpolate_pos_embed(
                    self.pos_embed, sz, self.INTERP_OFFSET, self.pos_embed.shape[-1]
                )
                self.pos_embeds[sz] = torch.cat((cls_pe.unsqueeze(0), patch_pe), dim=1)
            stored = self.pos_embeds[sz]
            class_pe = stored[:, 0, :]
            patch_pe = stored[:, 1:, :]
            selected = patch_pe[:, fine_mask.flatten(), :]
            fine_pes.append(selected)

        fine_all = torch.cat(fine_pes, dim=1)
        return torch.cat([class_pe.unsqueeze(0), fine_all], dim=1)

    def _drop_tokens(self, inp, pos, mask_ms=None):
        to_remove = inp.shape[0] - self._hidden_size
        if to_remove <= 0:
            return (inp, pos, mask_ms) if mask_ms is not None else (inp, pos)
        indices = random.sample(range(1, inp.shape[0]), to_remove)
        keep = torch.ones(inp.shape[0], dtype=torch.bool)
        keep[indices] = False
        if mask_ms is not None:
            return inp[keep], pos[keep], mask_ms[keep]
        return inp[keep], pos[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Load / unload
# ─────────────────────────────────────────────────────────────────────────────

def _load():
    global _model, _preprocessor, _device, _precision

    if _model is not None:
        return _model, _preprocessor, _device, _precision

    try:
        from huggingface_hub import hf_hub_download

        dev = get_device()

        ckpt_path = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)

        model = _FGAesQModel()
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
        model = model.to(dev)
        model.eval()

        preprocessor = _DiffToken(clip_model=model.clip_model)

        _model = model
        _preprocessor = preprocessor
        _device = dev
        _precision = "fp32"  # FGAesQ runs fp32 internally

    except Exception as exc:
        raise ModelLoadError(f"Failed to load FGAesQ model: {exc}") from exc

    return _model, _preprocessor, _device, _precision


def unload() -> None:
    global _model, _preprocessor, _device, _precision
    _model = None
    _preprocessor = None
    _device = None
    _precision = "fp32"
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def score_fgaesq(image_path: str) -> FGAesQScoreResult:
    """Score an image with FGAesQ (fine-grained aesthetic quality).

    Args:
        image_path: Path to a local image file.

    Returns:
        FGAesQScoreResult with technical_score, aesthetic_score, and subscores.

    Raises:
        TypeError: on bad argument type.
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
    model, preprocessor, device, precision = _load()

    with inference_guard():
        t0 = time.perf_counter()

        img = Image.open(image_path).convert("RGB")
        # Apply max-edge guard before the model's own ensure_large step
        w, h = img.size
        if max(w, h) > MAX_EDGE:
            scale = MAX_EDGE / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        patches, pos_embed, mask = preprocessor.process_image(img)

        patches = patches.unsqueeze(0).to(device)
        pos_embed = pos_embed.unsqueeze(0).to(device)
        mask = mask.unsqueeze(0).to(device)

        with torch.no_grad():
            scores_out, dist_probs = model(patches, pos_embed, mask)

        raw_score = float(scores_out.squeeze().item())
        dist_probs_list = dist_probs.squeeze().float().cpu().tolist()

        # technical_score: weighted mean over bins 1-5 (lower half)
        weights_tech = torch.arange(1, 6, dtype=torch.float32)
        probs_tech = torch.tensor(dist_probs_list[:5])
        probs_tech = probs_tech / (probs_tech.sum() + 1e-8)
        technical_score = float((probs_tech * weights_tech).sum().item())

        # aesthetic_score: full weighted mean (same as raw_score but normalized 1-10)
        aesthetic_score = raw_score

        subscores: Dict[str, float] = {
            f"bin_{i+1}": round(p, 6) for i, p in enumerate(dist_probs_list)
        }
        subscores["raw_score"] = round(raw_score, 6)

        latency_ms = (time.perf_counter() - t0) * 1000.0

    return FGAesQScoreResult(
        image_id=image_id,
        model_name=MODEL_NAME,
        model_version=MODEL_VERSION,
        latency_ms=latency_ms,
        device=str(device),
        precision=precision,
        technical_score=technical_score,
        aesthetic_score=aesthetic_score,
        subscores=subscores,
    )
