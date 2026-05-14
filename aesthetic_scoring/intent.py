"""
intent.py — Mask loading, policy application, boundary ring extraction, tile grid.

Mask format: single-channel PNG, uint8.  Pixel ≥ 128 → inside intent (True).
All outputs are numpy bool arrays unless otherwise noted.
CPU-only; no torch dependency.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

MASK_POLICIES = frozenset(
    {"whole_image", "subject", "background", "custom_intent", "none"}
)

FEATURE_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Mask loading
# ─────────────────────────────────────────────────────────────────────────────

def load_mask(mask_path: str, target_size: Tuple[int, int]) -> np.ndarray:
    """Load a PNG mask and resize to target_size (W, H).

    Returns a bool array of shape (H, W).  Pixel ≥ 128 → True (inside intent).
    """
    img = Image.open(mask_path).convert("L")
    img = img.resize(target_size, Image.NEAREST)
    return np.array(img) >= 128


def make_whole_image_mask(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def make_empty_mask(h: int, w: int) -> np.ndarray:
    return np.zeros((h, w), dtype=bool)


def resolve_masks(
    h: int,
    w: int,
    mask_policy: str,
    intent_mask_path: Optional[str] = None,
    subject_mask_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (inside_mask, outside_mask) booleans according to mask_policy.

    Returns (None, None) when policy is 'none' or masks are unavailable,
    indicating that mask-aware metrics should be skipped.
    """
    if mask_policy not in MASK_POLICIES:
        raise ValueError(
            f"mask_policy must be one of {sorted(MASK_POLICIES)}, got {mask_policy!r}"
        )

    target = (w, h)  # PIL uses (W, H)

    if mask_policy == "none":
        return None, None

    if mask_policy == "whole_image":
        inside = make_whole_image_mask(h, w)
        return inside, ~inside  # outside = nothing

    if mask_policy == "subject":
        if subject_mask_path is None:
            return None, None
        inside = load_mask(subject_mask_path, target)
        return inside, ~inside

    if mask_policy == "background":
        if subject_mask_path is None:
            return None, None
        subject = load_mask(subject_mask_path, target)
        inside = ~subject  # background is the intent region
        return inside, subject

    if mask_policy == "custom_intent":
        if intent_mask_path is None:
            return None, None
        inside = load_mask(intent_mask_path, target)
        return inside, ~inside

    return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Boundary ring
# ─────────────────────────────────────────────────────────────────────────────

def extract_boundary_ring(
    mask: np.ndarray, ring_width: int = 4
) -> np.ndarray:
    """Return a bool mask of the ring just outside and inside the mask boundary.

    Uses a simple erosion/dilation approach with a square structuring element.
    """
    if mask is None or not mask.any():
        return np.zeros_like(mask, dtype=bool)

    from scipy.ndimage import binary_dilation, binary_erosion  # lazy import

    struct = np.ones((ring_width * 2 + 1, ring_width * 2 + 1), dtype=bool)
    dilated = binary_dilation(mask, structure=struct)
    eroded = binary_erosion(mask, structure=struct)
    ring = dilated & ~eroded
    return ring


# ─────────────────────────────────────────────────────────────────────────────
# Tile grid
# ─────────────────────────────────────────────────────────────────────────────

def tile_grid(
    arr: np.ndarray, tile_size: int = 64
) -> List[np.ndarray]:
    """Split a (H, W) or (H, W, C) array into non-overlapping square tiles.

    Tiles that would overflow the edge are clipped.
    Returns a flat list of tile arrays.
    """
    h, w = arr.shape[:2]
    tiles = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            tile = arr[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
    return tiles


def tile_positions(h: int, w: int, tile_size: int = 64) -> List[Tuple[int, int, int, int]]:
    """Return (y0, y1, x0, x1) for each tile."""
    positions = []
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            positions.append((y, min(y + tile_size, h), x, min(x + tile_size, w)))
    return positions
