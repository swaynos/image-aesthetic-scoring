"""Sequential multi-model scoring that keeps peak VRAM within a small budget."""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from .clipscore import score_clipscore, unload as unload_clipscore
from .fgaesq import score_fgaesq, unload as unload_fgaesq
from .hpsv2 import score_hpsv2, unload as unload_hpsv2
from .laion import score_laion, unload as unload_laion
from .pickscore import score_pickscore, unload as unload_pickscore
from .types import BaseScoreResult, ImageScoreReport

DEFAULT_MODELS = ("laion", "fgaesq", "pickscore", "hpsv2", "clipscore")
PROMPT_MODELS = frozenset({"pickscore", "hpsv2", "clipscore"})


def score_images(
    image_paths: Sequence[str],
    evaluation_prompt: str | None = None,
    models: Sequence[str] | None = None,
) -> ImageScoreReport:
    """Run selected models sequentially and return their raw, incomparable outputs.

    ``evaluation_prompt`` should describe the desired visible result in ordinary
    language, rather than private LoRA trigger tokens.
    """
    if not isinstance(image_paths, (list, tuple)) or not image_paths:
        raise ValueError("image_paths must be a non-empty sequence of paths")
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"Image(s) not found: {', '.join(missing)}")
    if models is None:
        selected = tuple(DEFAULT_MODELS)
    else:
        # Preserve caller order but drop duplicates: a repeated model would
        # otherwise load and score twice, and the later result would silently
        # overwrite the earlier one in ``results``.
        selected = tuple(dict.fromkeys(models))
    unknown = set(selected).difference(DEFAULT_MODELS)
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown))}")
    if PROMPT_MODELS.intersection(selected) and (
        not isinstance(evaluation_prompt, str) or not evaluation_prompt.strip()
    ):
        raise ValueError("evaluation_prompt is required for prompt-conditioned models")

    results: dict[str, List[BaseScoreResult]] = {}
    for model in selected:
        if model == "laion":
            try:
                results[model] = [score_laion(path) for path in image_paths]
            finally:
                unload_laion()
        elif model == "fgaesq":
            try:
                results[model] = [score_fgaesq(path) for path in image_paths]
            finally:
                unload_fgaesq()
        elif model == "pickscore":
            try:
                results[model] = [score_pickscore(list(image_paths), evaluation_prompt)]
            finally:
                unload_pickscore()
        elif model == "hpsv2":
            try:
                results[model] = [score_hpsv2(path, evaluation_prompt) for path in image_paths]
            finally:
                unload_hpsv2()
        elif model == "clipscore":
            try:
                results[model] = [score_clipscore(list(image_paths), evaluation_prompt)]
            finally:
                unload_clipscore()
        else:
            # Unreachable while every DEFAULT_MODELS entry has a branch above.
            # Guards against a model being added to DEFAULT_MODELS without a
            # dispatch branch, which previously fell through to clipscore.
            raise ValueError(f"No scoring branch for model: {model}")

    return ImageScoreReport(
        image_ids=[Path(path).name for path in image_paths],
        evaluation_prompt=evaluation_prompt,
        results=results,
    )
