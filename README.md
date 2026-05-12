# aesthetic_scoring

GPU-backed Python library for image aesthetic/preference scoring using four models.

## Models

| Model | Function | VRAM (fp16) | Notes |
|---|---|---|---|
| LAION-Aesthetics v2.5 | `score_laion` | ~1.6 GiB | CLIP ViT-L/14 + MLP |
| PickScore | `score_pickscore` | ~3.8 GiB | CLIP-H/14, multi-image vs prompt |
| HPSv2.1 | `score_hpsv2` | ~3.7 GiB | CLIP-H/14 preference model |
| FGAesQ | `score_fgaesq` | ~0.6 GiB | CLIP ViT-B/16 fine-grained aesthetic |

## Requirements

- Python 3.11
- NVIDIA GPU, CUDA 12.1+
- **Dev tier:** RTX 4050 6 GB — all four models tested and passing
- **Production tier:** ≥16 GB VRAM recommended for headroom

## Installation

```bash
pip install git+https://github.com/openai/CLIP.git   # FGAesQ dependency
pip install -e ".[dev]" --no-deps                     # Install package
pip install torch torchvision transformers accelerate open_clip_torch \
    Pillow numpy hpsv2 scikit-image huggingface_hub pytest pytest-mock
```

> **Note:** The `hpsv2` PyPI package ships without the BPE vocab file needed by its
> vendored tokenizer. Copy it from `open_clip` if needed:
> ```bash
> cp $(python -c "import open_clip, os; print(os.path.dirname(open_clip.__file__))")/bpe_simple_vocab_16e6.txt.gz \
>    $(python -c "import hpsv2, os; print(os.path.dirname(hpsv2.__file__))")/src/open_clip/
> ```
> Our `hpsv2.py` module bypasses the `hpsv2` package internals entirely and uses
> `open_clip` directly, so this workaround is only needed if you use the `hpsv2`
> CLI separately.

## Quick Start

```python
from aesthetic_scoring import score_laion, score_pickscore, score_hpsv2, score_fgaesq

# LAION — single image, no prompt
result = score_laion("my_image.jpg")
print(result.aesthetic_score)   # float, ~1-10 scale

# PickScore — compare images against a prompt
result = score_pickscore(["img_a.jpg", "img_b.jpg"], "a cat in space")
print(result.probabilities)     # [0.72, 0.28] — sums to 1.0
print(result.ranked_image_ids)  # ["img_a.jpg", "img_b.jpg"]

# HPSv2 — image + prompt preference
result = score_hpsv2("my_image.jpg", "a cat in space")
print(result.preference_score)  # float ~0.2-0.3 typical range

# FGAesQ — fine-grained aesthetic quality
result = score_fgaesq("my_image.jpg")
print(result.aesthetic_score)   # float 1-10 range
print(result.technical_score)   # float 1-5 range (lower bins)
print(result.subscores)         # {"bin_1": ..., ..., "bin_10": ..., "raw_score": ...}
```

## Memory Management

Models are loaded lazily on first call. Call `unload()` to free VRAM between models:

```python
from aesthetic_scoring.laion import unload as laion_unload
from aesthetic_scoring.hpsv2 import unload as hpsv2_unload

result = score_laion("img.jpg")
laion_unload()   # free ~1.6 GiB before loading next model

result2 = score_hpsv2("img.jpg", "prompt")
hpsv2_unload()
```

**Defaults:**
- Precision: fp16 on CUDA (FGAesQ uses fp32 internally)
- Max input edge: 1024 px (images are downscaled before inference)

## Verification

```bash
# 1. Import surface check
python -c "import aesthetic_scoring; from aesthetic_scoring import score_laion, score_pickscore, score_hpsv2, score_fgaesq; from aesthetic_scoring.errors import ModelInferenceError, GpuMemoryError, ModelLoadError; from aesthetic_scoring.types import LaionScoreResult, PickScoreResult, HPSv2ScoreResult, FGAesQScoreResult"

# 2. Unit tests (no GPU required)
python -m pytest tests/unit -q

# 3. GPU smoke tests (requires CUDA device)
python -m pytest tests/smoke -q
```

## External Dependency Commit SHAs

| Dependency | Source | Pinned SHA |
|---|---|---|
| FG-IAA (FGAesQ source) | github.com/yzc-ippl/FG-IAA | `4bfd40fff7d935de1a613e3650815e0bb7a952e2` |
| HPSv2 | github.com/tgxs002/HPSv2 | `866735ecaae999fa714bd9edfa05aa2672669ee3` |

FGAesQ model code is inlined in `aesthetic_scoring/fgaesq.py` from the upstream
source at the commit above (no pip dependency on FG-IAA).

HPSv2 weights are downloaded from `xswu/HPSv2` on HuggingFace Hub
(`HPS_v2.1_compressed.pt`). The `hpsv2` PyPI package is listed as a dependency
but `aesthetic_scoring/hpsv2.py` uses `open_clip` directly for inference to
avoid the package's CPU-RAM overhead.

## Future Scope (v2+)

**HPSv3** is deferred. It uses a Qwen2-VL-7B backbone and requires ≥16 GB VRAM.
When implemented, it will follow the same interface pattern as HPSv2:
`score_hpsv3(image_path, prompt) -> HPSv3ScoreResult`.
