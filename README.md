# aesthetic_scoring

GPU-backed Python library for image aesthetic/preference scoring (v1), and a
CPU-based reference-vs-derivative image comparison scorer (v2).

- **v1** — four scoring models with a uniform typed API.  Device selected
  automatically: CUDA → MPS → CPU.  Weights are loaded lazily on first call;
  each module exposes `unload()` to free VRAM between models.
- **v2** — `score_reference_comparison` measures structural and perceptual
  divergence between a reference image and any derivative.  CPU-only, no model
  weights required by default (passthrough mode).  Pass `model_path=` to use a
  trained model from [aesthetic-model-training](https://github.com/anomalyco/aesthetic-model-training).

## Models

| Model | Function | VRAM (fp16) | Notes |
|---|---|---|---|
| LAION-Aesthetics v2.5 | `score_laion` | ~1.6 GiB | CLIP ViT-L/14 + MLP |
| PickScore | `score_pickscore` | ~3.8 GiB | CLIP-H/14, multi-image vs prompt |
| HPSv2.1 | `score_hpsv2` | ~3.7 GiB | CLIP-H/14 preference model |
| FGAesQ | `score_fgaesq` | ~0.6 GiB | CLIP ViT-B/16 fine-grained aesthetic |

## Requirements

- Python 3.11
- **CUDA (Linux, primary):** NVIDIA GPU, CUDA 12.1+
  - RTX 4050 6 GB — all four models tested and passing
- **MPS (macOS, supported):** Apple Silicon (M1/M2/M3/M4), macOS 13+, PyTorch ≥ 2.1

## Installation

### Linux (CUDA)

```bash
pip install git+https://github.com/openai/CLIP.git   # FGAesQ dependency
pip install -e ".[dev]" --no-deps                     # Install package
pip install torch torchvision transformers accelerate open_clip_torch \
    Pillow numpy hpsv2 scikit-image huggingface_hub pytest pytest-mock
```

### macOS — Apple Silicon (M1/M2/M3/M4)

```bash
pip install git+https://github.com/openai/CLIP.git   # FGAesQ dependency
pip install -e ".[dev]"                               # hpsv2 excluded automatically
pip install torch torchvision transformers accelerate open_clip_torch \
    Pillow numpy scikit-image huggingface_hub pytest pytest-mock
```

> **Note:** The `hpsv2` PyPI package is excluded on macOS via a `sys_platform != 'darwin'`
> marker. The library's `hpsv2.py` uses `open_clip` directly and does not need it.

> **MPS fallback:** On first import when Apple Silicon is detected, the library
> automatically sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. This allows ops that lack MPS
> kernels to run on CPU transparently. You can pre-set this variable before running
> if you want explicit control: `export PYTORCH_ENABLE_MPS_FALLBACK=1`.

## Quick Start

Each v1 function returns a typed dataclass with common metadata fields
(`image_id`, `model_name`, `model_version`, `latency_ms`, `device`, `precision`)
plus the model-specific score fields shown below.

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
- Precision: fp16 on CUDA and MPS (FGAesQ uses fp32 internally on all backends)
- Max input edge: 1024 px (images are downscaled before inference)
- MPS fallback env var: set automatically to `PYTORCH_ENABLE_MPS_FALLBACK=1`

**Expected latency (warm cache, weight download excluded):**

| Model | CUDA (RTX 4050) | MPS (M1 class) |
|---|---|---|
| LAION | < 30 s | < 90 s |
| PickScore | < 60 s | < 180 s |
| HPSv2 | < 60 s | < 180 s |
| FGAesQ | < 60 s | < 180 s |

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


## v2: Reference-Based Image Comparison

`score_reference_comparison` measures the structural and perceptual divergence
between a **reference image** and any **derivative** of it, with optional
mask-aware regional breakdown.

Applicable to any workflow that produces derivative images, including:
- iterative inpainting / editing
- generative model outputs vs. conditioning image
- encode → decode round-trips
- upscaling / super-resolution
- style transfer vs. content source
- compression / codec A/B testing
- render frame vs. golden frame regression

```python
from aesthetic_scoring import score_reference_comparison

result = score_reference_comparison(
    reference_path="source.png",
    derivative_path="output.png",
    mask_policy="custom_intent",          # whole_image|subject|background|custom_intent|none
    intent_mask_path="region_mask.png",   # PNG 0/255 — white = region of interest
    prior_reference_path="earlier.png",   # optional earlier reference for consistency scoring
    model_path="baseline_model.json",     # optional trained model from aesthetic-model-training
)

print(result.quality_score)               # float [0, 1] — structural fidelity; higher = closer
print(result.divergence_score)            # float [0, 1] — overall divergence; higher = more different
print(result.artifact_score)              # float [0, 1] — perceptual/technical artefact severity
print(result.temporal_consistency_score)  # float [0, 1] — stability vs prior reference
print(result.regional_breakdown)          # dict of per-region raw metrics
```

`prior_reference_path` and the mask parameters are optional.  `step_index` and
`intent_state` on the underlying `RegionalFeatureVector` are caller-defined
labels with no fixed semantics — use them to encode sequence position (edit pass,
frame number, variant index) and group/category (prompt variant, A/B group, style
bucket) as appropriate for your workflow.

Without `model_path`, scores are derived directly from image features (passthrough mode)
— no training data or external dependencies required.  To produce a trained model, use
[aesthetic-model-training](https://github.com/anomalyco/aesthetic-model-training), which
owns the dataset builder, pseudo-label generation, and baseline training scripts.  The two
repos are fully decoupled: training produces a `baseline_model.json` artefact; this library
only reads it at runtime and never imports from the training repo.


## Future Scope (v2+)

**HPSv3** is deferred. It uses a Qwen2-VL-7B backbone and requires ≥16 GB VRAM.
When implemented, it will follow the same interface pattern as HPSv2:
`score_hpsv3(image_path, prompt) -> HPSv3ScoreResult`.
