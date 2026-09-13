# aesthetic_scoring

GPU-backed Python scoring suite for ranking generated images. It exposes raw
scores from several models with different purposes. It does not combine them
into one "quality" number.

## Models

| Model | Prompt | What it measures | Typical VRAM |
|---|---:|---|---:|
| LAION-Aesthetics v2.5 | No | Broad visual appeal | 1.6 GiB |
| FGAesQ | No | Fine-grained aesthetic score | 2-3 GiB |
| PickScore | Yes | Relative human preference for candidates | 3.8 GiB |
| HPSv2.1 | Yes | General prompt-conditioned preference | 3.7 GiB |
| CLIPScore | Yes | Raw image-text alignment, not aesthetic quality | under 1 GiB |

Each model has its own scale, biases, and training data. Compare scores only
within the same model, prompt, and candidate set.

ImageReward was removed in 3.0.0. See DECISIONS.md for why.

## Requirements

- Python 3.11
- NVIDIA CUDA 12.1+ on Linux, or Apple Silicon with MPS
- A GPU with 6 GiB VRAM can run the built-in models one at a time

## Installation

```bash
pyenv virtualenv 3.11.10 image-aesthetic-scoring
pyenv local image-aesthetic-scoring
python -m pip install -e ".[dev]"
```

Weights download lazily on first use.

## Quick Start

```python
from aesthetic_scoring import score_images

report = score_images(
    ["candidate-a.png", "candidate-b.png"],
    evaluation_prompt="a professional portrait with short red hair and cinematic lighting",
)

for model, results in report.results.items():
    print(model, results)
```

The suite runs models sequentially and unloads each one before loading the
next. Select a subset to reduce run time:

```python
report = score_images(
    ["candidate-a.png", "candidate-b.png"],
    evaluation_prompt="a professional portrait with short red hair",
    models=["laion", "fgaesq", "pickscore", "clipscore"],
)
```

## Direct APIs

```python
from aesthetic_scoring import (
    score_clipscore,
    score_fgaesq,
    score_hpsv2,
    score_laion,
    score_pickscore,
)

score_laion("image.png")
score_fgaesq("image.png")
score_pickscore(["a.png", "b.png"], "a studio portrait")
score_hpsv2("image.png", "a studio portrait")
score_clipscore(["a.png", "b.png"], "a studio portrait")
```

Call `unload()` from an individual model module when using direct APIs in a
long-running process. `score_images()` does this automatically.

## Custom LoRA Triggers

Do not treat a private LoRA trigger such as `fwbugh4d5` as an evaluation
prompt. Prompt-conditioned scorers use separate frozen text encoders, so they
do not know what the trigger means. Store two fields instead:

- **Generation prompt:** may contain the private trigger.
- **Evaluation prompt:** describes the desired visible result in ordinary
  language, such as "a cinematic portrait of Jane with short red hair."

Use LAION and FGAesQ alongside prompt-conditioned models when the special
token cannot be described fully. Neither can verify that a LoRA-specific
identity or style was reproduced; that needs a reference or a dedicated model.

## Scope

This project scores images and ranks candidates. It does not provide:

- A universal combined score
- Technical image-quality assessment models such as TOPIQ or MANIQA
- Reference-based edit degradation scoring
- Identity, face, anatomy, or body-part scoring

Git history retains the former reference-comparison experiment.

## Verification

```bash
python -m pytest tests/unit -q
python -m pytest tests/smoke -q  # downloads weights and requires a GPU
```

## Model Sources

- LAION-Aesthetics: `christophschuhmann/improved-aesthetic-predictor`
- PickScore: `yuvalkirstain/PickScore_v1`
- HPSv2.1: `xswu/HPSv2`
- FGAesQ: `yzc002/FGAesQ`
- CLIPScore: CLIP ViT-B/32 cosine similarity

## License

GNU General Public License v3.0. Model weights and optional dependencies have
their own licenses; review them before redistribution or commercial use.
