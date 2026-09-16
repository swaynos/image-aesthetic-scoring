# aesthetic_scoring

GPU-backed Python scoring suite for ranking generated images. It exposes raw
scores from several models with different purposes. It does not combine them
into one "quality" number.

The distribution also contains local OWL-family object detection and an image
evaluation orchestrator. They preserve raw evidence for a later, explicit
policy; neither assigns a combined score.

## Packages

- `aesthetic_scoring` returns raw aesthetic, preference, and prompt-alignment
  measurements.
- `object_detection` runs text-conditioned OWL-ViT and OWLv2 detection.
- `image_evaluation` runs both layers sequentially and returns their evidence
  in one report for a separately defined policy.

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

The install fetches one dependency, OpenAI CLIP, straight from GitHub at a
pinned commit, so it needs network access and `git` on the machine. That
dependency also pulls `ftfy`, `regex`, and `tqdm`.

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

Paths may be relative; they resolve against the current working directory.
`evaluation_prompt` is required unless every selected model is prompt-free
(`laion`, `fgaesq`).

## OWL Detection And Evaluation

`object_detection` is separate from the scoring API because boxes and
confidence are evidence, not aesthetic scores. Both supported base models run
one at a time and are unloaded between comparison passes:

```python
from object_detection import compare_owl_models

results = compare_owl_models(
    "candidate.png",
    ["person", "bicycle"],
    threshold=0.2,
)
for result in results:
    print(result.model_id, result.detections)
```

Use `image_evaluation` when a later policy needs both kinds of evidence:

```python
from image_evaluation import evaluate_image

report = evaluate_image(
    "candidate.png",
    detection_queries=["person", "bicycle"],
    evaluation_prompt="a professional street portrait",
    scoring_models=["laion", "clipscore"],
)
```

The evaluation report does not contain a final score. A future policy must
state its detection conditions, weights, and handling for failed inference.
See `docs/IMAGE_EVALUATION.md` and `docs/OWL_DETECTION.md`.

## Reading the report

`score_images` returns an `ImageScoreReport` with three fields:

- `image_ids` — the input filenames, basename only (e.g. `candidate-a.png`).
- `evaluation_prompt` — the prompt you passed, or `None`.
- `results` — a dict keyed by model name. Each value is a list of result
  dataclasses.

**The list has two shapes, depending on the model.** Read it the wrong way and
you get the wrong number without an error:

| Models | Shape of `results[model]` | How to index |
|---|---|---|
| `laion`, `fgaesq`, `hpsv2` | one result per image, in input order | `results[model][i]` is image `i` |
| `pickscore`, `clipscore` | a single result covering all images | `results[model][0]`; its `scores` list is per image, in input order |

The per-image models each expose their own scalar field; the batch models
carry a `scores` list plus a `ranked_image_ids` list sorted best-first:

```python
laion = report.results["laion"]
for image_id, result in zip(report.image_ids, laion):
    print(image_id, result.aesthetic_score)        # LaionScoreResult

fgaesq = report.results["fgaesq"]
print(fgaesq[0].technical_score, fgaesq[0].aesthetic_score, fgaesq[0].subscores)

hpsv2 = report.results["hpsv2"]
print(hpsv2[0].preference_score)                    # per image, in input order

clipscore = report.results["clipscore"][0]          # one result for the batch
print(clipscore.scores)                             # one float per image, input order
print(clipscore.ranked_image_ids)                   # image_ids, best first

pickscore = report.results["pickscore"][0]
print(pickscore.scores, pickscore.probabilities, pickscore.ranked_image_ids)
```

Every result also carries `image_id`, `model_name`, `model_version`,
`latency_ms`, `device`, and `precision`. The whole report is JSON-serializable
through `dataclasses.asdict`. Field definitions live in
`aesthetic_scoring/types.py`.

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

Object detection remains available as evidence for a caller-defined evaluation
policy. It does not establish identity, aesthetic quality, or a final ranking.

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
