# image-aesthetic-scoring Specification

## Version

2.0.0

## Objective

Provide a lightweight, GPU-backed suite of image aesthetic, preference, and
prompt-alignment scorers for ranking generated-image candidates.

## Principles

- Aesthetic appeal, human preference, and prompt alignment are distinct
  measurements. The library returns raw outputs and never creates an arbitrary
  combined score.
- Prompt-conditioned scorers require an ordinary-language evaluation prompt.
  Private LoRA trigger tokens are generation metadata, not semantic inputs to
  another model's frozen text encoder.
- Models load lazily and the suite unloads each selected model before the next
  one loads, targeting a 6 GiB VRAM budget.
- ImageReward is an optional dependency. All other scorers are core features.

## Public API

- `score_laion(image_path) -> LaionScoreResult`
- `score_fgaesq(image_path) -> FGAesQScoreResult`
- `score_pickscore(image_paths, prompt) -> PickScoreResult`
- `score_hpsv2(image_path, prompt) -> HPSv2ScoreResult`
- `score_imagereward(image_paths, prompt) -> ImageRewardScoreResult`
- `score_clipscore(image_paths, prompt) -> CLIPScoreResult`
- `score_images(image_paths, evaluation_prompt=None, models=None) -> ImageScoreReport`

`score_images` runs selected models in the caller's order. It requires
`evaluation_prompt` when selecting PickScore, HPSv2, ImageReward, or CLIPScore.

## Included Models

| Identifier | Model | Prompt required | Output meaning |
|---|---|---:|---|
| `laion` | LAION-Aesthetics v2.5 | No | Broad aesthetic score |
| `fgaesq` | FGAesQ | No | Fine-grained aesthetic score |
| `pickscore` | PickScore v1 | Yes | Candidate-relative preference logits and probabilities |
| `hpsv2` | HPSv2.1 | Yes | Prompt-conditioned preference score |
| `imagereward` | ImageReward v1.0 | Yes | Prompt-conditioned expert-preference reward |
| `clipscore` | CLIP ViT-B/32 | Yes | Image-text cosine similarity |

## Out of Scope

- Reference-vs-derivative comparison and edit-degradation metrics
- Dataset construction or model training
- Technical IQA models, including TOPIQ, MUSIQ, MANIQA, and LIQE
- Q-ReAlign, VisionReward, HPSv3, and other models that exceed the lightweight
  hardware target or require an unresolved license decision
- Identity, face, anatomy, and body-part assessment

## Acceptance Criteria

1. All direct scorer functions and result types import without loading weights.
2. The suite rejects unknown model identifiers and prompt-conditioned requests
   without an evaluation prompt.
3. The suite unloads every selected model after its result is produced.
4. `ImageScoreReport` is JSON serializable through `dataclasses.asdict`.
5. Unit tests pass without a GPU, model weights, or network access.

## Verification

```bash
python -m pytest tests/unit -q
python -m pytest tests/smoke -q
```
