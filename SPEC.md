# SPEC.md: image-aesthetic-scoring

## Version
2.1.0

## Objective
Provide a GPU-backed Python library for image aesthetic and preference scoring
(v1), and a CPU-based reference-vs-derivative image comparison scorer (v2).

## Existing API Stability (Required)
These APIs must remain backward-compatible:
- `score_laion(image_path)`
- `score_pickscore(image_paths, prompt)`
- `score_hpsv2(image_path, prompt)`
- `score_fgaesq(image_path)`
- `score_reference_comparison(reference_path, derivative_path, ...)`

## v2 Thesis
A single whole-image aesthetic score cannot distinguish *how* one image differs
from another.  The v2 scorer measures divergence between a reference image and
any derivative of it, combining:
- global structural/perceptual divergence
- mask-aware inside-intent vs outside-intent regional behaviour
- boundary-ring seam stability
- tile-level local divergence
- optional temporal consistency against an earlier reference in the same group

## In Scope
- v1: GPU-backed aesthetic and preference scorers (LAION, PickScore, HPSv2, FGAesQ).
- v2: Deterministic reference-vs-derivative feature extraction (`features.py`, `intent.py`).
- v2: Runtime comparison API (`intent_api.py`) returning `ReferenceComparisonResult`.

## Out of Scope (lives in aesthetic-model-training)
- Dataset builder, pseudo-label generation, baseline model training and evaluation.
- Derivation-sequence manifests and oscillating-intent fixtures.

The two repos are fully decoupled: `aesthetic-model-training` produces a
`baseline_model.json` artefact; this library reads it via `model_path=` at
runtime and does not import from the training repo.

## Out of Scope (Future Plugin Modules)
- Facial-quality scoring.
- Identity-preservation scoring.
- Anatomy plausibility (hands, feet, pose, proportions).
- Task-specific body-part modules.
- Detector-owned segmentation as a required dependency.
- Human labeling or manual classification workflows.

## Modules
`aesthetic_scoring/` contains:
- `laion.py`, `pickscore.py`, `hpsv2.py`, `fgaesq.py` — v1 scorers
- `features.py` — `extract_reference_features()` → `RegionalFeatureVector`
- `intent.py` — mask loading, policies, boundary ring, tile grid
- `intent_api.py` — `score_reference_comparison()` runtime entry point
- `types.py` — all result dataclasses
- `errors.py` — shared exceptions
- `_device.py` — device detection and OOM wrapping

## Typed Schemas
Exported from `aesthetic_scoring/types.py`:
- `LaionScoreResult`, `PickScoreResult`, `HPSv2ScoreResult`, `FGAesQScoreResult`
- `RegionalFeatureVector`
- `ReferenceComparisonResult`

## Feature Extraction Contract
`extract_reference_features(reference_path, derivative_path, step_index=0, intent_state='', prior_reference_path=None, intent_mask_path=None, subject_mask_path=None, mask_policy='none') -> RegionalFeatureVector`

`step_index` and `intent_state` are caller-defined identifiers with no fixed
semantics.  Common uses: `step_index` as a sequence position (edit pass, frame
number, variant index); `intent_state` as a group/category label (prompt variant,
style bucket, A/B group).

Metrics produced:
- global: SSIM, LPIPS proxy, RGB L1 delta, edge retention, HF retention, grad ratio, banding
- mask-aware: inside/outside SSIM, LPIPS proxy, RGB L1, edge retention
- boundary: seam intensity, edge continuity across ring
- tile: mean, worst, variance, high-risk count
- temporal: SSIM, LPIPS proxy, RGB L1 vs prior reference

## Runtime API Contract
`score_reference_comparison(reference_path, derivative_path, prior_reference_path=None, intent_mask_path=None, subject_mask_path=None, mask_policy='none', model_path=None) -> ReferenceComparisonResult`

Result fields:
- `quality_score` — structural fidelity to reference; higher = closer
- `divergence_score` — overall divergence; higher = more different
- `artifact_score` — perceptual/technical artefact severity
- `temporal_consistency_score` — stability vs prior reference
- `regional_breakdown` — per-region raw metrics dict
- `feature_version`, `model_version`, `latency_ms`, `device`

`model_path` is optional.  If omitted, scores are derived from features directly
(passthrough).  Pass a `baseline_model.json` from aesthetic-model-training to
use learned weights.

## Acceptance Criteria
1. v1 scorer imports remain unchanged and passing.
2. `from aesthetic_scoring import score_reference_comparison` succeeds.
3. All unit tests pass without GPU.
4. Runtime API returns a typed, JSON-serializable result.

## Verification Commands
```bash
set -euo pipefail
cd /home/bendy/Git/image-aesthetic-scoring

python -c "from aesthetic_scoring import score_laion, score_pickscore, score_hpsv2, score_fgaesq"
python -c "from aesthetic_scoring import score_reference_comparison"

python -m pytest tests/unit -q
```
