# SPEC.md: Image Aesthetic Scoring Library (v1)

## Document Control
- **Version:** 1.2.0
- **Status:** Draft for implementation
- **Last Updated:** 2026-05-11
- **Owner:** Project maintainer

## Objective
Build a Python library module that exposes GPU-backed, individual scoring APIs
for four models:
1. LAION-Aesthetics v2.5
2. PickScore
3. HPSv2
4. FGAesQ

The module must be importable from other Python code and return typed Python
objects from each scoring call.

## Scope
- Library-only deliverable (no CLI in v1).
- Individual model functions (no required orchestration pipeline in v1).
- Deterministic, schema-stable return objects.
- Primary dev target: NVIDIA RTX 4050, 6 GB VRAM. All four v1 models must run
  and pass GPU smoke tests on this card.
- Sequential usage model; the caller is expected to invoke one model at a time.

## Out of Scope
- HPSv3 — requires ≥16 GB VRAM; deferred to v2 (see Future Scope below).
- ComfyUI-specific adapters or nodes.
- Web service/API server.
- Distributed or multi-GPU inference.
- Full benchmark suite for every checkpoint variant.

## Target Environment
- **Python:** 3.11
- **OS target:** Linux (primary runtime target)
- **GPU (primary dev tier):** NVIDIA RTX 4050, 6 GB VRAM
- **GPU (recommended production tier):** NVIDIA CUDA-capable device, ≥16 GB VRAM
- **CUDA runtime target:** 12.1+
- **Precision strategy:** fp16 on CUDA for all four models; bf16 permitted
  per model if numerically beneficial, documented in that module's docstring
- **Max input edge default:** 1024 px long edge (images downscaled before
  inference to protect the 6 GB budget)

## Functional Requirements

### FR-1: Module Structure
Provide a package with explicit model entry points:

```text
aesthetic_scoring/
  __init__.py       # re-exports public API and result types
  errors.py         # custom exception classes
  types.py          # dataclass result types
  _device.py        # CUDA detection, precision selection, OOM wrapping
  laion.py          # score_laion, unload
  pickscore.py      # score_pickscore, unload
  hpsv2.py          # score_hpsv2, unload
  fgaesq.py         # score_fgaesq, unload
tests/
  unit/             # mocked, no GPU needed
  smoke/            # real GPU inference
  fixtures/         # small sample images + prompts
```

### FR-2: Individual Scoring Functions
Implement one callable per model:
- `score_laion(image_path: str) -> LaionScoreResult`
- `score_pickscore(image_paths: list[str], prompt: str) -> PickScoreResult`
- `score_hpsv2(image_path: str, prompt: str) -> HPSv2ScoreResult`
- `score_fgaesq(image_path: str) -> FGAesQScoreResult`

Each module also exposes a corresponding `unload()` function that dereferences
the cached model and calls `torch.cuda.empty_cache()`. Calling `unload()` when
no model is loaded is a no-op.

### FR-3: Typed Return Objects
Each function must return a typed dataclass with fixed fields. All fields must
be JSON-serializable via `json.dumps(dataclasses.asdict(result))` without a
custom encoder.

Required shared fields across all result types:
- `image_id: str` — non-empty
- `model_name: str` — non-empty
- `model_version: str` — non-empty
- `latency_ms: float` — wall-clock inference time, > 0
- `device: str` — e.g. `"cuda:0"` or `"cpu"`
- `precision: str` — one of `"fp16"`, `"bf16"`, `"fp32"`

Model-specific required fields:
- **LAION:** `aesthetic_score: float`, `score_scale: str` (e.g. `"1-10"`)
- **PickScore:** `prompt: str`, `scores: list[float]` (raw logits, input
  order), `probabilities: list[float]` (softmax, input order),
  `ranked_image_ids: list[str]` (image IDs sorted by probability descending)
- **HPSv2:** `prompt: str`, `preference_score: float`
- **FGAesQ:** `technical_score: float`, `aesthetic_score: float`,
  `subscores: dict[str, float]`

### FR-4: Error Handling
Define and export from `aesthetic_scoring.errors`:
- `ModelInferenceError`
- `GpuMemoryError`
- `ModelLoadError`

Rules:
- Missing file path → `FileNotFoundError`.
- Wrong argument type or empty list for PickScore → `ValueError` or `TypeError`.
- CUDA OOM during inference → `GpuMemoryError`.
- Model load/init failures → `ModelLoadError`.
- Other runtime inference failures → `ModelInferenceError`.

### FR-5: Resource Behavior
- Do not require all four models loaded at once.
- Default precision fp16 and 1024 px max edge keep each model within the
  6 GB VRAM budget. Peak VRAM per model must stay below 5.5 GiB during
  smoke tests on the RTX 4050.
- Document memory-sensitive defaults (precision, max image edge) in
  the module docstring of each model module.

## Non-Functional Requirements
- **NFR-1 Stability:** Public function signatures are stable for v1.x.
- **NFR-2 Reproducibility:** Pin all dependency versions in `pyproject.toml`.
  Any external-repo dependency (FGAesQ source) is pinned by commit SHA and
  documented in `README.md`.
- **NFR-3 Observability:** All result objects include timing metadata
  (`latency_ms`).
- **NFR-4 Performance:** Single-image LAION smoke test must complete in under
  30 s on the dev box (weight download excluded). PickScore, HPSv2, and FGAesQ
  smoke tests must each complete in under 60 s (weight download excluded).

## API Contract (Normative)

```python
from dataclasses import dataclass
from typing import Dict, List

@dataclass
class BaseScoreResult:
    image_id: str
    model_name: str
    model_version: str
    latency_ms: float
    device: str
    precision: str

@dataclass
class LaionScoreResult(BaseScoreResult):
    aesthetic_score: float
    score_scale: str

@dataclass
class PickScoreResult(BaseScoreResult):
    prompt: str
    scores: List[float]
    probabilities: List[float]
    ranked_image_ids: List[str]

@dataclass
class HPSv2ScoreResult(BaseScoreResult):
    prompt: str
    preference_score: float

@dataclass
class FGAesQScoreResult(BaseScoreResult):
    technical_score: float
    aesthetic_score: float
    subscores: Dict[str, float]
```

All result types live in `aesthetic_scoring.types` and are re-exported from
the package root (`aesthetic_scoring`).

## Dependency Baseline
Pinned in `pyproject.toml`:
- `torch` (CUDA 12.1 build)
- `torchvision`
- `transformers`
- `accelerate`
- `open_clip_torch`
- `pillow`
- `numpy`
- `pytest`, `pytest-mock` (dev)

FGAesQ integration may require an additional pinned source dependency from an
upstream repo; if so, pin by commit SHA in `README.md`.

## Verification Strategy

### Required Test Types
1. **Unit tests (mocked):** input validation, exception mapping, schema shape
   and JSON round-trip. Must pass on any machine including CPU-only CI with no
   GPU access.
2. **GPU smoke tests (real inference):** one fixture-based real inference per
   model. Must pass on the RTX 4050 6 GB dev box for all four v1 models.

### Verification Commands
```bash
# 1. Import surface check
python -c "import aesthetic_scoring; from aesthetic_scoring import score_laion, score_pickscore, score_hpsv2, score_fgaesq; from aesthetic_scoring.errors import ModelInferenceError, GpuMemoryError, ModelLoadError; from aesthetic_scoring.types import LaionScoreResult, PickScoreResult, HPSv2ScoreResult, FGAesQScoreResult"

# 2. Unit tests — no GPU required
python -m pytest tests/unit -q

# 3. GPU smoke tests — must pass on RTX 4050 6 GB
python -m pytest tests/smoke -q
```

### Smoke Test Expectations
- LAION: returns numeric `aesthetic_score`; completes in < 30 s; peak VRAM < 5.5 GiB.
- PickScore: `probabilities` sum to ~1.0 (±1e-3); `ranked_image_ids` length
  equals input image count; completes in < 60 s; peak VRAM < 5.5 GiB.
- HPSv2: returns numeric `preference_score` for image + prompt; completes in
  < 60 s; peak VRAM < 5.5 GiB.
- FGAesQ: returns numeric `technical_score` and `aesthetic_score`; completes
  in < 60 s; peak VRAM < 5.5 GiB.

## Acceptance Criteria
- [ ] Package imports cleanly on Python 3.11.
- [ ] All four scoring functions exist and match specified signatures.
- [ ] All return types conform to the fixed typed schema.
- [ ] `json.dumps(dataclasses.asdict(result))` succeeds for any result
      without a custom encoder.
- [ ] Explicit custom exceptions are implemented and raised as specified.
- [ ] Unit test suite passes (no GPU required).
- [ ] All four GPU smoke tests pass on the RTX 4050 6 GB dev box.
- [ ] Peak VRAM during each smoke test stays below 5.5 GiB.
- [ ] Per-module `unload()` is implemented; calling it when no model is
      loaded is a no-op.
- [ ] `README.md` documents: installation, precision/image-size defaults,
      dev-tier vs production-tier VRAM, and all pinned external-repo
      commit SHAs.

## Implementation Notes
- Prefer lazy model loading per module to reduce startup cost.
- Ensure tensors are moved to GPU explicitly and safely released between calls.
- Keep model-specific pre/post-processing inside each model module.
- Use `_device.py` for shared CUDA detection and OOM-to-`GpuMemoryError`
  wrapping so each model module stays focused on inference logic.

## Risks and Mitigations
- **Risk:** Upstream model checkpoint or API changes.
  - **Mitigation:** Pin model IDs/versions and dependency versions.
- **Risk:** VRAM pressure on the 6 GB dev card.
  - **Mitigation:** fp16 defaults, 1024 px max edge guard, sequential use
    guidance, per-model `unload()`, and the 5.5 GiB VRAM ceiling in smoke tests.
- **Risk:** FGAesQ packaging inconsistency.
  - **Mitigation:** Lock known-good source revision by commit SHA; add smoke
    test gate.

## Future Scope (v2+)

### HPSv3
HPSv3 uses a large vision-language backbone (Qwen2-VL-7B class) and requires
approximately 8–14 GB VRAM at fp16, exceeding the 6 GB primary dev target.
It is deferred to v2.

When implemented in v2, the expected interface would follow the same pattern:
- `score_hpsv3(image_path: str, prompt: str) -> HPSv3ScoreResult`
- `HPSv3ScoreResult` extends `BaseScoreResult` with `prompt: str` and
  `preference_score: float` (same shape as `HPSv2ScoreResult`).
- Smoke test gated behind `AESTHETIC_SCORING_FULL_GPU=1` env var and a
  runtime VRAM check (≥15 GiB).
- Target hardware: ≥16 GB VRAM, CUDA 12.1+.

## Change Log
- **1.0.0 (2026-05-11):** Initial implementation-ready spec; 16 GB VRAM
  target; all four models including HPSv3.
- **1.1.0 (2026-05-11):** Rewrote for 6 GB RTX 4050 primary dev tier; HPSv3
  smoke test gated; per-module `unload()` contract; VRAM and latency ceilings
  added.
- **1.2.0 (2026-05-11):** Replaced HPSv3 with HPSv2 in v1 scope. HPSv3
  moved to Future Scope (v2+). All four v1 models (LAION, PickScore, HPSv2,
  FGAesQ) are built and GPU smoke-tested on the 6 GB dev box.
