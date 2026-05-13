# SPEC.md: Image Aesthetic Scoring Library (v1)

## Document Control
- **Version:** 1.3.0
- **Status:** Draft for implementation
- **Last Updated:** 2026-05-12
- **Owner:** Project maintainer

## Objective
Build a Python library module that exposes GPU-backed, individual scoring APIs
for four models:
1. LAION-Aesthetics v2.5
2. PickScore
3. HPSv2
4. FGAesQ

The module must be importable from other Python code and return typed Python
objects from each scoring call. The library must run on two hardware tiers:
NVIDIA CUDA (primary production target) and Apple Silicon via the PyTorch MPS
backend (supported developer target — specifically the user's M1 MacBook Pro).

## Scope
- Library-only deliverable (no CLI in v1).
- Individual model functions (no required orchestration pipeline in v1).
- Deterministic, schema-stable return objects.
- **Primary production target:** NVIDIA RTX 4050, 6 GB VRAM, CUDA 12.1+.
  All four v1 models must run and pass GPU smoke tests on this card.
- **Secondary supported target:** Apple Silicon (M1-class or newer) with macOS
  and PyTorch MPS backend. All four v1 models must run on MPS (with CPU
  fallback permitted for individual ops) and pass the MPS smoke test suite.
- Sequential usage model; the caller is expected to invoke one model at a time.

## Out of Scope
- HPSv3 — requires ≥16 GB VRAM; deferred to v2 (see Future Scope).
- ComfyUI-specific adapters or nodes.
- Web service/API server.
- Distributed or multi-GPU inference.
- Full benchmark suite for every checkpoint variant.
- Windows support beyond what `torch` wheels already provide; not tested.

## Target Environments

### CUDA (primary)
- **Python:** 3.11
- **OS:** Linux
- **GPU:** NVIDIA RTX 4050, 6 GB VRAM (dev); ≥16 GB VRAM recommended (prod).
- **CUDA runtime:** 12.1+
- **Precision:** fp16 (default). bf16 permitted per module if numerically
  beneficial; documented in that module's docstring.
- **Max input edge default:** 1024 px long edge.
- **Peak VRAM ceiling:** < 5.5 GiB per model during smoke tests.

### MPS (Apple Silicon, supported dev tier)
- **Python:** 3.11
- **OS:** macOS 13+ on Apple Silicon (M1/M2/M3/M4 family).
- **GPU:** Apple Silicon integrated GPU via PyTorch MPS backend.
- **Precision:** fp16 (default). Ops that MPS does not yet implement must
  fall back to CPU via `PYTORCH_ENABLE_MPS_FALLBACK=1`, which the library
  sets automatically in `_device.py` on import when MPS is the chosen device.
- **Max input edge default:** 1024 px long edge (same as CUDA).
- **Memory ceiling:** none enforced by the library (unified memory; OS
  manages pressure). Smoke tests do not assert a VRAM budget on MPS.

### CPU (degraded fallback)
- The library must not crash when neither CUDA nor MPS is available. It
  selects CPU with fp32 precision. Smoke tests are skipped on CPU-only hosts.

## Functional Requirements

### FR-1: Module Structure
```text
aesthetic_scoring/
  __init__.py       # re-exports public API and result types
  errors.py         # custom exception classes
  types.py          # dataclass result types
  _device.py        # device detection (cuda|mps|cpu), precision selection,
                    # OOM/OOR wrapping, MPS fallback env setup
  laion.py          # score_laion, unload
  pickscore.py      # score_pickscore, unload
  hpsv2.py          # score_hpsv2, unload
  fgaesq.py         # score_fgaesq, unload
tests/
  unit/             # mocked, no GPU needed
  smoke/            # real inference (CUDA or MPS)
  fixtures/         # small sample images + prompts
```

### FR-2: Individual Scoring Functions
- `score_laion(image_path: str) -> LaionScoreResult`
- `score_pickscore(image_paths: list[str], prompt: str) -> PickScoreResult`
- `score_hpsv2(image_path: str, prompt: str) -> HPSv2ScoreResult`
- `score_fgaesq(image_path: str) -> FGAesQScoreResult`

Each module also exposes `unload()` that dereferences the cached model and
calls the device-appropriate cache-clear (`torch.cuda.empty_cache()` on CUDA,
`torch.mps.empty_cache()` on MPS, no-op on CPU). Calling `unload()` when no
model is loaded is a no-op.

### FR-3: Typed Return Objects
Each function returns a typed dataclass with fixed fields. All fields must be
JSON-serializable via `json.dumps(dataclasses.asdict(result))` without a
custom encoder.

Shared required fields:
- `image_id: str` — non-empty
- `model_name: str` — non-empty
- `model_version: str` — non-empty
- `latency_ms: float` — wall-clock inference time, > 0
- `device: str` — one of `"cuda:0"`, `"mps"`, `"cpu"`
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
Exported from `aesthetic_scoring.errors`:
- `ModelInferenceError`
- `GpuMemoryError`
- `ModelLoadError`

Rules:
- Missing file path → `FileNotFoundError`.
- Wrong argument type or empty list for PickScore → `ValueError` or `TypeError`.
- CUDA OOM during inference → `GpuMemoryError`.
- MPS out-of-memory (`MPSNDArray.mm` / allocator errors) → `GpuMemoryError`.
- Model load/init failures → `ModelLoadError`.
- Other runtime inference failures → `ModelInferenceError`.

### FR-5: Device Selection and Fallback
`_device.py` must implement:
1. `get_device()` → returns `cuda:0` if `torch.cuda.is_available()`, else
   `mps` if `torch.backends.mps.is_available() and torch.backends.mps.is_built()`,
   else `cpu`.
2. `get_precision(device)` → `"fp16"` for `cuda` and `mps`, `"fp32"` for `cpu`.
3. `empty_cache(device)` → calls the backend-appropriate cache drain.
4. On first import when the selected device is `mps`, set
   `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")` so any ops
   missing MPS kernels silently execute on CPU rather than raising.
5. `inference_guard()` context manager maps CUDA OOM → `GpuMemoryError`, MPS
   allocator errors (substring `"MPS"` and `"memory"` in the message) →
   `GpuMemoryError`, other `RuntimeError`/`Exception` → `ModelInferenceError`.

### FR-6: Resource Behavior
- Models are loaded lazily on first call.
- On CUDA, peak VRAM per model must stay < 5.5 GiB in smoke tests (dev box).
- On MPS, no VRAM ceiling is enforced; smoke tests only assert correctness
  and latency.
- Document memory-sensitive defaults (precision, max image edge, MPS fallback
  env var) in each module's docstring and in `README.md`.

## Non-Functional Requirements
- **NFR-1 Stability:** Public function signatures are stable for v1.x.
- **NFR-2 Reproducibility:** Pin dependency versions in `pyproject.toml`.
  External-repo dependencies (FGAesQ source, HPSv2) are pinned by commit SHA
  in `README.md`.
- **NFR-3 Observability:** All result objects include `latency_ms`.
- **NFR-4 Performance:**
  - **CUDA:** LAION smoke < 30 s; PickScore/HPSv2/FGAesQ smoke < 60 s each.
  - **MPS:** LAION smoke < 90 s; PickScore/HPSv2/FGAesQ smoke < 180 s each
    (3× the CUDA budget to accommodate kernel maturity and CPU fallback).
  - Weight download time is excluded from all budgets.
- **NFR-5 Installability on macOS:**
  - `pip install -e ".[dev]"` must succeed on macOS arm64 without requiring
    CUDA wheels or any package that fails to build on darwin.
  - Mac-hostile dependencies (`hpsv2` PyPI package, CUDA-only torch builds)
    must be guarded by PEP 508 environment markers so they are not resolved
    on macOS. The library's `hpsv2.py` already uses `open_clip` directly and
    does not need the `hpsv2` package at runtime; the dependency is retained
    only on non-darwin platforms.

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
    device: str        # "cuda:0" | "mps" | "cpu"
    precision: str     # "fp16" | "bf16" | "fp32"

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
the package root.

## Dependency Baseline
Pinned in `pyproject.toml`:
- `torch` — CUDA 12.1 build on Linux; default PyPI wheel (CPU+MPS capable)
  on macOS via environment markers.
- `torchvision`
- `transformers`
- `accelerate`
- `open_clip_torch`
- `Pillow`
- `numpy`
- `scikit-image`
- `huggingface_hub`
- `hpsv2` — **`sys_platform != "darwin"`** only; not installed on macOS.
- Dev: `pytest`, `pytest-mock`.

FGAesQ source is vendored in `aesthetic_scoring/fgaesq.py`; no pip dep on it.

## Verification Strategy

### Required Test Types
1. **Unit tests (mocked):** input validation, exception mapping, schema
   shape, JSON round-trip, device-selection logic (parametrized over
   `cuda`, `mps`, `cpu` using `monkeypatch`). Must pass on any machine
   including CPU-only CI and macOS arm64.
2. **Smoke tests (real inference):** one fixture-based real inference per
   model. Must pass on:
   - the RTX 4050 6 GB dev box (CUDA path), and
   - an M1-class MacBook Pro (MPS path).
   Smoke tests auto-skip when neither CUDA nor MPS is available.

### Verification Commands
```bash
# 1. Import surface check
python -c "import aesthetic_scoring; from aesthetic_scoring import score_laion, score_pickscore, score_hpsv2, score_fgaesq; from aesthetic_scoring.errors import ModelInferenceError, GpuMemoryError, ModelLoadError; from aesthetic_scoring.types import LaionScoreResult, PickScoreResult, HPSv2ScoreResult, FGAesQScoreResult"

# 2. Unit tests (no GPU required; must pass on macOS arm64 and Linux)
python -m pytest tests/unit -q

# 3. Smoke tests (auto-selects CUDA or MPS; skipped on CPU-only hosts)
python -m pytest tests/smoke -q
```

### Smoke Test Expectations

**CUDA path (RTX 4050 6 GB):**
- LAION: numeric `aesthetic_score`; < 30 s; peak VRAM < 5.5 GiB.
- PickScore: `probabilities` sum ~1.0 (±1e-3); `ranked_image_ids` length
  equals input count; < 60 s; peak VRAM < 5.5 GiB.
- HPSv2: numeric `preference_score`; < 60 s; peak VRAM < 5.5 GiB.
- FGAesQ: numeric `technical_score` and `aesthetic_score`; < 60 s;
  peak VRAM < 5.5 GiB.

**MPS path (M1 MacBook Pro):**
- LAION: numeric `aesthetic_score`; < 90 s.
- PickScore: `probabilities` sum ~1.0 (±1e-3); `ranked_image_ids` length
  equals input count; < 180 s.
- HPSv2: numeric `preference_score`; < 180 s.
- FGAesQ: numeric `technical_score` and `aesthetic_score`; < 180 s.
- No VRAM assertion on MPS.
- `result.device == "mps"` for every result.

## Acceptance Criteria
1. Package imports cleanly on Python 3.11 on both Linux and macOS arm64.
2. `pip install -e ".[dev]"` completes with exit 0 on macOS arm64 without
   attempting to install `hpsv2` or any CUDA-only wheel.
3. All four scoring functions exist with the signatures in FR-2.
4. All return types conform to the fixed typed schema in the API Contract.
5. `json.dumps(dataclasses.asdict(result))` succeeds for any result without
   a custom encoder.
6. `ModelInferenceError`, `GpuMemoryError`, `ModelLoadError` are exported
   and raised per FR-4. CUDA OOM and MPS allocator errors both raise
   `GpuMemoryError`.
7. Unit test suite passes on both Linux CPU CI and macOS arm64.
8. `_device.get_device()` returns `"mps"` on an M1 MacBook Pro with a PyTorch
   build that reports `torch.backends.mps.is_available() == True`.
9. On import, when the selected device is `mps`, the environment variable
   `PYTORCH_ENABLE_MPS_FALLBACK` is set to `"1"` (unless the caller pre-set it).
10. All four smoke tests pass on the RTX 4050 6 GB dev box; peak VRAM per
    test stays below 5.5 GiB; latency budgets in NFR-4 (CUDA) are met.
11. All four smoke tests pass on an M1 MacBook Pro using the MPS backend;
    latency budgets in NFR-4 (MPS) are met; each result's `device` field is
    `"mps"` and `precision` is `"fp16"`.
12. Smoke tests auto-skip (not fail) on a CPU-only host.
13. Per-module `unload()` is implemented; calling it when no model is loaded
    is a no-op; it uses the correct backend cache-clear for the active device.
14. `README.md` documents installation for both Linux (CUDA) and macOS
    (Apple Silicon), precision and image-size defaults, the MPS fallback
    env var, latency expectations per backend, and all pinned external-repo
    commit SHAs.

## Verification

```bash
# Import surface
python -c "import aesthetic_scoring; from aesthetic_scoring import score_laion, score_pickscore, score_hpsv2, score_fgaesq; from aesthetic_scoring.errors import ModelInferenceError, GpuMemoryError, ModelLoadError; from aesthetic_scoring.types import LaionScoreResult, PickScoreResult, HPSv2ScoreResult, FGAesQScoreResult"

# macOS install check (run on arm64 Mac)
pip install -e ".[dev]"

# Unit tests (Linux CPU CI and macOS arm64)
python -m pytest tests/unit -q

# Smoke tests (auto-selects CUDA or MPS; skips on CPU-only)
python -m pytest tests/smoke -q

# Device-selection sanity check on M1
python -c "from aesthetic_scoring._device import get_device, get_precision; d = get_device(); print(d, get_precision(d)); assert str(d) in ('mps','cuda:0','cpu')"
```

## Implementation Checklist

### pyproject.toml
- [ ] Split `torch` and `torchvision` pins with PEP 508 markers: CUDA-specific
      index or version on `sys_platform == "linux"`; default PyPI wheel on
      `sys_platform == "darwin"`.
- [ ] Gate `hpsv2` dependency with `sys_platform != "darwin"`.
- [ ] Verify `scikit-image`, `Pillow`, `numpy`, `accelerate`, `transformers`,
      `open_clip_torch`, `huggingface_hub` all resolve on macOS arm64 at the
      pinned versions.

### `aesthetic_scoring/_device.py`
- [ ] Replace the current CUDA-only `get_device()` with a three-way selector:
      `cuda:0` → `mps` → `cpu`.
- [ ] Update `get_precision()` to return `"fp16"` for both `cuda` and `mps`;
      `"fp32"` for `cpu`.
- [ ] Add `empty_cache(device)` helper that dispatches to
      `torch.cuda.empty_cache()`, `torch.mps.empty_cache()`, or no-op.
- [ ] On module import, if the selected device is `mps`, call
      `os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")`.
- [ ] Extend `inference_guard()` to catch MPS allocator errors (detect by
      message containing `"MPS"` and (`"memory"` or `"out of memory"`)) and
      raise `GpuMemoryError`.
- [ ] Guard `vram_used_gib()` and `reset_peak_vram()` so they no-op on MPS
      and CPU without raising.

### Per-model modules (`laion.py`, `pickscore.py`, `hpsv2.py`, `fgaesq.py`)
- [ ] Replace any hard-coded `"cuda"` device strings with `get_device()`.
- [ ] Ensure `.to(device)` and `.to(dtype)` calls work when device is `mps`
      (no `pin_memory=True` on DataLoader; no `non_blocking=True` cross-device
      copies into MPS).
- [ ] Replace any `torch.cuda.empty_cache()` in `unload()` with
      `_device.empty_cache(self._device)`.
- [ ] Populate `device` and `precision` fields in each result from the
      actual device used at inference time.
- [ ] FGAesQ: confirm the vendored model runs on MPS at fp32 (current behavior)
      or document the precision choice in the module docstring.
- [ ] HPSv2: confirm `open_clip` HPSv2.1 checkpoint loads and runs on MPS at
      fp16; document any op that falls back to CPU via the env var.

### Tests
- [ ] `tests/conftest.py`: add a `device_available` fixture returning the
      active backend string; add an `xfail`/`skip` marker for CPU-only hosts.
- [ ] `tests/unit/test_device.py`: parametrize over `cuda`, `mps`, `cpu`
      using `monkeypatch` on `torch.cuda.is_available` and
      `torch.backends.mps.is_available`; assert `get_device`, `get_precision`,
      and MPS fallback env var behavior.
- [ ] `tests/smoke/conftest.py` (or equivalent): skip the whole smoke module
      when neither CUDA nor MPS is available.
- [ ] `tests/smoke/test_*`: assert device-specific latency budgets (≤ 30/60 s
      on CUDA, ≤ 90/180 s on MPS) and assert the VRAM ceiling only on CUDA.
- [ ] Add an MPS-only assertion in each smoke test: when `device == "mps"`,
      `result.device == "mps"` and `result.precision == "fp16"`.

### README
- [ ] Add a "Running on Apple Silicon (M1/M2/M3)" section covering:
      install command (no `hpsv2`, no CUDA torch), `PYTORCH_ENABLE_MPS_FALLBACK`
      (set automatically), expected latency ranges, and that smoke tests run
      on MPS.
- [ ] Update the Requirements table to list both CUDA and MPS tiers.
- [ ] Keep all external-repo commit SHAs current.

## Risks and Mitigations
- **Risk:** MPS lacks a required kernel for an op inside CLIP/HPSv2/FGAesQ.
  - **Mitigation:** `PYTORCH_ENABLE_MPS_FALLBACK=1` set automatically; smoke
    test validates end-to-end execution; document any observed CPU-fallback
    ops in the module docstring.
- **Risk:** fp16 numerical drift on MPS producing scores outside documented
  ranges.
  - **Mitigation:** Smoke tests assert `isfinite` and basic range sanity
    (not exact parity with CUDA). If a model is unstable at fp16 on MPS,
    that module may switch to fp32 and document it.
- **Risk:** `hpsv2` PyPI package fails to install on macOS (known — ships
  CUDA-specific helpers).
  - **Mitigation:** PEP 508 marker `sys_platform != "darwin"` excludes it;
    the library bypasses the package at runtime already.
- **Risk:** HuggingFace Hub checkpoint downloads are slow or blocked on
  first run.
  - **Mitigation:** Weight-download time excluded from latency budgets;
    document a warm-cache prerequisite for smoke tests.

## Future Scope (v2+)

### HPSv3
HPSv3 uses a Qwen2-VL-7B-class backbone and requires ~8–14 GB VRAM at fp16,
exceeding the 6 GB primary CUDA dev target. It is deferred to v2.

When implemented, the interface mirrors HPSv2:
- `score_hpsv3(image_path: str, prompt: str) -> HPSv3ScoreResult`
- `HPSv3ScoreResult` extends `BaseScoreResult` with `prompt: str` and
  `preference_score: float`.
- Smoke test gated behind `AESTHETIC_SCORING_FULL_GPU=1` and a runtime
  VRAM check (≥15 GiB on CUDA; MPS feasibility re-evaluated at v2 time).

## Change Log
- **1.0.0 (2026-05-11):** Initial implementation-ready spec; 16 GB VRAM
  target; all four models including HPSv3.
- **1.1.0 (2026-05-11):** Rewrote for 6 GB RTX 4050 primary dev tier;
  HPSv3 smoke test gated; per-module `unload()` contract; VRAM and latency
  ceilings added.
- **1.2.0 (2026-05-11):** Replaced HPSv3 with HPSv2 in v1 scope. HPSv3
  moved to Future Scope (v2+). All four v1 models (LAION, PickScore, HPSv2,
  FGAesQ) are built and GPU smoke-tested on the 6 GB dev box.
- **1.3.0 (2026-05-12):** Added Apple Silicon (MPS) as a supported dev
  target alongside CUDA. `_device.py` now selects `cuda → mps → cpu` and
  auto-sets `PYTORCH_ENABLE_MPS_FALLBACK=1`. Mac-hostile dependencies
  (`hpsv2`) gated by PEP 508 markers. MPS smoke-test latency budgets set
  at 3× CUDA; no MPS VRAM ceiling. New acceptance criteria 1, 2, 8, 9, 11,
  12 cover the Mac path. No VRAM ceiling enforced on MPS.
