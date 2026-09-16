"""GPU smoke test for OWL-family object detection models.

Tests google/owlvit-base-patch32, google/owlv2-base-patch16-ensemble,
serial comparison with memory release, and the image evaluation pipeline.
Skipped automatically when neither CUDA nor MPS is available.
"""
from __future__ import annotations

import dataclasses
import json
import time
import pytest
import torch

PHOTO_A = "tests/fixtures/photo_a.jpg"
QUERIES = ["gradient", "color", "background", "sky"]
PROMPT = "a vibrant colorful gradient abstract image"

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    and not (torch.backends.mps.is_available() and torch.backends.mps.is_built()),
    reason="No GPU backend (CUDA or MPS) available",
)


def test_owlvit_base_smoke():
    from object_detection.owl import OWLVIT_BASE, _load, detect_owl, unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()

    # Pre-warm: trigger weight download/load before the timed section
    _load(OWLVIT_BASE)

    reset_peak_vram()
    t0 = time.perf_counter()

    result = detect_owl(PHOTO_A, QUERIES, OWLVIT_BASE, threshold=0.01)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    # Correctness
    assert result.image_id == "photo_a.jpg"
    assert result.model_id == OWLVIT_BASE
    assert result.latency_ms > 0
    assert result.image_width == 512
    assert result.image_height == 512
    assert result.queries == QUERIES
    assert isinstance(result.detections, list)
    for det in result.detections:
        assert det.label in QUERIES
        assert 0.0 <= det.score <= 1.0
        assert len(det.box_xyxy) == 4

    json.dumps(dataclasses.asdict(result))

    if is_cuda:
        assert elapsed < 60, f"OWL-ViT smoke took {elapsed:.1f}s, expected < 60s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"
        assert result.device == "cuda:0"
        assert result.precision == "fp16"

    print(
        f"\nOWL-ViT Base detections: {len(result.detections)}, "
        f"device: {result.device}, precision: {result.precision}, "
        f"latency: {result.latency_ms:.1f}ms, VRAM/peak: {peak_gib:.2f} GiB"
    )


def test_owlv2_base_smoke():
    from object_detection.owl import OWLV2_BASE, _load, detect_owl, unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()

    # Pre-warm: trigger weight download/load before the timed section
    _load(OWLV2_BASE)

    reset_peak_vram()
    t0 = time.perf_counter()

    result = detect_owl(PHOTO_A, QUERIES, OWLV2_BASE, threshold=0.01)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    # Correctness
    assert result.image_id == "photo_a.jpg"
    assert result.model_id == OWLV2_BASE
    assert result.latency_ms > 0
    assert result.image_width == 512
    assert result.image_height == 512
    assert result.queries == QUERIES
    assert isinstance(result.detections, list)
    for det in result.detections:
        assert det.label in QUERIES
        assert 0.0 <= det.score <= 1.0
        assert len(det.box_xyxy) == 4

    json.dumps(dataclasses.asdict(result))

    if is_cuda:
        assert elapsed < 60, f"OWLv2 Base smoke took {elapsed:.1f}s, expected < 60s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"
        assert result.device == "cuda:0"
        assert result.precision == "fp16"

    print(
        f"\nOWLv2 Base detections: {len(result.detections)}, "
        f"device: {result.device}, precision: {result.precision}, "
        f"latency: {result.latency_ms:.1f}ms, VRAM/peak: {peak_gib:.2f} GiB"
    )


def test_compare_owl_models_smoke():
    from object_detection.owl import OWLVIT_BASE, OWLV2_BASE, compare_owl_models, unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()

    reset_peak_vram()
    t0 = time.perf_counter()

    results = compare_owl_models(PHOTO_A, QUERIES, threshold=0.01)

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    assert len(results) == 2
    assert results[0].model_id == OWLVIT_BASE
    assert results[1].model_id == OWLV2_BASE

    for result in results:
        assert result.image_id == "photo_a.jpg"
        assert result.latency_ms > 0
        json.dumps(dataclasses.asdict(result))

    if is_cuda:
        assert elapsed < 90, f"compare_owl_models took {elapsed:.1f}s, expected < 90s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"

    print(
        f"\ncompare_owl_models results: {len(results)}, "
        f"elapsed: {elapsed:.2f}s, VRAM/peak: {peak_gib:.2f} GiB"
    )


def test_image_evaluation_smoke():
    from image_evaluation.pipeline import evaluate_image
    from object_detection.owl import OWLVIT_BASE, OWLV2_BASE, unload
    from aesthetic_scoring._device import reset_peak_vram, vram_used_gib

    is_cuda = torch.cuda.is_available()

    reset_peak_vram()
    t0 = time.perf_counter()

    report = evaluate_image(
        PHOTO_A,
        QUERIES,
        evaluation_prompt=PROMPT,
        scoring_models=["laion"],
        threshold=0.01,
    )

    elapsed = time.perf_counter() - t0
    peak_gib = vram_used_gib()

    unload()

    assert report.image_id == "photo_a.jpg"
    assert report.status == "complete"
    assert len(report.detections) == 2
    assert report.detections[0].model_id == OWLVIT_BASE
    assert report.detections[1].model_id == OWLV2_BASE
    assert report.scores.image_ids == ["photo_a.jpg"]
    assert "laion" in report.scores.results
    assert len(report.scores.results["laion"]) == 1

    json.dumps(dataclasses.asdict(report))

    if is_cuda:
        assert elapsed < 90, f"evaluate_image took {elapsed:.1f}s, expected < 90s (CUDA)"
        assert peak_gib < 5.5, f"Peak VRAM {peak_gib:.2f} GiB exceeded 5.5 GiB"

    print(
        f"\nevaluate_image status: {report.status}, detections: {len(report.detections)}, "
        f"scores: {list(report.scores.results.keys())}, elapsed: {elapsed:.2f}s, VRAM/peak: {peak_gib:.2f} GiB"
    )
