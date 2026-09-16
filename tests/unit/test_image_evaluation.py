import dataclasses
import json
import pytest
from aesthetic_scoring.types import ImageScoreReport, LaionScoreResult
from image_evaluation.pipeline import evaluate_image
from object_detection.types import Detection, OwlDetectionResult


def test_evaluation_preserves_raw_detection_and_scoring_evidence(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"dummy")

    detection = OwlDetectionResult(
        image_id="image.png",
        model_id="google/owlv2-base-patch16-ensemble",
        latency_ms=12.0,
        device="cuda:0",
        precision="fp16",
        queries=["person"],
        image_width=100,
        image_height=80,
        detections=[Detection(label="person", score=0.8, box_xyxy=(1.0, 2.0, 30.0, 40.0))],
    )
    score_report = ImageScoreReport(
        image_ids=["image.png"],
        evaluation_prompt=None,
        results={
            "laion": [
                LaionScoreResult(
                    image_id="image.png",
                    model_name="laion",
                    model_version="v2.5",
                    latency_ms=4.0,
                    device="cuda:0",
                    precision="fp16",
                    aesthetic_score=6.5,
                    score_scale="1-10",
                )
            ]
        },
    )

    captured_kwargs = {}

    def mock_compare(path, queries, **kwargs):
        captured_kwargs.update(kwargs)
        return [detection]

    monkeypatch.setattr("image_evaluation.pipeline.compare_owl_models", mock_compare)
    monkeypatch.setattr("image_evaluation.pipeline.score_images", lambda paths, **kwargs: score_report)

    # Test positional call with detection_kwargs forwarded
    report = evaluate_image(
        str(image_path),
        ["person"],
        scoring_models=["laion"],
        threshold=0.2,
    )

    assert report.image_id == "image.png"
    assert report.detections == [detection]
    assert report.scores is score_report
    assert report.status == "complete"
    assert captured_kwargs == {"threshold": 0.2}

    # Verify JSON serialization
    serialized_dict = report.to_dict()
    assert serialized_dict["detections"][0]["detections"][0]["label"] == "person"
    json_str_dict = json.dumps(serialized_dict)
    assert "person" in json_str_dict

    json_str_asdict = json.dumps(dataclasses.asdict(report))
    assert "person" in json_str_asdict


def test_evaluate_image_early_validation(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"dummy")

    with pytest.raises(TypeError, match="image_path must be str"):
        evaluate_image(123, ["person"])

    with pytest.raises(FileNotFoundError, match="Image not found"):
        evaluate_image("nonexistent.png", ["person"])

    with pytest.raises(ValueError, match="Unknown model"):
        evaluate_image(str(image_path), ["person"], scoring_models=["unknown"])

    with pytest.raises(ValueError, match="evaluation_prompt is required"):
        evaluate_image(str(image_path), ["person"], scoring_models=["pickscore"])

