from aesthetic_scoring.types import ImageScoreReport, LaionScoreResult
from image_evaluation.pipeline import evaluate_image
from object_detection.types import Detection, OwlDetectionResult


def test_evaluation_preserves_raw_detection_and_scoring_evidence(monkeypatch):
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

    monkeypatch.setattr(
        "image_evaluation.pipeline.compare_owl_models", lambda path, queries: [detection]
    )
    monkeypatch.setattr(
        "image_evaluation.pipeline.score_images", lambda paths, **kwargs: score_report
    )

    report = evaluate_image(
        "image.png", detection_queries=["person"], scoring_models=["laion"]
    )

    assert report.image_id == "image.png"
    assert report.detections == [detection]
    assert report.scores is score_report
    assert report.status == "complete"
    assert report.to_dict()["detections"][0]["detections"][0]["label"] == "person"
