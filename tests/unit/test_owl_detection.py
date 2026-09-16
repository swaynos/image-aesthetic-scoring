import torch
import pytest
from PIL import Image

from object_detection.owl import OWLV2_BASE, compare_owl_models, detect_owl
from object_detection.types import Detection, OwlDetectionResult


def _result(model_id: str) -> OwlDetectionResult:
    return OwlDetectionResult(
        image_id="image.png",
        model_id=model_id,
        latency_ms=10.0,
        device="cuda:0",
        precision="fp16",
        queries=["person"],
        image_width=100,
        image_height=80,
        detections=[
            Detection(label="person", score=0.9, box_xyxy=(1.0, 2.0, 30.0, 40.0))
        ],
    )


def test_compare_runs_models_in_order_and_unloads(monkeypatch):
    calls = []

    def detect(image_path, queries, model, **kwargs):
        calls.append(("detect", model, image_path, queries))
        return _result(model)

    monkeypatch.setattr("object_detection.owl.detect_owl", detect)
    monkeypatch.setattr("object_detection.owl.unload", lambda: calls.append(("unload",)))

    results = compare_owl_models("image.png", ["person"])

    assert [result.model_id for result in results] == [
        "google/owlvit-base-patch32",
        "google/owlv2-base-patch16-ensemble",
    ]
    assert calls == [
        ("detect", "google/owlvit-base-patch32", "image.png", ["person"]),
        ("unload",),
        ("detect", "google/owlv2-base-patch16-ensemble", "image.png", ["person"]),
        ("unload",),
    ]


def test_detection_result_is_json_serializable():
    result = _result("google/owlvit-base-patch32")

    assert result.to_dict() == {
        "image_id": "image.png",
        "model_id": "google/owlvit-base-patch32",
        "latency_ms": 10.0,
        "device": "cuda:0",
        "precision": "fp16",
        "queries": ["person"],
        "image_width": 100,
        "image_height": 80,
        "detections": [
            {"label": "person", "score": 0.9, "box_xyxy": [1.0, 2.0, 30.0, 40.0]}
        ],
    }


def test_detect_owl_normalizes_and_filters_huggingface_output(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (100, 80)).save(image_path)

    class Processor:
        def __call__(self, **kwargs):
            return {
                "pixel_values": torch.zeros((1, 3, 8, 8)),
                "input_ids": torch.ones((1, 2), dtype=torch.long),
            }

        def post_process_grounded_object_detection(self, *args, **kwargs):
            assert kwargs["text_labels"] == [["person"]]
            return [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 30.0, 40.0], [2.0, 3.0, 29.0, 39.0]]),
                    "scores": torch.tensor([0.9, 0.8]),
                    "text_labels": ["person", "person"],
                }
            ]

    class Model:
        def __call__(self, **kwargs):
            assert set(kwargs) == {"pixel_values", "input_ids"}
            return object()

    monkeypatch.setattr(
        "object_detection.owl._load",
        lambda model: (Model(), Processor(), torch.device("cpu"), "fp32", torch.float32),
    )

    result = detect_owl(str(image_path), ["person"], OWLV2_BASE, threshold=0.5)

    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection.label == "person"
    assert detection.score == pytest.approx(0.9)
    assert detection.box_xyxy == (1.0, 2.0, 30.0, 40.0)


def test_detect_owl_validation(tmp_path):
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"placeholder")

    with pytest.raises(TypeError, match="image_path must be str"):
        detect_owl(123, ["person"])

    with pytest.raises(FileNotFoundError, match="Image not found"):
        detect_owl("nonexistent.png", ["person"])

    with pytest.raises(ValueError, match="Unsupported OWL model"):
        detect_owl(str(image_path), ["person"], model="unsupported-model")

    with pytest.raises(ValueError, match="queries must be a non-empty list"):
        detect_owl(str(image_path), [])

    with pytest.raises(ValueError, match="queries must be a non-empty list"):
        detect_owl(str(image_path), "not-a-list")

    with pytest.raises(ValueError, match="queries must contain only non-empty strings"):
        detect_owl(str(image_path), ["person", "   "])

    with pytest.raises(ValueError, match="threshold and nms_threshold"):
        detect_owl(str(image_path), ["person"], threshold=-0.1)

    with pytest.raises(ValueError, match="threshold and nms_threshold"):
        detect_owl(str(image_path), ["person"], nms_threshold=1.5)

    with pytest.raises(ValueError, match="max_detections must be positive"):
        detect_owl(str(image_path), ["person"], max_detections=0)


def test_detect_owl_returns_empty_when_no_detections_pass_threshold(monkeypatch, tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (50, 50)).save(image_path)

    class Processor:
        def __call__(self, **kwargs):
            return {"pixel_values": torch.zeros((1, 3, 8, 8)), "input_ids": torch.ones((1, 2), dtype=torch.long)}

        def post_process_grounded_object_detection(self, *args, **kwargs):
            return [{"boxes": torch.empty((0, 4)), "scores": torch.empty(0), "text_labels": []}]

    monkeypatch.setattr(
        "object_detection.owl._load",
        lambda model: (lambda **kwargs: object(), Processor(), torch.device("cpu"), "fp32", torch.float32),
    )

    result = detect_owl(str(image_path), ["person"], OWLV2_BASE, threshold=0.5)
    assert result.detections == []


def test_compare_owl_models_ignores_model_in_kwargs(monkeypatch):
    calls = []

    def detect(image_path, queries, model, **kwargs):
        calls.append(model)
        assert "model" not in kwargs
        return _result(model)

    monkeypatch.setattr("object_detection.owl.detect_owl", detect)
    monkeypatch.setattr("object_detection.owl.unload", lambda: None)

    results = compare_owl_models("image.png", ["person"], model="overridden", threshold=0.3)
    assert len(results) == 2
    assert calls == ["google/owlvit-base-patch32", "google/owlv2-base-patch16-ensemble"]


def test_unload_clears_module_state(monkeypatch):
    import object_detection.owl as owl_mod

    owl_mod._model = object()
    owl_mod._processor = object()
    owl_mod._model_id = "test"
    owl_mod._device = torch.device("cpu")
    owl_mod._precision = "fp32"

    owl_mod.unload()

    assert owl_mod._model is None
    assert owl_mod._processor is None
    assert owl_mod._model_id is None
    assert owl_mod._device is None
    assert owl_mod._precision is None

