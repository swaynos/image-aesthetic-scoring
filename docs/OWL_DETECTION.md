# OWL Detection

`object_detection` provides local, text-conditioned object detection for
evaluation evidence. It is not an aesthetic or identity scorer.

## Supported Models

- `google/owlvit-base-patch32`
- `google/owlv2-base-patch16-ensemble`

Both models load through Hugging Face Transformers. Weights download on first
use and remain in the Hugging Face cache. The detector runs in FP16 on CUDA and
FP32 on CPU.

## API

```python
from object_detection import detect_owl

result = detect_owl(
    "candidate.png",
    ["person", "bicycle"],
    threshold=0.2,
    nms_threshold=0.3,
)
```

Each result records the source filename, model ID, latency, device, precision,
queries, source-image dimensions, and detections. A detection contains its
queried label, confidence, and `[x1, y1, x2, y2]` box in source-image pixels.

The detector applies class-aware non-maximum suppression after the model's raw
post-processing. A missing detection means no box met the chosen threshold; it
does not prove the object is absent.

## Comparison And Memory

`compare_owl_models` always runs OWL-ViT Base, releases it, then runs OWLv2
Base. This is the supported mode for a 6 GiB GPU. Do not keep a generation
model resident during the comparison.

Before calling OWLv2 Large supported, record a real run's device name, PyTorch
version, input dimensions, precision, batch size, allocated peak, reserved
peak, and latency. PyTorch's allocated-memory measurement alone does not cover
display or driver memory on a laptop GPU.
