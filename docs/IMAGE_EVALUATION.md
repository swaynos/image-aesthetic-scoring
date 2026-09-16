# Image Evaluation

`image_evaluation.evaluate_image` runs OWL detection and selected raw scorers
sequentially, returning an `ImageEvaluationReport`. It does not assign a final
score.

```text
image + detection queries + optional evaluation prompt
  -> OWL-ViT evidence, then OWLv2 evidence
  -> selected raw aesthetic/preference/alignment scores
  -> evidence report
  -> caller-defined policy and final score
```

The report keeps raw boxes, labels, confidence, model IDs, queries, device and
precision metadata, plus the unchanged `ImageScoreReport`. A policy can be
recomputed from saved evidence when only its weighting changes.

## Policy Rules

A future policy must state:

- Which detection labels, counts, locations, sizes, and score thresholds apply.
- How raw model scores are normalized and weighted.
- Its version and an explanation for every final score.
- How no qualifying detection differs from detector failure.
- Which evidence changes require fresh inference, such as a new query or a
  lower saved confidence threshold.

Do not treat a failed model run as an object absence. Let the error stop the
evaluation or record it as incomplete before any policy is applied.
