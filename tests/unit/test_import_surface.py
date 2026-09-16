"""Test the public import surface of the package."""


def test_package_imports():
    import aesthetic_scoring  # noqa


def test_scoring_functions_exported():
    from aesthetic_scoring import (
        score_clipscore,
        score_fgaesq,
        score_hpsv2,
        score_laion,
        score_pickscore,
        score_images,
    )
    assert callable(score_laion)
    assert callable(score_pickscore)
    assert callable(score_hpsv2)
    assert callable(score_fgaesq)
    assert callable(score_clipscore)
    assert callable(score_images)


def test_result_types_exported():
    from aesthetic_scoring import (
        LaionScoreResult,
        PickScoreResult,
        HPSv2ScoreResult,
        CLIPScoreResult,
        FGAesQScoreResult,
        ImageScoreReport,
    )


def test_exceptions_exported():
    from aesthetic_scoring.errors import ModelInferenceError, GpuMemoryError, ModelLoadError


def test_types_importable_from_submodule():
    from aesthetic_scoring.types import (
        BaseScoreResult,
        LaionScoreResult,
        PickScoreResult,
        HPSv2ScoreResult,
        CLIPScoreResult,
        FGAesQScoreResult,
    )


def test_object_detection_imports_and_exports():
    import object_detection
    from object_detection import (
        Detection,
        OwlDetectionResult,
        OWLVIT_BASE,
        OWLV2_BASE,
        detect_owl,
        compare_owl_models,
        unload,
    )
    from object_detection.types import Detection as DetType, OwlDetectionResult as ResType

    assert callable(detect_owl)
    assert callable(compare_owl_models)
    assert callable(unload)
    assert isinstance(OWLVIT_BASE, str)
    assert isinstance(OWLV2_BASE, str)
    assert DetType is Detection
    assert ResType is OwlDetectionResult


def test_image_evaluation_imports_and_exports():
    import image_evaluation
    from image_evaluation import ImageEvaluationReport, evaluate_image
    from image_evaluation.types import ImageEvaluationReport as ReportType

    assert callable(evaluate_image)
    assert ReportType is ImageEvaluationReport

